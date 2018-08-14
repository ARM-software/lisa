# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

from math import isnan

from test import LisaTest, experiment_test
from unittest import SkipTest
from trace import Trace

from bart.common.Utils import area_under_curve
from energy_model import EnergyModel, EnergyModelCapacityError
from perf_analysis import PerfAnalysis

WORKLOAD_PERIOD_MS =  16
SET_IS_BIG_LITTLE = True
SET_INITIAL_TASK_UTIL = True

class _EnergyModelTest(LisaTest):
    """
    "Abstract" base class for generic EAS tests using the EnergyModel class

    Subclasses should provide a .workloads member to populate the 'wloads' field
    of the experiments_conf for the Executor. A set of helper methods are
    provided for making assertions about behaviour, most importantly the _test*
    methods which make assertions in a generic way.
    """

    test_conf = {
        "ftrace" : {
            "events" : [
                "sched_overutilized",
                "sched_energy_diff",
                "sched_load_avg_task",
                "sched_load_avg_cpu",
                "sched_migrate_task",
                "sched_switch",
                "cpu_frequency",
                "cpu_idle",
                "cpu_capacity",
            ],
        },
        "modules": ["cgroups"],
    }

    negative_slack_allowed_pct = 15
    """Percentage of RT-App task activations with negative slack allowed"""

    energy_est_threshold_pct = 5
    """
    Allowed margin for error in estimated energy cost for task placement,
    compared to optimal placment.
    """

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(_EnergyModelTest, cls).runExperiments(*args, **kwargs)

        cls._log = logging.getLogger('_EnergyModelTest')

    @classmethod
    def _getExperimentsConf(cls, test_env):
        if not test_env.nrg_model:
            try:
                test_env.nrg_model = EnergyModel.from_target(test_env.target)
            except Exception as e:
                raise SkipTest(
                    'This test requires an EnergyModel for the platform. '
                    'Either provide one manually or ensure it can be read '
                    'from the filesystem: {}'.format(e))

        conf = {
            'tag' : 'energy_aware',
            'flags' : ['ftrace', 'freeze_userspace'],
        }

        if 'cpufreq' in test_env.target.modules:
            available_govs = test_env.target.cpufreq.list_governors(0)
            if 'schedutil' in available_govs:
                conf['cpufreq'] = {'governor' : 'schedutil'}
            elif 'sched' in available_govs:
                conf['cpufreq'] = {'governor' : 'sched'}

        return {
            'wloads' : cls.workloads,
            'confs' : [conf],
        }

    @classmethod
    def _experimentsInit(cls, *args, **kwargs):
        super(_EnergyModelTest, cls)._experimentsInit(*args, **kwargs)

        if SET_IS_BIG_LITTLE:
            # This flag doesn't exist on mainline-integration kernels, so
            # don't worry if the file isn't present (hence verify=False)
            cls.target.write_value(
                "/proc/sys/kernel/sched_is_big_little", 1, verify=False)

        if SET_INITIAL_TASK_UTIL:
            # This flag doesn't exist on all kernels, so don't worry if the file
            # isn't present (hence verify=False)
            cls.target.write_value(
                "/proc/sys/kernel/sched_initial_task_util", 1024, verify=False)


    def get_task_utils_df(self, experiment):
        """
        Get a DataFrame with the *expected* utilization of each task over time

        :param experiment: The :class:Experiment to examine
        :returns: A Pandas DataFrame with a column for each task, showing how
                  the utilization of that task varies over time
        """
        util_scale = self.te.nrg_model.capacity_scale

        transitions = {}
        def add_transition(time, task, util):
            if time not in transitions:
                transitions[time] = {task: util}
            else:
                transitions[time][task] = util

        # First we'll build a dict D {time: {task_name: util}} where D[t][n] is
        # the expected utilization of task n from time t.
        for task, params in experiment.wload.params['profile'].iteritems():
            time = self.get_start_time(experiment) + params['delay']
            add_transition(time, task, 0)
            for _ in range(params.get('loops', 1)):
                for phase in params['phases']:
                    util = (phase.duty_cycle_pct * util_scale / 100.)
                    add_transition(time, task, util)
                    time += phase.duration_s
            add_transition(time, task, 0)

        index = sorted(transitions.keys())
        df = pd.DataFrame([transitions[k] for k in index], index=index)
        return df.fillna(method='ffill')

    def get_task_cpu_df(self, experiment):
        """
        Get a DataFrame mapping task names to the CPU they ran on

        Use the sched_switch trace event to find which CPU each task ran
        on. Does not reflect idleness - tasks not running are shown as running
        on the last CPU they woke on.

        :param experiment: The :class:Experiment to examine
        :returns: A Pandas DataFrame with a column for each task, showing the
                  CPU that the task was "on" at each moment in time
        """
        tasks = experiment.wload.tasks.keys()
        trace = self.get_trace(experiment)

        df = trace.ftrace.sched_switch.data_frame[['next_comm', '__cpu']]
        df = df[df['next_comm'].isin(tasks)]
        df = df.pivot(index=df.index, columns='next_comm').fillna(method='ffill')
        cpu_df = df['__cpu']
        # Drop consecutive duplicates
        cpu_df = cpu_df[(cpu_df.shift(+1) != cpu_df).any(axis=1)]
        return cpu_df

    def _sort_power_df_columns(self, df):
        """
        Helper method to re-order the columns of a power DataFrame

        This has no significance for code, but when examining DataFrames by hand
        they are easier to understand if the columns are in a logical order.
        """
        node_cpus = [node.cpus for node in self.te.nrg_model.root.iter_nodes()]
        return pd.DataFrame(df, columns=[c for c in node_cpus if c in df])

    def get_power_df(self, experiment):
        """
        Considering only the task placement, estimate power usage over time

        Examine a trace and use :meth:EnergyModel.estimate_from_cpu_util to get
        a DataFrame showing the estimated power usage over time. This assumes
        perfect cpuidle and cpufreq behaviour.

        :param experiment: The :class:Experiment to examine
        :returns: A Pandas DataFrame with a column node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) Shows
                  the estimated power over time.
        """
        task_cpu_df = self.get_task_cpu_df(experiment)
        task_utils_df = self.get_task_utils_df(experiment)

        tasks = experiment.wload.tasks.keys()

        # Create a combined DataFrame with the utilization of a task and the CPU
        # it was running on at each moment. Looks like:
        #                       utils                  cpus
        #          task_wmig0 task_wmig1 task_wmig0 task_wmig1
        # 2.375056      102.4      102.4        NaN        NaN
        # 2.375105      102.4      102.4        2.0        NaN

        df = pd.concat([task_utils_df, task_cpu_df],
                       axis=1, keys=['utils', 'cpus'])
        df = df.sort_index().fillna(method='ffill')
        nrg_model = self.executor.te.nrg_model

        # Now make a DataFrame with the estimated power at each moment.
        def est_power(row):
            cpu_utils = [0 for cpu in nrg_model.cpus]
            for task in tasks:
                cpu = row['cpus'][task]
                util = row['utils'][task]
                if not isnan(cpu):
                    cpu_utils[int(cpu)] += util
            power = nrg_model.estimate_from_cpu_util(cpu_utils)
            columns = power.keys()
            return pd.Series([power[c] for c in columns], index=columns)
        return self._sort_power_df_columns(df.apply(est_power, axis=1))

    def get_expected_power_df(self, experiment):
        """
        Estimate *optimal* power usage over time

        Examine a trace and use :meth:get_optimal_placements and
        :meth:EnergyModel.estimate_from_cpu_util to get a DataFrame showing the
        estimated power usage over time under ideal EAS behaviour.

        :meth:get_optimal_placements returns several optimal placements. They
        are usually equivalent, but can be drastically different in some cases.
        Currently only one of those placements is used (the first in the list).

        :param experiment: The :class:Experiment to examine
        :returns: A Pandas DataFrame with a column each node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) and a
                  "power" column with the sum of other columns. Shows the
                  estimated *optimal* power over time.
        """
        task_utils_df = self.get_task_utils_df(experiment)
        nrg_model = self.te.nrg_model

        data = []
        index = []

        def exp_power(row):
            task_utils = row.to_dict()
            expected_utils = nrg_model.get_optimal_placements(task_utils)[0]
            power = nrg_model.estimate_from_cpu_util(expected_utils)
            columns = power.keys()

            # Assemble a dataframe to plot the expected utilization
            data.append(expected_utils)
            index.append(row.name)

            return pd.Series([power[c] for c in columns], index=columns)

        res_df = self._sort_power_df_columns(
            task_utils_df.apply(exp_power, axis=1))

        self.plot_expected_util(experiment, pd.DataFrame(data, index=index))

        return res_df

    def plot_expected_util(self, experiment, util_df):
        """
        Create a plot of the expected per-CPU utilization for the experiment
        The plot is then outputted to the test results directory.

        :param experiment: The :class:Experiment to examine
        :param util_df: A Pandas Dataframe with a column per CPU giving their
                        (expected) utilization at each timestamp.
        """

        nrg_model = self.te.nrg_model
        trace = self.get_trace(experiment)

        fig, ax = plt.subplots(len(nrg_model.cpus), 1, figsize=(16, 1.8 * len(nrg_model.cpus)))
        fig.suptitle('Per-CPU expected utilization')

        # Check if big.LITTLE data is available for a more detailled plot
        if trace.has_big_little:
            little_cap = self.te.platform['nrg_model']['little']['cpu']['cap_max']
            little_cpus = self.te.platform['clusters']['little']

        for cpu in nrg_model.cpus:
            tdf = util_df[cpu]

            # big.LITTLE-specific beautifying
            if trace.has_big_little and cpu in little_cpus:
                ax[cpu].set_ylim((0, little_cap))
                color = 'blue'
            else:
                ax[cpu].set_ylim((0, 1024))
                color = 'red'

            tdf.plot(ax=ax[cpu], drawstyle='steps-post', title="CPU{}".format(cpu), color=color)
            ax[cpu].set_ylabel('Utilization')

            # Grey-out areas where utilization == 0
            ffill = False
            prev = 0.0
            for time, util in tdf.iteritems():
                if ffill:
                    ax[cpu].axvspan(prev, time, facecolor='gray', alpha=0.1, linewidth=0.0)
                    ffill = False
                if util == 0.0:
                    ffill = True

                prev = time

        figname = os.path.join(experiment.out_dir, 'expected_placement.png')
        pl.savefig(figname, bbox_inches='tight')
        plt.close()

    def _test_slack(self, experiment, tasks):
        """
        Assert that the RTApp workload was given enough performance

        Use :class:PerfAnalysis to find instances where the experiment's RT-App
        workload wasn't able to complete its activations (i.e. its reported
        "slack" was negative). Assert that this happened less that
        ``negative_slack_allowed_pct`` percent of the time.

        :meth:_test_task_placement asserts that estimated energy usage was
        low. That will pass for runs where too *little* energy was used,
        compromising performance. This method provides a separate test to
        counteract that problem.
        """

        pa = PerfAnalysis(experiment.out_dir)
        for task in tasks:
            slack = pa.df(task)["Slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100. / len(slack)

            msg = 'task {} missed {}% of activations ({}% allowed)'.format(
                    task, bad_activations_pct, self.negative_slack_allowed_pct)

            if bad_activations_pct > self.negative_slack_allowed_pct:
                raise AssertionError(msg)
            else:
                self._log.info(msg)

    def _test_task_placement(self, experiment, tasks):
        """
        Test that task placement was energy-efficient

        Use :meth:get_expected_power_df and :meth:get_power_df to estimate
        optimal and observed power usage for task placements of the experiment's
        workload. Assert that the observed power does not exceed the optimal
        power by more than :attr:energy_est_threshold_pct percents.
        """
        exp_power = self.get_expected_power_df(experiment)
        est_power = self.get_power_df(experiment)

        exp_energy = area_under_curve(exp_power.sum(axis=1), method='rect')
        est_energy = area_under_curve(est_power.sum(axis=1), method='rect')

        msg = 'Estimated {} bogo-Joules to run workload, expected {}'.format(
            est_energy, exp_energy)
        threshold = exp_energy * (1 + (self.energy_est_threshold_pct / 100.))
        self.assertLess(est_energy, threshold, msg=msg)

class OneSmallTask(_EnergyModelTest):
    """
    Test EAS for a single 20% task over 2 seconds
    """
    workloads = {
        'one_small' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 20,
                    'duration_s': 2,
                    'period_ms': WORKLOAD_PERIOD_MS,
                },
                'tasks' : 1,
                'prefix' : 'many',
            },
        },
    }
    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)

class ThreeSmallTasks(_EnergyModelTest):
    """
    Test EAS for 3 20% tasks over 2 seconds
    """

    # The energy estimation for this test is probably not very accurate and this
    # isn't a very realistic workload. It doesn't really matter if we pick an
    # "ideal" task placement for this workload, we just want to avoid using big
    # CPUs in a big.LITTLE system. So use a larger energy threshold that
    # hopefully prevents too much use of big CPUs but otherwise is flexible in
    # allocation of LITTLEs.
    energy_est_threshold_pct = 20

    workloads = {
        'three_small' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 20,
                    'duration_s': 2,
                    'period_ms': WORKLOAD_PERIOD_MS,
                },
                'tasks' : 3,
                'prefix' : 'many',
            },
        },
    }
    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)

class TwoBigTasks(_EnergyModelTest):
    """
    Test EAS for 2 80% tasks over 2 seconds
    """
    workloads = {
        'two_big' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 80,
                    'duration_s': 2,
                    'period_ms': WORKLOAD_PERIOD_MS,
                },
                'tasks' : 2,
                'prefix' : 'many',
            },
        },
    }
    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)

class TwoBigThreeSmall(_EnergyModelTest):
    """
    Test EAS for 2 70% tasks and 3 10% tasks over 2 seconds
    """
    workloads = {
        'two_big_three_small' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'profile',
                'params' : {
                    'large' : {
                        'kind' : 'Periodic',
                        'params' : {
                            'duty_cycle_pct': 70,
                            'duration_s': 2,
                            'period_ms': WORKLOAD_PERIOD_MS,
                        },
                        'tasks' : 2,
                    },
                    'small' : {
                        'kind' : 'Periodic',
                        'params' : {
                            'duty_cycle_pct': 10,
                            'duration_s': 2,
                            'period_ms': WORKLOAD_PERIOD_MS,
                        },
                        'tasks' : 3,
                    },
                },
            },
        },
    }
    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)

class RampUp(_EnergyModelTest):
    """
    Test EAS for a task ramping from 5% up to 70%
    """
    # The main purpose of this test is to ensure that as it grows in load, a
    # task is migrated from LITTLE to big CPUs on a big.LITTLE system.
    # This migration naturally happens some time _after_ it could possibly be
    # done, since there must be some hysteresis to avoid a performance cost.
    # Therefore allow a larger energy usage threshold
    energy_est_threshold_pct = 15

    workloads = {
        "ramp_up" : {
            "type": "rt-app",
            "conf" : {
                "class"  : "profile",
                "params"  : {
                    "r5_10-60" : {
                        "kind"   : "Ramp",
                        "params" : {
                            'period_ms': WORKLOAD_PERIOD_MS,
                            "start_pct" :  5,
                            "end_pct"   : 70,
                            "delta_pct" :  5,
                            "time_s"    : .5,
                         },
                    },
                },
            },
        },
    }

    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)

class RampDown(_EnergyModelTest):
    """
    Test EAS for a task ramping from 70% down to 5% over 2 seconds
    """
    # The main purpose of this test is to ensure that as it reduces in load, a
    # task is migrated from big to LITTLE CPUs on a big.LITTLE system.
    # This migration naturally happens some time _after_ it could possibly be
    # done, since there must be some hysteresis to avoid a performance cost.
    # Therefore allow a larger energy usage threshold
    #
    # The number below has been found by trial and error on the platforms
    # generally used for testing EAS (at the time of writing: Juno r0, Juno r2,
    # Hikey960 and TC2). It would be better to estimate the amount of energy
    # 'wasted' in the hysteresis (the overutilized band) and compute a threshold
    # based on that. But implementing this isn't easy because it's very platform
    # dependent, so until we have a way to do that easily in test classes, let's
    # stick with the arbitrary threshold.
    energy_est_threshold_pct = 18

    workloads = {
        "ramp_down" : {
            "type": "rt-app",
            "conf" : {
                "class"  : "profile",
                "params"  : {
                    "r5_10-60" : {
                        "kind"   : "Ramp",
                        "params" : {
                            'period_ms': WORKLOAD_PERIOD_MS,
                            "start_pct" : 70,
                            "end_pct"   :  5,
                            "delta_pct" :  5,
                            "time_s"    : .5,
                         },
                    },
                },
            },
        },
    }

    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)

class EnergyModelWakeMigration(_EnergyModelTest):
    """
    Test EAS for tasks alternating beetween 10% and 70%
    """
    workloads = {
        'em_wake_migration' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'profile',
                'params' : {
                    'wmig' : {
                        'kind' : 'Step',
                        'params' : {
                            "period_ms" : WORKLOAD_PERIOD_MS,
                            'start_pct': 10,
                            'end_pct': 70,
                            'time_s': 2,
                            'loops': 2
                        },
                        # Create one task for each big cpu
                        'tasks' : 'big',
                    },
                },
            },
        },
    }
    @experiment_test
    def test_slack(self, experiment, tasks):
        self._test_slack(experiment, tasks)
    @experiment_test
    def test_task_placement(self, experiment, tasks):
        self._test_task_placement(experiment, tasks)
