# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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
from math import isnan

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

from bart.common.Utils import area_under_curve

from wlgen.rta import RTA, Periodic, Ramp, Step
from trace import Trace
from test_workload import Metric, ResultBundle, TestBundle
from perf_analysis import PerfAnalysis

class GenericTestBundle(TestBundle):
    """
    "Abstract" class for generic synthetic tests.

    :param nrg_model: The energy model of the platform the synthetic workload
      was run on
    :type nrg_model: EnergyModel

    :param rtapp_params: The rtapp parameters used to create the synthetic
      workload. That happens to be what is returned by :meth:`create_rtapp_params`
    :type rtapp_params: dict

    This class provides :meth:`test_slack` and :meth:`test_task_placement` to
    validate the basic behaviour of EAS.
    """

    ftrace_conf = {
        "events" : ["sched_switch"],
    }
    """
    The FTrace configuration used to record a trace while the synthetic workload
    is being run.
    """

    @property
    def trace(self):
        """

        :returns: a Trace

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`to_path`
        """
        if not self._trace:
            self._trace = Trace(self.res_dir, events=self.ftrace_conf["events"])

        return self._trace

    def __init__(self, res_dir, nrg_model, rtapp_params):
        super(GenericTestBundle, self).__init__(res_dir)

        # self.trace = Trace(res_dir, events=self.ftrace_conf["events"])
        #EnergyModel.from_path(os.path.join(res_dir, "nrg_model.yaml"))
        self._trace = None
        self.nrg_model = nrg_model
        self.rtapp_params = rtapp_params

    @classmethod
    def create_rtapp_params(cls, te):
        """
        :returns: a :class:`dict` with task names as keys and :class:`RTATask` as values

        This is the method you want to override to specify what is
        your synthetic workload.
        """
        raise NotImplementedError()

    @classmethod
    def _from_target(cls, te, res_dir):
        rtapp_params = cls.create_rtapp_params(te)

        wload = RTA(te.target, "rta_{}".format(cls.__name__.lower()), te.calibration())
        wload.conf(kind='profile', params=rtapp_params, work_dir=res_dir)

        trace_path = os.path.join(res_dir, "trace.dat")
        te.configure_ftrace(**cls.ftrace_conf)

        with te.record_ftrace(trace_path):
            with te.freeze_userspace():
                wload.run(out_dir=res_dir)

        return cls(res_dir, te.nrg_model, rtapp_params)

    @classmethod
    def min_cpu_capacity(cls, te):
        """
        The smallest CPU capacity on the target

        :type te: TestEnv

        :returns: int
        """
        return min(te.target.sched.get_capacities().values())

    @classmethod
    def max_cpu_capacity(cls, te):
        """
        The highest CPU capacity on the target

        :type te: TestEnv

        :returns: int
        """
        return max(te.target.sched.get_capacities().values())

    def test_slack(self, negative_slack_allowed_pct=15):
        """
        Assert that the RTApp workload was given enough performance

        :param out_dir: Output directory for test artefacts
        :type out_dir: str

        :param negative_slack_allowed_pct: Allowed percentage of RT-app task
            activations with negative slack.
        :type negative_slack_allowed_pct: int

        Use :class:`PerfAnalysis` to find instances where the RT-App workload
        wasn't able to complete its activations (i.e. its reported "slack"
        was negative). Assert that this happened less than
        :attr:`negative_slack_allowed_pct` percent of the time.
        """
        pa = PerfAnalysis(self.res_dir)

        slacks = {}
        passed = True

        # Data is only collected for rt-app tasks, so it's safe to iterate over
        # all of them
        for task in pa.tasks():
            slack = pa.df(task)["Slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100. / len(slack)
            if bad_activations_pct > negative_slack_allowed_pct:
                passed = False

            slacks[task] = bad_activations_pct

        res = ResultBundle(passed)
        for task, slack in slacks.iteritems():
            res.add_metric(Metric("slack_{}".format(task), slack,
                                  units='%', lower_is_better=True))

        return res

    def _get_start_time(self):
        """
        Get the time where the first task spawned
        """
        tasks = self.rtapp_params.keys()
        sdf = self.trace.data_frame.trace_event('sched_switch')
        start_time = self.trace.start_time + self.trace.time_range

        for task in tasks:
            pid = self.trace.getTaskByName(task)
            assert len(pid) == 1, "getTaskByName returned more than one PID"
            pid = pid[0]
            start_time = min(start_time, sdf[sdf.next_pid == pid].index[0])

        return start_time

    def _get_expected_task_utils_df(self):
        """
        Get a DataFrame with the *expected* utilization of each task over time

        :returns: A Pandas DataFrame with a column for each task, showing how
                  the utilization of that task varies over time
        """
        util_scale = self.nrg_model.capacity_scale

        transitions = {}
        def add_transition(time, task, util):
            if time not in transitions:
                transitions[time] = {task: util}
            else:
                transitions[time][task] = util

        # First we'll build a dict D {time: {task_name: util}} where D[t][n] is
        # the expected utilization of task n from time t.
        for task, params in self.rtapp_params.iteritems():
            # time = self.get_start_time(experiment) + params.get('delay', 0)
            time = params.delay_s
            add_transition(time, task, 0)
            for _ in range(params.loops):
                for phase in params.phases:
                    util = (phase.duty_cycle_pct * util_scale / 100.)
                    add_transition(time, task, util)
                    time += phase.duration_s
            add_transition(time, task, 0)

        index = sorted(transitions.keys())
        df = pd.DataFrame([transitions[k] for k in index], index=index)
        return df.fillna(method='ffill')

    def _get_task_cpu_df(self):
        """
        Get a DataFrame mapping task names to the CPU they ran on

        Use the sched_switch trace event to find which CPU each task ran
        on. Does not reflect idleness - tasks not running are shown as running
        on the last CPU they woke on.

        :returns: A Pandas DataFrame with a column for each task, showing the
                  CPU that the task was "on" at each moment in time
        """
        tasks = self.rtapp_params.keys()

        df = self.trace.ftrace.sched_switch.data_frame[['next_comm', '__cpu']]
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
        node_cpus = [node.cpus for node in self.nrg_model.root.iter_nodes()]
        return pd.DataFrame(df, columns=[c for c in node_cpus if c in df])

    def _plot_expected_util(self, util_df):
        """
        Create a plot of the expected per-CPU utilization for the experiment
        The plot is then outputted to the test results directory.

        :param experiment: The :class:Experiment to examine
        :param util_df: A Pandas Dataframe with a column per CPU giving their
                        (expected) utilization at each timestamp.
        """

        fig, ax = plt.subplots(
            len(self.nrg_model.cpus), 1, figsize=(16, 1.8 * len(self.nrg_model.cpus))
        )
        fig.suptitle('Per-CPU expected utilization')

        for cpu in self.nrg_model.cpus:
            tdf = util_df[cpu]

            ax[cpu].set_ylim((0, 1024))
            tdf.plot(ax=ax[cpu], drawstyle='steps-post', title="CPU{}".format(cpu), color='red')
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

        figname = os.path.join(self.res_dir, 'expected_placement.png')
        pl.savefig(figname, bbox_inches='tight')
        plt.close()

    def _get_expected_power_df(self):
        """
        Estimate *optimal* power usage over time

        Examine a trace and use :meth:get_optimal_placements and
        :meth:EnergyModel.estimate_from_cpu_util to get a DataFrame showing the
        estimated power usage over time under ideal EAS behaviour.

        :meth:get_optimal_placements returns several optimal placements. They
        are usually equivalent, but can be drastically different in some cases.
        Currently only one of those placements is used (the first in the list).

        :returns: A Pandas DataFrame with a column each node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) and a
                  "power" column with the sum of other columns. Shows the
                  estimated *optimal* power over time.
        """
        task_utils_df = self._get_expected_task_utils_df()

        data = []
        index = []

        def exp_power(row):
            task_utils = row.to_dict()
            expected_utils = self.nrg_model.get_optimal_placements(task_utils)[0]
            power = self.nrg_model.estimate_from_cpu_util(expected_utils)
            columns = power.keys()

            # Assemble a dataframe to plot the expected utilization
            data.append(expected_utils)
            index.append(row.name)

            return pd.Series([power[c] for c in columns], index=columns)

        res_df = self._sort_power_df_columns(
            task_utils_df.apply(exp_power, axis=1))

        #self._plot_expected_util(pd.DataFrame(data, index=index))

        return res_df

    def _get_estimated_power_df(self):
        """
        Considering only the task placement, estimate power usage over time

        Examine a trace and use :meth:EnergyModel.estimate_from_cpu_util to get
        a DataFrame showing the estimated power usage over time. This assumes
        perfect cpuidle and cpufreq behaviour.

        :returns: A Pandas DataFrame with a column node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) Shows
                  the estimated power over time.
        """
        task_cpu_df = self._get_task_cpu_df()
        task_utils_df = self._get_expected_task_utils_df()
        task_utils_df.index = [time + self._get_start_time() for time in task_utils_df.index]
        tasks = self.rtapp_params.keys()

        # Create a combined DataFrame with the utilization of a task and the CPU
        # it was running on at each moment. Looks like:
        #                       utils                  cpus
        #          task_wmig0 task_wmig1 task_wmig0 task_wmig1
        # 2.375056      102.4      102.4        NaN        NaN
        # 2.375105      102.4      102.4        2.0        NaN

        df = pd.concat([task_utils_df, task_cpu_df],
                       axis=1, keys=['utils', 'cpus'])
        df = df.sort_index().fillna(method='ffill')

        # Now make a DataFrame with the estimated power at each moment.
        def est_power(row):
            cpu_utils = [0 for cpu in self.nrg_model.cpus]
            for task in tasks:
                cpu = row['cpus'][task]
                util = row['utils'][task]
                if not isnan(cpu):
                    cpu_utils[int(cpu)] += util
            power = self.nrg_model.estimate_from_cpu_util(cpu_utils)
            columns = power.keys()
            return pd.Series([power[c] for c in columns], index=columns)
        return self._sort_power_df_columns(df.apply(est_power, axis=1))


    def test_task_placement(self, energy_est_threshold_pct=5):
        """
        Test that task placement was energy-efficient

        :param energy_est_threshold_pct: Allowed margin for estimated vs
            optimal task placement energy cost
        :type energy_est_threshold_pct: int

        Compute optimal energy consumption (energy-optimal task placement)
        and compare to energy consumption estimated from the trace.
        Check that the estimated energy does not exceed the optimal energy by
        more than :attr:`energy_est_threshold_pct` percents.
        """
        exp_power = self._get_expected_power_df()
        est_power = self._get_estimated_power_df()

        exp_energy = area_under_curve(exp_power.sum(axis=1), method='rect')
        est_energy = area_under_curve(est_power.sum(axis=1), method='rect')

        msg = 'Estimated {} bogo-Joules to run workload, expected {}'.format(
            est_energy, exp_energy)
        threshold = exp_energy * (1 + (energy_est_threshold_pct / 100.))

        passed = est_energy < threshold
        res = ResultBundle(passed)
        res.add_metric(Metric("estimated_energy", est_energy, units='bogo-joules',
                              lower_is_better=True))
        res.add_metric(Metric("energy_threshold", threshold, units='bogo-joules',
                              lower_is_better=True))
        return res


# TODO: factorize this crap out of these classes
class OneSmallTask(GenericTestBundle):
    """
    A single 'small' task
    """

    task_name = "small"

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        duty = int((cls.min_cpu_capacity(te) / 1024.) * 50)

        rtapp_params = {}
        rtapp_params[cls.task_name] = Periodic(
            duty_cycle_pct=duty,
            duration_s=1,
            period_ms=16
        )

        return rtapp_params

class ThreeSmallTasks(GenericTestBundle):
    """
    Three 'small' tasks
    """
    task_prefix = "small"

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        duty = int((cls.min_cpu_capacity(te) / 1024.) * 50)

        rtapp_params = {}
        for i in range(3):
            rtapp_params["{}_{}".format(cls.task_prefix, i)] = Periodic(
                duty_cycle_pct=duty,
                duration_s=1,
                period_ms=16
            )

        return rtapp_params

class TwoBigTasks(GenericTestBundle):
    """
    Two 'big' tasks
    """

    task_prefix = "big"

    @classmethod
    def create_rtapp_params(cls, te):
        # 80% of the biggest CPU's capacity
        duty = int((cls.max_cpu_capacity(te) / 1024.) * 80)

        rtapp_params = {}
        for i in range(2):
            rtapp_params["{}_{}".format(cls.task_prefix, i)] = Periodic(
                duty_cycle_pct=duty,
                duration_s=1,
                period_ms=16
            )

        return rtapp_params

class TwoBigThreeSmall(GenericTestBundle):
    """
    A mix of 'big' and 'small' tasks
    """

    small_prefix = "small"
    big_prefix = "big"

    @classmethod
    def create_rtapp_params(cls, te):
        # 50% of the smallest CPU's capacity
        small_duty = int((cls.min_cpu_capacity(te) / 1024.) * 50)
        # 80% of the biggest CPU's capacity
        big_duty = int((cls.max_cpu_capacity(te) / 1024.) * 80)

        rtapp_params = {}

        for i in range(3):
            rtapp_params["{}_{}".format(cls.small_prefix, i)] = Periodic(
                duty_cycle_pct=small_duty,
                duration_s=1,
                period_ms=16
            )

        for i in range(2):
            rtapp_params["{}_{}".format(cls.big_prefix, i)] = Periodic(
                duty_cycle_pct=big_duty,
                duration_s=1,
                period_ms=16
            )

        return rtapp_params

class RampUp(GenericTestBundle):
    """
    A single task whose utilisation slowly ramps up
    """
    task_name = "ramp_up"

    @classmethod
    def create_rtapp_params(cls, te):
        rtapp_params = {}
        rtapp_params[cls.task_name] = Ramp(
            start_pct=5,
            end_pct=70,
            delta_pct=5,
            time_s=.5,
            period_ms=16
        )

        return rtapp_params

class RampDown(GenericTestBundle):
    """
    A single task whose utilisation slowly ramps down
    """
    task_name = "ramp_down"

    @classmethod
    def create_rtapp_params(cls, te):
        rtapp_params = {}
        rtapp_params[cls.task_name] = Ramp(
            start_pct=70,
            end_pct=5,
            delta_pct=5,
            time_s=.5,
            period_ms=16
        )

        return rtapp_params

class EnergyModelWakeMigration(GenericTestBundle):
    """
    One task per big CPU, alternating between two phases:
      * Low utilization phase (should run on LITTLE CPUs)
      * High utilization phase (should run on a big CPU)
    """
    task_prefix = "emwm"

    @classmethod
    def create_rtapp_params(cls, te):
        rtapp_params = {}
        capacities = te.target.sched.get_capacities()
        max_capa = cls.max_cpu_capacity(te)
        bigs = [cpu for cpu, capacity in capacities.items() if capacity == max_capa]

        for i in range(len(bigs)):
            rtapp_params["{}_{}".format(cls.task_prefix, i)] = Step(
                start_pct=10,
                end_pct=70,
                time_s=2,
                loops=2,
                period_ms=16
            )

        return rtapp_params
