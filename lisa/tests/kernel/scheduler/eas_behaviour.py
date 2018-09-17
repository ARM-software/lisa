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
from math import isnan

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

from bart.common.Utils import area_under_curve

from lisa.wlgen.rta import RTA, Periodic, Ramp, Step
from lisa.trace import Trace
from lisa.tests.kernel.test_bundle import RTATestBundle, ResultBundle
from lisa.perf_analysis import PerfAnalysis

class EASBehaviour(RTATestBundle):
    """
    "Abstract" class for EAS behavioural testing.

    :param rtapp_profile: The rtapp parameters used to create the synthetic
      workload. That happens to be what is returned by :meth:`create_rtapp_profile`
    :type rtapp_profile: dict

    :param nrg_model: The energy model of the platform the synthetic workload
      was run on
    :type nrg_model: EnergyModel

    This class provides :meth:`test_task_placement` to validate the basic
    behaviour of EAS. The implementations of this class have been developed to
    verify patches supporting Arm's big.LITTLE in the Linux scheduler. You can
    see these test results being published
    `here <https://developer.arm.com/open-source/energy-aware-scheduling/eas-mainline-development>`_.
    """

    def __init__(self, res_dir,rtapp_profile, nrg_model):
        super(EASBehaviour, self).__init__(res_dir, rtapp_profile)

        self.nrg_model = nrg_model

    @classmethod
    def _from_target(cls, te, res_dir):
        rtapp_profile = cls.create_rtapp_profile(te)

        # EAS doesn't make a lot of sense without schedutil,
        # so make sure this is what's being used
        with te.target.cpufreq.use_governor("schedutil"):
            cls._run_rtapp(te, res_dir, rtapp_profile)

        return cls(res_dir, rtapp_profile, te.nrg_model)

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

    def _get_start_time(self):
        """
        Get the time where the first task spawned
        """
        tasks = list(self.rtapp_profile.keys())
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
        for task, params in self.rtapp_profile.items():
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
        tasks = list(self.rtapp_profile.keys())

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
            for time, util in tdf.items():
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
            columns = list(power.keys())

            # Assemble a dataframe to plot the expected utilization
            data.append(expected_utils)
            index.append(row.name)

            return pd.Series([power[c] for c in columns], index=columns)

        res_df = self._sort_power_df_columns(
            task_utils_df.apply(exp_power, axis=1))

        self._plot_expected_util(pd.DataFrame(data, index=index))

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
        tasks = list(self.rtapp_profile.keys())

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
            columns = list(power.keys())
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
        res.add_metric("estimated_energy", est_energy, 'bogo-joules')
        res.add_metric("energy_threshold", threshold, 'bogo-joules')
        return res


    @classmethod
    def unscaled_utilization(cls, capacity, utilization_pct):
        """
        Convert a scaled utilization value to a 'raw', unscaled one.

        :param capacity: The capacity of the CPU :attr:`utilization_pct` is scaled
          against
        :type capacity: int

        :param utilization_pct: The scaled utilization in %
        :type utilization_pct: int
        """
        # TODO(?): use te.nrg_model.capacity_scale
        return int((capacity / 1024.) * utilization_pct)

# TODO: factorize this crap out of these classes
class OneSmallTask(EASBehaviour):
    """
    A single 'small' task
    """

    task_name = "small"

    @classmethod
    def create_rtapp_profile(cls, te):
        # 50% of the smallest CPU's capacity
        duty = cls.unscaled_utilization(cls.min_cpu_capacity(te), 50)

        rtapp_profile = {}
        rtapp_profile[cls.task_name] = Periodic(
            duty_cycle_pct=duty,
            duration_s=1,
            period_ms=cls.TASK_PERIOD_MS
        )

        return rtapp_profile

class ThreeSmallTasks(EASBehaviour):
    """
    Three 'small' tasks
    """
    task_prefix = "small"

    @classmethod
    def create_rtapp_profile(cls, te):
        # 50% of the smallest CPU's capacity
        duty = cls.unscaled_utilization(cls.min_cpu_capacity(te), 50)

        rtapp_profile = {}
        for i in range(3):
            rtapp_profile["{}_{}".format(cls.task_prefix, i)] = Periodic(
                duty_cycle_pct=duty,
                duration_s=1,
                period_ms=cls.TASK_PERIOD_MS
            )

        return rtapp_profile

class TwoBigTasks(EASBehaviour):
    """
    Two 'big' tasks
    """

    task_prefix = "big"

    @classmethod
    def create_rtapp_profile(cls, te):
        # 80% of the biggest CPU's capacity
        duty = cls.unscaled_utilization(cls.max_cpu_capacity(te), 80)

        rtapp_profile = {}
        for i in range(2):
            rtapp_profile["{}_{}".format(cls.task_prefix, i)] = Periodic(
                duty_cycle_pct=duty,
                duration_s=1,
                period_ms=cls.TASK_PERIOD_MS
            )

        return rtapp_profile

class TwoBigThreeSmall(EASBehaviour):
    """
    A mix of 'big' and 'small' tasks
    """

    small_prefix = "small"
    big_prefix = "big"

    @classmethod
    def create_rtapp_profile(cls, te):
        # 50% of the smallest CPU's capacity
        small_duty = cls.unscaled_utilization(cls.min_cpu_capacity(te), 50)
        # 80% of the biggest CPU's capacity
        big_duty = cls.unscaled_utilization(cls.max_cpu_capacity(te), 80)

        rtapp_profile = {}

        for i in range(3):
            rtapp_profile["{}_{}".format(cls.small_prefix, i)] = Periodic(
                duty_cycle_pct=small_duty,
                duration_s=1,
                period_ms=cls.TASK_PERIOD_MS
            )

        for i in range(2):
            rtapp_profile["{}_{}".format(cls.big_prefix, i)] = Periodic(
                duty_cycle_pct=big_duty,
                duration_s=1,
                period_ms=cls.TASK_PERIOD_MS
            )

        return rtapp_profile

class RampUp(EASBehaviour):
    """
    A single task whose utilisation slowly ramps up
    """
    task_name = "ramp_up"

    @classmethod
    def create_rtapp_profile(cls, te):
        start_pct = cls.unscaled_utilization(cls.min_cpu_capacity(te), 10)
        end_pct = cls.unscaled_utilization(cls.max_cpu_capacity(te), 70)

        rtapp_profile = {
            cls.task_name : Ramp(
                start_pct=start_pct,
                end_pct=end_pct,
            delta_pct=5,
            time_s=.5,
                period_ms=cls.TASK_PERIOD_MS
        )
        }

        return rtapp_profile

class RampDown(EASBehaviour):
    """
    A single task whose utilisation slowly ramps down
    """
    task_name = "ramp_down"

    @classmethod
    def create_rtapp_profile(cls, te):
        start_pct = cls.unscaled_utilization(cls.max_cpu_capacity(te), 70)
        end_pct = cls.unscaled_utilization(cls.min_cpu_capacity(te), 10)
        rtapp_profile = {
            cls.task_name : Ramp(
                start_pct=start_pct,
                end_pct=end_pct,
            delta_pct=5,
            time_s=.5,
                period_ms=cls.TASK_PERIOD_MS
        )
        }

        return rtapp_profile

class EnergyModelWakeMigration(EASBehaviour):
    """
    One task per big CPU, alternating between two phases:
      * Low utilization phase (should run on a LITTLE CPU)
      * High utilization phase (should run on a big CPU)
    """
    task_prefix = "emwm"

    @classmethod
    def create_rtapp_profile(cls, te):
        rtapp_profile = {}
        capacities = te.target.sched.get_capacities()
        bigs = [cpu for cpu, capacity in list(capacities.items())
                if capacity == cls.max_cpu_capacity(te)]

        start_pct = cls.unscaled_utilization(cls.min_cpu_capacity(te), 20)
        end_pct = cls.unscaled_utilization(cls.max_cpu_capacity(te), 70)

        for i in range(len(bigs)):
            rtapp_profile["{}_{}".format(cls.task_prefix, i)] = Step(
                start_pct=start_pct,
                end_pct=end_pct,
                time_s=2,
                loops=2,
                period_ms=cls.TASK_PERIOD_MS
            )

        return rtapp_profile

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
