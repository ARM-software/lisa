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

from math import ceil

import pandas as pd

from devlib.module.sched import SchedDomain, SchedDomainFlag

from lisa.utils import memoized, ArtifactPath
from lisa.datautils import df_squash, df_add_delta
from lisa.trace import Trace, FtraceConf, requires_events
from lisa.wlgen.rta import RTAPhase, RunWload, SleepWload
from lisa.tests.base import TestBundle, RTATestBundle, Result, ResultBundle, TestMetric
from lisa.target import Target
from lisa.analysis.tasks import TasksAnalysis, TaskState
from lisa.analysis.idle import IdleAnalysis
from lisa.analysis.rta import RTAEventsAnalysis


class MisfitMigrationBase(RTATestBundle, TestBundle):
    """
    Abstract class for Misfit behavioural testing

    This class provides some helpers for features related to Misfit.
    """

    @classmethod
    def _has_asym_cpucapacity(cls, target):
        """
        :returns: Whether the target has asymmetric CPU capacities
        """
        return len(set(target.plat_info["cpu-capacities"]['orig'].values())) > 1

    @classmethod
    def _get_max_lb_interval(cls, plat_info):
        """
        Get the value of maximum_load_balance_interval.

        The kernel computes it so:
            HZ*num_online_cpus()/10;
        (https://elixir.bootlin.com/linux/v4.15/source/kernel/sched/fair.c#L9101)

        Here we don't do any hotplugging so we consider all CPUs to be online.

        :returns: The absolute maximum load-balance interval in seconds
        """
        HZ = plat_info['kernel']['config']['CONFIG_HZ']
        return ((HZ * plat_info['cpus-count']) // 10) * (1. / HZ)

    @classmethod
    def _get_lb_interval(cls, plat_info):
        # Regular interval is 1 ms * nr_cpus, rounded to closest jiffy multiple
        jiffy = 1 / plat_info['kernel']['config']['CONFIG_HZ']
        interval = 1e-3 * plat_info["cpus-count"]

        return ceil(interval / jiffy) * jiffy

class StaggeredFinishes(MisfitMigrationBase):
    """
    One 100% task per CPU, with staggered completion times.

    By spawning one task per CPU on an asymmetric system, we expect the tasks
    running on the higher-performance CPUs to complete first. At this point,
    the misfit logic should kick in and they should pull tasks from
    lower-performance CPUs.

    The tasks have staggered completion times to prevent having several of them
    completing at the same time, which can cause some unwanted noise (e.g. some
    sshd or systemd activity at the end of the task).

    The end result should look something like this on big.LITTLE::

      a,b,c,d are CPU-hogging tasks
      _ signifies idling

      LITTLE_0 | a a a a _ _ _
      LITTLE_1 | b b b b b _ _
      ---------|--------------
        big_0  | c c c c a a a
        big_1  | d d d d d b b

    """

    task_prefix = "msft"

    PIN_DELAY = 0.001
    """
    How long the tasks will be pinned to their "starting" CPU. Doesn't have
    to be long (we just have to ensure they spawn there), so arbitrary value
    """

    # Let us handle things ourselves
    _BUFFER_PHASE_DURATION_S=0

    IDLING_DELAY = 1
    """
    A somewhat arbitray delay - long enough to ensure
    rq->avg_idle > sysctl_sched_migration_cost
    """

    @property
    def src_cpus(self):
        return self.plat_info['capacity-classes'][0]

    @property
    def dst_cpus(self):
        cpu_classes = self.plat_info['capacity-classes']

        # XXX: Might need to check the tasks can fit on all of those, rather
        # than just pick all but the smallest CPUs
        dst_cpus = []
        for group in cpu_classes[1:]:
            dst_cpus += group
        return dst_cpus

    @property
    def end_time(self):
        return self.trace.end

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    @memoized
    @RTAEventsAnalysis.df_rtapp_phases_start.used_events
    def start_time(self):
        """
        The tasks don't wake up at the same exact time, find the task that is
        the last to wake up (after the idling phase).

        .. note:: We don't want to redefine
            :meth:`~lisa.tests.base.RTATestBundle.trace_window` here because we
            still need the first wakeups to be visible.
        """
        phase_df = self.trace.analysis.rta.df_rtapp_phases_start(wlgen_profile=self.rtapp_profile)
        return phase_df[
            phase_df.index.get_level_values('phase') == 'test/pinned'
        ]['Time'].max()

    @classmethod
    def check_from_target(cls, target):
        if not cls._has_asym_cpucapacity(target):
            ResultBundle.raise_skip(
                "Target doesn't have asymmetric CPU capacities")

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        cpus = list(range(plat_info['cpus-count']))

        # We're pinning stuff in the first phase, so give it ample time to
        # clean the pinned logic out of balance_interval
        free_time_s = 1.1 * cls._get_max_lb_interval(plat_info)

        # Ideally we'd like the different tasks not to complete at the same time
        # (hence the "staggered" name), but this depends on a lot of factors
        # (capacity ratios, available frequencies, thermal conditions...) so the
        # best we can do is wing it.
        stagger_s = cls._get_lb_interval(plat_info) * 1.5

        return {
            f"{cls.task_prefix}{cpu}": (
                RTAPhase(
                    prop_name='idling',
                    prop_wload=SleepWload(cls.IDLING_DELAY),
                    prop_cpus=[cpu],
                ) +
                RTAPhase(
                    prop_name='pinned',
                    prop_wload=RunWload(cls.PIN_DELAY),
                    prop_cpus=[cpu],
                ) +
                RTAPhase(
                    prop_name='staggered',
                    prop_wload=RunWload(
                        # Introduce staggered task completions
                        free_time_s + cpu * stagger_s
                    ),
                    prop_cpus=cpus,
                )
            )
            for cpu in cpus
        }

    def _trim_state_df(self, state_df):
        if state_df.empty:
            return state_df

        return df_squash(state_df, self.start_time,
                         state_df.index[-1] + state_df['delta'].iloc[-1], "delta")

    @requires_events('sched_switch', TasksAnalysis.df_task_states.used_events)
    def test_preempt_time(self, allowed_preempt_pct=1) -> ResultBundle:
        """
        Test that tasks are not being preempted too much
        """

        sdf = self.trace.df_event('sched_switch')
        task_state_dfs = {
            task: self.trace.analysis.tasks.df_task_states(task)
            for task in self.rtapp_tasks
        }

        res = ResultBundle.from_bool(True)
        for task, state_df in task_state_dfs.items():
            # The sched_switch dataframe where the misfit task
            # is replaced by another misfit task
            preempt_sdf = sdf[
                (sdf.prev_comm == task) &
                (sdf.next_comm.str.startswith(self.task_prefix))
            ]

            state_df = self._trim_state_df(state_df)
            state_df = state_df[
                (state_df.index.isin(preempt_sdf.index)) &
                # Ensure this is a preemption and not just the task ending
                (state_df.curr_state == TaskState.TASK_INTERRUPTIBLE)
            ]

            preempt_time = state_df.delta.sum()
            preempt_pct = (preempt_time / self.duration) * 100

            res.add_metric(f"{task} preemption", {
                "ratio": TestMetric(preempt_pct, "%"),
                "time": TestMetric(preempt_time, "seconds")})

            if preempt_pct > allowed_preempt_pct:
                res.result = Result.FAILED

        return res

    @memoized
    @IdleAnalysis.signal_cpu_active.used_events
    def _get_active_df(self, cpu):
        """
        :returns: A dataframe that describes the idle status (on/off) of 'cpu'
        """
        active_df = pd.DataFrame(
            self.trace.analysis.idle.signal_cpu_active(cpu), columns=['state']
        )
        df_add_delta(active_df, inplace=True, window=self.trace.window)
        return active_df

    @_get_active_df.used_events
    def _max_idle_time(self, start, end, cpus):
        """
        :returns: The maximum idle time of 'cpus' in the [start, end] interval
        """
        max_time = 0
        max_cpu = 0

        for cpu in cpus:
            busy_df = self._get_active_df(cpu)
            busy_df = df_squash(busy_df, start, end)
            busy_df = busy_df[busy_df.state == 0]

            if busy_df.empty:
                continue

            local_max = busy_df.delta.max()
            if local_max > max_time:
                max_time = local_max
                max_cpu = cpu

        return max_time, max_cpu

    @_max_idle_time.used_events
    def _test_cpus_busy(self, task_state_dfs, cpus, allowed_idle_time_s):
        """
        Test that for every window in which the tasks are running, :attr:`cpus`
        are not idle for more than :attr:`allowed_idle_time_s`
        """
        if allowed_idle_time_s is None:
            allowed_idle_time_s = self._get_lb_interval(self.plat_info)

        res = ResultBundle.from_bool(True)

        for task, state_df in task_state_dfs.items():
            # Have a look at every task activation
            task_idle_times = [self._max_idle_time(index, index + row.delta, cpus)
                               for index, row in state_df.iterrows()]

            if not task_idle_times:
                continue

            max_time, max_cpu = max(task_idle_times)
            res.add_metric(f"{task} max idle", data={
                "time": TestMetric(max_time, "seconds"), "cpu": TestMetric(max_cpu)})

            if max_time > allowed_idle_time_s:
                res.result = Result.FAILED

        return res

    @TasksAnalysis.df_task_states.used_events
    @_test_cpus_busy.used_events
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    def test_throughput(self, allowed_idle_time_s=None) -> ResultBundle:
        """
        Test that big CPUs are not idle when there are misfit tasks to upmigrate

        :param allowed_idle_time_s: How much time should be allowed between a
          big CPU going idle and a misfit task ending on that CPU. In theory
          a newidle balance should lead to a null delay, but in practice
          there's a tiny one, so don't set that to 0 and expect the test to
          pass.

          Furthermore, we're not always guaranteed to get a newidle pull, so
          allow time for a regular load balance to happen.

          When ``None``, this defaults to (1ms x number_of_cpus) to mimic the
          default balance_interval (balance_interval = sd_weight), see
          kernel/sched/topology.c:sd_init().
        :type allowed_idle_time_s: int
        """
        task_state_dfs = {}
        for task in self.rtapp_tasks:
            # This test is all about throughput: check that every time a task
            # runs on a little it's because bigs are busy
            df = self.trace.analysis.tasks.df_task_states(task)
            # Trim first to keep coherent deltas
            df = self._trim_state_df(df)
            task_state_dfs[task] = df[
                # Task is active
                (df.curr_state == TaskState.TASK_ACTIVE) &
                # Task needs to be upmigrated
                (df.cpu.isin(self.src_cpus))
            ]

        return self._test_cpus_busy(task_state_dfs, self.dst_cpus, allowed_idle_time_s)
