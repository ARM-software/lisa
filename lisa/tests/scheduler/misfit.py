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

import pandas as pd

from devlib.module.sched import SchedDomain, SchedDomainFlag

from lisa.utils import memoized, ArtifactPath
from lisa.trace import Trace, FtraceConf, FtraceCollector, requires_events
from lisa.wlgen.rta import Periodic
from lisa.tests.base import RTATestBundle, Result, ResultBundle, CannotCreateError, TestMetric
from lisa.target import Target
from lisa.analysis.tasks import TasksAnalysis, TaskState

class MisfitMigrationBase(RTATestBundle):
    """
    Abstract class for Misfit behavioural testing

    This class provides some helpers for features related to Misfit.
    """

    ftrace_conf = FtraceConf({
        "events" : [
            "sched_switch",
            "sched_wakeup",
            "cpu_idle"
        ]
    }, __qualname__)

    @classmethod
    def _has_asym_cpucapacity(cls, target):
        """
        :returns: Whether the target has SD_ASYM_CPUCAPACITY set on any of its sd
        """
        # Just try to find at least one instance of that flag
        sd_info = target.sched.get_sd_info()

        for cpu, domain_node in sd_info.cpus.items():
            for domain in domain_node.domains.values():
                if SchedDomainFlag.SD_ASYM_CPUCAPACITY in domain.flags:
                    return True

        return False

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
    def from_target(cls, target:Target, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'MisfitMigrationBase':
        """
        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in
        :meth:`~lisa.tests.base.RTATestBundle.get_rtapp_profile`
        """
        return super().from_target(target, res_dir, ftrace_coll=ftrace_coll)

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

    task_prefix = "misfit"

    PIN_DELAY_S = 0.001
    """
    How long the tasks will be pinned to their "starting" CPU. Doesn't have
    to be long (we just have to ensure they spawn there), so arbitrary value
    """

    IDLING_DELAY_S = 1
    """
    A somewhat arbitray delay - long enough to ensure
    rq->avg_idle > sysctl_sched_migration_cost
    """

    def __init__(self, res_dir, plat_info, rtapp_profile):
        super().__init__(res_dir, plat_info, rtapp_profile)

        cpu_classes = plat_info['capacity-classes']

        sdf = self.trace.df_events('sched_switch')
        # Get the time where the first rt-app task spawns
        init_start = sdf[sdf.next_comm.str.contains(self.task_prefix)].index[0]

        # The tasks don't wake up at the same exact time, find the task that is
        # the last to wake up.
        last_start = 0

        sdf = sdf[init_start + self.IDLING_DELAY_S * 0.9 :]

        for task in self.rtapp_profile.keys():
            task_cpu = int(task.strip("{}_".format(self.task_prefix)))
            task_start = sdf[(sdf.next_comm == task) & (sdf["__cpu"] == task_cpu)].index[0]
            last_start = max(last_start, task_start)

        self.start_time = last_start

        self.end_time = sdf[sdf.prev_comm.str.contains(self.task_prefix)].index[-1]
        self.duration = self.end_time - self.start_time

        self.src_cpus = cpu_classes[0]
        # XXX: Might need to check the tasks can fit on all of those, rather
        # than just pick all but the smallest CPUs
        self.dst_cpus = []
        for group in cpu_classes[1:]:
            self.dst_cpus += group

    @classmethod
    def check_from_target(cls, target):
        if not cls._has_asym_cpucapacity(target):
            raise CannotCreateError(
                "Target doesn't have SD_ASYM_CPUCAPACITY on any sched_domain")

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        cpus = list(range(plat_info['cpus-count']))

        # We're pinning stuff in the first phase, so give it ample time to
        # clean the pinned logic out of balance_interval
        free_time_s = 1.1 * cls._get_max_lb_interval(plat_info)
        stagger_s = free_time_s // (10 * len(cpus))

        profile = {}

        for cpu in cpus:
            profile["{}_{}".format(cls.task_prefix, cpu)] = (
                Periodic(
                    duty_cycle_pct=100,
                    duration_s=cls.PIN_DELAY_S,
                    delay_s=cls.IDLING_DELAY_S,
                    period_ms=cls.TASK_PERIOD_MS,
                    cpus=[cpu]
                ) + Periodic(
                    duty_cycle_pct=100,
                    # Introduce staggered task completions
                    duration_s=free_time_s + cpu * stagger_s,
                    period_ms=cls.TASK_PERIOD_MS,
                    cpus=cpus
                )
            )

        return profile

    def _trim_state_df(self, state_df):
        if state_df.empty:
            return state_df

        return Trace.squash_df(state_df, self.start_time,
                               state_df.index[-1] + state_df.delta.values[-1], "delta")

    @requires_events('sched_switch', TasksAnalysis.df_task_states.used_events)
    def test_preempt_time(self, allowed_preempt_pct=1) -> ResultBundle:
        """
        Test that tasks are not being preempted too much
        """

        sdf = self.trace.df_events('sched_switch')
        task_state_dfs = {
            task : self.trace.analysis.tasks.df_task_states(task)
            for task in self.rtapp_profile.keys()
        }

        res = ResultBundle.from_bool(True)
        for task, state_df in task_state_dfs.items():
            # The sched_switch dataframe where the misfit task
            # is replaced by another misfit task
            preempt_sdf = sdf[
                (sdf.prev_comm == task) &
                (sdf.next_comm.str.startswith(self.task_prefix))
            ]

            state_df = self._trim_state_df(
                state_df[
                    (state_df.index.isin(preempt_sdf.index)) &
                    # Ensure this is a preemption and not just the task ending
                    (state_df.curr_state == TaskState.TASK_INTERRUPTIBLE)
                ]
            )

            preempt_time = state_df.delta.sum()
            preempt_pct = (preempt_time / self.duration) * 100

            res.add_metric("{} preemption".format(task), {
                "ratio" : TestMetric(preempt_pct, "%"),
                "time" : TestMetric(preempt_time, "seconds")})

            if preempt_pct > allowed_preempt_pct:
                res.result = Result.FAILED

        return res

    @memoized
    def _get_active_df(self, cpu):
        """
        :returns: A dataframe that describes the idle status (on/off) of 'cpu'
        """
        active_df = pd.DataFrame(self.trace.get_cpu_active_signal(cpu), columns=['state'])
        self.trace.add_events_deltas(active_df)
        return active_df

    def _max_idle_time(self, start, end, cpus):
        """
        :returns: The maximum idle time of 'cpus' in the [start, end] interval
        """
        max_time = 0
        max_cpu = 0

        for cpu in cpus:
            busy_df = self._get_active_df(cpu)
            busy_df = Trace.squash_df(busy_df, start, end)
            busy_df = busy_df[busy_df.state == 0]

            if busy_df.empty:
                continue

            local_max = busy_df.delta.max()
            if local_max > max_time:
                max_time = local_max
                max_cpu = cpu

        return max_time, max_cpu

    def _test_cpus_busy(self, task_state_dfs, cpus, allowed_idle_time_s):
        """
        Test that for every window in which the tasks are running, :attr:`cpus`
        are not idle for more than :attr:`allowed_idle_time_s`
        """
        if allowed_idle_time_s is None:
            allowed_idle_time_s = 1e-3 * self.plat_info["cpus-count"]

        res = ResultBundle.from_bool(True)

        for task, state_df in task_state_dfs.items():
            # Have a look at every task activation
            task_idle_times = [self._max_idle_time(index, index + row.delta, cpus)
                               for index, row in state_df.iterrows()]

            if not task_idle_times:
                continue

            max_time, max_cpu = max(task_idle_times)
            res.add_metric("{} max idle".format(task), data={
                "time" : TestMetric(max_time, "seconds"), "cpu" : TestMetric(max_cpu)})

            if max_time > allowed_idle_time_s:
                res.result = Result.FAILED

        return res

    @TasksAnalysis.df_task_states.used_events
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
        for task in self.rtapp_profile.keys():
            # This test is all about throughput: check that every time a task
            # runs on a little it's because bigs are busy
            df = self.trace.analysis.tasks.df_task_states(task)
            task_state_dfs[task] = self._trim_state_df(df[
                # Task is active
                (df.curr_state == TaskState.TASK_ACTIVE) &
                # Task needs to be upmigrated
                (df.cpu.isin(self.src_cpus))
            ])

        return self._test_cpus_busy(task_state_dfs, self.dst_cpus, allowed_idle_time_s)
