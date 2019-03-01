# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

from bart.common.Utils import area_under_curve

from lisa.wlgen.rta import RTATask, Periodic
from lisa.tests.base import TestMetric, ResultBundle, CannotCreateError
from lisa.tests.scheduler.load_tracking import (
    UTIL_AVG_CONVERGENCE_TIME_S,
    UTIL_SCALE,
    LoadTrackingBase
)
from lisa.target import Target, ArtifactPath
from lisa.trace import FtraceCollector

class CPUMigrationBase(LoadTrackingBase):
    """
    Base class for migration-related load tracking tests

    The idea here is to run several rt-app tasks and to have them pinned to
    a single CPU for a single phase. They can change CPUs in a new phase,
    and we can then inspect the CPU utilization - it should match the
    sum of the utilization of all the tasks running on it.

    **Design notes:**

    Since we sum up the utilization of each task, make sure not to overload the
    CPU - IOW, there should always be some idle cycles.

    The code assumes all tasks have the same number of phases, and that those
    phases are all aligned.
    """

    PHASE_DURATION_S = 3 * UTIL_AVG_CONVERGENCE_TIME_S
    """
    The duration of a single phase
    """

    @classmethod
    def _run_rtapp(cls, target, res_dir, profile, ftrace_coll):
        # Just do some validation on the profile
        for name, task in profile.items():
            for phase in task.phases:
                if len(phase.cpus) != 1:
                    raise RuntimeError("Each phase must be tied to a single CPU. "
                                       "Task \"{}\" violates this".format(name))

        super()._run_rtapp(target, res_dir, profile, ftrace_coll)

    def __init__(self, res_dir, plat_info, rtapp_profile):
        super().__init__(res_dir, plat_info, rtapp_profile)

        self.cpus = set()

        self.reference_task = list(self.rtapp_profile.values())[0]
        self.nr_phases = len(self.reference_task.phases)

        for task in self.rtapp_profile.values():
            for phase in task.phases:
                self.cpus.update(phase.cpus)

        self.phases_durations = [phase.duration_s
                                 for phase in self.reference_task.phases]
    @classmethod
    def from_target(cls, target:Target, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'CPUMigrationBase':
        """
        Factory method to create a bundle using a live target
        """
        return super().from_target(target=target, res_dir=res_dir, ftrace_coll=ftrace_coll)

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)

        try:
            target.plat_info["cpu-capacities"]
        except KeyError as e:
            raise CannotCreateError(str(e))

    def get_expected_cpu_util(self):
        """
        Get the per-phase average CPU utilization expected from the rtapp profile

        :returns: A dict of the shape {cpu : {phase_id : expected_util}}
        """
        cpu_util = {cpu : {phase_id : 0 for phase_id in range(self.nr_phases)}
                    for cpu in self.cpus}

        for task in self.rtapp_profile.values():
            for phase_id, phase in enumerate(task.phases):
                cpu_util[phase.cpus[0]][phase_id] += UTIL_SCALE * (phase.duty_cycle_pct / 100)

        return cpu_util

    def get_trace_cpu_util(self):
        """
        Get the per-phase average CPU utilization read from the trace

        :returns: A dict of the shape {cpu : {phase_id : trace_util}}
        """
        cpu_util = {cpu : {phase_id : 0 for phase_id in range(self.nr_phases)}
                    for cpu in self.cpus}
        df = self.trace.analysis.load_tracking.df_cpus_signals()
        sw_df = self.trace.df_events("sched_switch")

        phase_start = sw_df[sw_df.next_comm == list(self.rtapp_profile.keys())[0]].index[0]

        for phase in range(self.nr_phases):
            # Start looking at signals once they should've converged
            start = phase_start + UTIL_AVG_CONVERGENCE_TIME_S
            # Trim the end a bit, otherwise we could have one or two events
            # from the next phase
            end = phase_start + self.phases_durations[phase] * .9

            phase_df = df[start:end]
            phase_duration = end - start

            for cpu in self.cpus:
                util = phase_df[phase_df.cpu == cpu].util
                cpu_util[cpu][phase] = area_under_curve(util) / (phase_duration)

            phase_start += self.phases_durations[phase]

        return cpu_util

    def test_util_task_migration(self, allowed_error_pct=5) -> ResultBundle:
        """
        Test that a migrated task properly propagates its utilization at the CPU level

        :param allowed_error_pct: How much the trace averages can stray from the
          expected values
        :type allowed_error_pct: float
        """
        expected_cpu_util = self.get_expected_cpu_util()
        trace_cpu_util = self.get_trace_cpu_util()

        passed = True

        expected_metrics = {}
        trace_metrics = {}
        deltas = {}

        for cpu in self.cpus:
            cpu_str = "cpu{}".format(cpu)

            expected_metrics[cpu_str] = TestMetric({})
            trace_metrics[cpu_str] = TestMetric({})
            deltas[cpu_str] = TestMetric({})

            for phase in range(self.nr_phases):
                if not self.is_almost_equal(
                        trace_cpu_util[cpu][phase],
                        expected_cpu_util[cpu][phase],
                        allowed_error_pct):
                    passed = False

                # Just some verbose metric collection...
                phase_str = "phase{}".format(phase)

                expected = expected_cpu_util[cpu][phase]
                trace = trace_cpu_util[cpu][phase]
                delta = 100 * (trace - expected) / expected

                expected_metrics[cpu_str].data[phase_str] = TestMetric(expected)
                trace_metrics[cpu_str].data[phase_str] = TestMetric(trace)
                deltas[cpu_str].data[phase_str] = TestMetric(delta, "%")

        res = ResultBundle.from_bool(passed)
        res.add_metric("Expected utilization", expected_metrics)
        res.add_metric("Trace utilization", trace_metrics)
        res.add_metric("Utilization deltas", deltas)

        return res

class OneTaskCPUMigration(CPUMigrationBase):
    """
    Some tasks on two big CPUs, one of them migrates in its second phase.
    """

    NR_REQUIRED_CPUS = 2
    """
    The number of CPUs of same capacity involved in the test
    """

    @classmethod
    def get_migration_cpus(cls, plat_info):
        """
        :returns: :attr:`NR_REQUIRED_CPUS` CPUs of same capacity.
        """
        # Iterate over descending CPU capacity groups
        for cpus in reversed(plat_info["capacity-classes"]):
            if len(cpus) >= cls.NR_REQUIRED_CPUS:
                return cpus[:cls.NR_REQUIRED_CPUS]

        return []

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)

        cpus = cls.get_migration_cpus(target.plat_info)
        if not len(cpus) == cls.NR_REQUIRED_CPUS:
            raise CannotCreateError(
                "This workload requires {} CPUs of identical capacity".format(
                    cls.NR_REQUIRED_CPUS))

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        profile = {}
        cpus = cls.get_migration_cpus(plat_info)

        for task in ["migrating", "static0", "static1"]:
            # An empty RTATask just to sum phases up
            profile[task] = RTATask()

        for i in range(2):
            # A task that will migrate to another CPU
            profile["migrating"] += Periodic(
                duty_cycle_pct=cls.unscaled_utilization(plat_info, cpus[i], 20),
                duration_s=cls.PHASE_DURATION_S, period_ms=cls.TASK_PERIOD_MS,
                cpus=[cpus[i]])

            # Just some tasks that won't move to get some background utilization
            profile["static0"] += Periodic(
                duty_cycle_pct=cls.unscaled_utilization(plat_info, cpus[0], 30),
                duration_s=cls.PHASE_DURATION_S, period_ms=cls.TASK_PERIOD_MS,
                cpus=[cpus[0]])

            profile["static1"] += Periodic(
                duty_cycle_pct=cls.unscaled_utilization(plat_info, cpus[1], 20),
                duration_s=cls.PHASE_DURATION_S, period_ms=cls.TASK_PERIOD_MS,
                cpus=[cpus[1]])

        return profile

class TwoTasksCPUMigration(OneTaskCPUMigration):
    """
    Two tasks on two big CPUs, swap their CPU in the second phase
    """

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        profile = {}
        cpus = cls.get_migration_cpus(plat_info)

        for task in ["migrating0", "migrating1"]:
            # An empty RTATask just to sum phases up
            profile[task] = RTATask()

        for i in range(2):
            # A task that will migrate from CPU A to CPU B
            profile["migrating0"] += Periodic(
                duty_cycle_pct=20, duration_s=cls.PHASE_DURATION_S,
                period_ms=cls.TASK_PERIOD_MS, cpus=[cpus[i]])

            # A task that will migrate from CPU B to CPU A
            profile["migrating1"] += Periodic(
                duty_cycle_pct=20, duration_s=cls.PHASE_DURATION_S,
                period_ms=cls.TASK_PERIOD_MS, cpus=[cpus[1 - i]])

        return profile
