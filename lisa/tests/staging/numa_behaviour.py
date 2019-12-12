# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Linaro and contributors.
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

from lisa.wlgen.rta import Periodic
from lisa.tests.base import ResultBundle, RTATestBundle, TestMetric, CannotCreateError
from lisa.datautils import df_deduplicate
from lisa.analysis.tasks import TasksAnalysis

class NUMABehaviour(RTATestBundle):
    """
    Abstract class for NUMA related scheduler testing.
    """
    @classmethod
    def check_from_target(cls, target):
        if target.number_of_nodes < 2:
            raise CannotCreateError(
                "Target doesn't have at least two NUMA nodes")

    @TasksAnalysis.df_task_states.used_events
    def _get_task_cpu_df(self, task_id):
        """
        Get a DataFrame for task migrations

        Use the sched_switch trace event to find task migration from one CPU to another.

        :returns: A Pandas DataFrame for the task, showing the
                  CPU's that the task was migrated to
        """
        df = self.trace.analysis.tasks.df_task_states(task_id)
        cpu_df = df_deduplicate(df, cols=['cpu'], keep='first', consecutives=True)

        return cpu_df

    @_get_task_cpu_df.used_events
    def test_task_remains(self) -> ResultBundle:
        """
        Test that task remains on the same core
        """
        test_passed = True
        metrics = {}

        for task_id in self.rtapp_task_ids:
            cpu_df = self._get_task_cpu_df(task_id)
            core_migrations = len(cpu_df.index)
            metrics[task_id] = TestMetric(core_migrations)

            # Ideally, task with 50% utilization
            # should stay on the same core
            if  core_migrations > 1:
                test_passed = False

        res = ResultBundle.from_bool(test_passed)
        res.add_metric("Migrations", metrics)

        return res

class NUMASmallTaskPlacement(NUMABehaviour):
    """
    A single task with 50% utilization
    """

    task_prefix = "tsk"

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        rtapp_profile = {}
        rtapp_profile[cls.task_prefix] = Periodic(
            duty_cycle_pct=50,
            duration_s=30,
            period_ms=cls.TASK_PERIOD_MS
        )

        return rtapp_profile

class NUMAMultipleTasksPlacement(NUMABehaviour):
    """
    Multiple tasks with 50% utilization
    """
    task_prefix = "tsk"

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        # Four CPU's is enough to demonstrate task migration problem
        cpu_count = min(4, plat_info["cpus-count"])

        rtapp_profile = {}
        for cpu in range(cpu_count):
            rtapp_profile["{}{}".format(cls.task_prefix, cpu)] = Periodic(
                duty_cycle_pct=50,
                duration_s=30,
                period_ms=cls.TASK_PERIOD_MS
            )

        return rtapp_profile
# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
