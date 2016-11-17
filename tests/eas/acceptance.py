# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

import json
import operator
import os
import trappy
import unittest

from bart.sched.SchedAssert import SchedAssert

from devlib.target import TargetError

from env import TestEnv
from test import LisaTest, experiment_test

# Global test configuration parameters
WORKLOAD_DURATION_S = 5
WORKLOAD_PERIOD_MS =  10
SWITCH_WINDOW_HALF = 0.5
SMALL_DCYCLE = 10
BIG_DCYCLE = 100
STEP_HIGH_DCYCLE = 50
STEP_LOW_DCYCLE = 10
EXPECTED_RESIDENCY_PCT = 85
OFFLOAD_EXPECTED_BUSY_TIME_PCT = 97
SET_IS_BIG_LITTLE = True
SET_INITIAL_TASK_UTIL = True
OFFLOAD_MIGRATION_MIGRATOR_DELAY = 1

energy_aware_conf = {
    "tag" : "energy_aware",
    "flags" : "ftrace",
    "sched_features" : "ENERGY_AWARE",
}

class EasTest(LisaTest):
    """
    Base class for EAS tests
    """

    test_conf = {
        "ftrace" : {
            "events" : [
                "sched_overutilized",
                "sched_energy_diff",
                "sched_load_avg_task",
                "sched_load_avg_cpu",
                "sched_migrate_task",
                "sched_switch"
            ],
        },
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(EasTest, cls)._init(*args, **kwargs)

    @classmethod
    def _experimentsInit(cls, *args, **kwargs):
        super(EasTest, cls)._experimentsInit(*args, **kwargs)

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

    def _do_test_first_cpu(self, experiment, tasks):
        """Test that all tasks start on a big CPU"""

        sched_assert = self.get_multi_assert(experiment)

        self.assertTrue(
            sched_assert.assertFirstCpu(
                self.target.bl.bigs,
                rank=len(tasks)),
            msg="Not all the new generated tasks started on a big CPU")

class ForkMigration(EasTest):
    """
    Goal
    ====

    Check that newly created threads start on a big CPU

    Detailed Description
    ====================

    The test spawns as many threads as there are cores in the system.
    It then checks that all threads started on a big core.

    Expected Behaviour
    ==================

    The threads start on a big core.
    """

    experiments_conf = {
        "wloads" : {
            # Create N 100% tasks and M 10% tasks which run in parallel, where N
            # is the number of big CPUs and M is the number of LITTLE CPUs.
            "fmig" : {
                "type" : "rt-app",
                "conf" : {
                    "class" : "profile",
                    "params" : {
                        "small" : {
                            "kind" : "Periodic",
                            "params" : {
                                "duty_cycle_pct": 10,
                                "duration_s": WORKLOAD_DURATION_S,
                                "period_ms": WORKLOAD_PERIOD_MS,
                            },
                            "prefix" : "small",
                            "tasks" : "big",
                        },
                    },
                },
            },
        },
        "confs" : [energy_aware_conf]
    }

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Fork Migration: Test First CPU"""
        self._do_test_first_cpu(experiment, tasks)

class SmallTaskPacking(EasTest):
    """
    Goal
    ====

    Many small tasks are packed in little cpus

    Detailed Description
    ====================

    The tests spawns as many tasks as there are cpus in the system.
    The tasks are small, so none of them should be run on big cpus and
    the scheduler should pack them on little cpus.

    Expected Behaviour
    ==================

    All tasks run on little cpus.
    """

    experiments_conf = {
        "wloads" : {
            "small_tasks" : {
                "type" : "rt-app",
                "conf" : {
                    "class" : "periodic",
                    "params" : {
                        "duty_cycle_pct": 10,
                        "duration_s": WORKLOAD_DURATION_S,
                        "period_ms": WORKLOAD_PERIOD_MS,
                    },
                    # Create one task for each CPU
                    "tasks" : "cpus",
                },
            },
        },
        "confs" : [energy_aware_conf]
    }

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Small Task Packing: test first CPU"""
        self._do_test_first_cpu(experiment, tasks)

    @experiment_test
    def test_small_task_residency(self, experiment, tasks):
        "Small Task Packing: Test Residency (Little Cluster)"

        sched_assert = self.get_multi_assert(experiment)

        self.assertTrue(
            sched_assert.assertResidency(
                "cluster",
                self.target.bl.littles,
                EXPECTED_RESIDENCY_PCT,
                operator.ge,
                percent=True,
                rank=len(tasks)),
            msg="Not all tasks are running on LITTLE cores for at least {}% of their execution time"\
                    .format(EXPECTED_RESIDENCY_PCT))

class OffloadMigrationAndIdlePull(EasTest):
    """
    Goal
    ====

    Big cpus pull big tasks from little cpus when they become idle

    Detailed Description
    ====================

    This test runs twice as many tasks are there are big cpus.  All
    these tasks are big tasks.  Half of them are called
    "early_starter" and the other half "migrator".  The migrator tasks
    start 1 second after the early_starter tasks.  As the big cpus are
    fully utilized when the migrator tasks start, some tasks are
    offloaded to the little cpus.  As the big cpus finish their tasks,
    they pull tasks from the little to complete them.

    Expected Behaviour
    ==================

    As there are as many early_starter tasks as there are big cpus,
    the early_starter tasks should run in the big cpus until they
    finish.  When the migrator tasks start, there is no spare capacity
    in the big cpus so they run on the little cpus.  Once the big cpus
    finish with the early_starters, they should pull the migrator
    tasks and run them.

    It is possible that when the migrator tasks start they do it on
    big cpus and they end up displacing the early starters.  This is
    acceptable behaviour.  As long as big cpus are fully utilized
    running big tasks, the scheduler is doing a good job.

    That is why this test doesn't test for migrations of the migrator
    tasks to the bigs when we expect that the early starters have
    finished.  Instead, it tests that:

      * The big cpus were fully loaded as long as there are tasks left
        to run in the system

      * The little cpus run tasks while the bigs are busy (offload migration)

      * All tasks get a chance on a big cpu (either because they
        started there or because of idle pull)

      * All tasks are finished off in a big cpu.

    """

    conf_basename = "acceptance_offload_idle_pull.config"

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Offload Migration and Idle Pull: Test First CPU"""
        self._do_test_first_cpu(experiment, tasks)

    @experiment_test
    def test_big_cpus_fully_loaded(self, experiment, tasks):
        """Offload Migration and Idle Pull: Big cpus are fully loaded as long as there are tasks left to run in the system"""
        num_big_cpus = len(self.target.bl.bigs)

        sched_assert = self.get_multi_assert(experiment)

        end_times = sorted(self.get_end_times(experiment).values())

        # Window of time until the first migrator finishes
        window = (self.get_start_time(experiment), end_times[-num_big_cpus])
        busy_time = sched_assert.getCPUBusyTime("cluster",
                                            self.target.bl.bigs,
                                            window=window, percent=True)

        msg = "Big cpus were not fully loaded while there were enough big tasks to fill them"
        self.assertGreater(busy_time, OFFLOAD_EXPECTED_BUSY_TIME_PCT, msg=msg)

        # As the migrators start finishing, make sure that the tasks
        # that are left are running on the big cpus
        for i in range(num_big_cpus-1):
            big_cpus_left = num_big_cpus - i - 1
            window = (end_times[-num_big_cpus+i], end_times[-num_big_cpus+i+1])
            busy_time = sched_assert.getCPUBusyTime("cluster",
                                                    self.target.bl.bigs,
                                                    window=window, percent=True)

            expected_busy_time = OFFLOAD_EXPECTED_BUSY_TIME_PCT * \
                                 big_cpus_left / num_big_cpus
            msg = "Big tasks were not running on big cpus from {} to {}".format(
                window[0], window[1])

            self.assertGreater(busy_time, expected_busy_time, msg=msg)

    @experiment_test
    def test_little_cpus_run_tasks(self, experiment, tasks):
        """Offload Migration and Idle Pull: Little cpus run tasks while bigs are busy"""

        num_offloaded_tasks = len(tasks) / 2

        end_times = self.get_end_times(experiment).values()
        first_task_finish_time = min(end_times)

        migrators_assert = self.get_multi_assert(experiment, "migrator")
        start_time = min(t["starttime"]
                         for t in migrators_assert.getStartTime().itervalues())
        migrator_activation_time = start_time + OFFLOAD_MIGRATION_MIGRATOR_DELAY

        window = (migrator_activation_time, first_task_finish_time)

        all_tasks_assert = self.get_multi_assert(experiment)

        busy_time = all_tasks_assert.getCPUBusyTime("cluster",
                                                    self.target.bl.littles,
                                                    window=window)

        window_len = window[1] - window[0]
        expected_busy_time = window_len * num_offloaded_tasks * \
                             OFFLOAD_EXPECTED_BUSY_TIME_PCT / 100.
        msg = "Little cpus did not pick up big tasks while big cpus were fully loaded"

        self.assertGreater(busy_time, expected_busy_time, msg=msg)

    @experiment_test
    def test_all_tasks_run_on_a_big_cpu(self, experiment, tasks):
        """Offload Migration and Idle Pull: All tasks run on a big cpu at some point

        Note: this test may fail in big.LITTLE platforms in which the
        little cpus are almost as performant as the big ones.

        """
        for task in tasks:
            sa = SchedAssert(experiment.out_dir, self.te.topology, execname=task)
            end_times = self.get_end_times(experiment)
            window = (0, end_times[task])
            big_residency = sa.getResidency("cluster", self.target.bl.bigs,
                                            window=window, percent=True)

            msg = "Task {} didn't run on a big cpu.".format(task)
            self.assertGreater(big_residency, 0, msg=msg)

    @experiment_test
    def test_all_tasks_finish_on_a_big_cpu(self, experiment, tasks):
        """Offload Migration and Idle Pull: All tasks finish on a big cpu

        Note: this test may fail in big.LITTLE systems where the
        little cpus' performance is comparable to the bigs' and they
        can take almost the same time as a big cpu to complete a
        task.

        """
        for task in tasks:
            sa = SchedAssert(experiment.out_dir, self.te.topology, execname=task)

            msg = "Task {} did not finish on a big cpu".format(task)
            self.assertIn(sa.getLastCpu(), self.target.bl.bigs, msg=msg)


class WakeMigration(EasTest):
    """
    Goal
    ====

    A task that switches between being high and low utilization moves
    to big and little cores accordingly

    Detailed Description
    ====================

    This test creates as many tasks as there are big cpus.  The tasks
    alternate between high and low utilization.  They start being
    small load for 5 seconds, they become big for another 5 seconds,
    then small for another 5 seconds and finally big for the last 5
    seconds.

    Expected Behaviour
    ==================

    The tasks should run on the litlle cpus when they are small and in
    the big cpus when they are big.
    """

    conf_basename = "acceptance_wake_migration.config"

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Wake Migration: Test First CPU"""
        self._do_test_first_cpu(experiment, tasks)

    def _assert_switch(self, experiment, expected_switch_to, phases):
        if expected_switch_to == "big":
            switch_from = self.target.bl.littles
            switch_to   = self.target.bl.bigs
        elif expected_switch_to == "little":
            switch_from = self.target.bl.bigs
            switch_to   = self.target.bl.littles
        else:
            raise ValueError("Invalid expected_switch_to")

        sched_assert = self.get_multi_assert(experiment)

        expected_time = (self.get_start_time(experiment)
                         + phases*WORKLOAD_DURATION_S)
        switch_window = (max(expected_time - SWITCH_WINDOW_HALF, 0),
                         expected_time + SWITCH_WINDOW_HALF)

        fmt = "Not all tasks wake-migrated to {} cores in the expected window: {}"
        msg = fmt.format(expected_switch_to, switch_window)

        self.assertTrue(
            sched_assert.assertSwitch(
                "cluster",
                switch_from,
                switch_to,
                window=switch_window,
                rank=len(experiment.wload.tasks)),
            msg=msg)

    @experiment_test
    def test_little_big_switch1(self, experiment, tasks):
        """Wake Migration: LITTLE -> BIG: 1"""
        self._assert_switch(experiment, "big", 1)

    @experiment_test
    def test_little_big_switch2(self, experiment, tasks):
        """Wake Migration: LITTLE -> BIG: 2"""

        # little - big - little - big
        #                       ^
        # We want to test that this little to big migration happens.  So we skip
        # the first three phases.
        self._assert_switch(experiment, "big", 3)

    @experiment_test
    def test_big_little_switch1(self, experiment, tasks):
        """Wake Migration: BIG -> LITLLE: 1"""
        self._assert_switch(experiment, "little", 0)

    @experiment_test
    def test_big_little_switch2(self, experiment, tasks):
        """Wake Migration: BIG -> LITLLE: 2"""

        # little - big - little - big
        #              ^
        # We want to test that this big to little migration happens.  So we skip
        # the first two phases.
        self._assert_switch(experiment, "little", 2)
