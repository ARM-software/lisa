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

from env import TestEnv
from wlgen import RTA, Periodic, Step
from devlib.target import TargetError

import trappy
import os
import unittest
import logging
import json
import logging

logging.basicConfig(level=logging.INFO)
# Read the config file and update the globals
CONF_FILE = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "acceptance.config")

with open(CONF_FILE, "r") as fh:
    conf_vars = json.load(fh)
    globals().update(conf_vars)


def local_setup(env):
    env.target.cpufreq.set_all_governors("performance")

    if ENABLE_EAS:
        env.target.execute(
            "echo ENERGY_AWARE > /sys/kernel/debug/sched_features")

    if SET_IS_BIG_LITTLE:
        try:
            env.target.write_value("/proc/sys/kernel/sched_is_big_little", 1)
        except TargetError:
            # That flag doesn't exist on mainline-integration kernels, so don't
            # worry if the file isn't present.
            pass

SMALL_WORKLOAD = {

    "duty_cycle_pct": SMALL_DCYCLE,
    "duration_s": WORKLOAD_DURATION_S,
    "period_ms": WORKLOAD_PERIOD_MS,
}

BIG_WORKLOAD = {

    "duty_cycle_pct": BIG_DCYCLE,
    "duration_s": WORKLOAD_DURATION_S,
    "period_ms": WORKLOAD_PERIOD_MS,
}

STEP_WORKLOAD = {

    "start_pct": STEP_LOW_DCYCLE,
    "end_pct": STEP_HIGH_DCYCLE,
    "time_s": WORKLOAD_DURATION_S,
    "loops": 2
}

from bart.sched.SchedAssert import SchedAssert
from bart.sched.SchedMultiAssert import SchedMultiAssert
import operator
import json


def log_result(data, log_fh):
    result_str = json.dumps(data, indent=3)
    logging.info(result_str)
    log_fh.write(result_str)


class ForkMigration(unittest.TestCase):
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

    @classmethod
    def setUpClass(cls):
        cls.params = {}
        cls.task_prefix = "fmig"
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.trace_file = os.path.join(cls.env.res_dir, "fork_migration.dat")
        cls.log_file = os.path.join(cls.env.res_dir, "fork_migration.json")
        cls.populate_params()
        cls.tasks = cls.params.keys()
        cls.num_tasks = len(cls.tasks)
        local_setup(cls.env)
        cls.run_workload()
        cls.log_fh = open(os.path.join(cls.env.res_dir, cls.log_file), "w")

    @classmethod
    def tearDownClass(cls):
        cls.log_fh.close()

    @classmethod
    def populate_params(cls):
        big_prefix = cls.task_prefix + "_big"
        for idx in range(len(cls.env.target.bl.bigs)):
            task = big_prefix + str(idx)
            cls.params[task] = Periodic(**BIG_WORKLOAD).get()

        little_prefix = cls.task_prefix + "_little"
        for idx in range(len(cls.env.target.bl.littles)):
            task = little_prefix + str(idx)
            cls.params[task] = Periodic(**SMALL_WORKLOAD).get()

    @classmethod
    def run_workload(cls):
        wload = RTA(
            cls.env.target,
            "fork_migration",
            calibration=cls.env.calibration())
        wload.conf(kind="profile", params=cls.params)
        cls.env.ftrace.start()
        wload.run(
            out_dir=cls.env.res_dir,
            background=False)
        cls.env.ftrace.stop()
        trace = cls.env.ftrace.get_trace(cls.trace_file)

    def test_first_cpu(self):
        "Fork Migration: Test First CPU"

        logging.info("Fork Migration: Test First CPU")
        f_assert = SchedMultiAssert(
            self.trace_file,
            self.env.topology,
            execnames=self.tasks)

        log_result(
            f_assert.getFirstCpu(), self.log_fh)

        self.assertTrue(
            f_assert.assertFirstCpu(
                self.env.target.bl.bigs,
                rank=self.num_tasks),
            msg="Not all the new generated tasks started on a big CPU")


class SmallTaskPacking(unittest.TestCase):
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

    @classmethod
    def setUpClass(cls):
        cls.params = {}
        cls.task_prefix = "stp"
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.trace_file = os.path.join(
            cls.env.res_dir,
            "small_task_packing.dat")
        cls.log_file = os.path.join(cls.env.res_dir, "small_task_packing.json")
        cls.num_tasks = len(cls.env.target.bl.bigs + cls.env.target.bl.littles)
        cls.populate_params()
        cls.tasks = cls.params.keys()
        local_setup(cls.env)
        cls.run_workload()
        cls.s_assert = SchedMultiAssert(
            cls.trace_file,
            cls.env.topology,
            execnames=cls.tasks)
        cls.log_fh = open(os.path.join(cls.env.res_dir, cls.log_file), "w")

    @classmethod
    def tearDownClass(cls):
        cls.log_fh.close()

    @classmethod
    def populate_params(cls):
        for i in range(cls.num_tasks):
            task = cls.task_prefix + str(i)
            cls.params[task] = Periodic(**SMALL_WORKLOAD).get()

    @classmethod
    def run_workload(cls):
        wload = RTA(
            cls.env.target,
            "small_task_packing",
            calibration=cls.env.calibration())
        wload.conf(kind="profile", params=cls.params)
        cls.env.ftrace.start()
        wload.run(
            out_dir=cls.env.res_dir,
            background=False)
        cls.env.ftrace.stop()
        trace = cls.env.ftrace.get_trace(cls.trace_file)

    def test_small_task_pack_first_cpu(self):
        "Small Task Packing: First CPU: BIG"

        logging.info("Small Task Packing: First CPU: BIG\n")
        log_result(self.s_assert.getFirstCpu(), self.log_fh)
        self.assertTrue(
            self.s_assert.assertFirstCpu(
                self.env.target.bl.bigs,
                rank=self.num_tasks),
            msg="Not all the new generated tasks started on a big CPU")

    def test_small_task_residency(self):
        "Small Task Packing: Test Residency (Little Cluster)"

        logging.info("Small Task Packing: Test Residency (Little Cluster)")
        log_result(
            self.s_assert.getResidency(
                "cluster",
                self.env.target.bl.littles,
                percent=True), self.log_fh)

        self.assertTrue(
            self.s_assert.assertResidency(
                "cluster",
                self.env.target.bl.littles,
                EXPECTED_RESIDENCY_PCT,
                operator.ge,
                percent=True,
                rank=self.num_tasks),
            msg="Not all tasks are running on LITTLE cores for at least {}% of their execution time"\
                    .format(EXPECTED_RESIDENCY_PCT))


class OffloadMigrationAndIdlePull(unittest.TestCase):
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

    @classmethod
    def setUpClass(cls):
        cls.params = {}
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.trace_file = os.path.join(cls.env.res_dir, "offload_idle_pull.dat")
        cls.log_file = os.path.join(cls.env.res_dir, "offload_idle_pull.json")
        cls.early_starters = []
        cls.migrators = []
        cls.num_tasks = len(cls.env.target.bl.bigs)
        cls.populate_tasks()
        local_setup(cls.env)
        cls.run_workload()

        cls.trace = trappy.FTrace(cls.trace_file)
        cls.m_assert = SchedMultiAssert(cls.trace, cls.env.topology,
                                        execnames=cls.migrators)
        cls.e_assert = SchedMultiAssert(cls.trace, cls.env.topology,
                                        execnames=cls.early_starters)

        all_tasks = cls.early_starters + cls.migrators
        cls.a_assert = SchedMultiAssert(cls.trace, cls.env.topology,
                                        execnames=all_tasks)
        cls.offset = cls.get_offset()

        cls.end_times = cls.calculate_end_times()
        cls.log_fh = open(os.path.join(cls.env.res_dir, cls.log_file), "w")

    @classmethod
    def tearDownClass(cls):
        cls.log_fh.close()

    @classmethod
    def populate_tasks(cls):
        migrator_workload = BIG_WORKLOAD.copy()
        migrator_workload["duration_s"] = 9
        migrator_workload["delay_s"] = OFFLOAD_MIGRATION_MIGRATOR_DELAY

        for idx in range(cls.num_tasks):
            task = "early_starters" + str(idx)
            cls.params[task] = Periodic(**BIG_WORKLOAD).get()
            cls.early_starters.append(task)

            # Tasks that will be idle pulled
            task = "migrator" + str(idx)
            cls.params[task] = Periodic(**migrator_workload).get()
            cls.migrators.append(task)

    @classmethod
    def run_workload(cls):

        wload = RTA(
            cls.env.target,
            "offload_idle_pull",
            calibration=cls.env.calibration())
        wload.conf(kind="profile", params=cls.params)
        cls.env.ftrace.start()
        wload.run(
            out_dir=cls.env.res_dir,
            background=False)
        cls.env.ftrace.stop()
        trace = cls.env.ftrace.get_trace(cls.trace_file)

    @classmethod
    def get_offset(cls):
        task_start_times = cls.a_assert.getStartTime().values()
        return min([t['starttime'] for t in task_start_times])

    @classmethod
    def calculate_end_times(cls):

        end_times = {}
        for task in cls.params.keys():
            sched_assert = SchedAssert(cls.trace, cls.env.topology,
                                       execname=task)
            end_times[task] = sched_assert.getEndTime()

        return end_times

    def get_migrator_activation_time(self):
        start_times_dict = self.m_assert.getStartTime()
        start_time = min(t['starttime'] for t in start_times_dict.itervalues())

        return start_time + OFFLOAD_MIGRATION_MIGRATOR_DELAY

    def test_first_cpu_early_starters(self):
        """Offload Migration and Idle Pull: Test First CPU (Early Starters)"""

        logging.info(
            "Offload Migration and Idle Pull: Test First CPU (Early Starters)")
        log_result(
            self.e_assert.getFirstCpu(), self.log_fh)

        self.assertTrue(
            self.e_assert.assertFirstCpu(
                self.env.target.bl.bigs,
                rank=self.num_tasks),
            msg="Not all the new 'early starter' tasks started on a big CPU")

    def test_first_cpu_migrators(self):
        "Offload Migration and Idle Pull: Test First CPU (Migrators)"

        logging.info(
            "Offload Migration and Idle Pull: Test First CPU (Migrators)")

        log_result(
            self.m_assert.getFirstCpu(), self.log_fh)

        self.assertTrue(
            self.m_assert.assertFirstCpu(
                self.env.target.bl.bigs,
                rank=self.num_tasks),
            msg="Not all the new 'migrator' tasks started on a big CPU")

    def test_big_cpus_fully_loaded(self):
        """Offload Migration and Idle Pull: Big cpus are fully loaded as long as there are tasks left to run in the system"""
        num_big_cpus = len(self.env.target.bl.bigs)

        end_times = sorted(self.end_times.values())

        # Window of time until the first migrator finishes
        window = (self.offset, end_times[-num_big_cpus])
        busy_time = self.a_assert.getCPUBusyTime("cluster",
                                                 self.env.target.bl.bigs,
                                                 window=window, percent=True)
        msg = "Big cpus were not fully loaded while there were enough big tasks to fill them"
        self.assertGreater(busy_time, OFFLOAD_EXPECTED_BUSY_TIME_PCT, msg=msg)

        # As the migrators start finishing, make sure that the tasks
        # that are left are running on the big cpus
        for i in range(num_big_cpus-1):
            big_cpus_left = num_big_cpus - i - 1
            window = (end_times[-num_big_cpus+i], end_times[-num_big_cpus+i+1])
            busy_time = self.a_assert.getCPUBusyTime("cluster",
                                                     self.env.target.bl.bigs,
                                                     window=window, percent=True)

            expected_busy_time = OFFLOAD_EXPECTED_BUSY_TIME_PCT * \
                                 big_cpus_left / num_big_cpus
            msg = "Big tasks were not running on big cpus from {} to {}".format(
                window[0], window[1])

            self.assertGreater(busy_time, expected_busy_time, msg=msg)

    def test_little_cpus_run_tasks(self):
        """Offload Migration and Idle Pull: Little cpus run tasks while bigs are busy"""
        tasks = self.params.keys()
        num_offloaded_tasks = len(tasks) / 2

        first_task_finish_time = None
        for task in tasks:
            end_time = self.end_times[task]
            if not first_task_finish_time or (end_time < first_task_finish_time):
                first_task_finish_time = end_time

        window = (self.get_migrator_activation_time(), first_task_finish_time)
        busy_time = self.a_assert.getCPUBusyTime("cluster",
                                                 self.env.target.bl.littles,
                                                 window=window)

        window_len = window[1] - window[0]
        expected_busy_time = window_len * num_offloaded_tasks * \
                             OFFLOAD_EXPECTED_BUSY_TIME_PCT / 100.
        msg = "Little cpus did not pick up big tasks while big cpus were fully loaded"

        self.assertGreater(busy_time, expected_busy_time, msg=msg)

    def test_all_tasks_run_on_a_big_cpu(self):
        """Offload Migration and Idle Pull: All tasks run on a big cpu at some point

        Note: this test may fail in big.LITTLE platforms in which the
        little cpus are almost as performant as the big ones.

        """

        for task in self.params.keys():
            sa = SchedAssert(self.trace, self.env.topology, execname=task)
            window = (0, self.end_times[task])
            big_residency = sa.getResidency("cluster", self.env.target.bl.bigs,
                                            window=window, percent=True)
            log_result(big_residency, self.log_fh)

            msg = "Task {} didn't run on a big cpu.".format(task)
            self.assertGreater(big_residency, 0, msg=msg)

    def test_all_tasks_finish_on_a_big_cpu(self):
        """Offload Migration and Idle Pull: All tasks finish on a big cpu

        Note: this test may fail in big.LITTLE systems where the
        little cpus' performance is comparable to the bigs' and they
        can take almost the same time as a big cpu to complete a
        task.

        """

        for task in self.params.keys():
            sa = SchedAssert(self.trace, self.env.topology, execname=task)

            msg = "Task {} did not finish on a big cpu".format(task)
            self.assertIn(sa.getLastCpu(), self.env.target.bl.bigs, msg=msg)


class WakeMigration(unittest.TestCase):
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

    @classmethod
    def setUpClass(cls):
        cls.params = {}
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.task_prefix = "wmig"
        cls.trace_file = os.path.join(cls.env.res_dir, "wake_migration.dat")
        cls.log_file = os.path.join(cls.env.res_dir, "wake_migration.json")
        cls.populate_params()
        cls.tasks = cls.params.keys()
        cls.num_tasks = len(cls.tasks)
        local_setup(cls.env)
        cls.run_workload()
        cls.s_assert = SchedMultiAssert(
            cls.trace_file,
            cls.env.topology,
            execnames=cls.tasks)
        cls.offset = cls.get_offset(cls.tasks[0])
        cls.log_fh = open(os.path.join(cls.env.res_dir, cls.log_file), "w")

    @classmethod
    def tearDownClass(cls):
        cls.log_fh.close()

    @classmethod
    def populate_params(cls):
        num_big_cpus = len(cls.env.target.bl.bigs)

        for i in range(num_big_cpus):
            task_name = "{}_{}".format(cls.task_prefix, i)
            cls.params[task_name] = Step(**STEP_WORKLOAD).get()

        cls.phase_duration = STEP_WORKLOAD["time_s"]

    @classmethod
    def run_workload(cls):
        wload = RTA(
            cls.env.target,
            "wake_migration",
            calibration=cls.env.calibration())
        wload.conf(kind="profile", params=cls.params)
        cls.env.ftrace.start()
        wload.run(
            out_dir=cls.env.res_dir,
            background=False)
        cls.env.ftrace.stop()
        trace = cls.env.ftrace.get_trace(cls.trace_file)

    @classmethod
    def get_offset(cls, task_name):
        return SchedAssert(
            cls.trace_file,
            cls.env.topology,
            execname=task_name).getStartTime()

    def test_first_cpu(self):
        """Wake Migration: Test First CPU"""

        logging.info("Wake Migration: Test First CPU")

        log_result(self.s_assert.getFirstCpu(), self.log_fh)

        self.assertTrue(
            self.s_assert.assertFirstCpu(
                self.env.target.bl.bigs,
                rank=self.num_tasks),
            msg="Not all the new generated tasks started on a big CPU")


    def test_little_big_switch1(self):
        """Wake Migration: LITTLE -> BIG: 1"""
        expected_time = self.offset + self.phase_duration
        switch_window = (
            expected_time -
            SWITCH_WINDOW_HALF,
            expected_time +
            SWITCH_WINDOW_HALF)

        logging.info(
            "Wake Migration: LITTLE -> BIG Window: {}".format(switch_window))

        log_result(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.littles,
                self.env.target.bl.bigs,
                window=switch_window), self.log_fh)

        self.assertTrue(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.littles,
                self.env.target.bl.bigs,
                rank=self.num_tasks,
                window=switch_window),
            msg="Not all tasks are wake-migrated to big cores in the expected window: {}"\
                    .format(switch_window))

    def test_little_big_switch2(self):
        """Wake Migration: LITTLE -> BIG: 2"""

        # little - big - little - big
        #                       ^
        # We want to test that this little to big migration happens.  So we skip
        # the first three phases.
        expected_time = self.offset + 3 * self.phase_duration
        switch_window = (
            expected_time -
            SWITCH_WINDOW_HALF,
            expected_time +
            SWITCH_WINDOW_HALF)

        logging.info(
            "Wake Migration: LITTLE -> BIG Window: {}".format(switch_window))

        log_result(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.littles,
                self.env.target.bl.bigs,
                window=switch_window), self.log_fh)

        self.assertTrue(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.littles,
                self.env.target.bl.bigs,
                rank=self.num_tasks,
                window=switch_window),
            msg="Not all tasks are wake-migrated to big cores in the expected window: {}"\
                    .format(switch_window))

    def test_big_little_switch1(self):
        """Wake Migration: BIG -> LITLLE: 1"""
        expected_time = self.offset
        switch_window = (
            max(expected_time - SWITCH_WINDOW_HALF, 0), expected_time + SWITCH_WINDOW_HALF)

        logging.info(
            "Wake Migration: BIG -> LITTLE Window: {}".format(switch_window))

        log_result(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.bigs,
                self.env.target.bl.littles,
                window=switch_window), self.log_fh)

        self.assertTrue(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.bigs,
                self.env.target.bl.littles,
                rank=self.num_tasks,
                window=switch_window),
            msg="Not all tasks are wake-migrated to LITTLE cores in the expected window: {}"\
                    .format(switch_window))

    def test_big_little_switch2(self):
        """Wake Migration: BIG -> LITLLE: 2"""

        # little - big - little - big
        #              ^
        # We want to test that this big to little migration happens.  So we skip
        # the first two phases.
        expected_time = self.offset + 2 * self.phase_duration
        switch_window = (
            expected_time -
            SWITCH_WINDOW_HALF,
            expected_time +
            SWITCH_WINDOW_HALF)

        logging.info(
            "Wake Migration: BIG -> LITTLE Window: {}".format(switch_window))

        log_result(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.bigs,
                self.env.target.bl.littles,
                window=switch_window), self.log_fh)

        self.assertTrue(
            self.s_assert.assertSwitch(
                "cluster",
                self.env.target.bl.bigs,
                self.env.target.bl.littles,
                rank=self.num_tasks,
                window=switch_window),
            msg="Not all tasks are wake-migrated to LITTLE cores in the expected window: {}"\
                    .format(switch_window))
