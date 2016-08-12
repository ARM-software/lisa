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
import trappy
import shutil
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
    "hmp_parity.config")

with open(CONF_FILE, "r") as fh:
    conf_vars = json.load(fh)
    globals().update(conf_vars)


def local_setup(env):
    env.target.cpufreq.set_all_governors("performance")

    if ENABLE_EAS:
        env.target.execute(
            "echo ENERGY_AWARE > /sys/kernel/debug/sched_features")


def between_threshold_pct(a, b):
    THRESHOLD_PERCENT = 3
    lower = b - THRESHOLD_PERCENT
    upper = b + THRESHOLD_PERCENT

    if a >= lower and a <= upper:
        return True
    return False


def between_threshold_abs(a, b):
    THRESHOLD = 0.25
    lower = b - THRESHOLD
    upper = b + THRESHOLD

    if a >= lower and a <= upper:
        return True
    return False


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
    The tasks are small, so none of them should be run on big cpus,
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
    start 1 second after the early_starter tasks.  The test checks
    that when the big cpus finish executing the early starter tasks,
    the scheduler moves the migrator tasks to the big cpus.

    Expected Behaviour
    ==================

    As there are as many early_starter tasks as there are big cpus,
    the early_starter tasks should run in the big cpus until they
    finish.  When the migrator tasks start, there is no spare capacity
    in the big cpus so they run on the little cpus.  Once the big cpus
    finish with the early_starters, they should pull the migrator
    tasks and run them.
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

        cls.offset = cls.get_offset(cls.early_starters[0])

        cls.m_assert = SchedMultiAssert(
            cls.trace_file,
            cls.env.topology,
            execnames=cls.migrators)

        cls.e_assert = SchedMultiAssert(
            cls.trace_file,
            cls.env.topology,
            execnames=cls.early_starters)
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
    def get_offset(cls, task_name):
        return SchedAssert(
            cls.trace_file,
            cls.env.topology,
            execname=task_name).getStartTime()

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

    def test_little_res_migrators(self):
        "Offload Migration and Idle Pull: Test Little Residency (Migrators)"
        little_residency_window = (self.offset + 1, self.offset + 5)

        logging.info(
            "Offload Migration and Idle Pull: Test Little Residency (Migrators)")

        log_result(
            self.m_assert.getResidency(
                "cluster",
                self.env.target.bl.littles,
                percent=True,
                window=little_residency_window
            ), self.log_fh)

        self.assertTrue(
            self.m_assert.assertResidency(
                "cluster",
                self.env.target.bl.littles,
                EXPECTED_RESIDENCY_PCT,
                operator.ge,
                percent=True,
                window=little_residency_window,
                rank=self.num_tasks),
            msg="Not all 'migrator' tasks are running on LITTLE cores for at least {}% of their execution time"\
                    .format(EXPECTED_RESIDENCY_PCT))

    def test_big_res_migrators(self):
        "Offload Migration and Idle Pull: Test Big Residency (Migrators)"
        big_residency_window = (self.offset + 5, self.offset + 10)

        logging.info(
            "Offload Migration and Idle Pull: Test Big Residency (Migrators)")

        log_result(
            self.m_assert.getResidency(
                "cluster",
                self.env.target.bl.bigs,
                percent=True,
                window=big_residency_window
            ), self.log_fh)

        self.assertTrue(
            self.m_assert.assertResidency(
                "cluster",
                self.env.target.bl.bigs,
                EXPECTED_RESIDENCY_PCT,
                operator.ge,
                percent=True,
                window=big_residency_window,
                rank=self.num_tasks),
            msg="Not all 'migrator' tasks are running on big cores for at least {}% of their execution time"\
                    .format(EXPECTED_RESIDENCY_PCT))


    def test_migrators_switch(self):
        "Offload Migration and Idle Pull: Test LITTLE -> BIG Idle Pull Switch (Migrators)"
        switch_window = (self.offset + 4.5, self.offset + 5.5)

        logging.info(
            "Offload Migration and Idle Pull: Test LITTLE -> BIG Idle Pull Switch (Migrators)")
        log_result(
            self.m_assert.assertSwitch(
                "cluster",
                self.env.target.bl.littles,
                self.env.target.bl.bigs,
                window=switch_window), self.log_fh)

        self.assertTrue(
            self.m_assert.assertSwitch(
                "cluster",
                self.env.target.bl.littles,
                self.env.target.bl.bigs,
                window=switch_window,
                rank=self.num_tasks),
            msg="Not all 'migrator' tasks are pulled by idle big cores when expected")


    def test_big_res_early_starters(self):
        """Offload Migration and Idle Pull: Test Big Residency (EarlyStarters)"""
        logging.info(
            "Offload Migration and Idle Pull: Test Big Residency (EarlyStarters)")

        big_residency_window = (self.offset, self.offset + 5)

        log_result(self.e_assert.getResidency(
            "cluster",
            self.env.target.bl.bigs,
            percent=True,
            window=big_residency_window), self.log_fh)

        self.assertTrue(
            self.e_assert.assertResidency(
                "cluster",
                self.env.target.bl.bigs,
                EXPECTED_RESIDENCY_PCT,
                operator.ge,
                percent=True,
                window=big_residency_window,
                rank=self.num_tasks),
            msg="Not all 'early starter' tasks are running on big cores for at least {}% of their execution time"\
                    .format(EXPECTED_RESIDENCY_PCT))


class WakeMigration(unittest.TestCase):
    """
    Goal
    ====

    A task that switches between being big and small moves to big and little cores accordingly

    Detailed Description
    ====================

    This test creates two tasks that alternate between being big and
    small workload.  They start being small load for 5 seconds, they
    become big for another 5 seconds, then small for another 5 seconds
    and finally big for the last 5 seconds.

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
        cls.params[cls.task_prefix] = Step(**STEP_WORKLOAD).get()
        cls.params[cls.task_prefix + "1"] = Step(**STEP_WORKLOAD).get()

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
        expected_time = self.offset + 5
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

        expected_time = self.offset + 15
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

        expected_time = self.offset + 10
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
