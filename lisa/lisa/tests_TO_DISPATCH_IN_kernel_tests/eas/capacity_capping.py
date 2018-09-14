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

from bart.sched.SchedMultiAssert import SchedMultiAssert
from env import TestEnv
import json
import time
import trappy
import unittest
import os
from wlgen import Periodic, RTA

# Read the config file and update the globals
CONF_FILE = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "capacity_capping.config")

with open(CONF_FILE, "r") as fh:
    CONF_VARS = json.load(fh)
    globals().update(CONF_VARS)

class CapacityCappingTest(unittest.TestCase):
    """
    Goal
    ====

    Verify that dynamic CPU capacity capping works in the system.

    Detailed Description
    ====================

    The maximum frequency of a core can be restricted to a lower value
    than its absolute maximum frequency.  This may happen because of
    thermal management or as a request from userspace via sysfs.
    Dynamic CPU capacity capping provides PELT and the scheduler CPU
    capacity management with a maximum frequency scaling corrector
    which describes the influence of running a CPU with a current
    maximum frequency lower than the absolute maximum frequency.

    The test creates as many busy threads as there are big cpus.
    These busy threads have high load and should run in the CPUs with
    highest capacity.  The test has three phases of equal length.  In
    the first phase, the system runs unconstrained.  In the second
    phase, the maximum frequency of the big cpus is limited to the
    lowest frequency that the big frequency domain can run at.
    Finally, in the third phase, the maximum frequency of the big cpus
    is restored to its absolute maximum, i.e. the system is
    unconstrained again.

    This test assumes that the lowest OPPs of the big cpus have less
    capacity than the highest OPP of the little cpus.  If that is not
    the case, this test will fail.  Arguably, capacity capping is not
    needed in such a system.

    Expected Behaviour
    ==================

    The threads have high load, so they should always run in the CPUs
    with the highest capacity of the system.  In the first phase the
    system is unconstrained, so they should run on the big CPUs.  In
    the second phase, the big cluster's maximum frequency is limited
    and the little CPUs have higher capacity.  Therefore, in the
    second phase of the test, the threads should migrate to the little
    cpus.  In the third phase the maximum frequency of the big cpus is
    restored, so they become again the CPUs with the higest capacity
    in the system.  The busy threads must run on the big cpus in the
    third phase.

    """


    @classmethod
    def setUpClass(cls):
        cls.params = {}
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.trace_file = os.path.join(cls.env.res_dir, "cap_cap.dat")
        cls.populate_params()

        with cls.env.freeze_userspace():
            cls.run_workload()

        trace = trappy.FTrace(cls.trace_file)
        cls.sa = SchedMultiAssert(trace, cls.env.topology,
                                  execnames=cls.params.keys())
        times = cls.sa.getStartTime()
        cls.wload_start_time = min(t["starttime"] for t in times.itervalues())

    @classmethod
    def populate_params(cls):
        for idx in range(len(cls.env.target.bl.bigs)):
            task_name = "busy_thread{}".format(idx)
            cls.params[task_name] = Periodic(
                duty_cycle_pct=BIG_DCYCLE,
                duration_s=WORKLOAD_DURATION_S,
                period_ms=WORKLOAD_PERIOD_MS,
            ).get()

    @classmethod
    def run_workload(cls):
        big_cpu = cls.env.target.bl.bigs[0]
        big_cpufreq = "/sys/devices/system/cpu/cpu{}/cpufreq".format(big_cpu)
        max_freq_path = os.path.join(big_cpufreq, "scaling_max_freq")
        available_freqs_path = os.path.join(big_cpufreq,
                                            "scaling_available_frequencies")

        available_freqs_str = cls.env.target.read_value(available_freqs_path)
        available_freqs = available_freqs_str.split()
        min_frequency = available_freqs[0]
        max_frequency = available_freqs[-1]

        wload = RTA(cls.env.target, "busy_threads",
                    calibration=cls.env.calibration())
        wload.conf(kind="profile", params=cls.params)
        phase_duration = WORKLOAD_DURATION_S / 3.

        cls.env.ftrace.start()

        wload.run(out_dir=cls.env.res_dir, background=True)
        time.sleep(phase_duration)

        # Writing values on the target can take a non-negligible amount of time.
        # To prevent this from shifting the transitions between
        # constrained/unconstrained phases, measure this write latency and
        # reduce our sleep time by that amount.
        def write_and_sleep(max_freq):
            time_before = time.time()
            cls.env.target.write_value(max_freq_path, max_freq)
            write_latency = time.time() - time_before
            if (write_latency > phase_duration):
                raise ValueError(
                    "Latency of Target.write_value greater than phase duration! "
                    "Increase WORKLOAD_DURATION_S or speed up target connection")
            time.sleep(phase_duration - write_latency)

        write_and_sleep(min_frequency)
        write_and_sleep(max_frequency)

        cls.env.ftrace.stop()
        cls.env.ftrace.get_trace(cls.trace_file)

    def check_residencies(self, cpus, cpus_name, window, phase_description):
        """Helper function to check the residencies of all busy threads on a
        given set of cpus for a period of time."""

        residency_dict = self.sa.getResidency("cluster", cpus, window=window,
                                              percent=True)

        for pid, task_res in residency_dict.iteritems():
            msg = "Pid {} ran in {} cpus only {:.2f}% percent of the time when the system was {} (expected {:.2f}%)" \
                .format(pid, cpus_name, task_res["residency"],
                        phase_description, EXPECTED_BUSY_TIME_PCT)

            self.assertGreater(task_res["residency"], EXPECTED_BUSY_TIME_PCT,
                               msg)

    def test_tasks_starts_on_big(self):
        """All busy threads run in the beginning in big cpus"""

        phase_duration = WORKLOAD_DURATION_S / 3.
        unconstrained_window = (self.wload_start_time,
                                self.wload_start_time + phase_duration)
        self.check_residencies(self.env.target.bl.bigs, "big",
                               unconstrained_window, "unconstrained")

    def test_task_migrates_to_little_when_constrained(self):
        """Busy threads migrate to little in the thermally constrained phase"""

        phase_duration = WORKLOAD_DURATION_S / 3.
        mig_start = self.wload_start_time + phase_duration
        mig_end = mig_start + MIGRATION_WINDOW
        num_tasks = len(self.params)

        msg = "One or more of the busy threads didn't migrate to a little cpu between {} and {}" \
              .format(mig_start, mig_end)
        self.assertTrue(self.sa.assertSwitch("cluster", self.env.target.bl.bigs,
                                             self.env.target.bl.littles,
                                             window=(mig_start, mig_end),
                                             rank=num_tasks),
                        msg=msg)

        # The tasks must have migrated by the end of the
        # migration_window and they should not move until the end of
        # the phase.
        constrained_window = (mig_end,
                              self.wload_start_time + (2 * phase_duration))
        self.check_residencies(self.env.target.bl.littles, "little",
                               constrained_window, "thermally constrained")

    def test_task_returns_to_big_when_back_to_unconstrained(self):
        """Busy threads return to big when system goes back to unconstrained

        In the last phase, when the frequency capping is released, busy threads
        return to the big cpus"""

        phase_duration = WORKLOAD_DURATION_S / 3.
        mig_start = self.wload_start_time + 2 * phase_duration
        mig_end = mig_start + MIGRATION_WINDOW
        num_tasks = len(self.params)

        msg = "One of the busy threads didn't return to a big cpu"
        self.assertTrue(self.sa.assertSwitch("cluster",
                                             self.env.target.bl.littles,
                                             self.env.target.bl.bigs,
                                             window=(mig_start, mig_end),
                                             rank=num_tasks),
                        msg=msg)

        # The tasks must have migrated by the end of the
        # migration_window and they should continue to run on bigs
        # until the end of the run.
        last_phase = (mig_end, self.wload_start_time + WORKLOAD_DURATION_S)
        self.check_residencies(self.env.target.bl.bigs, "big",
                               last_phase, "unconstrained")

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
