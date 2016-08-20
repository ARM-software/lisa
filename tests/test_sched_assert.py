#    Copyright 2015-2016 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from bart.sched.SchedAssert import SchedAssert
from bart.sched.SchedMultiAssert import SchedMultiAssert
import trappy
from trappy.stats.Topology import Topology
import unittest

import utils_tests


@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestSchedAssert(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):

        self.BIG = [1,2]
        self.LITTLE = [0, 3, 4, 5]
        self.clusters = [self.BIG, self.LITTLE]
        self.topology = Topology(clusters=self.clusters)
        super(TestSchedAssert, self).__init__(
             [("raw_trace.dat", "trace.dat")],
             *args,
             **kwargs)

    def test_get_runtime(self):

        r = trappy.FTrace()
        # The ls process is process we are
        # testing against with pre calculated
        # values
        process = "ls"

        # Complete duration
        expected_time = 0.0034740000264719129
        s = SchedAssert(r, self.topology, execname=process)
        self.assertAlmostEqual(s.getRuntime(), expected_time, places=9)
        self.assertAlmostEqual(s.getRuntime(), expected_time, places=9)

        # Non Interrupted Window
        window = (0.0034, 0.003525)
        expected_time = 0.000125
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time,
                               places=9)

        # Interrupted Window
        window = (0.0030, 0.0032)
        expected_time = 0.000166
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time,
                               places=9)

        # A window with multiple interruptions
        window = (0.0027, 0.0036)
        expected_time = 0.000817
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time,
                               places=9)

    def test_get_last_cpu(self):
        """SchedAssert.getLastCpu() gives you the last cpu in which a task ran"""
        expected_last_cpu = 5

        sa = SchedAssert("trace.dat", self.topology, execname="ls")
        self.assertEqual(sa.getLastCpu(), expected_last_cpu)

class TestSchedMultiAssert(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        self.big = [1,2]
        self.little = [0, 3, 4, 5]
        self.clusters = [self.big, self.little]
        self.all_cpus = sorted(self.big + self.little)
        self.topology = Topology(clusters=self.clusters)
        super(TestSchedMultiAssert, self).__init__(
             [("raw_trace.dat", "trace.dat")],
             *args,
             **kwargs)

    def test_cpu_busy_time(self):
        """SchedMultiAssert.getCPUBusyTime() work"""

        # precalculated values against these processes in the trace
        pids = [4729, 4734]
        first_time = .000214
        last_time = .003171

        tr = trappy.FTrace()
        sma = SchedMultiAssert(tr, self.topology, pids=pids)

        expected_busy_time = 0.0041839999754810708
        busy_time = sma.getCPUBusyTime("all", self.all_cpus, window=(first_time, last_time))
        self.assertAlmostEqual(busy_time, expected_busy_time)

        # percent calculation
        expected_busy_pct = 23.582459561949445
        busy_pct= sma.getCPUBusyTime("all", self.all_cpus, percent=True,
                                     window=(first_time, last_time))
        self.assertAlmostEqual(busy_pct, expected_busy_pct)

        # percent without a window
        expected_busy_pct = 23.018818156540004
        busy_pct= sma.getCPUBusyTime("cluster", self.little, percent=True)
        self.assertAlmostEqual(busy_pct, expected_busy_pct)
