#    Copyright 2015-2015 ARM Limited
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


import utils_tests
import unittest
import cr2
from sheye.SchedAssert import SchedAssert
from cr2.stats.Topology import Topology

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

        r = cr2.Run()
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
        window=(0.0034, 0.003525)
        expected_time=0.000125
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time, places=9)

        # Interrupted Window
        window=(0.0030, 0.0032)
        expected_time=0.000166
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time, places=9)

        # A window with multiple interruptions
        window = (0.0027, 0.0036)
        expected_time = 0.000817
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time, places=9)
