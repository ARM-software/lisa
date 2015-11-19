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


import os
import sys

import utils_tests
import trappy

sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "trappy"))

class BaseTestSched(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(BaseTestSched, self).__init__(
             [("trace_sched.txt", "trace.txt")],
             *args,
             **kwargs)

class TestSchedLoadAvgSchedGroup(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgSchedGroup creates a proper data_frame"""
        dfr = trappy.Run().sched_load_avg_sched_group.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpus"].iloc[0], "00000002")
        self.assertEquals(dfr["load"].iloc[0], 0)
        self.assertEquals(dfr["utilization"].iloc[0], 0)

class TestSchedLoadAvgTask(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgTask creates a proper data_frame"""
        dfr = trappy.Run().sched_load_avg_task.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["comm"].iloc[0], "sshd")
        self.assertEquals(dfr["pid"].iloc[0], 2962)
        self.assertEquals(dfr["load"].iloc[0], 0)
        self.assertEquals(dfr["utilization"].iloc[0], 0)
        self.assertEquals(dfr["runnable_avg_sum"].iloc[0], 0)
        self.assertEquals(dfr["running_avg_sum"].iloc[0], 0)
        self.assertEquals(dfr["avg_period"].iloc[0], 48595)

class TestSchedLoadAvgCpu(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgCpu creates a proper data_frame"""
        dfr = trappy.Run().sched_load_avg_cpu.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["load"].iloc[0], 13)
        self.assertEquals(dfr["utilization"].iloc[0], 18)

class TestSchedContribScaleFactor(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedContribScaleFactor creates a proper data_frame"""
        dfr = trappy.Run().sched_contrib_scale_factor.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["freq_scale_factor"].iloc[0], 426)
        self.assertEquals(dfr["cpu_scale_factor"].iloc[0], 1024)

class TestSchedCpuCapacity(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedCpuCapacity creates a proper data_frame"""
        dfr = trappy.Run().sched_cpu_capacity.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 3)
        self.assertEquals(dfr["capacity"].iloc[0], 430)
        self.assertEquals(dfr["rt_capacity"].iloc[0], 1024)

class TestSchedCpuFrequency(BaseTestSched):

    def test_get_dataframe(self):
        """Test that CpuFrequency creates a proper data_frame"""
        dfr = trappy.Run().sched_cpu_frequency.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["frequency"].iloc[0], 600000)
        self.assertFalse("cpu_id" in dfr.columns)

class TestGetFilters(BaseTestSched):

    def test_get_filters(self):
        """Test that Run::get_filters returns correct list of filters"""

        run = trappy.Run()
        classes = run.class_definitions
        filters = run.get_filters()
        self.assertTrue(len(classes) == len(filters))
        self.assertTrue(sorted(classes) == sorted(filters))

        sched_classes = run.sched_classes
        sched_filters = run.get_filters("sched")
        self.assertTrue(len(sched_classes) == len(sched_filters))
        self.assertTrue(sorted(sched_classes) == sorted(sched_filters))

class TestSpacedValueAttributes(BaseTestSched):

    def test_spaced_value_attr(self):
        """Test that Run object parses spaced value attributes correctly"""

        with open("trace.txt", "a") as fout:
            fout.write("       <...>-2971  [004]  6550.056871: sched_load_avg_task:  comm=AsyncTask #2 pid=6163 ")

        dfr = trappy.Run().sched_load_avg_task.data_frame
        self.assertTrue(len(dfr) == 2)
        self.assertEquals(dfr["comm"].iloc[1], "AsyncTask #2")
        self.assertEquals(dfr["pid"].iloc[1], 6163)

class TestNoSchedTraces(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestNoSchedTraces, self).__init__(
             [("trace_empty.txt", "trace.txt")],
             *args,
             **kwargs)

    def test_empty_trace_txt(self):
        """Test that empty objects are created with empty trace file"""

        run = trappy.Run()

        for attr in run.sched_classes.iterkeys():
            self.assertTrue(len(getattr(run, attr).data_frame) == 0)
