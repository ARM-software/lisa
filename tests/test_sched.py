#    Copyright 2015-2017 ARM Limited
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
             [("trace_sched.txt", "trace.txt"),
              ("trace_sched.txt", "trace.raw.txt")],
             *args,
             **kwargs)

class TestSchedLoadAvgSchedGroup(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgSchedGroup creates a proper data_frame"""
        dfr = trappy.FTrace().sched_load_avg_sg.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpus"].iloc[0], "00000002")
        self.assertEquals(dfr["load"].iloc[0], 0)
        self.assertEquals(dfr["utilization"].iloc[0], 0)

class TestSchedLoadAvgTask(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgTask creates a proper data_frame"""
        dfr = trappy.FTrace().sched_load_avg_task.data_frame

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
        dfr = trappy.FTrace().sched_load_avg_cpu.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["load"].iloc[0], 13)
        self.assertEquals(dfr["utilization"].iloc[0], 18)

class TestSchedContribScaleFactor(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedContribScaleFactor creates a proper data_frame"""
        dfr = trappy.FTrace().sched_contrib_scale_factor.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["freq_scale_factor"].iloc[0], 426)
        self.assertEquals(dfr["cpu_scale_factor"].iloc[0], 1024)

class TestSchedCpuCapacity(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedCpuCapacity creates a proper data_frame"""
        dfr = trappy.FTrace().cpu_capacity.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 3)
        self.assertEquals(dfr["capacity"].iloc[0], 430)
        self.assertEquals(dfr["rt_capacity"].iloc[0], 1024)

class TestSchedCpuFrequency(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedCpuFrequency creates a proper data_frame"""
        dfr = trappy.FTrace().cpu_frequency.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["frequency"].iloc[0], 600000)
        self.assertFalse("cpu_id" in dfr.columns)

class TestSchedWakeup(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedWakeup creates a proper data_frame"""
        dfr = trappy.FTrace().sched_wakeup.data_frame

        self.assertTrue(len(dfr) == 2)
        self.assertEquals(dfr["comm"].iloc[0], "rcu_preempt")
        self.assertEquals(dfr["pid"].iloc[0], 7)
        self.assertEquals(dfr["prio"].iloc[0], 120)
        self.assertEquals(dfr["success"].iloc[0], 1)
        self.assertEquals(dfr["target_cpu"].iloc[0], 1)

class TestSchedWakeupNew(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedWakeupNew creates a proper data_frame"""
        dfr = trappy.FTrace().sched_wakeup_new.data_frame

        self.assertTrue(len(dfr) == 2)
        self.assertEquals(dfr["comm"].iloc[0], "shutils")
        self.assertEquals(dfr["pid"].iloc[0], 19428)
        self.assertEquals(dfr["prio"].iloc[0], 120)
        self.assertEquals(dfr["success"].iloc[0], 1)
        self.assertEquals(dfr["target_cpu"].iloc[0], 2)


class TestGetFilters(BaseTestSched):

    def test_get_filters(self):
        """Test that FTrace::get_filters returns correct list of filters"""

        trace = trappy.FTrace()
        classes = trace.class_definitions
        filters = trace.get_filters()
        self.assertTrue(len(classes) == len(filters))
        self.assertTrue(sorted(classes) == sorted(filters))

        sched_classes = trace.sched_classes.copy()
        sched_filters = trace.get_filters("sched")

        # cpu_capacity and cpu_frequency are in the sched scope but they should
        # not be captured by get_filters("sched")
        del sched_classes["cpu_capacity"]
        del sched_classes["cpu_frequency"]

        self.assertTrue(len(sched_classes) == len(sched_filters))
        self.assertTrue(sorted(sched_classes) == sorted(sched_filters))

class TestSpacedValueAttributes(BaseTestSched):

    def test_spaced_value_attr(self):
        """Test that FTrace object parses spaced value attributes correctly"""

        with open("trace.txt", "a") as fout:
            fout.write("       <...>-2971  [004]  6550.056871: sched_load_avg_task:  comm=AsyncTask #2 pid=6163 ")

        dfr = trappy.FTrace().sched_load_avg_task.data_frame
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

        trace = trappy.FTrace()

        for attr in trace.sched_classes.iterkeys():
            self.assertTrue(len(getattr(trace, attr).data_frame) == 0)
