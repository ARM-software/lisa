#!/usr/bin/python

import os
import sys

import utils_tests
import cr2

sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "cr2"))

class BaseTestSched(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(BaseTestSched, self).__init__(
             [("trace_sched.txt", "trace.txt")],
             *args,
             **kwargs)

class TestSchedLoadAvgSchedGroup(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgSchedGroup creates a proper data_frame"""
        dfr = cr2.Run().sched_load_avg_sched_group.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpus"].iloc[0], "00000002")
        self.assertEquals(dfr["load"].iloc[0], 0)
        self.assertEquals(dfr["utilization"].iloc[0], 0)

class TestSchedLoadAvgTask(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedLoadAvgTask creates a proper data_frame"""
        dfr = cr2.Run().sched_load_avg_task.data_frame

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
        dfr = cr2.Run().sched_load_avg_cpu.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["load"].iloc[0], 13)
        self.assertEquals(dfr["utilization"].iloc[0], 18)

class TestSchedContribScaleFactor(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedContribScaleFactor creates a proper data_frame"""
        dfr = cr2.Run().sched_contrib_scale_factor.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["freq_scale_factor"].iloc[0], 426)
        self.assertEquals(dfr["cpu_scale_factor"].iloc[0], 1024)

class TestSchedCpuCapacity(BaseTestSched):

    def test_get_dataframe(self):
        """Test that SchedCpuCapacity creates a proper data_frame"""
        dfr = cr2.Run().sched_cpu_capacity.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 3)
        self.assertEquals(dfr["capacity"].iloc[0], 430)
        self.assertEquals(dfr["rt_capacity"].iloc[0], 1024)

class TestSchedCpuFrequency(BaseTestSched):

    def test_get_dataframe(self):
        """Test that CpuFrequency creates a proper data_frame"""
        dfr = cr2.Run().sched_cpu_frequency.data_frame

        self.assertTrue(len(dfr) == 1)
        self.assertEquals(dfr["cpu"].iloc[0], 0)
        self.assertEquals(dfr["state"].iloc[0], 600000)
        self.assertFalse("cpu_id" in dfr.columns)

class TestNoSchedTraces(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestNoSchedTraces, self).__init__(
             [("trace_empty.txt", "trace.txt")],
             *args,
             **kwargs)

    def test_empty_trace_txt(self):
        """Test that empty objects are created with empty trace file"""

        run = cr2.Run()

        for attr in run.sched_classes.iterkeys():
            self.assertTrue(len(getattr(run, attr).data_frame) == 0)
