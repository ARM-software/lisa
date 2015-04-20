#!/usr/bin/python

import matplotlib
import os
import re
import shutil
import tempfile
import unittest

from test_thermal import BaseTestThermal
import cr2
import utils_tests

class TestRun(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestRun, self).__init__(*args, **kwargs)
        self.map_label = {"00000000,00000006": "A57", "00000000,00000039": "A53"}

    def test_run_has_all_classes(self):
        """The Run() class has members for all classes"""

        run = cr2.Run()

        for attr in run.class_definitions.iterkeys():
            self.assertTrue(hasattr(run, attr))

    def test_run_has_all_classes_scope_all(self):
        """The Run() class has members for all classes with scope=all"""

        run = cr2.Run(scope="all")

        for attr in run.class_definitions.iterkeys():
            self.assertTrue(hasattr(run, attr))

    def test_run_has_all_classes_scope_thermal(self):
        """The Run() class has only members for thermal classes with scope=thermal"""

        run = cr2.Run(scope="thermal")

        for attr in run.thermal_classes.iterkeys():
            self.assertTrue(hasattr(run, attr))

        for attr in run.sched_classes.iterkeys():
            self.assertFalse(hasattr(run, attr))

    def test_run_has_all_classes_scope_sched(self):
        """The Run() class has only members for sched classes with scope=sched"""

        run = cr2.Run(scope="sched")

        for attr in run.thermal_classes.iterkeys():
            self.assertFalse(hasattr(run, attr))

        for attr in run.sched_classes.iterkeys():
            self.assertTrue(hasattr(run, attr))

    def test_run_accepts_name(self):
        """The Run() class has members for all classes"""

        run = cr2.Run(name="foo")

        self.assertEquals(run.name, "foo")

    def test_fail_if_no_trace_dat(self):
        """Raise an IOError with the path if there's no trace.dat and trace.txt"""
        os.remove("trace.txt")
        self.assertRaises(IOError, cr2.Run)

        cwd = os.getcwd()

        try:
            cr2.Run(cwd)
        except IOError as exception:
            pass

        self.assertTrue(cwd in str(exception))

    def test_other_directory(self):
        """Run() can grab the trace.dat from other directories"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        dfr = cr2.Run(self.out_dir).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertEquals(os.getcwd(), other_random_dir)

    def test_run_arbitrary_trace_txt(self):
        """Run() works if the trace is called something other than trace.txt"""
        arbitrary_trace_name = "my_trace.txt"
        shutil.move("trace.txt", arbitrary_trace_name)

        dfr = cr2.Run(arbitrary_trace_name).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertFalse(os.path.exists("trace.txt"))
        # As there is no raw trace requested. The mytrace.raw.txt
        # Should not have been generated
        self.assertFalse(os.path.exists("mytrace.raw.txt"))

    def test_run_autonormalize_time(self):
        """Run() normalizes by default"""

        run = cr2.Run()

        self.assertEquals(round(run.thermal.data_frame.index[0], 7), 0)

    def test_run_dont_normalize_time(self):
        """Run() doesn't normalize if asked not to"""

        run = cr2.Run(normalize_time=False)

        self.assertNotEquals(round(run.thermal.data_frame.index[0], 7), 0)

    def test_run_basetime(self):
        """Test that basetime calculation is correct"""

        run = cr2.Run(normalize_time=False)

        basetime = run.thermal.data_frame.index[0]

        self.assertEqual(run.get_basetime(), basetime)

    def test_run_normalize_time(self):
        """Run().normalize_time() works accross all classes"""

        run = cr2.Run(normalize_time=False)

        prev_inpower_basetime = run.in_power.data_frame.index[0]
        prev_inpower_last = run.in_power.data_frame.index[-1]

        basetime = run.thermal.data_frame.index[0]
        run.normalize_time(basetime)

        self.assertEquals(round(run.thermal.data_frame.index[0], 7), 0)

        exp_inpower_first = prev_inpower_basetime - basetime
        self.assertEquals(round(run.in_power.data_frame.index[0] - exp_inpower_first, 7), 0)

        exp_inpower_last = prev_inpower_last - basetime
        self.assertEquals(round(run.in_power.data_frame.index[-1] - exp_inpower_last, 7), 0)

    def test_get_all_freqs_data(self):
        """Test get_all_freqs_data()"""

        allfreqs = cr2.Run().get_all_freqs_data(self.map_label)

        self.assertEquals(allfreqs[1][1]["A53_freq_out"].iloc[3], 850)
        self.assertEquals(allfreqs[1][1]["A53_freq_in"].iloc[1], 850)
        self.assertEquals(allfreqs[0][1]["A57_freq_out"].iloc[2], 1100)

        # Make sure there are no NaNs in the middle of the array
        self.assertTrue(allfreqs[0][1]["A57_freq_in"].notnull().all())

    def test_plot_freq_hists(self):
        """Test that plot_freq_hists() doesn't bomb"""

        run = cr2.Run()

        _, axis = matplotlib.pyplot.subplots(nrows=2)
        run.plot_freq_hists(self.map_label, axis)
        matplotlib.pyplot.close('all')

    def test_plot_load(self):
        """Test that plot_load() doesn't explode"""
        run = cr2.Run()
        run.plot_load(self.map_label, title="Util")

        _, ax = matplotlib.pyplot.subplots()
        run.plot_load(self.map_label, ax=ax)

    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        run = cr2.Run()

        run.plot_allfreqs(self.map_label)
        matplotlib.pyplot.close('all')

        _, axis = matplotlib.pyplot.subplots(nrows=2)

        run.plot_allfreqs(self.map_label, ax=axis)
        matplotlib.pyplot.close('all')

@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestRunRawDat(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestRunRawDat, self).__init__(
             [("raw_trace.dat", "trace.dat")],
             *args,
             **kwargs)

    def test_raw_dat(self):
        """Tests an event that relies on raw parsing"""

        cr2.register_dynamic("SchedSwitch", "sched_switch", scope="sched", parse_raw=True)
        run = cr2.Run()
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)
        self.assertTrue("prev_comm" in run.sched_switch.data_frame.columns)

    def test_raw_dat_arb_name(self):
        """Tests an event that relies on raw parsing with arbitrary .dat file name"""

        arbitrary_name = "my_trace.dat"
        shutil.move("trace.dat", arbitrary_name)

        cr2.register_dynamic("SchedSwitch", "sched_switch", scope="sched", parse_raw=True)
        run = cr2.Run(arbitrary_name)
        self.assertTrue(os.path.isfile("my_trace.raw.txt"))
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)

class TestRunRawBothTxt(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestRunRawBothTxt, self).__init__(
             [("raw_trace.txt", "trace.txt"),
              ("raw_trace.raw.txt", "trace.raw.txt")],
             *args,
             **kwargs)

    def test_both_txt_files(self):
        """test raw parsing for txt files"""

        self.assertFalse(os.path.isfile("trace.dat"))
        cr2.register_dynamic("SchedSwitch", "sched_switch", scope="sched", parse_raw=True)
        run = cr2.Run()
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)

    def test_both_txt_arb_name(self):
        """Test raw parsing for txt files arbitrary name"""

        arbitrary_name = "my_trace.txt"
        arbitrary_name_raw = "my_trace.raw.txt"

        shutil.move("trace.txt", arbitrary_name)
        shutil.move("trace.raw.txt", arbitrary_name_raw)

        cr2.register_dynamic("SchedSwitch", "sched_switch", scope="sched", parse_raw=True)
        run = cr2.Run(arbitrary_name)
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)

class TestRunSched(utils_tests.SetupDirectory):
    """Tests using a trace with only sched info and no (or partial) thermal"""

    def __init__(self, *args, **kwargs):
        super(TestRunSched, self).__init__(
             [("trace_empty.txt", "trace.txt")],
             *args,
             **kwargs)

    def test_run_basetime_empty(self):
        """Test that basetime is 0 if data frame of all data objects is empty"""

        run = cr2.Run(normalize_time=False)

        self.assertEqual(run.get_basetime(), 0)

    def test_run_normalize_some_tracepoints(self):
        """Test that normalizing time works if not all the tracepoints are in the trace"""

        with open("trace.txt", "a") as fil:
            fil.write("     kworker/4:1-1219  [004]   508.424826: thermal_temperature:  thermal_zone=exynos-therm id=0 temp_prev=24000 temp=24000")

        run = cr2.Run()

        self.assertEqual(run.thermal.data_frame.index[0], 0)

@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestTraceDat(utils_tests.SetupDirectory):
    """Test that trace.dat handling work"""
    def __init__(self, *args, **kwargs):
        super(TestTraceDat, self).__init__(
            [("trace.dat", "trace.dat")],
            *args, **kwargs)

    def test_do_txt_if_not_there(self):
        """Create trace.txt if it's not there"""
        self.assertFalse(os.path.isfile("trace.txt"))

        cr2.Run()

        found = False
        with open("trace.txt") as fin:
            for line in fin:
                if re.search("thermal", line):
                    found = True
                    break

        self.assertTrue(found)

    def test_do_raw_txt_if_not_there(self):
        """Create trace.raw.txt if it's not there"""
        self.assertFalse(os.path.isfile("trace.raw.txt"))

        cr2.Run()

        found = False
        with open("trace.raw.txt") as fin:
            for line in fin:
                if re.search("thermal", line):
                    found = True
                    break

    def test_run_arbitrary_trace_dat(self):
        """Run() works if asked to parse a binary trace with a filename other than trace.dat"""
        arbitrary_trace_name = "my_trace.dat"
        shutil.move("trace.dat", arbitrary_trace_name)

        dfr = cr2.Run(arbitrary_trace_name).thermal.data_frame

        self.assertTrue(os.path.exists("my_trace.txt"))
        self.assertTrue(os.path.exists("my_trace.raw.txt"))
        self.assertTrue(len(dfr) > 0)
        self.assertFalse(os.path.exists("trace.dat"))
        self.assertFalse(os.path.exists("trace.txt"))
        self.assertFalse(os.path.exists("trace.raw.txt"))
