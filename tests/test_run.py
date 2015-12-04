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


import matplotlib
import os
import pandas as pd
import re
import shutil
import subprocess
import tempfile
import time
import unittest

from test_thermal import BaseTestThermal
import trappy
import utils_tests

class TestRun(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestRun, self).__init__(*args, **kwargs)
        self.map_label = {"00000000,00000006": "A57", "00000000,00000039": "A53"}

    def test_run_has_all_classes(self):
        """The Run() class has members for all classes"""

        run = trappy.Run()

        for attr in run.class_definitions.iterkeys():
            self.assertTrue(hasattr(run, attr))

    def test_run_has_all_classes_scope_all(self):
        """The Run() class has members for all classes with scope=all"""

        run = trappy.Run(scope="all")

        for attr in run.class_definitions.iterkeys():
            self.assertTrue(hasattr(run, attr))

    def test_run_has_all_classes_scope_thermal(self):
        """The Run() class has only members for thermal classes with scope=thermal"""

        run = trappy.Run(scope="thermal")

        for attr in run.thermal_classes.iterkeys():
            self.assertTrue(hasattr(run, attr))

        for attr in run.sched_classes.iterkeys():
            self.assertFalse(hasattr(run, attr))

    def test_run_has_all_classes_scope_sched(self):
        """The Run() class has only members for sched classes with scope=sched"""

        run = trappy.Run(scope="sched")

        for attr in run.thermal_classes.iterkeys():
            self.assertFalse(hasattr(run, attr))

        for attr in run.sched_classes.iterkeys():
            self.assertTrue(hasattr(run, attr))

    def test_run_has_no_classes_scope_dynamic(self):
        """The Run() class has only dynamically registered classes with scope=custom"""

        run = trappy.Run(scope="custom")

        for attr in run.thermal_classes.iterkeys():
            self.assertFalse(hasattr(run, attr))

        for attr in run.sched_classes.iterkeys():
            self.assertFalse(hasattr(run, attr))

        trappy.register_dynamic("ADynamicEvent", "a_dynamic_event")
        run = trappy.Run(scope="custom")

        self.assertTrue(hasattr(run, "a_dynamic_event"))

    def test_run_can_add_parsed_event(self):
        """The Run() class can add parsed events to its collection of trace events"""
        run = trappy.Run(scope="custom")
        dfr = pd.DataFrame({"l1_misses": [24, 535,  41],
                            "l2_misses": [155, 11, 200],
                            "cpu":       [ 0,   1,   0]},
                           index=pd.Series([1.020, 1.342, 1.451], name="Time"))
        run.add_parsed_event("pmu_counters", dfr)

        self.assertEquals(len(run.pmu_counters.data_frame), 3)
        self.assertEquals(run.pmu_counters.data_frame["l1_misses"].iloc[0], 24)

        run.add_parsed_event("pivoted_counters", dfr, pivot="cpu")
        self.assertEquals(run.pivoted_counters.pivot, "cpu")

    def test_run_doesnt_overwrite_parsed_event(self):
        """Run().add_parsed_event() should not override an event that's already present"""
        run = trappy.Run()
        dfr = pd.DataFrame({"temp": [45000, 46724, 45520]},
                           index=pd.Series([1.020, 1.342, 1.451], name="Time"))

        with self.assertRaises(ValueError):
            run.add_parsed_event("sched_switch", dfr)

    def test_run_accepts_name(self):
        """The Run() class has members for all classes"""

        run = trappy.Run(name="foo")

        self.assertEquals(run.name, "foo")

    def test_fail_if_no_trace_dat(self):
        """Raise an IOError with the path if there's no trace.dat and trace.txt"""
        os.remove("trace.txt")
        self.assertRaises(IOError, trappy.Run)

        cwd = os.getcwd()

        try:
            trappy.Run(cwd)
        except IOError as exception:
            pass

        self.assertTrue(cwd in str(exception))

    def test_other_directory(self):
        """Run() can grab the trace.dat from other directories"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        dfr = trappy.Run(self.out_dir).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertEquals(os.getcwd(), other_random_dir)

    def test_run_arbitrary_trace_txt(self):
        """Run() works if the trace is called something other than trace.txt"""
        arbitrary_trace_name = "my_trace.txt"
        shutil.move("trace.txt", arbitrary_trace_name)

        dfr = trappy.Run(arbitrary_trace_name).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertFalse(os.path.exists("trace.txt"))
        # As there is no raw trace requested. The mytrace.raw.txt
        # Should not have been generated
        self.assertFalse(os.path.exists("mytrace.raw.txt"))

    def test_run_autonormalize_time(self):
        """Run() normalizes by default"""

        run = trappy.Run()

        self.assertEquals(round(run.thermal.data_frame.index[0], 7), 0)

    def test_run_dont_normalize_time(self):
        """Run() doesn't normalize if asked not to"""

        run = trappy.Run(normalize_time=False)

        self.assertNotEquals(round(run.thermal.data_frame.index[0], 7), 0)

    def test_run_basetime(self):
        """Test that basetime calculation is correct"""

        run = trappy.Run(normalize_time=False)

        basetime = run.thermal.data_frame.index[0]

        self.assertEqual(run.basetime, basetime)

    def test_run_duration(self):
        """Test that duration calculation is correct"""

        run = trappy.Run(normalize_time=False)

        duration = run.thermal_governor.data_frame.index[-1] - run.thermal.data_frame.index[0]

        self.assertEqual(run.get_duration(), duration)

    def test_run_normalize_time(self):
        """Run().normalize_time() works accross all classes"""

        run = trappy.Run(normalize_time=False)

        prev_inpower_basetime = run.cpu_in_power.data_frame.index[0]
        prev_inpower_last = run.cpu_in_power.data_frame.index[-1]

        run.normalize_time()

        self.assertEquals(round(run.thermal.data_frame.index[0], 7), 0)

        exp_inpower_first = prev_inpower_basetime - run.basetime
        self.assertEquals(round(run.cpu_in_power.data_frame.index[0] - exp_inpower_first, 7), 0)

        exp_inpower_last = prev_inpower_last - run.basetime
        self.assertEquals(round(run.cpu_in_power.data_frame.index[-1] - exp_inpower_last, 7), 0)

    def test_run_accepts_events(self):
        """The Run class accepts an events parameter with only the parameters interesting for a run"""

        run = trappy.Run(scope="custom", events=["cdev_update"])

        self.assertGreater(len(run.cdev_update.data_frame), 1)

        # If you specify events as a string by mistake, trappy does the right thing
        run = trappy.Run(scope="custom", events="foo")
        self.assertTrue(hasattr(run, "foo"))

    def test_run_already_registered_events_are_not_registered_again(self):
        """Run(events="foo") uses class for foo if it is a known class for trappy"""
        events = ["sched_switch", "sched_load_avg_sg"]
        run = trappy.Run(scope="custom", events=events)

        self.assertTrue(run.sched_switch.parse_raw)
        self.assertEquals(run.sched_load_avg_sg.pivot, "cpus")

    def test_get_all_freqs_data(self):
        """Test get_all_freqs_data()"""

        allfreqs = trappy.Run().get_all_freqs_data(self.map_label)

        self.assertEquals(allfreqs[1][1]["A53_freq_out"].iloc[3], 850)
        self.assertEquals(allfreqs[1][1]["A53_freq_in"].iloc[1], 850)
        self.assertEquals(allfreqs[0][1]["A57_freq_out"].iloc[2], 1100)
        self.assertTrue("gpu_freq_in" in allfreqs[2][1].columns)

        # Make sure there are no NaNs in the middle of the array
        self.assertTrue(allfreqs[0][1]["A57_freq_in"].notnull().all())

    def test_plot_freq_hists(self):
        """Test that plot_freq_hists() doesn't bomb"""

        run = trappy.Run()

        _, axis = matplotlib.pyplot.subplots(nrows=2)
        run.plot_freq_hists(self.map_label, axis)
        matplotlib.pyplot.close('all')

    def test_plot_load(self):
        """Test that plot_load() doesn't explode"""
        run = trappy.Run()
        run.plot_load(self.map_label, title="Util")

        _, ax = matplotlib.pyplot.subplots()
        run.plot_load(self.map_label, ax=ax)

    def test_plot_normalized_load(self):
        """Test that plot_normalized_load() doesn't explode"""

        run = trappy.Run()

        _, ax = matplotlib.pyplot.subplots()
        run.plot_normalized_load(self.map_label, ax=ax)

    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        run = trappy.Run()

        run.plot_allfreqs(self.map_label)
        matplotlib.pyplot.close('all')

        _, axis = matplotlib.pyplot.subplots(nrows=2)

        run.plot_allfreqs(self.map_label, ax=axis)
        matplotlib.pyplot.close('all')

    def test_trace_metadata(self):
        """Test if metadata gets populated correctly"""

        expected_metadata = {}
        expected_metadata["version"] = "6"
        expected_metadata["cpus"] = "6"

        run = trappy.Run()
        for key, value in expected_metadata.items():
            self.assertTrue(hasattr(run, "_" + key))
            self.assertEquals(getattr(run, "_" + key), value)

    def test_missing_metadata(self):
        """Test if trappy.Run() works with a trace missing metadata info"""
        lines = []

        with open("trace.txt", "r") as fil:
            lines += fil.readlines()
            lines = lines[7:]
            fil.close()

        with open("trace.txt", "w") as fil:
            fil.write("".join(lines))
            fil.close()

        run = trappy.Run()
        self.assertEquals(run._cpus, None)
        self.assertEquals(run._version, None)
        self.assertTrue(len(run.thermal.data_frame) > 0)

    def test_run_accepts_window(self):
        """Run class accepts a window parameter"""
        run = trappy.Run(window=(1.234726, 5.334726))
        self.assertEquals(run.thermal.data_frame.iloc[0]["temp"], 68989)
        self.assertEquals(run.thermal.data_frame.iloc[-1]["temp"], 69530)

    def test_run_accepts_abs_window(self):
        """Run class accepts an abs_window parameter"""
        run = trappy.Run(abs_window=(1585, 1589.1))
        self.assertEquals(run.thermal.data_frame.iloc[0]["temp"], 68989)
        self.assertEquals(run.thermal.data_frame.iloc[-1]["temp"], 69530)


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

        run = trappy.Run()
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)
        self.assertTrue("prev_comm" in run.sched_switch.data_frame.columns)

    def test_raw_dat_arb_name(self):
        """Tests an event that relies on raw parsing with arbitrary .dat file name"""

        arbitrary_name = "my_trace.dat"
        shutil.move("trace.dat", arbitrary_name)

        run = trappy.Run(arbitrary_name)
        self.assertTrue(os.path.isfile("my_trace.raw.txt"))
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)

    def test_raw_created_if_dat_and_txt_exist(self):
        """trace.raw.txt is created when both trace.dat and trace.txt exist"""

        # Create the trace.txt
        cmd = ["trace-cmd", "report", "trace.dat"]
        with open(os.devnull) as devnull:
            out = subprocess.check_output(cmd, stderr=devnull)

        with open("trace.txt", "w") as fout:
            fout.write(out)

        # Now check that the raw trace is created and analyzed when creating the run
        run = trappy.Run()

        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)
        self.assertTrue("prev_comm" in run.sched_switch.data_frame.columns)

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
        run = trappy.Run()
        self.assertTrue(hasattr(run, "sched_switch"))
        self.assertTrue(len(run.sched_switch.data_frame) > 0)

    def test_both_txt_arb_name(self):
        """Test raw parsing for txt files arbitrary name"""

        arbitrary_name = "my_trace.txt"
        arbitrary_name_raw = "my_trace.raw.txt"

        shutil.move("trace.txt", arbitrary_name)
        shutil.move("trace.raw.txt", arbitrary_name_raw)

        run = trappy.Run(arbitrary_name)
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

        run = trappy.Run(normalize_time=False)

        self.assertEqual(run.basetime, 0)

    def test_run_normalize_some_tracepoints(self):
        """Test that normalizing time works if not all the tracepoints are in the trace"""

        with open("trace.txt", "a") as fil:
            fil.write("     kworker/4:1-1219  [004]   508.424826: thermal_temperature:  thermal_zone=exynos-therm id=0 temp_prev=24000 temp=24000")

        run = trappy.Run()

        self.assertEqual(run.thermal.data_frame.index[0], 0)

@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestTraceDat(utils_tests.SetupDirectory):
    """Test that trace.dat handling work"""
    def __init__(self, *args, **kwargs):
        super(TestTraceDat, self).__init__(
            [("trace.dat", "trace.dat")],
            *args, **kwargs)

    def assert_thermal_in_trace(self, fname):
        """Assert that the thermal event is in the trace

        fname is the trace file, usually "trace.txt" or "trace.raw.txt"
        """

        found = False
        with open(fname) as fin:
            for line in fin:
                if re.search("thermal", line):
                    found = True
                    break

        self.assertTrue(found)

    def test_do_txt_if_not_there(self):
        """Create trace.txt if it's not there"""
        self.assertFalse(os.path.isfile("trace.txt"))

        trappy.Run()

        self.assert_thermal_in_trace("trace.txt")

    def test_do_raw_txt_if_not_there(self):
        """Create trace.raw.txt if it's not there"""
        self.assertFalse(os.path.isfile("trace.raw.txt"))

        trappy.Run()

        self.assert_thermal_in_trace("trace.raw.txt")

    def test_run_arbitrary_trace_dat(self):
        """Run() works if asked to parse a binary trace with a filename other than trace.dat"""
        arbitrary_trace_name = "my_trace.dat"
        shutil.move("trace.dat", arbitrary_trace_name)

        dfr = trappy.Run(arbitrary_trace_name).thermal.data_frame

        self.assertTrue(os.path.exists("my_trace.txt"))
        self.assertTrue(os.path.exists("my_trace.raw.txt"))
        self.assertTrue(len(dfr) > 0)
        self.assertFalse(os.path.exists("trace.dat"))
        self.assertFalse(os.path.exists("trace.txt"))
        self.assertFalse(os.path.exists("trace.raw.txt"))

    def test_regenerate_txt_if_outdated(self):
        """Regenerate the trace.txt if it's older than the trace.dat"""

        trappy.Run()

        # Empty the trace.txt
        with open("trace.txt", "w") as fout:
            fout.write("")

        # Set access and modified time of trace.txt to 10 seconds ago
        now = time.time()
        os.utime("trace.txt", (now - 10, now - 10))

        # touch trace.dat
        os.utime("trace.dat", None)

        trappy.Run()

        self.assert_thermal_in_trace("trace.txt")
