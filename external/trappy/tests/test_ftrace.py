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
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from builtins import str
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

class TestFTrace(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestFTrace, self).__init__(*args, **kwargs)
        self.map_label = {"00000000,00000006": "A57", "00000000,00000039": "A53"}

    def test_ftrace_has_all_classes(self):
        """The FTrace() class has members for all classes"""

        trace = trappy.FTrace()

        for attr in trace.class_definitions.keys():
            self.assertTrue(hasattr(trace, attr))

    def test_ftrace_has_all_classes_scope_all(self):
        """The FTrace() class has members for all classes with scope=all"""

        trace = trappy.FTrace(scope="all")

        for attr in trace.class_definitions.keys():
            self.assertTrue(hasattr(trace, attr))

    def test_ftrace_has_all_classes_scope_thermal(self):
        """The FTrace() class has only members for thermal classes with scope=thermal"""

        trace = trappy.FTrace(scope="thermal")

        for attr in trace.thermal_classes.keys():
            self.assertTrue(hasattr(trace, attr))

        for attr in trace.sched_classes.keys():
            self.assertFalse(hasattr(trace, attr))

    def test_ftrace_has_all_classes_scope_sched(self):
        """The FTrace() class has only members for sched classes with scope=sched"""

        trace = trappy.FTrace(scope="sched")

        for attr in trace.thermal_classes.keys():
            self.assertFalse(hasattr(trace, attr))

        for attr in trace.sched_classes.keys():
            self.assertTrue(hasattr(trace, attr))

    def test_ftrace_has_no_classes_scope_dynamic(self):
        """The FTrace() class has dynamically registered classes with scope=all"""

        trace = trappy.FTrace(scope="custom")

        for attr in trace.thermal_classes.keys():
            self.assertFalse(hasattr(trace, attr))

        for attr in trace.sched_classes.keys():
            self.assertFalse(hasattr(trace, attr))

        ftrace_parser = trappy.register_dynamic_ftrace("ADynamicEvent",
                                                       "a_dynamic_event")
        trace = trappy.FTrace(scope="all")

        self.assertTrue(hasattr(trace, "a_dynamic_event"))

        trappy.unregister_dynamic_ftrace(ftrace_parser)


    def test_ftrace_doesnt_overwrite_parsed_event(self):
        """FTrace().add_parsed_event() should not override an event that's already present"""
        trace = trappy.FTrace()
        dfr = pd.DataFrame({"temp": [45000, 46724, 45520]},
                           index=pd.Series([1.020, 1.342, 1.451], name="Time"))

        with self.assertRaises(ValueError):
            trace.add_parsed_event("sched_switch", dfr)

    def test_fail_if_no_trace_dat(self):
        """Raise an IOError with the path if there's no trace.dat and trace.txt"""
        os.remove("trace.txt")
        self.assertRaises(IOError, trappy.FTrace)

        cwd = os.getcwd()

        try:
            trappy.FTrace(cwd)
        except IOError as exception:
            self.assertTrue(cwd in str(exception))

    def test_other_directory(self):
        """FTrace() can grab the trace.dat from other directories"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        dfr = trappy.FTrace(self.out_dir).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertEqual(os.getcwd(), other_random_dir)

    def test_ftrace_arbitrary_trace_txt(self):
        """FTrace() works if the trace is called something other than trace.txt"""
        arbitrary_trace_name = "my_trace.txt"
        shutil.move("trace.txt", arbitrary_trace_name)

        dfr = trappy.FTrace(arbitrary_trace_name).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertFalse(os.path.exists("trace.txt"))
        # As there is no raw trace requested. The mytrace.raw.txt
        # Should not have been generated
        self.assertFalse(os.path.exists("mytrace.raw.txt"))

    def test_ftrace_autonormalize_time(self):
        """FTrace() normalizes by default"""

        trace = trappy.FTrace()

        self.assertEqual(round(trace.thermal.data_frame.index[0], 7), 0)

    def test_ftrace_dont_normalize_time(self):
        """FTrace() doesn't normalize if asked not to"""

        trace = trappy.FTrace(normalize_time=False)

        self.assertNotEquals(round(trace.thermal.data_frame.index[0], 7), 0)

    def test_ftrace_basetime(self):
        """Test that basetime calculation is correct"""

        trace = trappy.FTrace(normalize_time=False)

        basetime = trace.thermal.data_frame.index[0]

        self.assertEqual(trace.basetime, basetime)

    def test_ftrace_duration(self):
        """Test get_duration: normalize_time=False"""

        trace = trappy.FTrace(normalize_time=True)

        duration = trace.thermal_governor.data_frame.index[-1] - trace.thermal.data_frame.index[0]

        self.assertEqual(trace.get_duration(), duration)

    def test_ftrace_duration_window(self):
        """Test that duration is correct with time window (normalize_time=True)"""
        trace = trappy.FTrace(normalize_time=True, window=[1, 5])

        duration = trace.thermal_governor.data_frame.index[-1] - trace.thermal.data_frame.index[0]

        self.assertEqual(trace.get_duration(), duration)

    def test_ftrace_duration_window_not_normalized(self):
        """Test that duration is correct with time window (normalize_time=False)"""
        trace = trappy.FTrace(normalize_time=False, window=[1, 5])

        duration = trace.thermal_governor.data_frame.index[-1] - trace.thermal.data_frame.index[0]

        self.assertEqual(trace.get_duration(), duration)

    def test_ftrace_duration_not_normalized(self):
        """Test get_duration: normalize_time=True"""

        trace = trappy.FTrace(normalize_time=False)

        duration = trace.thermal_governor.data_frame.index[-1] - trace.thermal.data_frame.index[0]

        self.assertEqual(trace.get_duration(), duration)


    def test_ftrace_normalize_time(self):
        """FTrace()._normalize_time() works accross all classes"""

        trace = trappy.FTrace(normalize_time=False)

        prev_inpower_basetime = trace.cpu_in_power.data_frame.index[0]
        prev_inpower_last = trace.cpu_in_power.data_frame.index[-1]

        trace._normalize_time()

        self.assertEqual(round(trace.thermal.data_frame.index[0], 7), 0)

        exp_inpower_first = prev_inpower_basetime - trace.basetime
        self.assertEqual(round(trace.cpu_in_power.data_frame.index[0] - exp_inpower_first, 7), 0)

        exp_inpower_last = prev_inpower_last - trace.basetime
        self.assertEqual(round(trace.cpu_in_power.data_frame.index[-1] - exp_inpower_last, 7), 0)

    def test_ftrace_accepts_events(self):
        """The FTrace class accepts an events parameter with only the parameters interesting for a trace"""

        trace = trappy.FTrace(scope="custom", events=["cdev_update"])

        self.assertGreater(len(trace.cdev_update.data_frame), 1)

        # If you specify events as a string by mistake, trappy does the right thing
        trace = trappy.FTrace(scope="custom", events="foo")
        self.assertTrue(hasattr(trace, "foo"))

    def test_ftrace_already_registered_events_are_not_registered_again(self):
        """FTrace(events="foo") uses class for foo if it is a known class for trappy"""
        events = ["sched_switch", "sched_load_avg_sg"]
        trace = trappy.FTrace(scope="custom", events=events)

        self.assertTrue(trace.sched_switch.parse_raw)
        self.assertEqual(trace.sched_load_avg_sg.pivot, "cpus")

    def test_get_all_freqs_data(self):
        """Test get_all_freqs_data()"""

        allfreqs = dict(trappy.FTrace().get_all_freqs_data(self.map_label))

        self.assertEqual(allfreqs["A53"]["A53_freq_out"].iloc[3], 850)
        self.assertEqual(allfreqs["A53"]["A53_freq_in"].iloc[1], 850)
        self.assertEqual(allfreqs["A57"]["A57_freq_out"].iloc[2], 1100)
        self.assertTrue("gpu_freq_in" in allfreqs["GPU"].columns)

        # Make sure there are no NaNs in the middle of the array
        self.assertTrue(allfreqs["A57"]["A57_freq_in"].notnull().all())

    def test_apply_callbacks(self):
        """Test apply_callbacks()"""

        counts = {
            "cpu_in_power": 0,
            "cpu_out_power": 0
        }

        def cpu_in_power_fn(data):
            counts["cpu_in_power"] += 1

        def cpu_out_power_fn(data):
            counts["cpu_out_power"] += 1

        fn_map = {
            "cpu_in_power": cpu_in_power_fn,
            "cpu_out_power": cpu_out_power_fn
        }
        trace = trappy.FTrace()

        trace.apply_callbacks(fn_map)

        self.assertEqual(counts["cpu_in_power"], 134)
        self.assertEqual(counts["cpu_out_power"], 134)

    def test_plot_freq_hists(self):
        """Test that plot_freq_hists() doesn't bomb"""

        trace = trappy.FTrace()

        _, axis = matplotlib.pyplot.subplots(nrows=2)
        trace.plot_freq_hists(self.map_label, axis)
        matplotlib.pyplot.close('all')

    def test_plot_load(self):
        """Test that plot_load() doesn't explode"""
        trace = trappy.FTrace()
        trace.plot_load(self.map_label, title="Util")

        _, ax = matplotlib.pyplot.subplots()
        trace.plot_load(self.map_label, ax=ax)

    def test_plot_normalized_load(self):
        """Test that plot_normalized_load() doesn't explode"""

        trace = trappy.FTrace()

        _, ax = matplotlib.pyplot.subplots()
        trace.plot_normalized_load(self.map_label, ax=ax)

    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        trace = trappy.FTrace()

        trace.plot_allfreqs(self.map_label)
        matplotlib.pyplot.close('all')

        _, axis = matplotlib.pyplot.subplots(nrows=2)

        trace.plot_allfreqs(self.map_label, ax=axis)
        matplotlib.pyplot.close('all')

    def test_plot_allfreqs_with_one_actor(self):
        """Check that plot_allfreqs() works with one actor"""

        in_data = """     kworker/4:1-397   [004]   720.741349: thermal_power_cpu_get_power: cpus=00000000,00000006 freq=1400000 raw_cpu_power=189 load={23, 12} power=14
     kworker/4:1-397   [004]   720.741679: thermal_power_cpu_limit: cpus=00000000,00000006 freq=1400000 cdev_state=1 power=14"""

        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        trace = trappy.FTrace()
        map_label = {"00000000,00000006": "A57"}
        _, axis = matplotlib.pyplot.subplots(nrows=1)

        trace.plot_allfreqs(map_label, ax=[axis])
        matplotlib.pyplot.close('all')

    def test_trace_metadata(self):
        """Test if metadata gets populated correctly"""

        expected_metadata = {}
        expected_metadata["version"] = "6"
        expected_metadata["cpus"] = "6"

        trace = trappy.FTrace()
        for key, value in expected_metadata.items():
            self.assertTrue(hasattr(trace, "_" + key))
            self.assertEqual(getattr(trace, "_" + key), value)

    def test_missing_metadata(self):
        """Test if trappy.FTrace() works with a trace missing metadata info"""
        lines = []

        with open("trace.txt", "r") as fil:
            lines += fil.readlines()
            lines = lines[7:]
            fil.close()

        with open("trace.txt", "w") as fil:
            fil.write("".join(lines))
            fil.close()

        trace = trappy.FTrace()
        self.assertEqual(trace._cpus, None)
        self.assertEqual(trace._version, None)
        self.assertTrue(len(trace.thermal.data_frame) > 0)

    def test_ftrace_accepts_window(self):
        """FTrace class accepts a window parameter"""
        trace = trappy.FTrace(window=(1.234726, 5.334726))
        self.assertEqual(trace.thermal.data_frame.iloc[0]["temp"], 68989)
        self.assertEqual(trace.thermal.data_frame.iloc[-1]["temp"], 69530)

    def test_ftrace_accepts_abs_window(self):
        """FTrace class accepts an abs_window parameter"""
        trace = trappy.FTrace(abs_window=(1585, 1589.1))
        self.assertEqual(trace.thermal.data_frame.iloc[0]["temp"], 68989)
        self.assertEqual(trace.thermal.data_frame.iloc[-1]["temp"], 69530)

    def test_parse_tracing_mark_write_events(self):
        """Check that tracing_mark_write events are parsed without errors"""

        in_data = """     sh-1379  [002]   353.397813: print:                tracing_mark_write: TRACE_MARKER_START
     shutils-1381  [001]   353.680439: print:                tracing_mark_write: cpu_frequency:        state=450000 cpu_id=5"""

        with open("trace.txt", "w") as fout:
            fout.write(in_data)

        try:
            trace = trappy.FTrace()
        except TypeError as e:
            self.fail("tracing_mark_write parsing failed with {} exception"\
                      .format(e.message))
        # The second event is recognised as a cpu_frequency event and therefore
        # put under trace.cpu_frequency
        self.assertEqual(trace.tracing_mark_write.data_frame.iloc[0]["string"],
                          "TRACE_MARKER_START")
        self.assertEqual(len(trace.tracing_mark_write.data_frame), 1)
        self.assertEqual(len(trace.cpu_frequency.data_frame), 1)

    def test_ftrace_metadata(self):
        """FTrace class correctly populates metadata"""
        trace = trappy.FTrace()

        self.assertEqual(int(trace._version), 6)
        self.assertEqual(int(trace._cpus), 6)

@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestFTraceRawDat(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestFTraceRawDat, self).__init__(
             [("raw_trace.dat", "trace.dat")],
             *args,
             **kwargs)

    def test_raw_dat(self):
        """Tests an event that relies on raw parsing"""

        trace = trappy.FTrace()
        self.assertTrue(hasattr(trace, "sched_switch"))
        self.assertTrue(len(trace.sched_switch.data_frame) > 0)
        self.assertTrue("prev_comm" in trace.sched_switch.data_frame.columns)

    def test_raw_dat_arb_name(self):
        """Tests an event that relies on raw parsing with arbitrary .dat file name"""

        arbitrary_name = "my_trace.dat"
        shutil.move("trace.dat", arbitrary_name)

        trace = trappy.FTrace(arbitrary_name)
        self.assertTrue(hasattr(trace, "sched_switch"))
        self.assertTrue(len(trace.sched_switch.data_frame) > 0)

class TestFTraceRawBothTxt(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestFTraceRawBothTxt, self).__init__(
             [("raw_trace.txt", "trace.txt"),],
             *args,
             **kwargs)

    def test_both_txt_files(self):
        """test raw parsing for txt files"""

        self.assertFalse(os.path.isfile("trace.dat"))
        trace = trappy.FTrace()
        self.assertTrue(hasattr(trace, "sched_switch"))
        self.assertTrue(len(trace.sched_switch.data_frame) > 0)

    def test_both_txt_arb_name(self):
        """Test raw parsing for txt files arbitrary name"""

        arbitrary_name = "my_trace.txt"

        shutil.move("trace.txt", arbitrary_name)

        trace = trappy.FTrace(arbitrary_name)
        self.assertTrue(hasattr(trace, "sched_switch"))
        self.assertTrue(len(trace.sched_switch.data_frame) > 0)

    def test_keep_txt(self):
        """Do not delete trace.txt if trace.dat isn't there"""
        self.assertFalse(os.path.isfile("trace.dat"))
        self.assertTrue(os.path.isfile("trace.txt"))
        trace = trappy.FTrace()
        self.assertTrue(os.path.isfile("trace.txt"))

class TestFTraceSched(utils_tests.SetupDirectory):
    """Tests using a trace with only sched info and no (or partial) thermal"""

    def __init__(self, *args, **kwargs):
        super(TestFTraceSched, self).__init__(
             [("trace_empty.txt", "trace.txt")],
             *args,
             **kwargs)

    def test_ftrace_unique_but_no_fields(self):
        """Test with a matching unique but no special fields"""
        version_parser = trappy.register_dynamic_ftrace("Version", "version")

        # Append invalid line to file
        with open("trace.txt", "a") as fil:
            fil.write("version = 6")

        trace = trappy.FTrace(scope="all")

        trappy.unregister_dynamic_ftrace(version_parser)

    def test_ftrace_normalize_some_tracepoints(self):
        """Test that normalizing time works if not all the tracepoints are in the trace"""

        with open("trace.txt", "a") as fil:
            fil.write("     kworker/4:1-1219  [004]   508.424826: thermal_temperature:  thermal_zone=exynos-therm id=0 temp_prev=24000 temp=24000")

        trace = trappy.FTrace(normalize_time=True)

        self.assertEqual(trace.thermal.data_frame.index[0], 473.527906)

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

    def test_do_not_txt(self):
        """Do not create trace.txt if trace.dat is there"""
        self.assertTrue(os.path.isfile("trace.dat"))
        self.assertFalse(os.path.isfile("trace.txt"))

        trappy.FTrace()

        self.assertFalse(os.path.isfile("trace.txt"))

    def test_ftrace_arbitrary_trace_dat(self):
        """FTrace() works if asked to parse a binary trace with a filename other than trace.dat"""
        arbitrary_trace_name = "my_trace.dat"
        shutil.move("trace.dat", arbitrary_trace_name)

        dfr = trappy.FTrace(arbitrary_trace_name).thermal.data_frame

        self.assertTrue(len(dfr) > 0)
        self.assertFalse(os.path.exists("trace.dat"))

class TestTraceTxtNoTrailingLine(utils_tests.SetupDirectory):
    """Test that we don't produce garbage when trace.txt has no trailing line"""
    def __init__(self, *args, **kwargs):
        super(TestTraceTxtNoTrailingLine, self).__init__(
             [("trace_no_trailing_line.txt", "trace.txt")],
             *args,
             **kwargs)

    def test_no_trailing_line(self):
        ftrace = trappy.FTrace()

        self.assertSetEqual(
            set(ftrace.cpu_idle.data_frame['cpu_id'].unique()),
            set([0, 3]))


class TestHelpfulAttributeError(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestHelpfulAttributeError, self).__init__(
            [("trace.dat", "trace.dat")],
            *args, **kwargs)

    def test_helpful_error(self):
        trace = trappy.FTrace(events=['sched_contrib_scale_f'])

        with self.assertRaises(AttributeError) as assert_raises:
            df = trace.sched_contrib_scale_f.data_frame

        exception = assert_raises.exception
        regex = (r'.*\.sched_contrib_scale_f", instead'
                 r'.*\.sched_contrib_scale_factor"..*')

        self.assertRegexpMatches(str(exception), regex)


class DummyEvent(trappy.base.Base):
    unique_word = 'dummy_unique_word'
    name = 'dummy_name'

class TestDynamicDoesntOverwrite(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestDynamicDoesntOverwrite, self).__init__(
            [("trace.dat", "trace.dat")],
            *args, **kwargs)

    def setUp(self):
        super(TestDynamicDoesntOverwrite, self).setUp()
        trappy.register_ftrace_parser(DummyEvent)

    def tearDown(self):
        super(TestDynamicDoesntOverwrite, self).tearDown()
        trappy.unregister_ftrace_parser(DummyEvent)

    def test_not_overwritten(self):
        trace = trappy.FTrace(events=['dummy_name'])
        self.assertIsInstance(trace.dummy_name, DummyEvent)
