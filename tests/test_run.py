#!/usr/bin/python

import matplotlib
import os
import re
import tempfile

from test_thermal import BaseTestThermal
import cr2

class TestRun(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestRun, self).__init__(*args, **kwargs)
        self.map_label = {"000000f0": "A15", "0000000f": "A7"}

    def test_run_has_all_classes(self):
        """The Run() class has members for all classes"""

        run = cr2.Run()

        self.assertTrue(len(run.thermal.data_frame) > 0)
        self.assertTrue(len(run.thermal_governor.data_frame) > 0)
        self.assertTrue(len(run.pid_controller.data_frame) > 0)
        self.assertTrue(len(run.in_power.data_frame) > 0)
        self.assertTrue(len(run.out_power.data_frame) > 0)

    def test_run_accepts_name(self):
        """The Run() class has members for all classes"""

        run = cr2.Run(name="foo")

        self.assertEquals(run.name, "foo")

    def test_do_txt_if_not_there(self):
        """Create trace.txt if it's not there"""
        cr2.Run()

        found = False
        with open("trace.txt") as f:
            for line in f:
                if re.search("thermal", line):
                    found = True
                    break

        self.assertTrue(found)

    def test_fail_if_no_trace_dat(self):
        """Raise an IOError with the path if there's no trace.dat and trace.txt"""
        os.remove("trace.dat")
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

    def test_run_normalize_time(self):
        """Run().normalize_time() works accross all classes"""

        run = cr2.Run()

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

        self.assertEquals(allfreqs[1][1]["A7_freq_out"].iloc[1], 1400)
        self.assertEquals(allfreqs[1][1]["A7_freq_in"].iloc[35], 1400)
        self.assertEquals(allfreqs[0][1]["A15_freq_out"].iloc[0], 1900)

        # Make sure there are no NaNs in the middle of the array
        self.assertTrue(allfreqs[0][1]["A15_freq_out"].notnull().all())

    def test_plot_freq_hists(self):
        """Test that plot_freq_hists() doesn't bomb"""

        run = cr2.Run()

        _, axis = matplotlib.pyplot.subplots(nrows=2)
        run.plot_freq_hists(self.map_label, axis)
        matplotlib.pyplot.close('all')

    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        run = cr2.Run()

        run.plot_allfreqs(self.map_label)
        matplotlib.pyplot.close('all')

        _, axis = matplotlib.pyplot.subplots(nrows=2)

        run.plot_allfreqs(self.map_label, ax=axis)
        matplotlib.pyplot.close('all')
