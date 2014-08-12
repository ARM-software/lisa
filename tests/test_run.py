#!/usr/bin/python

import matplotlib

from test_thermal import BaseTestThermal
from cr2 import Run

class TestRun(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestRun, self).__init__(*args, **kwargs)
        self.map_label = {"000000f0": "A15", "0000000f": "A7"}

    def test_run_has_all_classes(self):
        """The Run() class has members for all classes"""

        run = Run()

        self.assertTrue(len(run.thermal.data_frame) > 0)
        self.assertTrue(len(run.thermal_governor.data_frame) > 0)
        self.assertTrue(len(run.pid_controller.data_frame) > 0)
        self.assertTrue(len(run.in_power.data_frame) > 0)
        self.assertTrue(len(run.out_power.data_frame) > 0)

    def test_run_accepts_name(self):
        """The Run() class has members for all classes"""

        run = Run(name="foo")

        self.assertEquals(run.name, "foo")

    def test_run_normalize_time(self):
        """Run().normalize_time() works accross all classes"""

        run = Run()

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

        allfreqs = Run().get_all_freqs_data(self.map_label)

        self.assertEquals(allfreqs["A7"]["A7_freq_out"].iloc[1], 1400)
        self.assertEquals(allfreqs["A7"]["A7_freq_in"].iloc[35], 1400)
        self.assertEquals(allfreqs["A15"]["A15_freq_out"].iloc[0], 1900)

        # Make sure there are no NaNs in the middle of the array
        self.assertTrue(allfreqs["A15"]["A15_freq_out"].notnull().all())

    def test_plot_power_hists(self):
        """Test that plot_power_hists() doesn't bomb"""

        run = Run()

        run.plot_power_hists(self.map_label)
        matplotlib.pyplot.close('all')

    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        run = Run()

        run.plot_allfreqs(self.map_label)
        matplotlib.pyplot.close('all')

        _, axis = matplotlib.pyplot.subplots(ncols=2)

        run.plot_allfreqs(self.map_label, ax=axis)
        matplotlib.pyplot.close('all')
