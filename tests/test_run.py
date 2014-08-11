#!/usr/bin/python

import matplotlib

from test_thermal import BaseTestThermal
from cr2 import Run

class TestRun(BaseTestThermal):
    def test_run_has_all_classes(self):
        """The Run() class has members for all classes"""

        run = Run()

        self.assertTrue(len(run.thermal.data_frame) > 0)
        self.assertTrue(len(run.thermal_governor.data_frame) > 0)
        self.assertTrue(len(run.pid_controller.data_frame) > 0)
        self.assertTrue(len(run.in_power.data_frame) > 0)
        self.assertTrue(len(run.out_power.data_frame) > 0)

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

    def test_plot_power_hists(self):
        """Test that plot_power_hists() doesn't bomb"""

        run = Run()
        map_label = {"0000000f": "A7", "000000f0": "A15"}

        run.plot_power_hists(map_label)
        matplotlib.pyplot.close('all')

    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        run = Run()
        map_label = {"0000000f": "A7", "000000f0": "A15"}

        run.plot_allfreqs(map_label)
        matplotlib.pyplot.close('all')
