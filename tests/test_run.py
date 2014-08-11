#!/usr/bin/python

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
