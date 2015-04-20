#!/usr/bin/python

import matplotlib

from test_thermal import BaseTestThermal
import cr2

class TestPIDController(BaseTestThermal):
    def test_dataframe(self):
        """Test that PIDController() generates a valid data_frame"""
        pid = cr2.Run().pid_controller

        self.assertTrue(len(pid.data_frame) > 0)
        self.assertTrue("err_integral" in pid.data_frame.columns)
        self.assertEquals(pid.data_frame["err"].iloc[0], 3225)

    def test_plot_controller(self):
        """Test PIDController.plot_controller()

        As it happens with all plot functions, just test that it doesn't explode"""
        pid = cr2.Run().pid_controller

        pid.plot_controller()
        matplotlib.pyplot.close('all')

        pid.plot_controller(title="Antutu", width=20, height=5)
        matplotlib.pyplot.close('all')

        _, ax = matplotlib.pyplot.subplots()
        pid.plot_controller(ax=ax)
        matplotlib.pyplot.close('all')
