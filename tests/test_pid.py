#!/usr/bin/python

import matplotlib

from test_thermal import TestThermalBase
from cr2 import PIDController

class TestPIDController(TestThermalBase):
    def test_get_dataframe(self):
        """Test PIDController.get_data_frame()"""
        df = PIDController().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertTrue("err_integral" in df.columns)
        self.assertEquals(df["err"].iloc[0], 7)

    def test_plot_controller(self):
        """Test PIDController.plot_controller()

        As it happens with all plot functions, just test that it doesn't explode"""

        PIDController().plot_controller()
        PIDController().plot_controller(title="Antutu", width=20, height=5)
        matplotlib.pyplot.close('all')
