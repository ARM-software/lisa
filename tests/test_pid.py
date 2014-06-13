#!/usr/bin/python

from test_thermal import TestThermalBase
from cr2 import PIDController

class TestPIDController(TestThermalBase):
    def test_get_dataframe(self):
        """Test PIDController.get_data_frame()"""
        df = PIDController().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertTrue("err_integral" in df.columns)
        self.assertEquals(df["err"].iloc[0], 9)
