#!/usr/bin/python

from test_thermal import TestThermalBase
from cr2 import PIController

class TestPIController(TestThermalBase):
    def test_get_dataframe(self):
        """Test PIController.get_data_frame()"""
        df = PIController().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertTrue("err_integral" in df.columns)
        self.assertEquals(df["err"].iloc[0], 2304)
