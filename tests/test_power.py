#!/usr/bin/python

from test_thermal import TestThermalBase
from cr2 import OutPower

class TestPower(TestThermalBase):
    def test_outpower_get_dataframe(self):
        """Test OutPower.get_data_frame()"""
        df = OutPower().get_data_frame()

        self.assertEquals(df["power"].iloc[0], 5252)
        self.assertTrue("cdev_state" in df.columns)
