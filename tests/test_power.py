#!/usr/bin/python

from test_thermal import TestThermalBase
from cr2 import Power

class TestPower(TestThermalBase):
    def test_get_dataframe(self):
        """Test Power.get_data_frame()"""
        df = Power().get_data_frame()

        self.assertEquals(df["power"].iloc[0], 2898)
        self.assertTrue("cdev_state" in df.columns)
