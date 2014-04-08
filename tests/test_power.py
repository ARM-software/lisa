#!/usr/bin/python

import unittest
import os, sys

import utils_tests
from test_thermal import TestThermalBase
import power

class TestPower(TestThermalBase):
    def test_get_dataframe(self):
        """Test Power.get_data_frame()"""
        df = power.Power().get_data_frame()

        self.assertEquals(df["power"][0], 2898)
        self.assertTrue("cdev_state" in df.columns)
