#!/usr/bin/python

import unittest
import os, sys
from test_thermal import TestThermalBase

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(TESTS_DIRECTORY, "..", "cr2"))
import power

class TestPower(TestThermalBase):
    def test_get_dataframe(self):
        df = power.Power().get_data_frame()

        self.assertEquals(df["power"][0], 2898)
        self.assertTrue("cdev_state" in df.columns)
