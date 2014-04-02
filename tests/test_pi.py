#!/usr/bin/python

import unittest
import os, sys
from test_thermal import TestThermalBase

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(TESTS_DIRECTORY, "..", "cr2"))
import pi_controller

class TestPIController(TestThermalBase):
    def test_get_dataframe(self):
        df = pi_controller.PIController().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertTrue("err_integral" in df.columns)
        self.assertEquals(df["err"][0], 2304)
