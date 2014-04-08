#!/usr/bin/python

import unittest
import os, sys

import utils_tests
from test_thermal import TestThermalBase
import pi_controller

class TestPIController(TestThermalBase):
    def test_get_dataframe(self):
        """Test PIController.get_data_frame()"""
        df = pi_controller.PIController().get_data_frame()

        self.assertTrue(len(df) > 0)
        self.assertTrue("err_integral" in df.columns)
        self.assertEquals(df["err"][0], 2304)
