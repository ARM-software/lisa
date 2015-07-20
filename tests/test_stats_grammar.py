# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        test_stats_grammar.py
# ----------------------------------------------------------------
# $
#

from test_thermal import BaseTestThermal
import cr2
from cr2.stats.grammar import Parser
from pandas.util.testing import assert_series_equal
import numpy as np


class TestStatsGrammar(BaseTestThermal):

    def __init__(self, *args, **kwargs):
        super(TestStatsGrammar, self).__init__(*args, **kwargs)

    def test_sum_operator(self):
        """Test Addition And Subtraction: Numeric"""

        parser = Parser(cr2.Run())
        # Simple equation
        eqn = "10 + 2 - 3"
        self.assertEquals(parser.solve(eqn), 9)
        # Equation with bracket and unary ops
        eqn = "(10 + 2) - (-3 + 2)"
        self.assertEquals(parser.solve(eqn), 13)

    def test_accessors_sum(self):
        """Test Addition And Subtraction: Data"""

        parser = Parser(cr2.Run())
        # Equation with dataframe accessors
        eqn = "cr2.thermal.Thermal:temp + \
cr2.thermal.Thermal:temp"
        assert_series_equal(
            parser.solve(eqn),
            2 *
            parser.data.thermal.data_frame["temp"])

    def test_funcparams_sum(self):
        """Test Addition And Subtraction: Functions"""

        parser = Parser(cr2.Run())
        # Equation with functions as parameters (Mixed)
        eqn = "numpy.mean(cr2.thermal.Thermal:temp) + 1000"
        self.assertEquals(
            parser.solve(eqn),
            np.mean(
                parser.data.thermal.data_frame["temp"]) +
            1000)
        # Multiple func params
        eqn = "numpy.mean(cr2.thermal.Thermal:temp) + numpy.mean(cr2.thermal.Thermal:temp)"
        self.assertEquals(
            parser.solve(eqn),
            np.mean(
                parser.data.thermal.data_frame["temp"]) *
            2)

    def test_bool_ops_vector(self):
        """Test Logical Operations: Vector"""

        # The equation returns a vector mask
        parser = Parser(cr2.Run())
        eqn = "(cr2.thermal.Thermal:temp > 40000) & (cr2.cpu_power.CpuOutPower:power > 800)"
        mask = parser.solve(eqn)
        res = parser.ref(mask)
        self.assertEquals(len(res), 67)
        eqn = "(cr2.thermal.Thermal:temp > 69000) | (cr2.cpu_power.CpuOutPower:power < 800)"
        mask = parser.solve(eqn)
        res = parser.ref(~mask)
        self.assertTrue(len(res), 17)

    def test_bool_ops_scalar(self):
        """Test Logical Operations: Vector"""

        parser = Parser(cr2.Run())
        # The equation returns a boolean scalar
        eqn = "(numpy.mean(cr2.thermal.Thermal:temp) > 65000) && (numpy.mean(cr2.cpu_power.CpuOutPower) > 500)"
        self.assertTrue(parser.solve(eqn))
        eqn = "(numpy.mean(cr2.thermal.Thermal:temp) > 65000) || (numpy.mean(cr2.cpu_power.CpuOutPower) < 500)"
        self.assertTrue(parser.solve(eqn))

    def test_single_func_call(self):
        """Test Single Function Call"""

        parser = Parser(cr2.Run())
        eqn = "numpy.mean(cr2.thermal.Thermal:temp)"
        self.assertEquals(
            parser.solve(eqn),
            np.mean(
                parser.data.thermal.data_frame["temp"]))

    def test_mul_ops(self):
        """Test Mult and Division: Numeric"""

        parser = Parser(cr2.Run())
        eqn = "(10 * 2 / 10)"
        self.assertEquals(parser.solve(eqn), 2)
        eqn = "-2 * 2 + 2 * 10 / 10"
        self.assertEquals(parser.solve(eqn), -2)

    def test_funcparams_mul(self):
        """Test Mult and Division: Data"""

        parser = Parser(cr2.Run())
        eqn = "cr2.thermal.Thermal:temp * 10.0"
        series = parser.data.thermal.data_frame["temp"]
        assert_series_equal(parser.solve(eqn), series * 10.0)
        eqn = "cr2.thermal.Thermal:temp / cr2.thermal.Thermal:temp * 10"
        assert_series_equal(parser.solve(eqn), series / series * 10)

    def test_var_forward(self):
        """Test Forwarding: Variable"""

        pvars = {}
        pvars["control_temp"] = 78000
        parser = Parser(cr2.Run(), pvars=pvars)
        eqn = "numpy.mean(cr2.thermal.Thermal:temp) < control_temp"
        self.assertTrue(parser.solve(eqn))

    def test_func_forward(self):
        """Test Forwarding: Mixed"""

        pvars = {}
        pvars["mean"] = np.mean
        pvars["control_temp"] = 78000
        parser = Parser(cr2.Run(), pvars=pvars)
        eqn = "mean(cr2.thermal.Thermal:temp) < control_temp"
        self.assertTrue(parser.solve(eqn))
