#    Copyright 2015-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from test_thermal import BaseTestThermal
import trappy
from trappy.stats.grammar import Parser
from pandas.util.testing import assert_series_equal
import numpy as np
import pandas
from distutils.version import LooseVersion as V
import unittest


class TestStatsGrammar(BaseTestThermal):

    def __init__(self, *args, **kwargs):
        super(TestStatsGrammar, self).__init__(*args, **kwargs)

    def test_sum_operator(self):
        """Test Addition And Subtraction: Numeric"""

        parser = Parser(trappy.BareTrace())
        # Simple equation
        eqn = "10 + 2 - 3"
        self.assertEquals(parser.solve(eqn), 9)
        # Equation with bracket and unary ops
        eqn = "(10 + 2) - (-3 + 2)"
        self.assertEquals(parser.solve(eqn), 13)

    @unittest.skipIf(V(pandas.__version__) < V('0.16.1'),
                     "check_names is not supported in pandas < 0.16.1")
    def test_accessors_sum(self):
        """Test Addition And Subtraction: Data"""

        thermal_zone_id = 0
        parser = Parser(trappy.FTrace())
        # Equation with dataframe accessors
        eqn = "trappy.thermal.Thermal:temp + \
trappy.thermal.Thermal:temp"

        assert_series_equal(
            parser.solve(eqn)[thermal_zone_id],
            2 *
            parser.data.thermal.data_frame["temp"], check_names=False)

    def test_funcparams_sum(self):
        """Test Addition And Subtraction: Functions"""

        thermal_zone_id = 0
        parser = Parser(trappy.FTrace())
        # Equation with functions as parameters (Mixed)
        eqn = "numpy.mean(trappy.thermal.Thermal:temp) + 1000"
        self.assertEquals(
            parser.solve(eqn)[thermal_zone_id],
            np.mean(
                parser.data.thermal.data_frame["temp"]) +
            1000)
        # Multiple func params
        eqn = "numpy.mean(trappy.thermal.Thermal:temp) + numpy.mean(trappy.thermal.Thermal:temp)"
        self.assertEquals(
            parser.solve(eqn)[thermal_zone_id],
            np.mean(
                parser.data.thermal.data_frame["temp"]) *
            2)

    def test_parser_with_name(self):
        """Test equation using event name"""

        thermal_zone_id = 0
        parser = Parser(trappy.FTrace())
        # Equation with functions as parameters (Mixed)
        eqn = "numpy.mean(thermal:temp) + 1000"
        self.assertEquals(
            parser.solve(eqn)[thermal_zone_id],
            np.mean(
                parser.data.thermal.data_frame["temp"]) + 1000)

    def test_bool_ops_vector(self):
        """Test Logical Operations: Vector"""

        thermal_zone_id = 0
        # The equation returns a vector mask
        parser = Parser(trappy.FTrace())
        eqn = "(trappy.thermal.ThermalGovernor:current_temperature > 77000)\
                & (trappy.pid_controller.PIDController:output > 2500)"
        mask = parser.solve(eqn)
        self.assertEquals(len(parser.ref(mask.dropna()[0])), 0)

    def test_bool_ops_scalar(self):
        """Test Logical Operations: Vector"""

        thermal_zone_id=0
        parser = Parser(trappy.FTrace())
        # The equation returns a boolean scalar
        eqn = "(numpy.mean(trappy.thermal.Thermal:temp) > 65000) && (numpy.mean(trappy.cpu_power.CpuOutPower) > 500)"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])
        eqn = "(numpy.mean(trappy.thermal.Thermal:temp) > 65000) || (numpy.mean(trappy.cpu_power.CpuOutPower) < 500)"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])

    def test_super_indexing(self):
        "Test if super-indexing works correctly"""

        trace = trappy.FTrace()
        parser = Parser(trace)
        # The first event has less index values
        sol1 = parser.solve("trappy.thermal.Thermal:temp")
        # The second index has more index values
        sol2 = parser.solve("trappy.pid_controller.PIDController:output")
        # Super Indexing should result in len(sol2) > len(sol1)
        self.assertGreater(len(sol2), len(sol1))

    def test_single_func_call(self):
        """Test Single Function Call"""

        thermal_zone_id = 0
        parser = Parser(trappy.FTrace())
        eqn = "numpy.mean(trappy.thermal.Thermal:temp)"
        self.assertEquals(
            parser.solve(eqn)[thermal_zone_id],
            np.mean(
                parser.data.thermal.data_frame["temp"]))

    def test_mul_ops(self):
        """Test Mult and Division: Numeric"""

        parser = Parser(trappy.BareTrace())
        eqn = "(10 * 2 / 10)"
        self.assertEquals(parser.solve(eqn), 2)
        eqn = "-2 * 2 + 2 * 10 / 10"
        self.assertEquals(parser.solve(eqn), -2)
        eqn = "3.5 // 2"
        self.assertEquals(parser.solve(eqn), 1)
        eqn = "5 % 2"
        self.assertEquals(parser.solve(eqn), 1)

    def test_exp_ops(self):
        """Test exponentiation: Numeric"""
        parser = Parser(trappy.BareTrace())
        eqn = "3**3 * 2**4"
        self.assertEquals(parser.solve(eqn), 432)
        eqn = "3**(4/2)"
        self.assertEquals(parser.solve(eqn), 9)

    @unittest.skipIf(V(pandas.__version__) < V('0.16.1'),
                     "check_names is not supported in pandas < 0.16.1")
    def test_funcparams_mul(self):
        """Test Mult and Division: Data"""

        thermal_zone_id = 0
        parser = Parser(trappy.FTrace())
        eqn = "trappy.thermal.Thermal:temp * 10.0"
        series = parser.data.thermal.data_frame["temp"]
        assert_series_equal(parser.solve(eqn)[thermal_zone_id], series * 10.0, check_names=False)
        eqn = "trappy.thermal.Thermal:temp / trappy.thermal.Thermal:temp * 10"
        assert_series_equal(parser.solve(eqn)[thermal_zone_id], series / series * 10, check_names=False)

    def test_var_forward(self):
        """Test Forwarding: Variable"""

        thermal_zone_id = 0
        pvars = {}
        pvars["control_temp"] = 78000
        parser = Parser(trappy.FTrace(), pvars=pvars)
        eqn = "numpy.mean(trappy.thermal.Thermal:temp) < control_temp"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])

    def test_func_forward(self):
        """Test Forwarding: Mixed"""

        thermal_zone_id = 0
        pvars = {}
        pvars["mean"] = np.mean
        pvars["control_temp"] = 78000
        parser = Parser(trappy.FTrace(), pvars=pvars)
        eqn = "mean(trappy.thermal.Thermal:temp) < control_temp"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])

    def test_cls_forward(self):
        """Test Forwarding: Classes"""

        cls = trappy.thermal.Thermal
        pvars = {}
        pvars["mean"] = np.mean
        pvars["control_temp"] = 78000
        pvars["therm"] = cls

        thermal_zone_id = 0
        parser = Parser(trappy.FTrace(), pvars=pvars)
        eqn = "mean(therm:temp) < control_temp"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])

    def test_for_parsed_event(self):
        """Test if an added parsed event can be accessed"""

        trace = trappy.FTrace(scope="custom")
        dfr = pandas.DataFrame({"l1_misses": [24, 535,  41],
                                "l2_misses": [155, 11, 200],
                                "cpu":       [ 0,   1,   0]},
                           index=pandas.Series([1.020, 1.342, 1.451], name="Time"))
        trace.add_parsed_event("pmu_counters", dfr)

        p = Parser(trace)
        self.assertTrue(len(p.solve("pmu_counters:cpu")), 3)

    def test_windowed_parse(self):
        """Test that the parser can operate on a window of the trace"""
        trace = trappy.FTrace()

        prs = Parser(trace, window=(2, 3))
        dfr_res = prs.solve("thermal:temp")

        self.assertGreater(dfr_res.index[0], 2)
        self.assertLess(dfr_res.index[-1], 3)

        prs = Parser(trace, window=(4, None))
        dfr_res = prs.solve("thermal:temp")

        self.assertGreater(dfr_res.index[0], 4)
        self.assertEquals(dfr_res.index[-1], trace.thermal.data_frame.index[-1])

        prs = Parser(trace, window=(0, 1))
        dfr_res = prs.solve("thermal:temp")

        self.assertEquals(dfr_res.index[0], trace.thermal.data_frame.index[0])
        self.assertLess(dfr_res.index[-1], 1)

    def test_filtered_parse(self):
        """The Parser can filter a trace"""
        trace = trappy.FTrace()

        prs = Parser(trace, filters={"cdev_state": 3})
        dfr_res = prs.solve("devfreq_out_power:freq")
        self.assertEquals(len(dfr_res), 1)
