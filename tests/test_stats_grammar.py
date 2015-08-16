#    Copyright 2015-2015 ARM Limited
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


class TestStatsGrammar(BaseTestThermal):

    def __init__(self, *args, **kwargs):
        super(TestStatsGrammar, self).__init__(*args, **kwargs)

    def test_sum_operator(self):
        """Test Addition And Subtraction: Numeric"""

        parser = Parser(trappy.Run())
        # Simple equation
        eqn = "10 + 2 - 3"
        self.assertEquals(parser.solve(eqn), 9)
        # Equation with bracket and unary ops
        eqn = "(10 + 2) - (-3 + 2)"
        self.assertEquals(parser.solve(eqn), 13)

    def test_accessors_sum(self):
        """Test Addition And Subtraction: Data"""

        thermal_zone_id = 0
        parser = Parser(trappy.Run())
        # Equation with dataframe accessors
        eqn = "trappy.thermal.Thermal:temp + \
trappy.thermal.Thermal:temp"

        assert_series_equal(
            parser.solve(eqn)[thermal_zone_id],
            2 *
            parser.data.thermal.data_frame["temp"])

    def test_funcparams_sum(self):
        """Test Addition And Subtraction: Functions"""

        thermal_zone_id = 0
        parser = Parser(trappy.Run())
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

    def test_bool_ops_vector(self):
        """Test Logical Operations: Vector"""

        thermal_zone_id = 0
        # The equation returns a vector mask
        parser = Parser(trappy.Run())
        eqn = "(trappy.thermal.ThermalGovernor:current_temperature > 77000)\
                & (trappy.pid_controller.PIDController:output > 2500)"
        mask = parser.solve(eqn)
        self.assertEquals(len(parser.ref(mask.dropna()[0])), 0)

    def test_bool_ops_scalar(self):
        """Test Logical Operations: Vector"""

        thermal_zone_id=0
        parser = Parser(trappy.Run())
        # The equation returns a boolean scalar
        eqn = "(numpy.mean(trappy.thermal.Thermal:temp) > 65000) && (numpy.mean(trappy.cpu_power.CpuOutPower) > 500)"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])
        eqn = "(numpy.mean(trappy.thermal.Thermal:temp) > 65000) || (numpy.mean(trappy.cpu_power.CpuOutPower) < 500)"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])

    def test_single_func_call(self):
        """Test Single Function Call"""

        thermal_zone_id = 0
        parser = Parser(trappy.Run())
        eqn = "numpy.mean(trappy.thermal.Thermal:temp)"
        self.assertEquals(
            parser.solve(eqn)[thermal_zone_id],
            np.mean(
                parser.data.thermal.data_frame["temp"]))

    def test_mul_ops(self):
        """Test Mult and Division: Numeric"""

        parser = Parser(trappy.Run())
        eqn = "(10 * 2 / 10)"
        self.assertEquals(parser.solve(eqn), 2)
        eqn = "-2 * 2 + 2 * 10 / 10"
        self.assertEquals(parser.solve(eqn), -2)

    def test_funcparams_mul(self):
        """Test Mult and Division: Data"""

        thermal_zone_id = 0
        parser = Parser(trappy.Run())
        eqn = "trappy.thermal.Thermal:temp * 10.0"
        series = parser.data.thermal.data_frame["temp"]
        assert_series_equal(parser.solve(eqn)[thermal_zone_id], series * 10.0)
        eqn = "trappy.thermal.Thermal:temp / trappy.thermal.Thermal:temp * 10"
        assert_series_equal(parser.solve(eqn)[thermal_zone_id], series / series * 10)

    def test_var_forward(self):
        """Test Forwarding: Variable"""

        thermal_zone_id = 0
        pvars = {}
        pvars["control_temp"] = 78000
        parser = Parser(trappy.Run(), pvars=pvars)
        eqn = "numpy.mean(trappy.thermal.Thermal:temp) < control_temp"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])

    def test_func_forward(self):
        """Test Forwarding: Mixed"""

        thermal_zone_id = 0
        pvars = {}
        pvars["mean"] = np.mean
        pvars["control_temp"] = 78000
        parser = Parser(trappy.Run(), pvars=pvars)
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
        parser = Parser(trappy.Run(), pvars=pvars)
        eqn = "mean(therm:temp) < control_temp"
        self.assertTrue(parser.solve(eqn)[thermal_zone_id])
