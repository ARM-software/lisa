#    Copyright 2015-2016 ARM Limited
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

"""
An example file for usage of Analyzer for thermal assertions
"""
from bart.common.Analyzer import Analyzer
from trappy.stats.Topology import Topology
import unittest
import trappy


class TestThermal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # We can run a workload invocation script here
        # Which then copies the required traces for analysis to
        # the host.
        trace_file = "update_a_trace_path_here"
        ftrace = trappy.FTrace(trace_file, "test_run")

        # Define the parameters that you intend to use in the grammar
        config = {}
        config["THERMAL"] = trappy.thermal.Thermal
        config["OUT"] = trappy.cpu_power.CpuOutPower
        config["IN"] = trappy.cpu_power.CpuInPower
        config["PID"] = trappy.pid_controller.PIDController
        config["GOVERNOR"] = trappy.thermal.ThermalGovernor
        config["CONTROL_TEMP"] = 77000
        config["SUSTAINABLE_POWER"] = 2500
        config["EXPECTED_TEMP_QRT"] = 95
        config["EXPECTED_STD_PCT"] = 5

        # Define a Topology
        cls.BIG = '000000f0'
        cls.LITTLE = '0000000f'
        cls.tz = 0
        cls.analyzer = Analyzer(ftrace, config)

    def test_temperature_quartile(self):
        """Assert Temperature quartile"""

        self.assertTrue(self.analyzer.assertStatement(
            "numpy.percentile(THERMAL:temp, EXPECTED_TEMP_QRT) < (CONTROL_TEMP + 5000)"))

    def test_average_temperature(self):
        """Assert Average temperature"""

        self.assertTrue(self.analyzer.assertStatement(
            "numpy.mean(THERMAL:temp) < CONTROL_TEMP", select=self.tz))

    def test_temp_stdev(self):
        """Assert StdDev(temp) as % of mean"""

        self.assertTrue(self.analyzer.assertStatement(
            "(numpy.std(THERMAL:temp) * 100.0) / numpy.mean(THERMAL:temp)\
             < EXPECTED_STD_PCT", select=self.tz))

    def test_zero_load_input_power(self):
        """Test power demand when load is zero"""

        zero_load_power_big = self.analyzer.getStatement("((IN:load0 + IN:load1 + IN:load2 + IN:load3) == 0) \
                                                     & (IN:dynamic_power > 0)", reference=True, select=self.BIG)
        self.assertEquals(len(zero_load_power_big), 0)

        zero_load_power_little = self.analyzer.getStatement("((IN:load0 + IN:load1 + IN:load2 + IN:load3) == 0) \
                                                     & (IN:dynamic_power > 0)", reference=True, select=self.LITTLE)
        self.assertEquals(len(zero_load_power_little), 0)

    def test_sustainable_power(self):
        """temp > control_temp, allocated_power < sustainable_power"""

        self.analyzer.getStatement("(GOVERNOR:current_temperature > CONTROL_TEMP) &\
            (PID:output > SUSTAINABLE_POWER)", reference=True, select=0)
