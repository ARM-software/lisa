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


import unittest
import matplotlib
import pandas as pd
import tempfile
import os

from test_thermal import BaseTestThermal
import cr2


class TestPlotter(BaseTestThermal):

    """No Bombing testcases for plotter"""

    def __init__(self, *args, **kwargs):
        super(TestPlotter, self).__init__(*args, **kwargs)

    def test_plot_no_pivot(self):
        """Tests LinePlot with no pivot"""
        run1 = cr2.Run(name="first")
        l = cr2.LinePlot(run1, cr2.thermal.Thermal, column="temp")
        l.view(test=True)

    def test_plot_multi_run(self):
        """Tests LinePlot with no Pivot multi runs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            [run1, run2], cr2.thermal.Thermal, column="temp")
        l.view(test=True)

    def test_plot_multi(self):
        """Tests LinePlot with no Pivot multi attrs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot([run1,
                          run2],
                         [cr2.thermal.Thermal,
                          cr2.thermal.ThermalGovernor],
                         column=["temp",
                                 "power_range"])
        l.view(test=True)

    def test_plot_filter(self):
        """Tests LinePlot with no Pivot with filters"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot([run1,
                          run2],
                         [cr2.cpu_power.CpuOutPower],
                         column=["power"],
                         filters={"cdev_state": [1]})
        l.view(test=True)

    def test_plot_pivot(self):
        """Tests LinePlot with Pivot"""
        run1 = cr2.Run(name="first")
        l = cr2.LinePlot(
            run1,
            cr2.thermal.Thermal,
            column="temp",
            pivot="thermal_zone")
        l.view(test=True)

    def test_plot_multi_run_pivot(self):
        """Tests LinePlot with Pivot multi runs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            [run1, run2], cr2.cpu_power.CpuOutPower, column="power", pivot="cpus")
        l.view(test=True)

    def test_plot_multi_pivot(self):
        """Tests LinePlot with Pivot with multi attrs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot([run1,
                          run2],
                         [cr2.cpu_power.CpuInPower,
                          cr2.cpu_power.CpuOutPower],
                         column=["dynamic_power",
                                 "power"],
                         pivot="cpus")
        l.view(test=True)

    def test_plot_multi_pivot_filter(self):
        """Tests LinePlot with Pivot and filters"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            run1,
            cr2.cpu_power.CpuInPower,
            column=[
                "dynamic_power",
                "load1"],
            filters={
                "cdev_state": [
                    1,
                    0]},
            pivot="cpus")
        l.view(test=True)

    def test_plot_savefig(self):
        """Tests plotter: savefig"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            run1,
            cr2.cpu_power.CpuInPower,
            column=[
                "dynamic_power",
                "load1"],
            filters={
                "cdev_state": [
                    1,
                    0]},
            pivot="cpus")
        png_file = tempfile.mktemp(dir="/tmp", suffix=".png")
        l.savefig(png_file)
        self.assertTrue(os.path.isfile(png_file))
        os.remove(png_file)
