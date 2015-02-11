#!/usr/bin/env python

import unittest
import matplotlib
import pandas as pd

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
        l.view()
        matplotlib.pyplot.close('all')

    def test_plot_multi_run(self):
        """Tests LinePlot with no Pivot multi runs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            [run1, run2], cr2.thermal.Thermal, column="temp")
        l.view()
        matplotlib.pyplot.close('all')

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
        l.view()
        matplotlib.pyplot.close('all')

    def test_plot_filter(self):
        """Tests LinePlot with no Pivot with filters"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot([run1,
                          run2],
                         [cr2.power.OutPower],
                         column=["power"],
                         filters={"cdev_state": [1]})
        l.view()
        matplotlib.pyplot.close('all')

    def test_plot_pivot(self):
        """Tests LinePlot with Pivot"""
        run1 = cr2.Run(name="first")
        l = cr2.LinePlot(
            run1,
            cr2.thermal.Thermal,
            column="temp",
            pivot="thermal_zone")
        l.view()
        matplotlib.pyplot.close('all')

    def test_plot_multi_run_pivot(self):
        """Tests LinePlot with Pivot multi runs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            [run1, run2], cr2.power.OutPower, column="power", pivot="cpus")
        l.view()
        matplotlib.pyplot.close('all')

    def test_plot_multi_pivot(self):
        """Tests LinePlot with Pivot with multi attrs"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot([run1,
                          run2],
                         [cr2.power.InPower,
                          cr2.power.OutPower],
                         column=["dynamic_power",
                                 "power"],
                         pivot="cpus")
        l.view()
        matplotlib.pyplot.close('all')

    def test_plot_multi_pivot_filter(self):
        """Tests LinePlot with Pivot and filters"""
        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        l = cr2.LinePlot(
            run1,
            cr2.power.InPower,
            column=[
                "dynamic_power",
                "load1"],
            filters={
                "cdev_state": [
                    1,
                    0]},
            pivot="cpus")
        l.view()
        matplotlib.pyplot.close('all')
