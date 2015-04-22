#!/usr/bin/env python
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
# File:        test_plotter.py
# ----------------------------------------------------------------
# $
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
                         [cr2.power.OutPower],
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
            [run1, run2], cr2.power.OutPower, column="power", pivot="cpus")
        l.view(test=True)

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
        l.view(test=True)

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
        l.view(test=True)

    def test_plot_savefig(self):
        """Tests plotter: savefig"""
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
        png_file = tempfile.mktemp(dir="/tmp", suffix=".png")
        l.savefig(png_file)
        self.assertTrue(os.path.isfile(png_file))
        os.remove(png_file)
