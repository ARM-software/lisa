#!/usr/bin/python
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
# File:        test_plot_utils.py
# ----------------------------------------------------------------
# $
#

import unittest
import matplotlib
import pandas as pd

from test_thermal import BaseTestThermal
import cr2
import plot_utils

class TestPlotUtils(unittest.TestCase):
    def test_normalize_title(self):
        """Test normalize_title"""
        self.assertEquals(plot_utils.normalize_title("Foo", ""), "Foo")
        self.assertEquals(plot_utils.normalize_title("Foo", "Bar"), "Bar - Foo")

    def test_set_lim(self):
        """Test set_lim()"""

        class GetSet(object):
            def __init__(self):
                self.min = 1
                self.max = 2

            def get(self):
                return (self.min, self.max)

            def set(self, minimum, maximum):
                self.min = minimum
                self.max = maximum

        gs = GetSet()

        plot_utils.set_lim("default", gs.get, gs.set)
        self.assertEquals(gs.min, 1)
        self.assertEquals(gs.max, 2)

        plot_utils.set_lim("range", gs.get, gs.set)
        self.assertEquals(gs.min, 0.9)
        self.assertEquals(gs.max, 2.1)

        plot_utils.set_lim((0, 100), gs.get, gs.set)
        self.assertEquals(gs.min, 0)
        self.assertEquals(gs.max, 100)

    def test_set_ylim(self):
        """Test that set_ylim() doesn't bomb"""

        _, ax = matplotlib.pyplot.subplots()

        plot_utils.set_ylim(ax, "default")
        plot_utils.set_ylim(ax, (0, 5))

    def test_set_xlim(self):
        """Test that set_xlim() doesn't bomb"""

        _, ax = matplotlib.pyplot.subplots()

        plot_utils.set_xlim(ax, "default")
        plot_utils.set_xlim(ax, (0, 5))

    def test_pre_plot_setup(self):
        """Test that plot_utils.pre_plot_setup() doesn't bomb"""
        plot_utils.pre_plot_setup(None, None)
        plot_utils.pre_plot_setup(height=9, width=None)
        plot_utils.pre_plot_setup(height=None, width=9)
        plot_utils.pre_plot_setup(3, 9)

        axis = plot_utils.pre_plot_setup(ncols=2)
        self.assertEquals(len(axis), 2)

        axis = plot_utils.pre_plot_setup(nrows=2, ncols=3)
        self.assertEquals(len(axis), 2)
        self.assertEquals(len(axis[0]), 3)
        self.assertEquals(len(axis[1]), 3)

    def test_post_plot_setup(self):
        """Test that post_plot_setup() doesn't bomb"""

        _, ax = matplotlib.pyplot.subplots()

        plot_utils.post_plot_setup(ax)
        plot_utils.post_plot_setup(ax, title="Foo")
        plot_utils.post_plot_setup(ax, ylim=(0, 72))
        plot_utils.post_plot_setup(ax, ylim="range")
        plot_utils.post_plot_setup(ax, xlabel="Bar")
        plot_utils.post_plot_setup(ax, xlim=(0, 100))
        plot_utils.post_plot_setup(ax, xlim="default")

    def test_plot_hist(self):
        """Test that plost_hist doesn't bomb"""
        data = pd.Series([1, 1, 2, 4])

        _, ax = matplotlib.pyplot.subplots()
        plot_utils.plot_hist(data, ax, "Foo", "m", 20, "numbers", (0, 4), "default")

class TestPlotUtilsNeedTrace(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestPlotUtilsNeedTrace, self).__init__(*args, **kwargs)
        self.map_label = {"00000000,00000039": "A53", "00000000,00000006": "A57"}
        self.actor_order = ["GPU", "A57", "A53"]

    def test_number_freq_plots(self):
        """Calculate the number of frequency plots correctly"""
        trace_out = ""

        run = cr2.Run()
        self.assertEquals(plot_utils.number_freq_plots([run], self.map_label),
                          3)

        # Strip out devfreq traces
        with open("trace.txt") as fin:
            for line in fin:
                if ("thermal_power_devfreq_get_power:" not in line) and \
                   ("thermal_power_devfreq_limit:" not in line):
                    trace_out += line

        with open("trace.txt", "w") as fout:
            fout.write(trace_out)

        # Without devfreq there should only be two plots
        run = cr2.Run()
        self.assertEquals(plot_utils.number_freq_plots([run], self.map_label),
                          2)

    def test_plot_temperature(self):
        """Test that plot_utils.plot_temperature() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_temperature(runs, ylim="default")
        matplotlib.pyplot.close('all')

    def test_plot_load(self):
        """Test that plot_utils.plot_load() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_load(runs, self.map_label, height=5)
        matplotlib.pyplot.close('all')

    def test_plot_load_single_run(self):
        """plot_utils.plot_load() can be used with a single run"""
        run = cr2.Run()

        plot_utils.plot_load([run], self.map_label)
        matplotlib.pyplot.close('all')

    def test_plot_allfreqs(self):
        """Test that plot_utils.plot_allfreqs() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_allfreqs(runs, self.map_label, width=20)
        matplotlib.pyplot.close('all')

    def test_plot_allfreqs_single_run(self):
        """plot_utils.plot_allfreqs() can be used with a single run"""
        run = cr2.Run()

        plot_utils.plot_allfreqs([run], self.map_label)
        matplotlib.pyplot.close('all')

    def test_plot_controller(self):
        """plot_utils.plot_controller() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_controller(runs, height=5)
        matplotlib.pyplot.close('all')

    def test_plot_input_power(self):
        """plot_utils.plot_input_power() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_input_power(runs, self.actor_order, width=20)
        matplotlib.pyplot.close('all')

    def test_plot_output_power(self):
        """plot_utils.plot_output_power() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_output_power(runs, self.actor_order, width=20)
        matplotlib.pyplot.close('all')

    def test_plot_freq_hists(self):
        """plot_utils.plot_freq_hists() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_freq_hists(runs, self.map_label)
        matplotlib.pyplot.close('all')

    def test_plot_freq_hists_single_run(self):
        """plot_utils.plot_freq_hists() works with a single run"""

        run = cr2.Run()

        plot_utils.plot_freq_hists([run], self.map_label)
        matplotlib.pyplot.close('all')

    def test_plot_temperature_hist(self):
        """plot_utils.plot_temperature_hist() doesn't bomb"""

        run1 = cr2.Run(name="first")
        run2 = cr2.Run(name="second")
        runs = [run1, run2]

        plot_utils.plot_temperature_hist(runs)
        matplotlib.pyplot.close('all')
