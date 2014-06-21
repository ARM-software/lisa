#!/usr/bin/python

import unittest

from test_thermal import TestThermalBase
import cr2
import plot_utils

class TestPlotUtils(unittest.TestCase):
    def test_set_plot_size(self):
        """Test that plot_utils.set_plot_size() doesn't bomb"""
        plot_utils.set_plot_size(None, None)
        plot_utils.set_plot_size(height=9, width=None)
        plot_utils.set_plot_size(height=None, width=9)
        plot_utils.set_plot_size(3, 9)

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

        ax = plot_utils.pre_plot_setup()

        plot_utils.set_ylim(ax, "default")
        plot_utils.set_ylim(ax, (0, 5))

    def test_set_xlim(self):
        """Test that set_xlim() doesn't bomb"""

        ax = plot_utils.pre_plot_setup()

        plot_utils.set_xlim(ax, "default")
        plot_utils.set_xlim(ax, (0, 5))

    def test_post_plot_setup(self):
        """Test that post_plot_setup() doesn't bomb"""

        ax = plot_utils.pre_plot_setup()

        plot_utils.post_plot_setup(ax)
        plot_utils.post_plot_setup(ax, title="Foo")
        plot_utils.post_plot_setup(ax, ylim=(0, 72))
        plot_utils.post_plot_setup(ax, ylim="range")
        plot_utils.post_plot_setup(ax, xlabel="Bar")
        plot_utils.post_plot_setup(ax, xlim=(0, 100))
        plot_utils.post_plot_setup(ax, xlim="default")

class TestPlotUtilsNeedTrace(TestThermalBase):
    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        inp = cr2.InPower()
        outp = cr2.OutPower()
        map_label = {"0000000f": "A7", "000000f0": "A15"}

        plot_utils.plot_allfreqs(inp, outp, map_label)

    def test_plot_power_hists(self):
        """Test that plot_power_hists() doesn't bomb"""

        inp = cr2.InPower()
        outp = cr2.OutPower()
        map_label = {"0000000f": "A7", "000000f0": "A15"}

        plot_utils.plot_power_hists(inp, outp, map_label)
