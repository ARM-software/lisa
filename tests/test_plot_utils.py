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

    def test_post_plot_setup(self):
        """Test that post_plot_setup() doesn't bomb"""

        ax = plot_utils.pre_plot_setup()

        plot_utils.post_plot_setup(ax)
        plot_utils.post_plot_setup(ax, title="Foo")
        plot_utils.post_plot_setup(ax, ylim=(0, 72))
        plot_utils.post_plot_setup(ax, xlabel="Bar")

class TestPlotUtilsNeedTrace(TestThermalBase):
    def test_plot_allfreqs(self):
        """Test that plot_allfreqs() doesn't bomb"""

        inp = cr2.InPower()
        outp = cr2.OutPower()
        map_label = {"0000000f": "A7", "000000f0": "A15"}

        plot_utils.plot_allfreqs(inp, outp, map_label)
