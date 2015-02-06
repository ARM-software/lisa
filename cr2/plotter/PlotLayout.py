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
# File:        PlotLayout.py
# ----------------------------------------------------------------
# $
#
"""This module is reponsible for creating a layout
of plots as a 2D axes and handling corener cases
and deleting empty plots
"""

import matplotlib.pyplot as plt
import AttrConf


class PlotLayout(object):

    """Cols is the number of columns to draw
       rows are calculated as 1D - 2D transformation
       the same transformation is used to index the
       axes array
    """

    def __init__(self, cols, num_plots, **kwargs):

        self.cols = cols
        self._attr = {}
        self.num_plots = num_plots
        self._single_plot = False
        if self.num_plots == 0:
            raise RuntimeError("No plots for the given constraints")

        if self.num_plots < self.cols:
            self.cols = self.num_plots
        self.rows = (self.num_plots / self.cols)
        # Avoid Extra Allocation (shows up in savefig!)
        if self.num_plots % self.cols != 0:
            self.rows += 1

        self.usecol = False
        self.userow = False
        self._set_defaults()

        for key in kwargs:
            self._attr[key] = kwargs[key]

        # Scale the plots if there is a single plot and
        # Set boolean variables
        if num_plots == 1:
            self._attr["width"] = int(self._attr["width"] * 2.5)
            self._attr["length"] = int(self._attr["length"] * 1.25)
            self._single_plot = True
        elif self.rows == 1:
            self.usecol = True
        elif self.cols == 1:
            self.userow = True

        self._attr["figure"], self._attr["axes"] = plt.subplots(
            self.rows, self.cols, figsize=(
                self._attr["width"] * self.cols,
                self._attr["length"] * self.rows))

    def _set_defaults(self):
        """set the default attrs"""
        self._attr["width"] = AttrConf.WIDTH
        self._attr["length"] = AttrConf.LENGTH

    def get_2d(self, linear_val):
        """Convert Linear to 2D coordinates"""
        if self.usecol:
            return linear_val % self.cols

        if self.userow:
            return linear_val % self.rows

        val_x = linear_val % self.cols
        val_y = linear_val / self.cols
        return val_y, val_x

    def finish(self, plot_index):
        """Delete the empty cells"""
        while plot_index < (self.rows * self.cols):
            self._attr["figure"].delaxes(
                self._attr["axes"][
                    self.get_2d(plot_index)])
            plot_index += 1

    def get_axis(self, plot_index):
        """Get the axes for the plots"""
        if self._single_plot:
            return self._attr["axes"]
        else:
            return self._attr["axes"][self.get_2d(plot_index)]

    def get_fig(self):
        """Return the matplotlib figure object"""
        return self._attr["figure"]
