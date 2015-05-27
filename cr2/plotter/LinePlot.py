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
# File:        LinePlot.py
# ----------------------------------------------------------------
# $
#
"""This module contains the class for plotting and
customizing Line Plots with a pandas dataframe input
"""

import matplotlib.pyplot as plt
import AttrConf
from Constraint import ConstraintManager
from PlotLayout import PlotLayout
from AbstractDataPlotter import AbstractDataPlotter
from ColorMap import ColorMap


class LinePlot(AbstractDataPlotter):

    """The plots are plotted by default against the dataframe index
       The column="col_name" specifies the name of the column to
       be plotted

       filters =
        {
          "pid": [ 3338 ],
          "cpu": [0, 2, 4],
        }

       The above filters will filter the column to be plotted as per the
       specified criteria.

       per_line input is used to control the number of graphs
       in each graph subplot row
       concat, Draws all the graphs on a single plot
    """

    def __init__(self, runs, templates=None, **kwargs):
        # Default keys, each can be overridden in kwargs
        self._attr = {}
        self.runs = runs
        self.templates = templates
        self.set_defaults()
        self._fig = None
        self._layout = None
        self._constraints = []

        self._check_data()

        for key in kwargs:
            if key in AttrConf.ARGS_TO_FORWARD:
                self._attr["args_to_forward"][key] = kwargs[key]
            else:
                self._attr[key] = kwargs[key]

        if "column" not in self._attr:
            raise RuntimeError("Value Column not specified")

        self.c_mgr = ConstraintManager(
            runs,
            self._attr["column"],
            templates,
            self._attr["pivot"],
            self._attr["filters"])
        self._constraints = self.c_mgr.constraints
        super(LinePlot, self).__init__()

    def savefig(self, *args, **kwargs):
        if self._fig == None:
            self.view()
        self._fig.savefig(*args, **kwargs)

    def view(self, test=False):
        """Displays the graph"""

        if test:
            self._attr["style"] = True
            AttrConf.MPL_STYLE["interactive"] = False

        if self._attr["concat"]:
            if self._attr["style"]:
                with plt.rc_context(AttrConf.MPL_STYLE):
                    self._plot_concat()
            else:
                self._plot_concat()
        else:
            if self._attr["style"]:
                with plt.rc_context(AttrConf.MPL_STYLE):
                    self._plot()
            else:
                self._plot()

    def set_defaults(self):
        """Sets the default attrs"""
        self._attr["width"] = AttrConf.WIDTH
        self._attr["length"] = AttrConf.LENGTH
        self._attr["per_line"] = AttrConf.PER_LINE
        self._attr["concat"] = AttrConf.CONCAT
        self._attr["fill"] = AttrConf.FILL
        self._attr["filters"] = {}
        self._attr["style"] = True
        self._attr["pivot"] = AttrConf.PIVOT
        self._attr["xlim"] = AttrConf.XLIM
        self._attr["ylim"] = AttrConf.XLIM
        self._attr["args_to_forward"] = {}

    def _plot(self):
        """Internal Method called to draw the plot"""
        pivot_vals = self.c_mgr.get_all_pivots()

        # Create a 2D Layout
        self._layout = PlotLayout(
            self._attr["per_line"],
            len(pivot_vals),
            width=self._attr["width"],
            length=self._attr["length"])

        self._fig = self._layout.get_fig()

        legend = [None] * len(self._constraints)
        legend_str = self.c_mgr.constraint_labels()
        constraint_index = 0
        cmap = ColorMap(len(self._constraints))

        for constraint in self._constraints:
            result = constraint.result
            plot_index = 0
            for pivot in pivot_vals:
                if pivot in result:
                    axis = self._layout.get_axis(plot_index)
                    line_2d_list = axis.plot(
                        result[pivot].index,
                        result[pivot].values,
                        color=cmap.cmap(constraint_index),
                        **self._attr["args_to_forward"])

                    if self._attr["fill"]:
                        drawstyle = line_2d_list[0].get_drawstyle()
                        # This has been fixed in upstream matplotlib
                        if drawstyle.startswith("steps"):
                            raise UserWarning("matplotlib does not support fill for step plots")

                        xdat, ydat = line_2d_list[0].get_data(orig=False)
                        axis.fill_between(xdat,
                            axis.get_ylim()[0],
                            ydat,
                            facecolor=cmap.cmap(constraint_index),
                            alpha=AttrConf.ALPHA)

                    legend[constraint_index] = line_2d_list[0]
                    if self._attr["xlim"] != None:
                        axis.set_xlim(self._attr["xlim"])
                    if self._attr["ylim"] != None:
                        axis.set_ylim(self._attr["ylim"])

                else:
                    axis = self._layout.get_axis(plot_index)
                    axis.plot([], [], **self._attr["args_to_forward"])
                plot_index += 1

            constraint_index += 1

        for l_idx, legend_line in enumerate(legend):
            if not legend_line:
                del legend[l_idx]
                del legend_str[l_idx]
        self._fig.legend(legend, legend_str)

        plot_index = 0
        for pivot_val in pivot_vals:
            if pivot_val != AttrConf.PIVOT_VAL:
                self._layout.get_axis(plot_index).set_title( \
                    self._attr["pivot"] + \
                    ":" + \
                    str(pivot_val))
            else:
                self._layout.get_axis(plot_index).set_title(
                    self._attr["column"])
            plot_index += 1

        self._layout.finish(len(pivot_vals))

    def _plot_concat(self):
        """Plot all lines on a single figure"""

        pivot_vals = self.c_mgr.get_all_pivots()
        num_lines = len(pivot_vals)

        cmap = ColorMap(num_lines)

        self._layout = PlotLayout(
            self._attr["per_line"],
            len(self._constraints),
            width=self._attr["width"],
            length=self._attr["length"])

        self._fig = self._layout.get_fig()

        pivot_index = 0
        legend = [None] * len(pivot_vals)
        legend_str = []
        for pivot in pivot_vals:
            plot_index = 0
            for constraint in self._constraints:
                result = constraint.result
                if pivot in result:
                    axis = self._layout.get_axis(plot_index)
                    line_2d_list = axis.plot(
                        result[pivot].index,
                        result[pivot].values,
                        color=cmap.cmap(pivot_index),
                        **self._attr["args_to_forward"])
                    if self._attr["xlim"] != None:
                        axis.set_xlim(self._attr["xlim"])
                    if self._attr["ylim"] != None:
                        axis.set_ylim(self._attr["ylim"])
                    legend[pivot_index] = line_2d_list[0]

                    if self._attr["fill"]:

                        drawstyle = line_2d_list[0].get_drawstyle()
                        if drawstyle.startswith("steps"):
                            # This has been fixed in upstream matplotlib
                            raise UserWarning("matplotlib does not support fill for step plots")

                        xdat, ydat = line_2d_list[0].get_data(orig=False)
                        axis.fill_between(xdat,
                            axis.get_ylim()[0],
                            ydat,
                            facecolor=cmap.cmap(pivot_index),
                            alpha=AttrConf.ALPHA)
                else:
                    axis = self._layout.get_axis(plot_index)
                    axis.plot(
                        [],
                        [],
                        color=cmap.cmap(pivot_index),
                        **self._attr["args_to_forward"])
                plot_index += 1

            if pivot != AttrConf.PIVOT_VAL:
                legend_str.append(self._attr["pivot"] + ":" + str(pivot))
            else:
                legend_str.append(self._attr["column"])
            pivot_index += 1

        self._fig.legend(legend, legend_str)
        plot_index = 0
        for constraint in self._constraints:
            self._layout.get_axis(plot_index).set_title(str(constraint))
            plot_index += 1

        self._layout.finish(len(self._constraints))
