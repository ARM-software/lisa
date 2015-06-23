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
from cr2.plotter import AttrConf
from cr2.plotter.Constraint import ConstraintManager
from cr2.plotter.PlotLayout import PlotLayout
from cr2.plotter.AbstractDataPlotter import AbstractDataPlotter
from cr2.plotter.ColorMap import ColorMap


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
       permute, draws one plot for each of the runs specified
    """

    def __init__(self, runs, templates=None, **kwargs):
        # Default keys, each can be overridden in kwargs
        self._attr = {}
        self.runs = runs
        self.templates = templates
        self.set_defaults()
        self._fig = None
        self._layout = None

        self._check_data()

        for key in kwargs:
            if key in AttrConf.ARGS_TO_FORWARD:
                self._attr["args_to_forward"][key] = kwargs[key]
            else:
                self._attr[key] = kwargs[key]

        if "column" not in self._attr:
            raise RuntimeError("Value Column not specified")

        zip_constraints = not self._attr["permute"]
        self.c_mgr = ConstraintManager(
            runs,
            self._attr["column"],
            templates,
            self._attr["pivot"],
            self._attr["filters"], zip_constraints)
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
                    self._plot(self._attr["permute"])
            else:
                self._plot(self._attr["permute"])

    def set_defaults(self):
        """Sets the default attrs"""
        self._attr["width"] = AttrConf.WIDTH
        self._attr["length"] = AttrConf.LENGTH
        self._attr["per_line"] = AttrConf.PER_LINE
        self._attr["concat"] = AttrConf.CONCAT
        self._attr["fill"] = AttrConf.FILL
        self._attr["filters"] = {}
        self._attr["style"] = True
        self._attr["permute"] = False
        self._attr["pivot"] = AttrConf.PIVOT
        self._attr["xlim"] = AttrConf.XLIM
        self._attr["ylim"] = AttrConf.XLIM
        self._attr["args_to_forward"] = {}

    def _plot(self, permute):
        """Internal Method called to draw the plot"""
        pivot_vals, len_pivots = self.c_mgr.generate_pivots(permute)

        # Create a 2D Layout
        self._layout = PlotLayout(
            self._attr["per_line"],
            len_pivots,
            width=self._attr["width"],
            length=self._attr["length"])

        self._fig = self._layout.get_fig()
        legend_str = []
        plot_index = 0

        if permute:
            legend = [None] * self.c_mgr._max_len
            cmap = ColorMap(self.c_mgr._max_len)
        else:
            legend = [None] * len(self.c_mgr)
            cmap = ColorMap(len(self.c_mgr))

        for p_val in pivot_vals:
            l_index = 0
            for constraint in self.c_mgr:
                if permute:
                    run_idx, pivot = p_val
                    if constraint.run_index != run_idx:
                        continue
                    legend_str.append(constraint._column)
                    l_index = self.c_mgr.get_column_index(constraint)
                    title = constraint.get_data_name() + ":"
                else:
                    pivot = p_val
                    legend_str.append(str(constraint))
                    title = ""

                result = constraint.result
                if pivot in result:
                    axis = self._layout.get_axis(plot_index)
                    line_2d_list = axis.plot(
                        result[pivot].index,
                        result[pivot].values,
                        color=cmap.cmap(l_index),
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
                            facecolor=cmap.cmap(l_index),
                            alpha=AttrConf.ALPHA)

                    legend[l_index] = line_2d_list[0]
                    if self._attr["xlim"] != None:
                        axis.set_xlim(self._attr["xlim"])
                    if self._attr["ylim"] != None:
                        axis.set_ylim(self._attr["ylim"])

                else:
                    axis = self._layout.get_axis(plot_index)
                    axis.plot([], [], **self._attr["args_to_forward"])

                l_index += 1

            if pivot == AttrConf.PIVOT_VAL:
                title += ",".join(self._attr["column"])
            else:
                title += "{0}: {1}".format(self._attr["pivot"], pivot)

            axis.set_title(title)
            plot_index += 1

        for l_idx, legend_line in enumerate(legend):
            if not legend_line:
                del legend[l_idx]
                del legend_str[l_idx]
        self._fig.legend(legend, legend_str)
        self._layout.finish(len_pivots)

    def _plot_concat(self):
        """Plot all lines on a single figure"""

        pivot_vals, len_pivots = self.c_mgr.generate_pivots()
        cmap = ColorMap(len_pivots)

        self._layout = PlotLayout(self._attr["per_line"], len(self.c_mgr),
                                  width=self._attr["width"],
                                  length=self._attr["length"])

        self._fig = self._layout.get_fig()
        legend = [None] * len_pivots
        legend_str = [""] * len_pivots
        plot_index = 0

        for constraint in self.c_mgr:
            result = constraint.result
            title = str(constraint)
            result = constraint.result
            pivot_index = 0
            for pivot in pivot_vals:

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

                    if pivot == AttrConf.PIVOT_VAL:
                        legend_str[pivot_index] = self._attr["column"]
                    else:
                        legend_str[pivot_index] = "{0}: {1}".format(self._attr["pivot"], pivot)

                else:
                    axis = self._layout.get_axis(plot_index)
                    axis.plot(
                        [],
                        [],
                        color=cmap.cmap(pivot_index),
                        **self._attr["args_to_forward"])
                pivot_index += 1
            plot_index += 1

        self._fig.legend(legend, legend_str)
        plot_index = 0
        for constraint in self.c_mgr:
            self._layout.get_axis(plot_index).set_title(str(constraint))
            plot_index += 1
        self._layout.finish(len(self.c_mgr))
