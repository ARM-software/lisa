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
# File:        ILinePlot.py
# ----------------------------------------------------------------
# $
#
"""This module contains the class for plotting and
customizing Line Plots with a pandas dataframe input
"""

import matplotlib.pyplot as plt
from cr2.plotter import AttrConf
from cr2.plotter.Constraint import ConstraintManager
from cr2.plotter.ILinePlotGen import ILinePlotGen
from cr2.plotter.AbstractDataPlotter import AbstractDataPlotter
from cr2.plotter.ColorMap import ColorMap
import pandas as pd

if not AttrConf.PLOTTER_IPYTHON:
    raise ImportError("Ipython Environment not Found")

class ILinePlot(AbstractDataPlotter):

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
        self._layout = None
        self._constraints = []

        self._check_data()
        for key in kwargs:
            self._attr[key] = kwargs[key]

        if "column" not in self._attr:
            raise RuntimeError("Value Column not specified")

        if self._attr["drawstyle"] and self._attr["drawstyle"].startswith("steps"):
            self._attr["step_plot"] = True

        self.c_mgr = ConstraintManager(
            runs,
            self._attr["column"],
            templates,
            self._attr["pivot"],
            self._attr["filters"])
        self._constraints = self.c_mgr.constraints
        super(ILinePlot, self).__init__()

    def savefig(self, *args, **kwargs):
        raise NotImplementedError("Not Available for ILinePlot")

    def view(self, test=False):
        """Displays the graph"""

        if self._attr["concat"]:
                self._plot_concat()
        else:
                self._plot()

    def set_defaults(self):
        """Sets the default attrs"""
        self._attr["per_line"] = AttrConf.PER_LINE
        self._attr["concat"] = AttrConf.CONCAT
        self._attr["filters"] = {}
        self._attr["pivot"] = AttrConf.PIVOT
        self._attr["drawstyle"] = None
        self._attr["step_plot"] = False
        self._attr["fill"] = AttrConf.FILL

    def _plot(self):
        """Internal Method called to draw the plot"""
        pivot_vals = self.c_mgr.get_all_pivots()

        self._layout = ILinePlotGen(self._attr["per_line"],
                                    len(pivot_vals),
                                    **self._attr)
        plot_index = 0

        for pivot in pivot_vals:
            data_frame = pd.Series()

            for constraint in self._constraints:
                result = constraint.result
                constraint_str = str(constraint)

                if pivot in result:
                    data_frame[constraint_str] = result[pivot]

            if pivot == AttrConf.PIVOT_VAL:
                title = self._attr["column"]
            else:
                title = "{0}: {1}".format(self._attr["pivot"], pivot)

            self._layout.add_plot(plot_index, data_frame, title)
            plot_index += 1

        self._layout.finish()

    def _plot_concat(self):
        """Plot all lines on a single figure"""

        pivot_vals = self.c_mgr.get_all_pivots()
        plot_index = 0

        self._layout = ILinePlotGen(self._attr["per_line"],
                                    len(self._constraints),
                                    **self._attr)

        for constraint in self._constraints:
            result = constraint.result
            title = str(constraint)
            data_frame = pd.Series()

            for pivot in pivot_vals:

                if pivot in result:
                    if pivot == AttrConf.PIVOT_VAL:
                        key = self._attr["column"]
                    else:
                        key = "{0}: {1}".format(self._attr["pivot"], pivot)

                    data_frame[key] = result[pivot]

            self._layout.add_plot(plot_index, data_frame, title)
            plot_index += 1

        self._layout.finish()
