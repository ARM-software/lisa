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

"""This module contains the class for plotting and
customizing Line Plots with a pandas dataframe input
"""

import matplotlib.pyplot as plt
from trappy.plotter import AttrConf
from trappy.plotter import Utils
from trappy.plotter.Constraint import ConstraintManager
from trappy.plotter.ILinePlotGen import ILinePlotGen
from trappy.plotter.AbstractDataPlotter import AbstractDataPlotter
from trappy.plotter.ColorMap import ColorMap
from trappy.plotter import IPythonConf
import pandas as pd

if not IPythonConf.check_ipython():
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
       permute, draws one plot for each of the runs specified
    """

    def __init__(self, runs, templates=None, **kwargs):
        # Default keys, each can be overridden in kwargs
        self._attr = {}
        self.runs = runs
        self.templates = templates
        self.set_defaults()
        self._layout = None

        self._check_data()
        for key in kwargs:
            self._attr[key] = kwargs[key]

        if "column" not in self._attr:
            raise RuntimeError("Value Column not specified")

        if self._attr["drawstyle"] and self._attr["drawstyle"].startswith("steps"):
            self._attr["step_plot"] = True

        zip_constraints = not self._attr["permute"]

        self.c_mgr = ConstraintManager(runs, self._attr["column"], templates,
                                       self._attr["pivot"],
                                       self._attr["filters"], zip_constraints)

        super(ILinePlot, self).__init__()

    def savefig(self, *args, **kwargs):
        raise NotImplementedError("Not Available for ILinePlot")

    def view(self, test=False):
        """Displays the graph"""

        if self._attr["concat"]:
            self._plot_concat()
        else:
            self._plot(self._attr["permute"])

    def set_defaults(self):
        """Sets the default attrs"""
        self._attr["per_line"] = AttrConf.PER_LINE
        self._attr["concat"] = AttrConf.CONCAT
        self._attr["filters"] = {}
        self._attr["pivot"] = AttrConf.PIVOT
        self._attr["permute"] = False
        self._attr["drawstyle"] = None
        self._attr["step_plot"] = False
        self._attr["fill"] = AttrConf.FILL

    def _plot(self, permute):
        """Internal Method called to draw the plot"""
        pivot_vals, len_pivots = self.c_mgr.generate_pivots(permute)

        self._layout = ILinePlotGen(self._attr["per_line"],
                                    len_pivots,
                                    **self._attr)
        plot_index = 0
        for p_val in pivot_vals:
            data_frame = pd.Series()
            for constraint in self.c_mgr:

                if permute:
                    run_idx, pivot = p_val
                    if constraint.run_index != run_idx:
                        continue
                    title = constraint.get_data_name() + ":"
                    legend = constraint._column
                else:
                    pivot = p_val
                    title = ""
                    legend = str(constraint)

                result = constraint.result
                if pivot in result:
                    data_frame[legend] = result[pivot]

            if pivot == AttrConf.PIVOT_VAL:
                title += ",".join(self._attr["column"])
            else:
                title += "{0}: {1}".format(self._attr["pivot"], pivot)

            self._layout.add_plot(plot_index, data_frame, title)
            plot_index += 1

        self._layout.finish()

    def _plot_concat(self):
        """Plot all lines on a single figure"""

        pivot_vals, _ = self.c_mgr.generate_pivots()
        plot_index = 0

        self._layout = ILinePlotGen(self._attr["per_line"], len(self.c_mgr),
                                    **self._attr)

        for constraint in self.c_mgr:
            result = constraint.result
            title = str(constraint)
            data_frame = pd.Series()

            for pivot in pivot_vals:
                if pivot in result:
                    if pivot == AttrConf.PIVOT_VAL:
                        key = ",".join(self._attr["column"])
                    else:
                        key = "{0}: {1}".format(self._attr["pivot"], pivot)

                    data_frame[key] = result[pivot]

            self._layout.add_plot(plot_index, data_frame, title)
            plot_index += 1

        self._layout.finish()
