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
"""This module implements functionality related to
the arrangement of the plots on the underlying
plotting backend.
"""

import matplotlib.pyplot as plt
from trappy.plotter import AttrConf


class PlotLayout(object):

    """
    :param cols: The number of columns to draw
    :type cols: int

    :param num_plots: The total number of plots
    :type num_plots: int

    The linear co-ordinate system :math:`[0, N_{plots}]` is
    mapped to a 2-D coordinate system with math:`N_{rows}`
    and :math:`N_{cols}` such that:

    .. math::

        N_{rows} = \\frac{N_{cols}}{N_{plots}}
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
            self._single_plot = True
            self._scale_plot()
        elif self.rows == 1:
            self.usecol = True
        elif self.cols == 1:
            self.userow = True
            self._scale_plot()

        self._attr["figure"], self._attr["axes"] = plt.subplots(
            self.rows, self.cols, figsize=(
                self._attr["width"] * self.cols,
                self._attr["length"] * self.rows))

    def _scale_plot(self):
        """Scale the graph in one
           plot per line use case"""

        self._attr["width"] = int(self._attr["width"] * 2.5)
        self._attr["length"] = int(self._attr["length"] * 1.25)

    def _set_defaults(self):
        """set the default attrs"""
        self._attr["width"] = AttrConf.WIDTH
        self._attr["length"] = AttrConf.LENGTH

    def get_2d(self, linear_val):
        """Convert Linear to 2D coordinates

        :param linear_val: The value in 1-D
            co-ordinate
        :type linear_val: int

        :return: Converted 2-D tuple
        """
        if self.usecol:
            return linear_val % self.cols

        if self.userow:
            return linear_val % self.rows

        val_x = linear_val % self.cols
        val_y = linear_val / self.cols
        return val_y, val_x

    def finish(self, plot_index):
        """Delete the empty cells

        :param plot_index: Linear index at which the
            last plot was created. This is used to
            delete the leftover empty plots that
            were generated.
        :type plot_index: int
        """
        while plot_index < (self.rows * self.cols):
            self._attr["figure"].delaxes(
                self._attr["axes"][
                    self.get_2d(plot_index)])
            plot_index += 1

    def get_axis(self, plot_index):
        """Get the axes for the plots

        :param plot_index: The index for
            which the axis is required. This
            internally is mapped to a 2-D co-ordinate

        :return: :mod:`matplotlib.axes.Axes`
            instance is returned
        """
        if self._single_plot:
            return self._attr["axes"]
        else:
            return self._attr["axes"][self.get_2d(plot_index)]

    def get_fig(self):
        """Return the matplotlib figure object

        :return: :mod:`matplotlib.figure`
        """
        return self._attr["figure"]
