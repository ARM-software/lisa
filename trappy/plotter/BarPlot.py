#    Copyright 2016-2016 ARM Limited
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
"""
This class sublclasses :mod:`trappy.plotter.StaticPlot.StaticPlot` to
implement a bar plot.

"""
import numpy as np
from trappy.plotter.StaticPlot import StaticPlot

class BarPlot(StaticPlot):
    """BarPlot can plot data as vertical bars

    Values are plotted against their position in the list of data.

    :param traces: The input data
    :type traces: A single instance or a list of :mod:`trappy.trace.FTrace`,
        :mod:`trappy.trace.SysTrace`, :mod:`trappy.trace.BareTrace` or
        :mod:`pandas.DataFrame`.

    :param column: specifies the name of the column to be plotted.
    :type column: str or list(str)

    :param templates: TRAPpy events

        .. note::

                This is not required if a :mod:`pandas.DataFrame` is
                used

    :type templates: :mod:`trappy.base.Base`

    :param signals: A string of the type event_name:column
        to indicate the value that needs to be plotted

        .. note::

            - Only one of `signals` or both `templates` and
              `columns` should be specified
            - Signals format won't work for :mod:`pandas.DataFrame`
              input

    :type signals: str or list(string)

    :param title: A title describing the generated plots
    :type title: str

    :param stacked: The series are grouped by default.  If you want a
        stacked plot, set stacked to True.
    :type stacked: bool

    :param spacing: A proportion of the size of each group which
        should be used as the spacing between the groups. e.g. 0.2
        (default) means that 1/5 of the groups total width is used as
        a spacing between groups.
    :type spacing: float
    """

    def __init__(self, traces, templates=None, **kwargs):
        # Default keys, each can be overridden in kwargs

        super(BarPlot, self).__init__(
            traces=traces,
            templates=templates,
            **kwargs)

    def set_defaults(self):
        """Sets the default attrs"""
        super(BarPlot, self).set_defaults()
        self._attr["spacing"] = 0.2
        self._attr["stacked"] = False

    def plot_axis(self, axis, series_list, permute, concat, args_to_forward):
        """Internal Method called to plot data (series_list) on a given axis"""
        stacked = self._attr["stacked"]
        #Figure out how many bars per group
        bars_in_group = 1 if stacked else len(series_list)

        #Get the width of a group
        group_width = 1.0 - self._attr["spacing"]
        bar_width = group_width / bars_in_group

        #Keep a list of the tops of bars to plot stacks
        #Start with a list of 0s to put the first bars at the bottom
        value_list = [c.result[p].values for (c, p) in series_list]
        end_of_previous = [0] * max(len(x) for x in value_list)

        for i, (constraint, pivot) in enumerate(series_list):
            result = constraint.result
            bar_anchor = np.arange(len(result[pivot].values))
            if not stacked:
                bar_anchor = bar_anchor + i * bar_width

            line_2d_list = axis.bar(
                bar_anchor,
                result[pivot].values,
                bottom=end_of_previous,
                width=bar_width,
                color=self._cmap.cmap(i),
                **args_to_forward
            )

            if stacked:
                end_of_previous = [x + y for (x, y) in zip(end_of_previous, result[pivot].values)]

            axis.set_title(self.make_title(constraint, pivot, permute, concat))

            self.add_to_legend(i, line_2d_list[0], constraint, pivot, concat, permute)
