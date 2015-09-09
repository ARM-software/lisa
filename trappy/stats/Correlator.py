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

"""The module responsible for correlation
and related functionality
"""
from trappy.plotter.PlotLayout import PlotLayout
from trappy.stats import StatConf
from trappy.stats.Indexer import get_unified_indexer
import numpy as np
import math


class Correlator(object):
    """Class that allows to align and correlate two
    runs
    :param first: First Aggregator
    :type first: :mod:`trappy.stats.Aggregator`

    :param second: Second Aggregator
    :type second: :mod:`trappy.stats.Aggregator`
    """

    def __init__(self, first, second, **kwargs):

        self._first_agg = first
        self._second_agg = second
        self.indexer = get_unified_indexer([first.indexer, second.indexer])
        self._corrfunc = kwargs.pop("corrfunc", None)
        self._agg_kwargs = kwargs
        self.corr_graphs = {}
        self._shift = self._align_top_level()

    def _resample(self, series, delta=StatConf.DELTA_DEFAULT):
        """Internal method to resample the series
        to a uniformly spaced index

        :param series: Series io be resampled
        :type series: :mod:`pandas.Series`

        :param delta: spacing between indices
        :type delta: float

        :return: resampled :mod:`pandas.Series`
        """

        new_index = self.indexer.get_uniform(delta)
        return series.reindex(index=new_index, method="pad")

    def correlate(self, level, resample=True):
        """This function returns the correlation between two
           runs

        :param level: The level at which the correlation is
            required
        :type level: str

        :param resample: Resample data
        :type resample: bool

        :return: A normalized correlation value is returned
            for each group in the level
        """
        result_1 = self._first_agg.aggregate(level=level, **self._agg_kwargs)
        result_2 = self._second_agg.aggregate(level=level, **self._agg_kwargs)


        corr_output = []
        weights = []

        for group_id, result_group in enumerate(result_1):
            series_x = result_group
            series_y = result_2[group_id]

            if resample:
                series_x = self._resample(series_x)
                series_y = self._resample(series_y)

            series_x, series_y = shift_series(series_x, series_y, self._shift)
            corr_output.append(self._correlate(series_x, series_y))
            weights.append(len(series_x[series_x != 0]) + len(series_y[series_y != 0]))

        total = 0
        for weight, corr in zip(weights, corr_output):
            if math.isnan(corr):
                continue
            total += (weight * corr) / sum(weights)

        return corr_output, total


    def plot(self, level, per_line=3):
        """Temporary function to plot data. Expected to be
        implemented in plotter

        :param level: Topological Level (level in :mod:`trappy.stats.Topology`)
        :type level: str

        :param per_line: Number of plots per line
        :type per_line: int
        """

        num_plots = self._first_agg.topology.level_span(level)
        result_1 = self._first_agg.aggregate(level=level, **self._agg_kwargs)
        result_2 = self._second_agg.aggregate(level=level, **self._agg_kwargs)
        layout = PlotLayout(per_line, num_plots)

        plot_index = 0

        for group_id, result_group in enumerate(result_1):
            s_x = result_group
            s_y = result_2[group_id]

            s_x = self._resample(s_x)
            s_y = self._resample(s_y)

            s_x, s_y = shift_series(s_x, s_y, self._shift)

            ymax = 1.25 + max(max(s_x.values), max(s_y.values)) + 1
            ymin = min(min(s_x.values), min(s_y.values)) - 1
            ylim = [ymin, ymax]
            ylim = [-1, 3]

            axis = layout.get_axis(plot_index)

            axis.plot(s_x.index, s_x.values)
            axis.plot(s_y.index, s_y.values)

            axis.set_ylim(ylim)
            plot_index += 1
        layout.finish(plot_index)

    def _correlate(self, s_x, s_y):

        if self._corrfunc != None:
            f = self._corrfunc
            return f(s_x, s_y)
        else:
            return s_x.corr(s_y)

    def _align_top_level(self):
        """Temporary function to plot data. Expected to be
        implemented in plotter
        """

        result_1 = self._first_agg.aggregate(level="all")
        result_2 = self._second_agg.aggregate(level="all")

        s_x = self._resample(result_1[0])
        s_y = self._resample(result_2[0])


        front_x, front_y, front_shift = align(s_x, s_y, mode="front")
        front_corr = self._correlate(front_x, front_y)

        back_x, back_y, back_shift = align(s_x, s_y, mode="back")
        back_corr = self._correlate(back_x, back_y)

        if math.isnan(back_corr):
            back_corr = 0
        if math.isnan(front_corr):
            front_corr = 0

        if front_corr >= back_corr:
            return front_shift
        else:
            return back_shift



def align(s_x, s_y, mode="front"):
    """Function to align the input series

    :param s_x: First Series
    :type s_x: :mod:`pandas.Series`

    :param s_y: Second Series
    :type s_y: :mod:`pandas.Series`

    :param mode: Align Front/Back
    :type mode: str
    """

    p_x = np.flatnonzero(s_x)
    p_y = np.flatnonzero(s_y)

    if not len(p_x) or not len(p_y):
        return s_x, s_y, 0

    if mode == "front":
        p_x = p_x[0]
        p_y = p_y[0]

    if mode == "back":
        p_x = p_x[-1]
        p_y = p_y[-1]

    shift = p_x - p_y

    s_x, s_y = shift_series(s_x, s_y, shift)
    return s_x, s_y, shift

def shift_series(s_x, s_y, shift):
    """Shift series to align
    :param s_x: First Series
    :type s_x: :mod:`pandas.Series`

    :param s_y: Second Series
    :type s_y: :mod:`pandas.Series`

    :param shift: The number of index
        positions to be shifted
    :type shift: int
    """

    if shift > 0:
        s_y = s_y.shift(shift)
    else:
        s_x = s_x.shift(-1 * shift)

    return s_x, s_y
