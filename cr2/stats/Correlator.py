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
# File:        Correlator.py
# ----------------------------------------------------------------
# $
#
"""The module responsible for correlation
and related functionality
"""
from cr2.plotter.PlotLayout import PlotLayout
from cr2.stats import StatConf
from cr2.stats.Indexer import get_unified_indexer
import numpy as np


class Correlator(object):
    """Class that allows to align and correlate two
       runs
    """

    def __init__(self, first, second):
        """
            Args:
                first (stat.Aggregator): First Aggregator
                second (stat.Aggregator): Second Aggregator
        """

        self._first_agg = first
        self._second_agg = second
        self.indexer = get_unified_indexer([first.indexer, second.indexer])
        self.corr_graphs = {}

    def _resample(self, series, delta=StatConf.DELTA_DEFAULT):
        """Internal method to resample the series
        to a uniformally spaces index

        Args:
            series (pandas.Series): Series io be resampled
            delta  (float): spacing between indices

        Returns:
            resampled (pandas.Series)
        """

        new_index = self.indexer.get_uniform(delta)
        return series.reindex(index=new_index, method="pad")

    def correlate(self, level, resample=True):
        """This function returns the correlation between two
           runs

            Args:
                level: The level at which the correlation is
                    required

            Returns:
                A normalized correlation value is returned
                for each group in the level

        """
        result_1 = self._first_agg.aggregate(level=level)
        result_2 = self._second_agg.aggregate(level=level)
        corr_output = []

        for group_id, result_group in enumerate(result_1):
            series_x = result_group
            series_y = result_2[group_id]

            if resample:
                series_x = self._resample(series_x)
                series_y = self._resample(series_y)

            front_x, front_y = align(series_x, series_y, mode="front")
            front_corr = front_x.corr(front_y)
            back_x, back_y = align(series_x, series_y, mode="back")
            back_corr = back_x.corr(back_y)
            corr_output.append(max(back_corr, front_corr))

        return corr_output


    def plot(self, level, per_line=3):
        """Temporary function to plot data. Expected to be
        implemented in plotter
        """

        num_plots = self._first_agg.topology.level_span(level)
        result_1 = self._first_agg.aggregate(level=level)
        result_2 = self._second_agg.aggregate(level=level)
        layout = PlotLayout(per_line, num_plots)

        plot_index = 0

        for group_id, result_group in enumerate(result_1):
            s_x = result_group
            s_y = result_2[group_id]

            s_x = self._resample(s_x)
            s_y = self._resample(s_y)

            ymax = 1.25 + max(max(s_x.values), max(s_y.values)) + 1
            ymin = min(min(s_x.values), min(s_y.values)) - 1
            ylim = [ymin, ymax]

            axis = layout.get_axis(plot_index)
            front_x, front_y = align(s_x, s_y, mode="front")
            front_corr = front_x.corr(front_y)
            back_x, back_y = align(s_x, s_y, mode="back")
            back_corr = back_x.corr(back_y)

            if front_corr > back_corr:
                axis.plot(front_x.index, front_x.values)
                axis.plot(front_y.index, front_y.values)
            else:
                axis.plot(back_x.index, back_x.values)
                axis.plot(back_y.index, back_y.values)

            axis.set_ylim(ylim)
            plot_index += 1
        layout.finish(plot_index)


def align(s_x, s_y, mode="front"):
    """Function to align the input series"""

    p_x = np.flatnonzero(s_x)
    p_y = np.flatnonzero(s_y)

    if not len(p_x) or not len(p_y):
        return s_x, s_y

    if mode == "front":
        p_x = p_x[0]
        p_y = p_y[0]

    if mode == "back":
        p_x = p_x[-1]
        p_y = p_y[-1]

    if p_x > p_y:
        s_y = s_y.shift(p_x - p_y)
    else:
        s_x = s_x.shift(p_y - p_x)

    return s_x, s_y
