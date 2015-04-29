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
# File:        SchedConf.py
# ----------------------------------------------------------------
# $
#
"""Scheduler specific Functionality for the
stats framework
"""

WINDOW_SIZE = 0.0001

def window_filt_cum_sum(series):
    """The following actions are done on the
    input series

        * A sched_out of duration < WindowSize
          is filtered out
        * The resultant series is summed cumulatively
          and returned

        Args:
            series (pandas.Series)
        Returns:
            pandas.Series
    """

    prev = None
    for index, value in series.iteritems():
        if value != 0:
            if prev is None:
                prev = index
                continue

            if index - prev < WINDOW_SIZE:
                if series[prev] == -1:
                    series[prev] = 0
                    series[index] = 0
                    prev = None
            else:
                prev = index


    return series.cumsum()

def residency_sum(series):
    """The input series is processed for
    intervals between a 1 and -1 in order
    to track additive residency of a task


        Args:
            series (pandas.Series)
        Returns:
            float (scalar)
    """

    duration = 0
    start = None
    for index, value in series.iteritems():
        if value == 1:
            start = index

        if value == -1:
            if start is not None:
                duration += index - start

            start = None


    return duration

def binary_correlate(series_x, series_y):
    """Function to Correlate binary Data"""

    if len(series_x) != len(series_y):
        raise ValueError("Cannot compute binary correlation for \
                          unequal vectors")

    agree = len(series_x[series_x == series_y])
    disagree = len(series_x[series_x != series_y])

    return (agree - disagree) / float(len(series_x))


