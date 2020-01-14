# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Notebook Analysis Module """

import sys
import pandas as pd
import functools
import operator

import __main__ as main

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_refit_index, df_filter


class NotebookAnalysis(TraceAnalysisBase):
    """
    Support for custom Notebook-defined plots

    Attribute lookup will be resolved in ``__main__`` module, which contains
    all names created in cells of Jupyter notebooks.

    Functions named ``plot_*`` have a special behavior: they are expected to
    take a :class:`lisa.trace.Trace` as first parameter and a named parameter
    :class:`matplotlib.axes.Axes` ``axis`` parameter to plot on.

    example::

        from lisa.trace import Trace
        trace = Trace('trace.dat', events=['sched_switch'])

        # Define a plot method in any cell
        def plot_foo(trace, y, axis):
            print('Plotting horizontal line at level: {}'.format(y))
            axis.axhline(y=y)

        # Just lookup the plot function
        trace.analysis.notebook.plot_foo(3)

    """

    name = 'notebook'

    def __getattr__(self, attr):
        val = getattr(main, attr)

        if attr.startswith('plot_'):
            f = val
            # swallow "local_fig" as it is usually not needed for the notebook
            # usage and pass the trace directly instead of the analysis
            @TraceAnalysisBase.plot_method(return_axis=False)
            @functools.wraps(f)
            def wrapper(self, *args, local_fig, **kwargs):
                return f(self.trace, *args, **kwargs)

            val = wrapper

        if callable(val):
            # bind the function to the analysis instance to give a bound method
            return val.__get__(self, type(self))
        else:
            return val

    @TraceAnalysisBase.plot_method(return_axis=False)
    def plot_event_field(self, event, field, axis, local_fig, filter_columns=None, filter_f=None):
        """
        Plot a signal represented by the filtered values of a field of an event.

        :param event: FTrace event name of interest.
        :type event: str

        :param field: Name of the field of ``event``.
        :type field: str

        :param filter_columns: Pre-filter the dataframe using
            :func:`lisa.datautils.df_filter`
        :type filter_columns: dict or None

        :param filter_f: Function used to filter the dataframe of the event.
            The function must take a dataframe as only parameter and return
            a filtered dataframe. It is applied after ``filter_columns`` filter.
        :type filter_f: collections.abc.Callable
        """
        trace = self.trace
        df = trace.df_events(event)

        if filter_columns:
            df = df_filter(df, filter_columns)

        if filter_f:
            df = filter_f(df)

        df = df_refit_index(df, trace.start, trace.end)
        df[[field]].plot(ax=axis, drawstyle='steps-post')


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
