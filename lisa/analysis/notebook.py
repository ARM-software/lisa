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

import pandas as pd
import functools

import __main__ as main

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_refit_index, df_filter, SignalDesc, df_update_duplicates
from lisa.utils import kwargs_forwarded_to


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
            print(f'Plotting horizontal line at level: {y}')
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

    @TraceAnalysisBase.cache
    def _df_all_events(self, events, field_sep=' ', fields_as_cols=None, event_as_col=True):
        """
        Split implementation to be able to use the cache
        """
        if fields_as_cols is None:
            fields_as_cols = ['__comm', '__pid', '__cpu']
        else:
            fields_as_cols = list(fields_as_cols)

        trace = self.trace

        if not events:
            return pd.DataFrame({'info': []})
        else:
            if event_as_col:
                fmt = '{fields}'
            else:
                fmt = '{{event:<{max_len}}}: {{fields}}'.format(
                    max_len=max(len(event) for event in events)
                )

            fields_as_cols_set = set(fields_as_cols)
            def make_info_row(row, event):
                fields = field_sep.join(
                    f'{key}={value}'
                    for key, value in row.iteritems()
                    if key not in fields_as_cols_set
                )
                return fmt.format(
                    event=event,
                    fields=fields,
                )

            def make_info_df(event):
                df = trace.df_event(event)
                df = pd.DataFrame(
                    {
                        'info': df.apply(make_info_row, axis=1, event=event),
                        **{
                            field: df[field]
                            for field in fields_as_cols
                        }
                    },
                    index=df.index,
                )

                if event_as_col:
                    df['event'] = event
                return df

            df = pd.concat(map(make_info_df, events) )
            df.sort_index(inplace=True)
            df_update_duplicates(df, inplace=True)
            return df

    @kwargs_forwarded_to(_df_all_events)
    def df_all_events(self, events=None, **kwargs):
        """
        Provide a dataframe with an ``info`` column containing the textual
        human-readable representation of the events fields.

        :param events: List of events to include. If ``None``, all parsed
            events will be used.

            .. note:: Since events can be parsed on-demand, passing ``None``
                might result in different results depending on what was done
                with the object. For reproducible behaviour, pass an explicit
                list of events.
        :type events: list(str) or None

        :param field_sep: String to use to separate fields.
        :type field_sep: str

        :param fields_as_cols: List of fields to keep as separate columns rather than
            merged in the ``info`` column. If ``None``, will default to a fixed
            set of columns.
        :type fields_as_cols: list(str) or None

        :param event_as_col: If ``True``, the event name is split in its own
            column.
        :type event_as_col: bool
        """
        if events is None:
            events = sorted(self.trace.available_events)

        return self._df_all_events(events=events, **kwargs)


    @TraceAnalysisBase.plot_method(return_axis=False)
    def plot_event_field(self, event: str, field: str, axis, local_fig, filter_columns=None, filter_f=None):
        """
        Plot a signal represented by the filtered values of a field of an event.

        :param event: FTrace event name of interest.
        :type event: str

        :param field: Name of the field of ``event``.
        :type field: str

        :param filter_columns: Pre-filter the dataframe using
            :func:`lisa.datautils.df_filter`. Also, a signal will be inferred
            from the column names being used and will be passed to
            :meth:`lisa.trace.Trace.df_event`.
        :type filter_columns: dict or None

        :param filter_f: Function used to filter the dataframe of the event.
            The function must take a dataframe as only parameter and return
            a filtered dataframe. It is applied after ``filter_columns`` filter.
        :type filter_f: collections.abc.Callable
        """
        trace = self.trace
        if filter_columns:
            signals = [SignalDesc(event, sorted(filter_columns.keys()))]
        else:
            signals = None

        df = trace.df_event(event, signals=signals)

        if filter_columns:
            df = df_filter(df, filter_columns)

        if filter_f:
            df = filter_f(df)

        df = df_refit_index(df, window=trace.window)
        df[[field]].plot(ax=axis, drawstyle='steps-post')


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
