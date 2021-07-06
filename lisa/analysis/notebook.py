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

import functools
import inspect
import warnings

import holoviews as hv
import pandas as pd

import __main__ as main

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_refit_index, df_filter, SignalDesc, df_update_duplicates
from lisa.utils import kwargs_forwarded_to, order_as
from lisa.notebook import plot_signal


class _CustomProxy:
    def __init__(self, ana):
        self.ana = ana

    def __getattr__(self, attr):
        val = getattr(main, attr)

        if attr.startswith('plot_') and callable(val):
            f = val
            first_param = list(inspect.signature(f).parameters)[0]

            @TraceAnalysisBase.plot_method
            @functools.wraps(f)
            def wrapper(**kwargs):
                # We cannot capture "self" in the signature directly as we need
                # to match the name of the first parameter of f, which could be
                # anything. It's therefore simpler to manually unpack it.
                #
                # Note: The lisa.analysis.proxy._AnalysisPreset will turn all
                # parameters into kwargs.
                kwargs[first_param] = kwargs[first_param].trace
                return f(**kwargs)

            ana = self.ana
            # bind the function to the analysis instance to give a bound method
            return wrapper.__get__(ana, type(ana))
        else:
            raise AttributeError('Only functions with a name starting with "plot_" are supported')


class NotebookAnalysis(TraceAnalysisBase):
    """
    Support for custom Notebook-defined plots

    This analysis provides a proxy object that can be used to turn any plot
    method defined in a notebook into a proper analysis plot method::

        import holoviews as hv
        from lisa.trace import Trace
        trace = Trace('trace.dat', events=['sched_switch'])

        # Define a plot method in any cell:
        # The first parameter will be the trace, other parameters are free-form
        def plot_foo(trace, y):
            print(f'Plotting horizontal line at level: {y}')
            return hv.HLine(y).options(color='red')

        # The plot function becomes available on the "notebook.custom" proxy
        # object.
        trace.ana.notebook.custom.plot_foo(0.5)

    """

    name = 'notebook'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom = _CustomProxy(self)

    def __getattr__(self, attr):
        x = getattr(self.custom, attr)
        warnings.warn(
            f'Calling plot methods on trace.ana.notebook.{attr}() is deprecated, please use trace.ana.notebook.custom.{attr}() instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return x

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
            df = pd.DataFrame(
                dict.fromkeys(
                    (
                        ['info'] +
                        fields_as_cols +
                        ['event'] if event_as_col else []
                    ),
                    []
                )
            )
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

            # Reorder the columns to provide a better kernelshark-like display
            columns_order = (
                [
                    col
                    for col in df.columns
                    if col.startswith('__')
                ] +
                (['event'] if event_as_col else []) +
                ['info']
            )
            df = df[order_as(df.columns, columns_order)]

        df.attrs['name'] = 'events'
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


    @TraceAnalysisBase.plot_method
    def plot_event_field(self, event: str, field: str, filter_columns=None, filter_f=None):
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
        return plot_signal(df[field], name=field)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
