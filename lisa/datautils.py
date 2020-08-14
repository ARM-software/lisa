# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

import re
import functools
import operator
import math
import itertools
import warnings
import contextlib
from operator import attrgetter

import numpy as np
import pandas as pd
import pandas.api.extensions
import scipy.integrate
import scipy.signal

from lisa.utils import TASK_COMM_MAX_LEN, groupby, deprecate


class DataAccessor:
    """
    Proxy class that allows extending the :class:`pandas.DataFrame` API.

    **Example**::

        # Define and register a dataframe accessor
        @DataFrameAccessor.register_accessor
        def df_foobar(df, baz):
            ...

        df = pandas.DataFrame()
        # Use the accessor with the "lisa" proxy
        df.lisa.foobar(baz=1)
    """
    def __init__(self, data):
        self.data = data

    @classmethod
    def register_accessor(cls, f):
        """
        Decorator to register an accessor function.

        The accessor name will be the name of the function, without the
        ``series_`` or ``df_`` prefix.
        """
        name = re.sub(r'^(?:df|series)_(.*)', r'\1', f.__name__)
        cls.FUNCTIONS[name] = f
        return f

    def __getattr__(self, attr):
        try:
            f = self.FUNCTIONS[attr]
        except KeyError:
            raise AttributeError('Unknown method name: {}'.format(attr))

        meth = f.__get__(self.data, self.__class__)
        return meth

    def __dir__(self):
        attrs = set(super().__dir__())
        attrs |= self.FUNCTIONS.keys()
        return sorted(attrs)


@pandas.api.extensions.register_dataframe_accessor('lisa')
class DataFrameAccessor(DataAccessor):
    FUNCTIONS = {}


@pandas.api.extensions.register_series_accessor('lisa')
class SeriesAccessor(DataAccessor):
    FUNCTIONS = {}


@SeriesAccessor.register_accessor
def series_refit_index(series, start=None, end=None, window=None, method='inclusive', clip_window=True):
    """
    Slice a series using :func:`series_window` and ensure we have a value at
    exactly the specified boundaries, unless the signal started after the
    beginning of the required window.

    :param df: Series to act on
    :type df: pandas.Series

    :param start: First index value to find in the returned series.
    :type start: object

    :param end: Last index value to find in the returned series.
    :type end: object

    :param window: ``window=(start, end)`` is the same as
        ``start=start, end=end``. These parameters styles are mutually
        exclusive.
    :type window: tuple(float or None, float or None) or None

    :param method: Windowing method used to select the first and last values of
        the series using :func:`series_window`. Defaults to ``pre``, which is
        suitable for signals where all the value changes have a corresponding
        row without any fixed sample-rate constraints. If they have been
        downsampled, ``nearest`` might be a better choice.).

    .. note:: If ``end`` is past the end of the data, the last row will
        be duplicated so that we can have a start and end index at the right
        location, without moving the point at which the transition to the last
        value happened. This also allows plotting series with only one item
        using matplotlib, which would otherwise be impossible.

    :param clip_window: Passed down to :func:`series_refit_index`.
    """
    window = _make_window(start, end, window)
    return _data_refit_index(series, window, method=method, clip_window=clip_window)


@DataFrameAccessor.register_accessor
def df_refit_index(df, start=None, end=None, window=None, method='inclusive', clip_window=True):
    """
    Same as :func:`series_refit_index` but acting on :class:`pandas.DataFrame`
    """
    window = _make_window(start, end, window)
    return _data_refit_index(df, window, method=method, clip_window=clip_window)

def _make_window(start, end, window):
    uses_separated = (start, end) != (None, None)

    if uses_separated:
        warnings.warn('start and end df_refit_index() parameters are deprecated, please use window=', DeprecationWarning, stacklevel=3)

    if window is not None and uses_separated:
        raise ValueError('window != None cannot be used along with start and end parameters')

    if window is None:
        return (start, end)
    else:
        return window


@DataFrameAccessor.register_accessor
def df_split_signals(df, signal_cols, align_start=False, window=None):
    """
    Yield subset of ``df`` that only contain one signal, along with the signal
    identification values.

    :param df: The dataframe to split.
    :type df: pandas.DataFrame

    :param signal_cols: Columns that uniquely identify a signal.
    :type signal_cols: list(str)

    :param window: Apply :func:`df_refit_index` on the yielded dataframes with
        the given window.
    :type window: tuple(float or None, float or None) or None

    :param align_start: If ``True``, same as ``window=(df.index[0], None)``.
        This makes sure all yielded signals start at the same index as the
        original dataframe.
    :type align_start: bool
    """
    if not signal_cols:
        yield ({}, df)
    else:
        if align_start:
            if window is not None:
                raise ValueError('align_start=True cannot be used with window != None')
            window = (df.index[0], None)

        for group, signal in df.groupby(signal_cols, observed=True, sort=False):
            # When only one column is looked at, the group is the value instead of
            # a tuple of values
            if len(signal_cols) < 2:
                cols_val = {signal_cols[0]: group}
            else:
                cols_val = dict(zip(signal_cols, group))

            if window:
                signal = df_refit_index(signal, window=window, method='inclusive')
            yield (cols_val, signal)


def _data_refit_index(data, window, method, clip_window):
    if data.empty:
        raise ValueError('Cannot refit the index of an empty dataframe or series')

    start, end = window
    duplicate_last = end > data.index[-1]
    data = _data_window(data, window, method=method, clip_window=clip_window)

    if data.empty:
        return data

    # When the end is after the end of the data, duplicate the last row so we
    # can push it to the right as much as we want without changing the point at
    # which the transition to that value happened
    if duplicate_last:
        data = data.append(data.iloc[-1:])
    else:
        # Shallow copy is enough, we only want to replace the index and not the
        # actual data
        data = data.copy(deep=False)

    index = data.index.to_series()

    # Only advance the beginning of the data, never move it in the past.
    # Otherwise, we "invent" a value for the signal that did not existed,
    # leading to various wrong results.
    if start is not None and index.iloc[0] < start:
        index.iloc[0] = start

    if end is not None:
        index.iloc[-1] = end

    data.index = index
    return data


@DataFrameAccessor.register_accessor
def df_squash(df, start, end, column='delta'):
    """
    Slice a dataframe of deltas in [start:end] and ensure we have
    an event at exactly those boundaries.

    The input dataframe is expected to have a "column" which reports
    the time delta between consecutive rows, as for example dataframes
    generated by :func:`df_add_delta`.

    The returned dataframe is granted to have an initial and final
    event at the specified "start" ("end") index values, which values
    are the same of the last event before (first event after) the
    specified "start" ("end") time.

    Examples:

    Slice a dataframe to [start:end], and work on the time data so that it
    makes sense within the interval.

    Examples to make it clearer::

        df is:
        Time len state
        15    1   1
        16    1   0
        17    1   1
        18    1   0
        -------------

        df_squash(df, 16.5, 17.5) =>

        Time len state
        16.5  .5   0
        17    .5   1

        df_squash(df, 16.2, 16.8) =>

        Time len state
        16.2  .6   0

    :returns: a new df that fits the above description
    """
    if df.empty:
        return df

    end = min(end, df.index[-1] + df[column].iloc[-1])
    res_df = pd.DataFrame(data=[], columns=df.columns)

    if start > end:
        return res_df

    # There's a few things to keep in mind here, and it gets confusing
    # even for the people who wrote the code. Let's write it down.
    #
    # It's assumed that the data is continuous, i.e. for any row 'r' within
    # the trace interval, we will find a new row at (r.index + r.len)
    # For us this means we'll never end up with an empty dataframe
    # (if we started with a non empty one)
    #
    # What's we're manipulating looks like this:
    # (| = events; [ & ] = start,end slice)
    #
    # |   [   |   ]   |
    # e0  s0  e1  s1  e2
    #
    # We need to push e0 within the interval, and then tweak its duration
    # (len column). The mathemagical incantation for that is:
    # e0.len = min(e1.index - s0, s1 - s0)
    #
    # This takes care of the case where s1 isn't in the interval
    # If s1 is in the interval, we just need to cap its len to
    # s1 - e1.index

    prev_df = df[:start]
    middle_df = df[start:end]

    # Tweak the closest previous event to include it in the slice
    if not prev_df.empty and not (start in middle_df.index):
        res_df = res_df.append(prev_df.tail(1))
        res_df.index = [start]
        e1 = end

        if not middle_df.empty:
            e1 = middle_df.index[0]

        res_df[column] = min(e1 - start, end - start)

    if not middle_df.empty:
        res_df = res_df.append(middle_df)
        if end in res_df.index:
            # e_last and s1 collide, ditch e_last
            res_df = res_df.drop([end])
        else:
            # Fix the delta for the last row
            delta = min(end - res_df.index[-1], res_df[column].iloc[-1])
            res_df.at[res_df.index[-1], column] = delta

    return res_df


@DataFrameAccessor.register_accessor
def df_filter(df, filter_columns):
    """
    Filter the content of a dataframe.

    :param df: DataFrame to filter
    :type df: pandas.DataFrame

    :param filter_columns: Dict of `{"column": value)` that rows has to match
        to be selected.
    :type filter_columns: dict(str, object)
    """
    key = functools.reduce(
        operator.and_,
        (
            df[col] == val
            for col, val in filter_columns.items()
        )
    )

    return df[key]


def df_merge(df_list, drop_columns=None, drop_inplace=False, filter_columns=None):
    """
    Merge a list of :class:`pandas.DataFrame`, keeping the index sorted.

    :param drop_columns: List of columns to drop prior to merging. This avoids
        ending up with extra renamed columns if some dataframes have column
        names in common.
    :type drop_columns: list(str)

    :param drop_inplace: Drop columns in the original dataframes instead of
        creating copies.
    :type drop_inplace: bool

    :param filter_columns: Dict of `{"column": value)` used to filter each
        dataframe prior to dropping columns. The columns are then dropped as
        they have a constant value.
    :type filter_columns: dict(str, object)
    """

    drop_columns = drop_columns if drop_columns else []

    if filter_columns:
        df_list = [
            df_filter(df, filter_columns)
            for df in df_list
        ]

        # remove the column to avoid duplicated useless columns
        drop_columns.extend(filter_columns.keys())
        # Since we just created dataframe slices, drop_inplace would give a
        # warning from pandas
        drop_inplace = False

    if drop_columns:
        def drop(df):
            filtered_df = df.drop(columns=drop_columns, inplace=drop_inplace)
            # when inplace=True, df.drop() returns None
            return df if drop_inplace else filtered_df

        df_list = [
            drop(df)
            for df in df_list
        ]

    def merge(df1, df2):
        return pd.merge(df1, df2, left_index=True, right_index=True, how='outer')

    return functools.reduce(merge, df_list)


def _resolve_x(y, x):
    """
    Resolve the `x` series to use for derivative and integral operations
    """

    if x is None:
        x = pd.Series(y.index)
        x.index = y.index
    return x


@SeriesAccessor.register_accessor
def series_derivate(y, x=None, order=1):
    """
    Compute a derivative of a :class:`pandas.Series` with respect to another
    series.

    :return: A series of `dy/dx`, where `x` is either the index of `y` or
        another series.

    :param y: Series with the data to derivate.
    :type y: pandas.DataFrame

    :param x: Series with the `x` data. If ``None``, the index of `y` will be
        used. Note that `y` and `x` are expected to have the same index.
    :type y: pandas.DataFrame or None

    :param order: Order of the derivative (1 is speed, 2 is acceleration etc).
    :type order: int
    """
    x = _resolve_x(y, x)

    for i in range(order):
        y = y.diff() / x.diff()

    return y


@SeriesAccessor.register_accessor
def series_integrate(y, x=None, sign=None, method='rect', rect_step='post'):
    """
    Compute the integral of `y` with respect to `x`.

    :return: A scalar :math:`\\int_{x=A}^{x=B} y \\, dx`, where `x` is either the
        index of `y` or another series.

    :param y: Series with the data to integrate.
    :type y: pandas.DataFrame

    :param x: Series with the `x` data. If ``None``, the index of `y` will be
        used. Note that `y` and `x` are expected to have the same index.
    :type y: pandas.DataFrame or None

    :param sign: Clip the data for the area in positive
        or negative regions. Can be any of:

        - ``+``: ignore negative data
        - ``-``: ignore positive data
        - ``None``: use all data

    :type sign: str or None

    :param method: The method for area calculation. This can
        be any of the integration methods supported in :mod:`numpy`
        or `rect`
    :type param: str

    :param rect_step: The step behaviour for `rect` method
    :type rect_step: str

    *Rectangular Method*

        - Step: Post

            Consider the following time series data::

                2            *----*----*----+
                             |              |
                1            |              *----*----+
                             |
                0  *----*----+
                   0    1    2    3    4    5    6    7

                import pandas as pd
                a = [0, 0, 2, 2, 2, 1, 1]
                s = pd.Series(a)

            The area under the curve is:

            .. math::

                \\sum_{k=0}^{N-1} (x_{k+1} - {x_k}) \\times f(x_k) \\\\
                (2 \\times 3) + (1 \\times 2) = 8

        - Step: Pre

            ::

                2       +----*----*----*
                        |              |
                1       |              +----*----*----+
                        |
                0  *----*
                   0    1    2    3    4    5    6    7

                import pandas as pd
                a = [0, 0, 2, 2, 2, 1, 1]
                s = pd.Series(a)

            The area under the curve is:

            .. math::

                \\sum_{k=1}^{N} (x_k - x_{k-1}) \\times f(x_k) \\\\
                (2 \\times 3) + (1 \\times 3) = 9
    """

    x = _resolve_x(y, x)

    if sign == "+":
        y = y.clip(lower=0)
    elif sign == "-":
        y = y.clip(upper=0)
    elif sign is None:
        pass
    else:
        raise ValueError('Unsupported "sign": {}'.format(sign))

    if method == "rect":
        dx = x.diff()

        if rect_step == "post":
            dx = dx.shift(-1)

        return (y * dx).sum()

    # Make a DataFrame to make sure all rows stay aligned when we drop NaN,
    # which is needed by all the below methods
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x = df['x']
    y = df['y']

    if method == 'trapz':
        return np.trapz(y, x)

    elif method == 'simps':
        return scipy.integrate.simps(y, x)

    else:
        raise ValueError('Unsupported integration method: {}'.format(method))


@SeriesAccessor.register_accessor
def series_mean(y, x=None, **kwargs):
    r"""
    Compute the average of `y` by integrating with respect to `x` and dividing
    by the range of `x`.

    :return: A scalar :math:`\int_{x=A}^{x=B} \frac{y}{| B - A |} \, dx`,
        where `x` is either the index of `y` or another series.

    :param y: Series with the data to integrate.
    :type y: pandas.DataFrame

    :param x: Series with the `x` data. If ``None``, the index of `y` will be
        used. Note that `y` and `x` are expected to have the same index.
    :type y: pandas.DataFrame or None

    :Variable keyword arguments: Forwarded to :func:`series_integrate`.
    """
    x = _resolve_x(y, x)
    integral = series_integrate(y, x, **kwargs)

    if len(y) > 1:
        mean = integral / (x.max() - x.min())
    # If there is only one data item, the mean is equal to it.
    else:
        mean = integral

    return mean


@SeriesAccessor.register_accessor
def series_window(series, window, method='pre', clip_window=True):
    """
    Select a portion of a :class:`pandas.Series`

    :param series: series to slice
    :type series: :class:`pandas.Series`

    :param window: two-tuple of index values for the start and end of the
        region to select.
    :type window: tuple(object)

    :param clip_window: Clip the requested window to the bounds of the index,
        otherwise raise exceptions if the window is too large.
    :type clip_window: bool

    :param method: Choose how edges are handled:

        * `inclusive`: When no exact match is found, include both the previous
            and next values around the window.
        * `exclusive`: When no exact match is found, only index values within
            the range are selected. This is the default pandas float slicing
            behavior.
        * `nearest`: When no exact match is found, take the nearest index value.
        * `pre`: When no exact match is found, take the previous index value.
        * `post`: When no exact match is found, take the next index value.

    .. note:: The index of `series` must be monotonic and without duplicates.
    """
    return _data_window(series, window, method, clip_window)


def _data_window(data, window, method, clip_window):
    """
    ``data`` can either be a :class:`pandas.DataFrame` or :class:`pandas.Series`

    .. warning:: This function assumes ``data`` has a sorted index.
    """

    index = data.index
    if clip_window:
        if data.empty:
            return data

        start, end = window
        first = index[0]
        last = index[-1]

        # Fill placeholders
        if start is None:
            start = first
        if end is None:
            end = last

        # Window is on the left
        if start <= first and end <= first:
            start = first
            end = first
        # Window is on the rigth
        elif start >= last and end >= last:
            start = last
            end = last
        # Overlapping window
        else:
            if start <= first:
                start = first

            if end >= last:
                end = last

        window = (start, end)

    if window[0] > window[1]:
        raise KeyError('The window starts after its end: {}'.format(window))

    if method == 'inclusive':
        method = ('ffill', 'bfill')

    elif method == 'exclusive':
        # Default slicing behaviour of pandas' Float64Index is to be exclusive,
        # so we can use that knowledge to enable a fast path.
        if isinstance(data.index, pd.Float64Index):
            return data[slice(*window)]

        method = ('bfill', 'ffill')

    elif method == 'nearest':
        method = ('nearest', 'nearest')

    elif method == 'pre':
        method = ('ffill', 'ffill')

    elif method == 'post':
        method = ('bfill', 'bfill')

    else:
        raise ValueError('Slicing method not supported: {}'.format(method))

    window = [
        _get_loc(index, x, method=method) if x is not None else None
        for x, method in zip(window, method)
    ]
    window = window[0], (window[1] + 1)

    return data.iloc[slice(*window)]


def _get_loc(index, x, method):
    """
    Emulate :func:`pandas.Index.get_loc` behavior with the much faster
    :func:`pandas.Index.searchsorted`.

    .. warning:: Passing a non-sorted index will destroy performance.
    """

    # Not a lot of use for nearest, so fall back on the slow but easy to use get_loc()
    #
    # Also, if the index is not sorted, we need to fall back on the slow path
    # as well. Checking is_monotonic is cheap so it's ok to do it here.
    if method == 'nearest' or not index.is_monotonic:
        return index.get_loc(x, method=method)
    else:
        if index.empty:
            raise KeyError(x)
        # get_loc() also raises an exception in these case
        elif method == 'ffill' and x < index[0]:
            raise KeyError(x)
        elif method == 'bfill' and x > index[-1]:
            raise KeyError(x)

        loc = index.searchsorted(x)
        try:
            val_at_loc = index[loc]
        # We are getting an index past the end. This is fine since we already
        # checked correct bounds before
        except IndexError:
            return loc - 1

        if val_at_loc == x:
            return loc
        elif val_at_loc < x:
            return loc if method == 'ffill' else loc + 1
        else:
            return loc - 1 if method == 'ffill' else loc


@DataFrameAccessor.register_accessor
def df_window(df, window, method='pre', clip_window=True):
    """
    Same as :func:`series_window` but acting on a :class:`pandas.DataFrame`
    """
    return _data_window(df, window, method, clip_window)


@DataFrameAccessor.register_accessor
def df_make_empty_clone(df):
    """
    Make an empty clone of the given dataframe.

    :param df: The template dataframe.
    :type df: pandas.DataFrame

    More specifically, the following aspects are cloned:

        * Column names
        * Column dtypes
    """
    return df.iloc[0:0].copy(deep=True)


@DataFrameAccessor.register_accessor
def df_window_signals(df, window, signals, compress_init=False, clip_window=True):
    """
    Similar to :func:`df_window` with ``method='pre'`` but guarantees that each
    signal will have a values at the beginning of the window.

    :param window: two-tuple of index values for the start and end of the
        region to select.
    :type window: tuple(object)

    :param signals: List of :class:`SignalDesc` describing the signals to
        fixup.
    :type signals: list(SignalDesc)

    :param compress_init: When ``False``, the timestamps of the init value of
        signals (right before the window) are preserved. If ``True``, they are
        changed into values as close as possible to the beginning of the window.
    :type compress_init: bool

    :param clip_window: See :func:`df_window`

    .. seealso:: :func:`df_split_signals`
    """

    def before(x):
        return np.nextafter(x, -math.inf)

    windowed_df = df_window(df, window, method='pre', clip_window=clip_window)

    # Split the extra rows that the method='pre' gave in a separate dataframe,
    # so we make sure we don't end up with duplication in init_df
    extra_window = (
        windowed_df.index[0],
        window[0],
    )
    if extra_window[0] >= extra_window[1]:
        extra_df = df_make_empty_clone(df)
    else:
        extra_df = df_window(windowed_df, extra_window, method='pre')

    # This time around, exclude anything before extra_window[1] since it will be provided by extra_df
    try:
        # Right boundary is exact, so failure can only happen if left boundary
        # is after the start of the dataframe, or if the window starts after its end.
        _window = (extra_window[1], windowed_df.index[-1])
        windowed_df = df_window(windowed_df, _window, method='post', clip_window=False)
    # The windowed_df did not contain any row in the given window, all the
    # actual data are in extra_df
    except KeyError:
        windowed_df = df_make_empty_clone(df)
    else:
        # Make sure we don't include the left boundary
        if windowed_df.index[0] == _window[0]:
            windowed_df = windowed_df.iloc[1:]

    def window_signal(signal_df):
        # Get the row immediately preceding the window start
        loc = _get_loc(signal_df.index, window[0], method='ffill')
        return signal_df.iloc[loc:loc + 1]

    # Get the value of each signal at the beginning of the window
    signal_df_list = [
        window_signal(signal_df)
        for signal, signal_df in itertools.chain.from_iterable(
            df_split_signals(df, signal.fields, align_start=False)
            for signal in signals
        )
        # Only consider the signal that are in the window. Signals that started
        # after the window are irrelevant.
        if not signal_df.empty and signal_df.index[0] <= window[0]
    ]

    if compress_init:
        def make_init_df_index(init_df):
            # Yield a sequence of numbers incrementing by the smallest amount
            # possible
            def smallest_increment(start, length):
                curr = start
                for _ in range(length):
                    curr = before(curr)
                    yield curr


            # If windowed_df is empty, we take the last bit right before the
            # beginning of the window
            try:
                start = windowed_df.index[0]
            except IndexError:
                start = extra_df.index[-1]

            index = list(smallest_increment(start, len(init_df)))
            index = pd.Float64Index(reversed(index))
            return index
    else:
        def make_init_df_index(init_df):
            return init_df.index

    # Get the last row before the beginning the window for each signal, in
    # timestamp order
    init_df = pd.concat([extra_df] + signal_df_list)
    init_df.sort_index(inplace=True)
    # Remove duplicated indices, meaning we selected the same row multiple
    # times because it's part of multiple signals
    init_df = init_df.loc[~init_df.index.duplicated(keep='first')]

    init_df.index = make_init_df_index(init_df)
    return pd.concat([init_df, windowed_df])


@SeriesAccessor.register_accessor
def series_align_signal(ref, to_align, max_shift=None):
    """
    Align a signal to an expected reference signal using their
    cross-correlation.

    :returns: `(ref, to_align)` tuple, with `to_align` shifted by an amount
        computed to align as well as possible with `ref`. Both `ref` and
        `to_align` are resampled to have a fixed sample rate.

    :param ref: reference signal.
    :type ref: pandas.Series

    :param to_align: signal to align
    :type to_align: pandas.Series

    :param max_shift: Maximum shift allowed to align signals, in index units.
    :type max_shift: object or None
    """
    if ref.isnull().any() or to_align.isnull().any():
        raise ValueError('NaN needs to be dropped prior to alignment')

    # Select the overlapping part of the signals
    start = max(ref.index.min(), to_align.index.min())
    end = min(ref.index.max(), to_align.index.max())

    # Resample so that we operate on a fixed sampled rate signal, which is
    # necessary in order to be able to do a meaningful interpretation of
    # correlation argmax
    def get_period(series): return pd.Series(series.index).diff().min()
    period = min(get_period(ref), get_period(to_align))
    num = math.ceil((end - start) / period)
    new_index = pd.Float64Index(np.linspace(start, end, num))

    to_align = to_align.reindex(new_index, method='ffill')
    ref = ref.reindex(new_index, method='ffill')

    # Compute the correlation between the two signals
    correlation = scipy.signal.signaltools.correlate(to_align, ref)
    # The most likely shift is the index at which the correlation is
    # maximum. correlation.argmax() can vary from 0 to 2*len(to_align), so we
    # re-center it.
    shift = correlation.argmax() - len(to_align)

    # Cap the shift value
    if max_shift is not None:
        assert max_shift >= 0

        # Turn max_shift into a number of samples in the resampled signal
        max_shift = int(max_shift / period)

        # Adjust the sign of max_shift to match shift
        max_shift *= -1 if shift < 0 else 1

        if abs(shift) > abs(max_shift):
            shift = max_shift

    # Compensate the shift
    return ref, to_align.shift(-shift)


@DataFrameAccessor.register_accessor
def df_filter_task_ids(df, task_ids, pid_col='pid', comm_col='comm', invert=False, comm_max_len=TASK_COMM_MAX_LEN):
    """
    Filter a dataframe using a list of :class:`lisa.trace.TaskID`

    :param task_ids: List of task IDs to filter
    :type task_ids: list(lisa.trace.TaskID)

    :param df: Dataframe to act on.
    :type df: pandas.DataFrame

    :param pid_col: Column name in the dataframe with PIDs.
    :type pid_col: str or None

    :param comm_col: Column name in the dataframe with comm.
    :type comm_col: str or None

    :param comm_max_len: Maximum expected length of the strings in
        ``comm_col``. The ``task_ids`` `comm` field will be truncated at that
        length before being matched.

    :param invert: Invert selection
    :type invert: bool
    """

    def make_filter(task_id):
        if pid_col and task_id.pid is not None:
            pid = (df[pid_col] == task_id.pid)
        else:
            pid = True
        if comm_col and task_id.comm is not None:
            comm = (df[comm_col] == task_id.comm[:comm_max_len])
        else:
            comm = True

        return pid & comm

    tasks_filters = map(make_filter, task_ids)

    # Combine all the task filters with OR
    tasks_filter = functools.reduce(operator.or_, tasks_filters, False)

    if invert:
        tasks_filter = ~tasks_filter

    return df[tasks_filter]


@SeriesAccessor.register_accessor
def series_local_extremum(series, kind):
    """
    Returns a series of local extremum.

    :param series: Series to look at.
    :type series: pandas.Series

    :param kind: Kind of extremum: ``min`` or ``max``.
    :type kind: str
    """
    if kind == 'min':
        comparator = np.less_equal
    elif kind == 'max':
        comparator = np.greater_equal
    else:
        raise ValueError('Unsupported kind: {}'.format(kind))

    ilocs = scipy.signal.argrelextrema(series.to_numpy(), comparator=comparator)
    return series.iloc[ilocs]


@SeriesAccessor.register_accessor
def series_envelope_mean(series):
    """
    Compute the average between the mean of local maximums and local minimums
    of the series.

    Assuming that the values are ranging inside a tunnel, this will give the
    average center of that tunnel.
    """
    maxs = series_local_extremum(series, kind='max')
    mins = series_local_extremum(series, kind='min')

    maxs_mean = series_mean(maxs)
    mins_mean = series_mean(mins)

    return (maxs_mean - mins_mean) / 2 + mins_mean

# Keep an alias in place for compatibility
@deprecate(replaced_by=series_envelope_mean, deprecated_in='2.0', removed_in='2.1')
def series_tunnel_mean(*args, **kwargs):
    return series_envelope_mean(*args, **kwargs)


@SeriesAccessor.register_accessor
def series_rolling_apply(series, func, window, window_float_index=True, center=False):
    """
    Apply a function on a rolling window of a series.

    :returns: The series of results of the function.

    :param series: Series to act on.
    :type series: pandas.Series

    :param func: Function to apply on each window. It must take a
        :class:`pandas.Series` as only parameter and return one value.
    :type func: collections.abc.Callable

    :param window: Rolling window width in seconds.
    :type window: float

    :param center: Label values generated by ``func`` with the center of the
        window, rather than the highest index in it.
    :type center: bool

    :param window_float_index: If ``True``, the series passed to ``func`` will
        be of type :class:`pandas.Float64Index`, in nanoseconds. Disabling is
        recommended if the index is not used by ``func`` since it will remove
        the need for a conversion.
    :type window_float_index: bool
    """
    orig_index = series.index

    # Wrap the func to turn the index into nanosecond Float64Index
    if window_float_index:
        def func(s, func=func):
            s.index = s.index.astype('int64') * 1e-9
            return func(s)

    # Use a timedelta index so that rolling gives time-based results
    index = pd.to_timedelta(orig_index, unit='s')
    series = pd.Series(series.array, index=index)

    window_ns = int(window * 1e9)
    rolling_window = '{}ns'.format(window_ns)
    values = series.rolling(rolling_window).apply(func, raw=False).values

    if center:
        new_index = orig_index - (window / 2)
    else:
        new_index = orig_index

    return pd.Series(values, index=new_index)


def _data_find_unique_bool_vector(data, cols, all_col, keep):
    if keep == 'first':
        shift = 1
    elif keep == 'last':
        shift = -1
    elif keep is None:
        shift = 1
    else:
        raise ValueError('Unknown keep value: {}'.format(keep))

    dedup_data = data[cols] if cols else data
    # Unique values will be True, duplicate False
    cond = dedup_data != dedup_data.shift(shift)
    if isinstance(data, pd.DataFrame):
        # (not (duplicate and duplicate))
        # (not ((not unique) and (not unique)))
        # (not (not (unique or unique)))
        # (unique or unique)
        if all_col:
            cond = cond.any(axis=1)
        # (not (duplicate or duplicate))
        # (not (duplicate or duplicate))
        # (not ((not unique) or (not unique)))
        # (not (not (unique and unique)))
        # (unique and unique)
        else:
            cond = cond.all(axis=1)

    # Also mark as duplicate the first row in a run
    if keep is None:
        cond &= cond.shift(-1).fillna(True)

    return cond


def _data_deduplicate(data, keep, consecutives, cols, all_col):
    if consecutives:
        return data.loc[_data_find_unique_bool_vector(data, cols, all_col, keep)]
    else:
        if not all_col:
            raise ValueError("all_col=False is not supported with consecutives=False")

        kwargs = dict(subset=cols) if cols else {}
        return data.drop_duplicates(keep=keep, **kwargs)


@SeriesAccessor.register_accessor
def series_deduplicate(series, keep, consecutives):
    """
    Remove duplicate values in a :class:`pandas.Series`.

    :param keep: Keep the first occurrences if ``first``, or the last if
        ``last``.
    :type keep: str

    :param consecutives: If ``True``, will only remove consecutive duplicates,
        for example::

            s = pd.Series([1,2,2,3,4,2], index=[1,2,20,30,40,50])
            s2 = series_deduplicate(s, keep='first', consecutives=True)
            assert (s2 == [1,2,3,4,2]).all()

            s3 = series_deduplicate(s, keep='first', consecutives=False)
            assert (s3 == [1,2,3,4]).all()

    :type consecutives: bool
    """
    return _data_deduplicate(series, keep=keep, consecutives=consecutives, cols=None, all_col=True)


@DataFrameAccessor.register_accessor
def df_deduplicate(df, keep, consecutives, cols=None, all_col=True):
    """
    Same as :func:`series_deduplicate` but for :class:`pandas.DataFrame`.

    :param cols: Only consider these columns when looking for duplicates.
        By default, all columns are considered
    :type cols: list(str) or None

    :param all_col: If ``True``, remove a row when all the columns have duplicated value.
        Otherwise, remove the row if any column is duplicated.
    :type all_col: bool
    """
    return _data_deduplicate(df, keep=keep, consecutives=consecutives, cols=cols, all_col=all_col)


@DataFrameAccessor.register_accessor
def df_update_duplicates(df, col=None, func=None, inplace=False):
    """
    Update a given column to avoid duplicated values.

    :param df: Dataframe to act on.
    :type df: pandas.DataFrame

    :param col: Column to update. If ``None``, the index is used.
    :type col: str or None

    :param func: The function used to update the column. It must take a
        :class:`pandas.Series` of duplicated entries to update as parameters,
        and return a new :class:`pandas.Series`. The function will be called as
        long as there are remaining duplicates. If ``None``, the column is
        assumed to be floating point and duplicated values will be incremented
        by the smallest amount possible.
    :type func: collections.abc.Callable

    :param inplace: If ``True``, the passed dataframe will be modified.
    :type inplace: bool
    """

    def increment(series):
        return pd.Series(np.nextafter(series.array, math.inf), index=series.index)

    def get_duplicated(series):
        # Keep the first, so we update the second duplicates
        locs = series.duplicated(keep='first')
        return locs, series.loc[locs]

    def copy(series):
        return series.copy() if inplace else series

    use_index = col is None
    # Indices already gets copied with to_series()
    use_copy = inplace and not use_index

    series = df.index.to_series() if use_index else df[col]
    series = series.copy() if use_copy else series
    func = func if func else increment

    # Update the values until there is no more duplication
    duplicated_locs, duplicated = get_duplicated(series)
    while duplicated_locs.any():
        updated = func(duplicated)
        # Change the values at the points of duplication. Otherwise, take the
        # initial value
        series.loc[duplicated_locs] = updated
        duplicated_locs, duplicated = get_duplicated(series)

    df = df if inplace else df.copy()
    if use_index:
        df.index = series
    else:
        df[col] = series

    return df


@DataFrameAccessor.register_accessor
def df_combine_duplicates(df, func, output_col, cols=None, all_col=True, prune=True, inplace=False):
    """
    Combine the duplicated rows using ``func`` and remove the duplicates.

    :param df: The dataframe to act on.
    :type df: pandas.DataFrame

    :param func: Function to combine a group of duplicates. It will be passed a
        :class:`pandas.DataFrame` corresponding to the group and must return
        either a :class:`pandas.Series` with the same index as its input dataframe,
        or a scalar depending on the value of ``prune``.
    :type func: collections.abc.Callable

    :param prune: If ``True``, ``func`` will be expected to return a single
        scalar that will be used instead of a whole duplicated group. Only the
        first row of the group is kept, the other ones are removed.

        If ``False``, ``func`` is expected to return a :class:`pandas.Series`
        that will be used as replacement for the group. No rows will be removed.
    :type prune: bool

    :param output_col: Column in which the output of ``func`` should be stored.
    :type output_col: str

    :param cols: Columns to use for duplicates detection
    :type cols: list(str) or None

    :param all_cols: If ``True``, all columns will be used.
    :type all_cols: bool

    :param inplace: If ``True``, the passed dataframe is modified.
    :type inplace: bool
    """
    init_df = df if inplace else df.copy()
    # We are going to add columns so make a copy
    df = df.copy(deep=False)

    # Find all rows where the active status is the same as the previous one
    duplicates_to_remove = ~_data_find_unique_bool_vector(df, cols, all_col, keep='first')
    # Then get only the first row in a run of duplicates
    first_duplicates = (~duplicates_to_remove) & duplicates_to_remove.shift(-1, fill_value=False)
    # We only kept them separate with keep='first' to be able to detect
    # correctly the beginning of a duplicate run to get a group ID, so now we
    # merge them
    duplicates = duplicates_to_remove | first_duplicates

    # Assign the group ID to each member of the group
    df.loc[first_duplicates, 'duplicate_group'] = first_duplicates.loc[first_duplicates].index
    df.loc[duplicates, 'duplicate_group'] = df.loc[duplicates, 'duplicate_group'].fillna(method='ffill')

    # For some reasons GroupBy.apply() will raise a KeyError if the index is a
    # Float64Index, go figure ...
    index = df.index
    df.reset_index(drop=True, inplace=True)

    # Apply the function to each group, and assign the result to the output
    # Note that we cannot use GroupBy.transform() as it currently cannot handle
    # NaN groups.
    output = df.groupby('duplicate_group', sort=False, as_index=True, group_keys=False, observed=True).apply(func)
    if not output.empty:
        init_df[output_col].update(output)

    # Ensure the column is created if it does not exists yet
    try:
        init_df[output_col]
    except KeyError:
        init_df[output_col] = np.NaN
    else:
        # Restore the index that we had to remove for apply()
        df.index = index
        try:
            fill = df[output_col]
        except KeyError:
            pass
        else:
            init_df[output_col].fillna(fill, inplace=True)

    if prune:
        # Only keep the first row of each duplicate run
        if inplace:
            removed_indices = duplicates_to_remove[duplicates_to_remove].index
            init_df.drop(removed_indices, inplace=True)
            return None
        else:
            return init_df.loc[~duplicates_to_remove]
    else:
        if inplace:
            return None
        else:
            return init_df


@DataFrameAccessor.register_accessor
def df_add_delta(df, col='delta', src_col=None, window=None, inplace=False):
    """
    Add a column containing the delta of the given other column.

    :param df: The dataframe to act on.
    :type df: pandas.DataFrame

    :param col: The name of the column to add.
    :type col: str

    :param src_col: Name of the column to compute the delta of. If ``None``,
        the index is used.
    :type src_col: str or None

    :param window: Optionally, a window. It will be used to compute the correct
        delta of the last row. If ``inplace=False``, the dataframe will be
        pre-filtered using :func:`df_refit_index`. This implies that the last
        row will have a NaN delta, but will be suitable e.g. for plotting, and
        aggregation functions that ignore delta such as
        :meth:`pandas.DataFrame.sum`.
    :type window: tuple(float or None, float or None) or None

    :param inplace: If ``True``, ``df`` is modified inplace to add the column
    :type inplace: bool
    """

    use_refit_index = window and not inplace

    if use_refit_index:
        df = df_refit_index(df, window=window)

    src = df[src_col] if src_col else df.index.to_series()
    delta = src.diff().shift(-1)

    # When use_refit_index=True, the last delta will already be sensible
    if not use_refit_index and window:
        start, end = window
        if end is not None:
            new_end = end - src.iloc[-1]
            new_end = new_end if new_end > 0 else 0
            delta.iloc[-1] = new_end

    if not inplace:
        df = df.copy()

    df[col] = delta

    return df


def series_combine(series_list, func, fill_value=None):
    """
    Same as :meth:`pandas.Series.combine` on a list of series rather than just
    two.
    """
    return _data_combine(series_list, func, fill_value)


def df_combine(series_list, func, fill_value=None):
    """
    Same as :meth:`pandas.DataFrame.combine` on a list of series rather than just
    two.
    """
    return _data_combine(series_list, func, fill_value)


def series_dereference(series, sources, inplace=False):
    """
    Replace each value in ``series`` by the value at the corresponding index by
    the source indicated by ``series``'s value.

    :param series: Series of "pointer" values.
    :type series: pandas.Series

    :param sources: Dictionary with keys corresponding to ``series`` values.
        For each value of ``series``, a source will be chosen and its value at the
        current index will be used. If a :class:`pandas.DataFrame` is passed,
        the column names will be used as keys and the column series as values.

        .. note:: Unless ``series`` and the ``sources`` share the same index,
            the ``sources`` will be reindexed with ``ffill`` method.
    :type sources: collections.abc.Mapping or pandas.DataFrame

    :param inplace: If ``True``, modify the series inplace.
    :type inplace: bool
    """
    def reindex(values):
        # Skip the reindex if they are in the same dataframe
        if values.index is not series.index:
            values = values.reindex(series.index, method='ffill')
        return values

    if isinstance(sources, pd.DataFrame):
        sources = reindex(sources)
        sources = {
            col: sources[col]
            for col in sources.columns
        }
    else:
        sources = {
            col: reindex(val)
            for col, val in sources.items()
        }

    for key, values in sources.items():
        _series = series.mask(series == key, values, inplace=inplace)
        series = series if inplace else _series

    return series


def df_dereference(df, col, pointer_col=None, sources=None, inplace=False):
    """
    Similar to :func:`series_dereference`.

    **Example**::

        df = pd.DataFrame({
            'ptr': ['A', 'B'],
            'A'  : ['A1', 'A2'],
            'B'  : ['B1', 'B2'],
        })
        df = df_dereference(df, 'dereferenced', pointer_col='ptr')
        #   ptr   A   B dereferenced
        # 0   A  A1  B1           A1
        # 1   B  A2  B2           B2


    :param df: Dataframe to act on.
    :type df: pandas.DataFrame

    :param col: Name of the column to create.
    :type col: str

    :param pointer_col: Name of the column containing "pointer" values.
        Defaults to the same value as ``col``.
    :type pointer_col: str or None

    :param sources: Same meaning as in :func:`series_dereference`. If omitted,
        ``df`` is used.
    :type sources: collections.abc.Mapping or pandas.DataFrame

    :param inplace: If ``True``, the dataframe is modified inplace.
    :type inplace: bool
    """
    pointer_col = pointer_col or col
    sources = df if sources is None else sources
    df = df if inplace else df.copy(deep=False)
    df[col] = series_dereference(df[pointer_col], sources, inplace=inplace)
    return df


def _data_combine(datas, func, fill_value=None):
    state = datas[0]
    for data in datas[1:]:
        state = state.combine(data, func=func, fill_value=fill_value)

    return state


class SignalDesc:
    """
    Define a signal to be used by various signal-oriented APIs.

    :param event: Name of the event that this signal is represented by.
    :type event: str

    :param fields: Fields that identify multiple signals multiplexed into one
        event. For example, a `frequency` signal would have a ``cpu_frequency``
        event and a ``cpu`` field since ``cpu_frequency`` multiplexes the
        signals for all CPUs.
    :type fields: list(str)
    """

    def __init__(self, event, fields):
        self.event = event
        self.fields = sorted(fields)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash(self.event) ^ hash(tuple(self.fields))

    @classmethod
    def from_event(cls, event, fields=None):
        """
        Return list of :class:`SignalDesc` for the given event.

        The hand-coded list is used first, and then some generic heuristics are
        used to detect per-cpu and per-task signals.
        """
        try:
            return cls._SIGNALS_MAP[event]
        except KeyError:
            if not fields:
                return [cls(event, fields=[])]
            else:
                fields = set(fields)
                # At most one set of each group will be taken
                default_field_sets = [
                    [
                        {'comm', 'pid'},
                        {'pid'},
                        {'comm'},
                    ],
                    [
                        {'cpu'},
                        {'cpu_id'},
                    ],
                ]

                selected = []
                for field_set_group in default_field_sets:
                    # Select at most one field set per group
                    for field_set in field_set_group:
                        # if fields is a non-strict superset of field_set
                        if fields >= field_set:
                            selected.append(field_set)
                            break

                return [
                    cls(event, fields=field_set)
                    for field_set in selected
                ]


# Before pandas <= 1.0.0 (Python <= 3.5):
# * 'string' dtype does not exist
# * nullable integer dtypes are not serializable before
_PANDAS_HIGHER_THAN_1_1_0 = tuple(map(int, pd.__version__.split('.'))) >= (1, 0, 0)

@SeriesAccessor.register_accessor
def series_convert(series, dtype):
    """
    Convert a :class:`pandas.Series` with a best effort strategy.

    Nullable types may be used if necessary and possible, otherwise ``object``
    dtype will be used.

    :param series: Series of another type than the target one. Strings are
        allowed.
    :type series: pandas.Series

    :param dtype: dtype to convert to. If it is a string ``uint8``, the
        following strategy will be used:

            1. Convert to the given dtype
            2. If it failed, try converting to an equivalent nullable dtype
            3. If it failed, try to parse it with an equivalent Python object
               constructor, and then convert it to the dtype.
            4. If an integer dtype was requested, parsing as hex string will be
               attempted too

        If it is a callable, it will be applied on the series, converting all
        values considered as nan by :func:`pandas.isna` into ``None`` values.
        The result will have ``object`` dtype. The callable has a chance to
        handle the conversion from nan itself.

        .. note:: In some cases, asking for an unsigned dtype might let through
            negative values, as there is no way to reliably distinguish between
            conversion failures reasons.
    :type dtype: str or collections.abc.Callable
    """

    if series.dtype.name == dtype:
        return series

    def to_object(x):
        x = x.astype('object', copy=True)
        # If we had any pandas <NA> values, they need to be turned into None
        # first, otherwise pyarrow will choke on them
        x.loc[x.isna()] = None
        return x

    astype = lambda dtype: lambda x: x.astype(dtype, copy=False)
    make_convert = lambda dtype: lambda x: series_convert(x, dtype)
    basic = astype(dtype)

    class Tree(list):
        """
        Tree of converters to guide what to do in case of failure
        """
        def __init__(self, *items, name=None):
            items = [
                item
                for item in items
                if item is not None
            ]
            super().__init__(items)
            self.name = name

    class Pipeline(Tree):
        """
        Sequence of converters that succeed as a whole or fail as a whole
        """
        def __call__(self, series):
            for x in self:
                series = x(series)
            return series

    class Alternative(Tree):
        """
        Sequence of converters to try in order until one works
        """
        def __call__(self, series):
            excep = ValueError('Empty alternative')
            for x in self:
                try:
                    return x(series)
                except (TypeError, ValueError, OverflowError) as e:
                    excep = e

            # Re-raise the last exception raised
            raise excep

    pipelines = Alternative(name='root')

    # If that is not a string
    with contextlib.suppress(AttributeError, TypeError):
        lower_dtype = dtype.lower()
        is_bool = ('bool' in lower_dtype)
        is_int = ('int' in lower_dtype)

    # types are callable too
    if callable(dtype):
        def convert(x):
            try:
                return dtype(x)
            except Exception:
                # Make sure None will be propagated as None.
                # note: We use an exception handler rather than checking first
                # in order to speed up the expected path where the conversion
                # won't fail.
                if pd.isna(x):
                    return None
                else:
                    raise

        # Use faster logic of pandas if possible, but not for bytes as it will
        # happily convert math.nan into b'nan'
        if dtype is not bytes:
            pipelines.append(basic)

        pipelines.append(
            # Otherwise fallback to calling the type directly
            lambda series: series.apply(convert, convert_dtype=False)
        )

    # Then try with a nullable type.
    # Floats are already nullable so we don't need to do anything
    elif is_bool or is_int:
        nullable_dtypes = {
            'int':    'Int64',
            'int8':   'Int8',
            'int16':  'Int16',
            'int32':  'Int32',
            'int64':  'Int64',

            'uint':   'UInt64',
            'uint8':  'UInt8',
            'uint16': 'UInt16',
            'uint32': 'UInt32',
            'uint64': 'UInt64',

            'bool':   'boolean',
        }

        # Bare nullable dtype
        if _PANDAS_HIGHER_THAN_1_1_0:
            # Already nullable
            if dtype[0].isupper():
                nullable = dtype
            else:
                nullable = nullable_dtypes[dtype]
            to_nullable = astype(nullable)
        else:
            # Make it fail so we don't end up with issues with missing parquet
            # support for nullable types down the line
            def to_nullable(series):
                raise ValueError('pandas version too old for nullable types')
            basic = astype(lower_dtype)

        if is_int:
            parse = Alternative(
                # Parse as integer
                make_convert(int),
                # Parse as hex int
                make_convert(functools.partial(int, base=16))
            )
        elif is_bool:
            parse = make_convert(bool)
        else:
            assert False

        # Strategy assuming it's already a numeric type
        from_numeric = Alternative(
            basic,
            to_nullable
        )

        pipelines.extend((
            from_numeric,
            # Maybe we were trying to parse some strings that turned out to
            # need to go through the Python int constructor to be parsed,
            # so do that first
            Pipeline(
                parse,
                Alternative(
                    from_numeric,
                    # Or just leave the output as it is if nothing else can be
                    # done, as we already have 'object' of an integer type
                    to_object,
                    name='convert parser output',
                ),
                name='parse',
            )
        ))

    elif dtype == 'string':

        # Sadly, pandas==1.1.1 (and maybe later) series.astype('string') turns
        # b'hello' into "b'hello'" instead of "hello", so basic decoder becomes
        # unusable
        if (
            series.dtype.name == 'object' and
            series.apply(isinstance, args=(bytes,), convert_dtype=False).any()
        ):
            string_basic = None
            # Handle mixed dtypes
            str_basic = lambda x : x.apply(
                lambda x: x.decode('ascii') if isinstance(x, bytes) else str(x),
                convert_dtype=False
            )
        else:
            string_basic = basic
            str_basic = make_convert(str)

        # Faster than Series.str.decode()
        basic_decode = lambda x : x.apply(bytes.decode, args=('ascii',), convert_dtype=False)

        # Significantly faster than Series.str.decode()
        def fast_decode(x):
            # Deduplicate the original values by turning into a category
            x = x.astype('category')
            cat = x.cat.categories.to_series()
            # Decode the deduplicated values.
            #
            # Since decoding is relatively expensive, doing it on fewer objects
            # is usually a win, especially since most strings are task names.
            #
            # This also has the advantage that the strings are deduplicated,
            # which is safe since they are immutable. This reduces the memory
            # used by the final series
            new_cat = basic_decode(cat)
            x.cat.rename_categories(new_cat, inplace=True)
            return astype('string')(x)

        pipelines.extend((
            string_basic,
            # We need to attempt conversion from bytes before using Python str,
            # otherwise it will include the b'' inside the string
            fast_decode,
            # Since decode() is complex, let's have the basic version in case
            # categories have unexpected limitations
            basic_decode,
            # If direct conversion to "string" failed, we need to turn
            # whatever the type was to actual strings using the Python
            # constructor
            Pipeline(
                str_basic,
                Alternative(
                    basic,
                    # basic might fail on older version of pandas where
                    # 'string' dtype does not exists
                    to_object,
                    name='convert parse output'
                ),
                name='parse'
            )
        ))

    elif dtype == 'bytes':
        pipelines.append(make_convert(bytes))
    else:
        # For floats, astype() works well and can even convert from strings and the like
        pipelines.append(basic)

    return pipelines(series)


# Defined outside SignalDesc as it references SignalDesc itself
_SIGNALS = [
    SignalDesc('sched_switch', ['next_comm', 'next_pid']),
    SignalDesc('sched_switch', ['prev_comm', 'prev_pid']),

    SignalDesc('sched_waking', ['target_cpu']),
    SignalDesc('sched_waking', ['comm', 'pid']),
    SignalDesc('sched_wakeup', ['target_cpu']),
    SignalDesc('sched_wakeup', ['comm', 'pid']),
    SignalDesc('sched_wakeup_new', ['target_cpu']),
    SignalDesc('sched_wakeup_new', ['comm', 'pid']),

    SignalDesc('cpu_idle', ['cpu_id']),
    SignalDesc('cpu_capacity', ['cpu_id']),
    SignalDesc('cpu_frequency', ['cpu_id']),
    SignalDesc('userspace@cpu_frequency_devlib', ['cpu_id']),
    SignalDesc('sched_compute_energy', ['comm', 'pid']),

    SignalDesc('sched_pelt_se', ['comm', 'pid']),
    SignalDesc('sched_load_se', ['comm', 'pid']),
    SignalDesc('sched_util_est_se', ['comm', 'pid']),

    SignalDesc('sched_util_est_cfs', ['cpu']),
    SignalDesc('sched_pelt_cfs', ['path', 'cpu']),
    SignalDesc('sched_load_cfs_rq', ['path', 'cpu']),
    SignalDesc('sched_pelt_irq', ['cpu']),
    SignalDesc('sched_pelt_rt', ['cpu']),
    SignalDesc('sched_pelt_dl', ['cpu']),

    SignalDesc('uclamp_util_se', ['pid', 'comm']),
    SignalDesc('uclamp_util_cfs', ['cpu']),

    SignalDesc('sched_overutilized', []),
    SignalDesc('sched_process_wait', ['comm', 'pid']),

    SignalDesc('schedutil_em_boost', ['cpu']),

    SignalDesc('thermal_temperature', ['id']),
    SignalDesc('thermal_zone_trip', ['id']),
]
"""
List of predefined :class:`SignalDesc`.
"""

SignalDesc._SIGNALS_MAP = {
    event: list(signal_descs)
    for event, signal_descs in groupby(_SIGNALS, key=attrgetter('event'))
}

def _data_extend_index(data, extension):
    """
    ``data`` can either be a :class:`pandas.DataFrame` or :class:`pandas.Series`
    """
    return data.join(extension, how='outer')

@SeriesAccessor.register_accessor
def series_extend_index(series, extension):
    """
    Extending the index of a :class:`pandas.Series

    :param: series: series to extend
    :type series: :class:`pandas.Series`

    :param extension: series holding the extension
    :type extension: class:`pandas.Series`
    """
    return _data_extend_index(series, extension)

@DataFrameAccessor.register_accessor
def df_extend_index(df, extension):
    """
    Same as :func:`series_extend_index` but acting on a :class:`pandas.DataFrame`
    """
    return _data_extend_index(df, extension)
# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
