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
"""
Dataframe utilities.
"""

import re
import functools
import operator
import math
import itertools
import warnings
import contextlib
import uuid
from operator import attrgetter
import decimal
from numbers import Number
import weakref
import threading

import polars as pl
import polars.selectors as cs
import numpy as np
import pandas as pd
import pandas.api.extensions
import scipy.integrate
import scipy.signal
import pyarrow

from lisa.utils import TASK_COMM_MAX_LEN, groupby, deprecate, order_as


class Timestamp(float):
    """
    Nanosecond-precision timestamp. It inherits from ``float`` and as such can
    be manipulating as a floating point number of seconds. The ``nanoseconds``
    attribute allows getting the exact timestamp regardless of the magnitude of
    the float, allowing for more precise computation.

    :param unit: Unit of the ``ts`` value being passed. One of ``"s"``,
        ``"ms"``, ``"us"`` and ``"ns"``.
    :type unit: str

    :param rounding: How to round the value when converting to float.
        Timestamps of large magnitude will suffer from the loss of least
        significant digits in their float value which will not have nanosecond
        precision. The rounding determines if a value below or above the actual
        nanosecond-precision timestamp should be used. One of ``"up"`` or
        ``"down"``.
    :type rounding: str

    """
    __slots__ = ('as_nanoseconds', '_Timestamp__rounding')

    _MUL = dict(
        ns=decimal.Decimal('1'),
        us=decimal.Decimal('1e3'),
        ms=decimal.Decimal('1e6'),
        s=decimal.Decimal('1e9'),
    )

    _1NS = decimal.Decimal('1e-9')

    def __new__(cls, ts, unit='s', rounding='down'):
        if isinstance(ts, cls):
            return cls(ts.as_nanoseconds, unit='ns', rounding=rounding)
        else:
            if isinstance(ts, np.integer):
                ts = int(ts)

            ts = decimal.Decimal(ts)
            try:
                mul = cls._MUL[unit]
            except KeyError:
                raise ValueError(f'Unknown unit={unit}')

            ns = mul * ts

            if rounding == 'up':
                ns = math.ceil(ns)
            elif rounding == 'down':
                ns = math.floor(ns)
            else:
                raise ValueError(f'Unknown rounding={rounding}')

            s = ns * cls._1NS
            float_s = float(s)

            if float_s > s and rounding == 'down':
                float_s = float(np.nextafter(float_s, -math.inf))
            elif float_s < s and rounding == 'up':
                float_s = float(np.nextafter(float_s, math.inf))

            self = super().__new__(cls, float_s)
            self.as_nanoseconds = ns
            self.__rounding = rounding
            return self

    def _with_ns(self, ns):
        return self.__class__(
            ns,
            unit='ns',
            rounding=self.__rounding
        )

    def __hash__(self):
        return hash(self.as_nanoseconds)

    def __cmp(self, other, op):
        try:
            other = Timestamp(other)
        except OverflowError:
            # +inf/-inf
            return op(float(self), other)
        else:
            return op(self.as_nanoseconds, other.as_nanoseconds)

    def __le__(self, other):
        return self.__cmp(other, operator.le)

    def __ge__(self, other):
        return self.__cmp(other, operator.ge)

    def __lt__(self, other):
        return self.__cmp(other, operator.lt)

    def __gt__(self, other):
        return self.__cmp(other, operator.gt)

    def __eq__(self, other):
        try:
            return self.__cmp(other, operator.eq)
        except TypeError:
            return NotImplemented
        except Exception:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __add__(self, other):
        ns = Timestamp(other).as_nanoseconds
        return self._with_ns(self.as_nanoseconds + ns)

    def __sub__(self, other):
        ns = Timestamp(other).as_nanoseconds
        return self._with_ns(self.as_nanoseconds - ns)

    def __mul__(self, other):
        return self._with_ns(self.as_nanoseconds * other)

    def __mod__(self, other):
        return self._with_ns(self.as_nanoseconds % other)

    def __truediv__(self, other):
        return self._with_ns(self.as_nanoseconds / other)

    def __floordiv__(self, other):
        return self._with_ns(self.as_nanoseconds // other)

    def __abs__(self):
        return self._with_ns(abs(self.as_nanoseconds))

    def __neg__(self):
        return self._with_ns(abs(-self.as_nanoseconds))

    def __pos__(self):
        return self

    def __invert__(self):
        return -self

    def to_polars_expr(self):
        return pl.duration(
            # https://github.com/pola-rs/polars/issues/11625
            nanoseconds=self.as_nanoseconds,
            # https://github.com/pola-rs/polars/issues/14751
            time_unit='ns'
        )


def _dispatch(polars_f, pandas_f, data, *args, **kwargs):
    if isinstance(data, (pl.LazyFrame, pl.DataFrame, pl.Series)):
        return polars_f(data, *args, **kwargs)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return pandas_f(data, *args, **kwargs)
    else:
        raise TypeError(f'Cannot find implementation for {data.__class__}')


def _polars_duration_expr(duration, unit='s', rounding='down'):
    if duration is None:
        return duration
    elif isinstance(duration, pl.Expr):
        return duration
    else:
        duration = Timestamp(
            duration,
            unit=unit,
            rounding=rounding
        )
        return duration.to_polars_expr()


def _polars_duration_window(window):
    start, end = window
    return (
        _polars_duration_expr(start, rounding='down'),
        _polars_duration_expr(end, rounding='up')
    )


_MEM_LAZYFRAMES_LOCK = threading.Lock()
_MEM_LAZYFRAMES = weakref.WeakValueDictionary()
def _polars_declare_in_memory(df):
        with _MEM_LAZYFRAMES_LOCK:
            _MEM_LAZYFRAMES[id(df)] = df

def _polars_df_in_memory(df):
    try:
        with _MEM_LAZYFRAMES_LOCK:
            _df = _MEM_LAZYFRAMES[id(df)]
    except KeyError:
        return False
    else:
        return _df is df


class _NoIndex:
    pass


NO_INDEX = _NoIndex()


def _polars_index_col(df, index=None):
    columns = df.collect_schema().names()

    if index is NO_INDEX:
        return None
    elif index in columns:
        return index
    else:
        return columns[0]


def _df_to_polars(df, index):
    in_memory = _polars_df_in_memory(df)

    if isinstance(df, pl.LazyFrame):
        index = _polars_index_col(df, index)
        if index is not None:
            schema = df.collect_schema()
            dtype = schema[index]
            # This skips a useless cast, saving some time on the common path
            if index == 'Time':
                if dtype != pl.Duration('ns'):
                    _index = pl.col(index)
                    if dtype.is_float():
                        # Convert to nanoseconds
                        df = df.with_columns(_index * 1_000_000_000)
                    elif dtype.is_integer() or dtype.is_temporal():
                        pass
                    else:
                        raise TypeError(f'Index dtype not handled: {dtype}')

                    df = df.with_columns(
                        _index.cast(pl.Duration('ns'))
                    )

            # Make the index column the first one
            df = df.select(order_as(list(df.collect_schema().names()), [index]))
    # TODO: once this is solved, we can just inspect the plan and see if the
    # data is backed by a "DataFrameScan" instead of a "Scan" of a file:
    # https://github.com/pola-rs/polars/issues/9771
    elif isinstance(df, pl.DataFrame):
        in_memory = True
        df = df.lazy()
        df = _df_to_polars(df, index=index)
    elif isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df, include_index=index is not NO_INDEX)
        df = _df_to_polars(df, index=index)
    else:
        raise ValueError(f'{df.__class__} not supported')

    if in_memory:
        _polars_declare_in_memory(df)

    return df


def _df_to_pandas(df, index):
    if isinstance(df, pd.DataFrame):
        return df
    else:
        assert isinstance(df, pl.LazyFrame)
        index = _polars_index_col(df, index)
        schema = df.collect_schema()
        has_time_index = index == 'Time' and schema[index].is_temporal()

        df = df.with_columns(
            cs.duration().dt.total_nanoseconds() * 1e-9
        )
        df = df.collect()

        # Make sure we get nullable dtypes:
        # https://arrow.apache.org/docs/python/pandas.html
        dtype_mapping = {
            pyarrow.int8(): pd.Int8Dtype(),
            pyarrow.int16(): pd.Int16Dtype(),
            pyarrow.int32(): pd.Int32Dtype(),
            pyarrow.int64(): pd.Int64Dtype(),
            pyarrow.uint8(): pd.UInt8Dtype(),
            pyarrow.uint16(): pd.UInt16Dtype(),
            pyarrow.uint32(): pd.UInt32Dtype(),
            pyarrow.uint64(): pd.UInt64Dtype(),
            pyarrow.bool_(): pd.BooleanDtype(),
            pyarrow.string(): pd.StringDtype(),
        }
        df = df.to_pandas(types_mapper=dtype_mapping.get)
        if index is not None:
            df.set_index(index, inplace=True)

        # Nullable dtypes are still not supported everywhere, so cast back to a
        # non-nullable dtype in cases where there is no null value:
        # https://github.com/holoviz/holoviews/issues/6142
        dtypes = {
            col: dtype.type
            for col in df.columns
            if getattr(
                (dtype := df[col].dtype),
                'na_value',
                None
            ) is pd.NA and not df[col].isna().any()
        }
        df = df.astype(dtypes, copy=False)

        if has_time_index:
            # Round trip polars -> pandas -> polars can be destructive as polars
            # will store timestamps at nanosecond precision in an integer. This
            # will wipe any sub-nanosecond difference in values, possibly leading
            # to duplicate timestamps.
            df.index = series_update_duplicates(df.index.to_series())

        return df


def _df_to(df, fmt, index=None):

    if isinstance(df, pd.DataFrame):
        df = _pandas_cleanup_df(df)

        if index is None:
            index = df.index.name
            # Default index in pandas, e.g. when using reset_index(). In that
            # case, there is no reason to include that index in anything we we
            # convert it to.
            if index is None and isinstance(df.index, pd.RangeIndex):
                index = NO_INDEX
        elif index is NO_INDEX:
            assert df.index.name is None
            assert isinstance(df.index, pd.RangeIndex)
        else:
            assert index == df.index.name

    if fmt == 'pandas':
        return _df_to_pandas(df, index=index)
    elif fmt == 'polars-lazyframe':
        # Note that this is not always a no-op even if the input is already a
        # LazyFrame, so it's important this does not get "optimized away".
        return _df_to_polars(df, index=index)
    else:
        raise ValueError(f'Unknown format {fmt}')


def _pandas_cleanup_df(df):
    assert isinstance(df, pd.DataFrame)

    # Ensure we only have string column names, as it is the only type that will
    # survive library conversions and serialization to parquet
    assert all(isinstance(col, str) for col in df.columns)

    # We need an index name if it's not just a default RangeIndex, otherwise we
    # cannot convert the dataframe to polars.
    assert isinstance(df.index, pd.RangeIndex) or df.index.name is not None

    # This will not survive conversion between dataframe types
    df.columns.name = None

    return df


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
        except KeyError as e:
            raise AttributeError(f'Unknown method name: {attr}') from e
        else:
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
def series_refit_index(series, start=None, end=None, window=None, clip_window=True):
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

    .. note:: If ``end`` is past the end of the data, the last row will
        be duplicated so that we can have a start and end index at the right
        location, without moving the point at which the transition to the last
        value happened. This also allows plotting series with only one item
        using matplotlib, which would otherwise be impossible.

    :param clip_window: Passed down to :func:`series_refit_index`.
    """
    window = _make_window(start, end, window)
    return _pandas_refit_index(series, window)


@DataFrameAccessor.register_accessor
def df_refit_index(df, start=None, end=None, window=None):
    """
    Same as :func:`series_refit_index` but acting on :class:`pandas.DataFrame`
    """
    window = _make_window(start, end, window)

    return _dispatch(
        _polars_refit_index,
        _pandas_refit_index,
        df, window
    )


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

        # Pandas chokes on common iterables like dict key views, so spoon feed
        # it a list
        signal_cols = list(signal_cols)

        # Avoid this warning:
        # FutureWarning: In a future version of pandas, a length 1 tuple will
        # be returned when iterating over a groupby with a grouper equal to a
        # list of length 1. Don't supply a list with a single grouper to avoid
        # this warning.
        if len(signal_cols) == 1:
            _signal_cols = signal_cols[0]
        else:
            _signal_cols = signal_cols

        for group, signal in df.groupby(_signal_cols, observed=True, sort=False, group_keys=False):
            # When only one column is looked at, the group is the value instead of
            # a tuple of values
            if isinstance(group, tuple) :
                cols_val = dict(zip(signal_cols, group))
            else:
                cols_val = {signal_cols[0]: group}

            if window:
                signal = df_refit_index(signal, window=window)
            yield (cols_val, signal)


def _polars_refit_index(data, window):
    # TODO: maybe expose that as a param
    index = _polars_index_col(data, index='Time')
    start, end = _polars_duration_window(window)

    index_col = pl.col(index)
    # Ensure the data is sorted, which should be free if they already are.
    data = data.sort(index_col)
    data = _polars_window(data, window, method='pre', col=index)

    if start is not None:
        data = data.with_columns(
            # Only advance the beginning of the data, never move it in the
            # past. Otherwise, we "invent" a value for the signal that did
            # not exist, leading to various wrong results.
            index_col.clip(lower_bound=start)
        )

    if end is not None:
        data = pl.concat([
            data.with_columns(
                index_col.clip(upper_bound=end)
            ),
            data.last().with_columns(
                end.alias(index)
            )
        ])

        # If it turns out the last value of the index was already "end" or if
        # "end" had a lower value than the unclipped index, we get rid of all
        # the excess rows.
        data = data.filter(
            (index_col != end) | (index_col != index_col.shift(1))
        )

    return data


def _pandas_refit_index(data, window):
    if data.empty:
        raise ValueError('Cannot refit the index of an empty dataframe or series')

    start, end = window
    if end is None:
        duplicate_last = False
    else:
        duplicate_last = end > data.index[-1]
    data = _pandas_window(data, window, method='pre')

    if data.empty:
        return data

    # When the end is after the end of the data, duplicate the last row so we
    # can push it to the right as much as we want without changing the point at
    # which the transition to that value happened
    if duplicate_last:
        data = pd.concat([data, data.iloc[-1:]])
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

    prev_df = df.loc[:start]
    middle_df = df.loc[start:end]

    # Tweak the closest previous event to include it in the slice
    if not prev_df.empty and start not in middle_df.index:
        res_df = pd.concat([res_df, prev_df.tail(1)])
        res_df.index = [start]
        e1 = end

        if not middle_df.empty:
            e1 = middle_df.index[0]

        res_df[column] = min(e1 - start, end - start)

    if not middle_df.empty:
        res_df = pd.concat([res_df, middle_df])
        if end in res_df.index:
            # e_last and s1 collide, ditch e_last
            res_df = res_df.drop([end])
        else:
            # Fix the delta for the last row
            delta = min(end - res_df.index[-1], res_df[column].iloc[-1])
            res_df.at[res_df.index[-1], column] = delta

    return res_df


@DataFrameAccessor.register_accessor
def df_filter(df, filter_columns, exclude=False):
    """
    Filter the content of a dataframe.

    :param df: DataFrame to filter
    :type df: pandas.DataFrame

    :param filter_columns: Dict of `{"column": value)` that rows has to match
        to be selected.
    :type filter_columns: dict(str, object)

    :param exclude: If ``True``, the matching rows will be excluded rather than
        selected.
    :type exclude: bool
    """
    if filter_columns:
        key = functools.reduce(
            operator.and_,
            (
                df[col] == val
                for col, val in filter_columns.items()
            )
        )
        return df[~key if exclude else key]
    else:
        if exclude:
            return df
        else:
            return df_make_empty_clone(df)



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

    df_list = list(df_list)
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

    if any(
        not (df1.columns.intersection(df2.columns)).empty
        for (df1, df2)  in itertools.combinations(df_list, 2)
    ):
        df = pd.concat(df_list)
        df.sort_index(inplace=True)
        return df
    else:
        df1, *other_dfs = df_list
        return df1.join(other_dfs, how='outer')


@DataFrameAccessor.register_accessor
def df_delta(pre_df, post_df, group_on=None):
    """
    pre_df and post_df containing paired/consecutive events indexed by time,
    df_delta() merges the two dataframes and adds a ``delta`` column
    containing the time spent between the two events.
    A typical usecase would be adding pre/post events at the entry/exit of a
    function.

    Rows from ``pre_df`` and ``post_df`` are grouped by the ``group_on``
    columns.
    E.g.: ``['pid', 'comm']`` to group by task.
    Except columns listed in ``group_on``, ``pre_df`` and ``post_df`` must
    have columns with different names.

    Events that cannot be paired are ignored.

    :param pre_df: Dataframe containing the events that start a record.
    :type pre_df: pandas.DataFrame

    :param post_df: Dataframe containing the events that end a record.
    :type post_df: pandas.DataFrame

    :param group_on: Columns used to group ``pre_df`` and ``post_df``.
        E.g.: This would be ``['pid', 'comm']`` to group by task.
    :type group_on: list(str)

    :returns: a :class:`pandas.DataFrame` indexed by the ``pre_df`` dataframe
        with:

        * All the columns from the ``pre_df`` dataframe.
        * All the columns from the ``post_df`` dataframe.
        * A ``delta`` column (duration between the emission of a 'pre' event
            and its consecutive 'post' event).
    """
    pre_df = pre_df.copy(deep=False)
    post_df = post_df.copy(deep=False)

    # Tag the rows to remember from which df they are coming from.
    pre_df["is_pre"] = True
    post_df["is_pre"] = False

    # Merge on columns common to the two dfs to avoid overlapping of names.
    on_col = sorted(pre_df.columns.intersection(post_df.columns))

    # Merging on nullable types converts columns to object.
    # Merging on non-nullable types converts integer/boolean to float.
    # Thus, let the on_col non-nullable and converts the others to nullable.
    pre_df_cols = sorted(set(pre_df) - set(on_col))
    post_df_cols = sorted(set(post_df) - set(on_col))
    pre_df[pre_df_cols] = df_convert_to_nullable(pre_df[pre_df_cols])
    post_df[post_df_cols] = df_convert_to_nullable(post_df[post_df_cols])

    # Merge. Don't allow column renaming.
    df = pd.merge(pre_df, post_df, left_index=True, right_index=True, on=on_col,
                  how='outer', suffixes=(False, False))

    # Save and replace the index name by a tmp name to avoid a clash
    # with column names.
    index_name = df.index.name
    index_tmp_name = uuid.uuid4().hex
    df.index.name = index_tmp_name
    df.reset_index(inplace=True)

    # In each group, search for a faulty sequence (where pre/post events are
    # not interleaving, e.g. pre1->pre2->post1->post2).
    if group_on:
        grouped = df.groupby(group_on, observed=True, sort=False, group_keys=False)
    else:
        grouped = df
    if grouped['is_pre'].transform(lambda x: x == x.shift()).any():
        raise ValueError('Unexpected sequence of pre and post event (more than one "pre" or "post" in a row)')

    # Create the 'delta' column and add the columns from post_df
    # in the rows coming from pre_df.
    new_columns = dict(
        delta=grouped[index_tmp_name].transform(lambda time: time.diff().shift(-1)),
    )
    new_columns.update({col: grouped[col].shift(-1) for col in post_df_cols})
    df = df.assign(**new_columns)

    df.set_index(index_tmp_name, inplace=True)
    df.index.name = index_name

    # Only keep the rows from the pre_df, they have all the necessary info.
    df = df.loc[df["is_pre"]]
    # Drop the rows from pre_df with not matching row from post_df.
    df.dropna(inplace=True)

    df.drop(columns=["is_pre"], inplace=True)

    return df


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
    :type x: pandas.DataFrame or None

    :param order: Order of the derivative (1 is speed, 2 is acceleration etc).
    :type order: int
    """
    x = _resolve_x(y, x)

    for _ in range(order):
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
    :type x: pandas.DataFrame or None

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

    * Step: Post

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

    * Step: Pre

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
        raise ValueError(f'Unsupported "sign": {sign}')

    if method == "rect":
        if len(x) <= 1:
            raise ValueError('Cannot integrate with less than 2 points')
        else:
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
        raise ValueError(f'Unsupported integration method: {method}')


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
    :type x: pandas.DataFrame or None

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

    :param clip_window: Only ``True`` value is now allwed: clip the requested
        window to the bounds of the index, otherwise raise exceptions if the
        window is too large.
    :type clip_window: bool

    :param method: Choose how edges are handled:

       * `inclusive`: When no exact match is found, include both the previous
         and next values around the window.
       * `exclusive`: When no exact match is found, only index values within
         the range are selected. This is the default pandas float slicing
         behavior.
       * `nearest`: Not supported with :mod:`polars` objects: when no exact
         match is found, take the nearest index value.
       * `pre`: When no exact match is found, take the previous index value.
       * `post`: When no exact match is found, take the next index value.

    .. note:: The index of `series` must be monotonic and without duplicates.
    """
    if not clip_window:
        raise ValueError(f'Only clip_window=True is supported')

    return _pandas_window(series, window, method)


def _polars_window(data, window, method, col=None):
    # TODO: relax that
    assert isinstance(data, pl.LazyFrame)

    if col is None:
        col = _polars_index_col(data, index='Time')

    col = pl.col(col)
    start, end = window

    def pre():
        if start is None:
            filter_ = col <= end
        else:
            if end is None:
                filter_ = col > start
            else:
                filter_ = col.is_between(
                    lower_bound=start,
                    upper_bound=end,
                    closed='right',
                )

            filter_ = filter_ | filter_.shift(-1)

        return filter_

    def post():
        if end is None:
            filter_ = col >= start
        else:
            if start is None:
                filter_ = col < end
            else:
                filter_ = col.is_between(
                    lower_bound=start,
                    upper_bound=end,
                    closed='left',
                )

            filter_ = filter_ | filter_.shift(+1)
        return filter_

    if start is None and end is None:
        filter_ = True
    else:
        start, end = _polars_duration_window((start, end))

        if method == 'exclusive':
            if start is None:
                filter_ = col <= end
            elif end is None:
                filter_ = col >= start
            else:
                filter_ = col.is_between(
                    lower_bound=start,
                    upper_bound=end,
                    closed='both',
                )

        elif method == 'inclusive':
            filter_ = pre() | post()
        elif method == 'pre':
            filter_ = pre()
        elif method == 'post':
            filter_ = post()
        else:
            raise ValueError(f'Slicing method not supported: {method}')

    return data.filter(filter_)


def _pandas_window(data, window, method):
    """
    ``data`` can either be a :class:`pandas.DataFrame` or :class:`pandas.Series`

    .. warning:: This function assumes ``data`` has a sorted index.
    """

    index = data.index
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
    # Window is on the right
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

    if None not in window and window[0] > window[1]:
        raise KeyError(f'The window starts after its end: {window}')

    if method == 'inclusive':
        method = ('ffill', 'bfill')

    elif method == 'exclusive':
        # Default slicing behaviour of pandas' float index is to be exclusive,
        # so we can use that knowledge to enable a fast path.
        if data.index.dtype.kind == 'f':
            return data[slice(*window)]

        method = ('bfill', 'ffill')

    elif method == 'nearest':
        method = ('nearest', 'nearest')

    elif method == 'pre':
        method = ('ffill', 'ffill')

    elif method == 'post':
        method = ('bfill', 'bfill')

    else:
        raise ValueError(f'Slicing method not supported: {method}')

    sides = ('left', 'right')

    window = [
        _get_loc(index, x, method=method, side=side) if x is not None else None
        for x, method, side in zip(window, method, sides)
    ]
    window = window[0], (window[1] + 1)

    return data.iloc[slice(*window)]


def _get_loc(index, x, method, side):
    """
    Emulate :func:`pandas.Index.get_loc` behavior with the much faster
    :func:`pandas.Index.searchsorted`.

    .. warning:: Passing a non-sorted index will destroy performance.
    """

    # Not a lot of use for nearest, so fall back on the slow but easy to use get_loc()
    #
    # Also, if the index is not sorted, we need to fall back on the slow path
    # as well. Checking is_monotonic is cheap so it's ok to do it here.
    if method == 'nearest' or not index.is_monotonic_increasing:
        return index.get_indexer([x], method=method)[0]
    else:
        if index.empty:
            raise KeyError(x)
        # get_loc() also raises an exception in these case
        elif method == 'ffill' and x < index[0]:
            raise KeyError(x)
        elif method == 'bfill' and x > index[-1]:
            raise KeyError(x)

        loc = index.searchsorted(x, side=side)
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
    if not clip_window:
        raise ValueError(f'Only clip_window=True is supported')

    return _dispatch(
        _polars_window,
        _pandas_window,
        df, window, method
    )


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
    if not clip_window:
        raise ValueError(f'Only clip_window=True is supported')

    return _dispatch(
        _polars_window_signals,
        _pandas_window_signals,
        df, window, signals, compress_init
    )

def _polars_window_signals(df, window, signals, compress_init):
    index = _polars_index_col(df, index='Time')
    schema = df.collect_schema()
    assert schema[index].is_temporal()

    start, end = _polars_duration_window(window)

    if start is not None:
        if end is None:
            post_filter = pl.col(index) >= start
            pre_filter = pl.lit(True)
        else:
            post_filter = pl.col(index).is_between(
                lower_bound=start,
                upper_bound=end,
                closed='both'
            )
            pre_filter = (pl.col(index) < end)

        pre_filter &= ~post_filter

        post_df = df.filter(post_filter)
        pre_df = df.filter(pre_filter)

        signals_init = [
            pre_df.group_by(fields).last()
            for signal in set(signals)
            if (fields := signal.fields)
        ]

        if signals_init:
            pre_df = pl.concat(
                signals_init,
                how='diagonal',
            )

            if compress_init:
                first_row = post_df.select(index).head(1).collect()
                try:
                    first_time = first_row.item()
                except ValueError:
                    pass
                else:
                    pre_df.with_columns(Time=pl.lit(first_time))

            # We could have multiple signals for the same event, so we want to
            # avoid duplicate events occurrences.
            pre_df = pre_df.unique()
            pre_df = pre_df.sort(index)

            return pl.concat(
                [
                    pre_df,
                    post_df,
                ],
                how='diagonal',
            )

    df = _polars_window(
        df,
        window=window,
        method='pre',
    )
    return df

def _pandas_window_signals(df, window, signals, compress_init=False):

    def before(x):
        return x - 1e-9

    windowed_df = df_window(df, window, method='pre')

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
        # is after the end of the dataframe, or if the window starts after its
        # end.
        _window = (extra_window[1], windowed_df.index[-1])
        windowed_df = df_window(windowed_df, _window, method='post')
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
        loc = _get_loc(signal_df.index, window[0], method='ffill', side='left')
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
                    prev = curr
                    while int(prev * 1e9) == int(curr * 1e9):
                        curr = before(curr)
                    yield curr


            # If windowed_df is empty, we take the last bit right before the
            # beginning of the window
            try:
                start = windowed_df.index[0]
            except IndexError:
                start = extra_df.index[-1]

            index = list(smallest_increment(start, len(init_df)))
            index = pd.Index(reversed(index), dtype='float64')
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
    def get_period(series):
        return pd.Series(series.index).diff().min()
    period = min(get_period(ref), get_period(to_align))
    num = math.ceil((end - start) / period)
    new_index = pd.Index(np.linspace(start, end, num), dtype='float64')

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
    Filter a dataframe using a list of :class:`lisa.analysis.tasks.TaskID`

    :param task_ids: List of task IDs to filter
    :type task_ids: list(lisa.analysis.tasks.TaskID)

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

    return _dispatch(
        _polars_filter_task_ids,
        _pandas_filter_task_ids,
        df,

        task_ids=task_ids,
        pid_col=pid_col,
        comm_col=comm_col,
        invert=invert,
        comm_max_len=comm_max_len,
    )

def _pandas_filter_task_ids(df, task_ids, pid_col, comm_col, invert, comm_max_len):
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

    tasks_filters = list(map(make_filter, task_ids))
    if tasks_filters:
        # Combine all the task filters with OR
        tasks_filter = functools.reduce(operator.or_, tasks_filters)

        if invert:
            tasks_filter = ~tasks_filter

        return df[tasks_filter]
    else:
        return df if invert else df.iloc[0:0]


def _polars_filter_task_ids(df, task_ids, pid_col, comm_col, invert, comm_max_len):
    def make_filter(task_id):
        if pid_col and task_id.pid is not None:
            pid = (pl.col(pid_col) == pl.lit(task_id.pid))
        else:
            pid = pl.lit(True)

        if comm_col and task_id.comm is not None:
            comm = (pl.col(comm_col) == pl.lit(task_id.comm[:comm_max_len]))
        else:
            comm = pl.lit(True)

        return pid & comm

    tasks_filters = list(map(make_filter, task_ids))
    # Combine all the task filters with OR
    tasks_filter = functools.reduce(operator.or_, tasks_filters, pl.lit(False))
    if invert:
        tasks_filter = ~tasks_filter

    return df.filter(tasks_filter)


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
        raise ValueError(f'Unsupported kind: {kind}')

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

    first_val = series.iat[0]
    # Remove constant values, otherwise they would be accounted in both max and
    # min, which can bias the result
    series = series_deduplicate(series, keep='first', consecutives=True)

    # If the series was constant, just return that constant
    if series.empty:
        return first_val
    else:
        maxs = series_local_extremum(series, kind='max')
        mins = series_local_extremum(series, kind='min')

        maxs_mean = series_mean(maxs)
        mins_mean = series_mean(mins)

        return (maxs_mean - mins_mean) / 2 + mins_mean

# Keep an alias in place for compatibility
@deprecate(replaced_by=series_envelope_mean, deprecated_in='2.0', removed_in='4.0')
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
        be of type :class:`pandas.Index` (float64), in nanoseconds. Disabling is
        recommended if the index is not used by ``func`` since it will remove
        the need for a conversion.
    :type window_float_index: bool
    """
    orig_index = series.index

    # Wrap the func to turn the index into nanosecond Float64Index
    if window_float_index:
        def func(s, func=func):
            # pylint: disable=function-redefined
            s.index = s.index.astype('int64') * 1e-9
            return func(s)

    # Use a timedelta index so that rolling gives time-based results
    index = pd.to_timedelta(orig_index, unit='s')
    series = pd.Series(series.array, index=index)

    window_ns = int(window * 1e9)
    rolling_window = f'{window_ns}ns'
    values = series.rolling(rolling_window).apply(func, raw=False).values

    if center:
        new_index = orig_index - (window / 2)
    else:
        new_index = orig_index

    return pd.Series(values, index=new_index)


def _pandas_find_unique_bool_vector(data, cols, all_col, keep):
    if keep == 'first':
        shift = 1
    elif keep == 'last':
        shift = -1
    elif keep is None:
        shift = 1
    else:
        raise ValueError(f'Unknown keep value: {keep}')

    dedup_data = data[cols] if cols else data
    # Unique values will be True, duplicate False
    cond = dedup_data != dedup_data.shift(shift)
    cond = cond.fillna(True)
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


def _pandas_deduplicate(data, keep, consecutives, cols, all_col):
    if consecutives:
        return data.loc[_pandas_find_unique_bool_vector(data, cols, all_col, keep)]
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
    return _pandas_deduplicate(series, keep=keep, consecutives=consecutives, cols=None, all_col=True)


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
    return _pandas_deduplicate(df, keep=keep, consecutives=consecutives, cols=cols, all_col=all_col)


@DataFrameAccessor.register_accessor
def series_update_duplicates(series, func=None):
    """
    Update a given series to avoid duplicated values.

    :param series: Series to act on.
    :type series: pandas.Series

    :param func: The function used to update the column. It must take a
        :class:`pandas.Series` of duplicated entries to update as parameters,
        and return a new :class:`pandas.Series`. The function will be called as
        long as there are remaining duplicates. If ``None``, the column is
        assumed to be floating point number of seconds and will be updated so
        that the no duplicated timestamps exist once translated to an integer
        number of nanoseconds.
    :type func: collections.abc.Callable or None
    """
    if func:
        def preprocess(series):
            return series
    else:
        def func(series):
            return series + 1e-9

        def preprocess(series):
            return (series * 1e9).astype('int64')

    def get_duplicated(series):
        # Keep the first, so we update the second duplicates
        locs = preprocess(series).duplicated(keep='first')
        return locs, series.loc[locs]

    # Update the values until there is no more duplication
    duplicated_locs, duplicated = get_duplicated(series)
    while not duplicated.empty:
        updated = func(duplicated)
        # Change the values at the points of duplication. Otherwise, take the
        # initial value
        series.loc[duplicated_locs] = updated
        duplicated_locs, duplicated = get_duplicated(series)

    return series


@DataFrameAccessor.register_accessor
def df_update_duplicates(df, col=None, func=None, inplace=False):
    """
    Same as :func:`series_update_duplicates` but on a :class:`pandas.DataFrame`.

    :param df: Dataframe to act on.
    :type df: pandas.DataFrame

    :param col: Column to update. If ``None``, the index is used.
    :type col: str or None

    :param func: See :func:`series_update_duplicates`.
    :type func: collections.abc.Callable or None

    :param inplace: If ``True``, the passed dataframe will be modified.
    :type inplace: bool
    """
    use_index = col is None

    series = df.index.to_series() if use_index else df[col].copy()
    series = series_update_duplicates(series, func=func)

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
    duplicates_to_remove = ~_pandas_find_unique_bool_vector(df, cols, all_col, keep='first')
    # Then get only the first row in a run of duplicates
    first_duplicates = (~duplicates_to_remove) & duplicates_to_remove.shift(-1, fill_value=False)
    # We only kept them separate with keep='first' to be able to detect
    # correctly the beginning of a duplicate run to get a group ID, so now we
    # merge them
    duplicates = duplicates_to_remove | first_duplicates

    # Assign the group ID to each member of the group
    df.loc[first_duplicates, 'duplicate_group'] = first_duplicates.loc[first_duplicates].index
    df.loc[duplicates, 'duplicate_group'] = df.loc[duplicates, 'duplicate_group'].ffill()

    # For some reasons GroupBy.apply() will raise a KeyError if the index is a
    # Float64Index, go figure ...
    index = df.index
    df.reset_index(drop=True, inplace=True)

    # Apply the function to each group, and assign the result to the output
    # Note that we cannot use GroupBy.transform() as it currently cannot handle
    # NaN groups.
    output = df.groupby('duplicate_group', sort=False, as_index=True, group_keys=False, observed=True)[df.columns].apply(func)
    if not output.empty:
        init_df[output_col].update(output)

    # Ensure the column is created if it does not exists yet
    try:
        init_df[output_col]
    except KeyError:
        init_df[output_col] = np.nan
    else:
        # Restore the index that we had to remove for apply()
        df.index = index
        try:
            fill = df[output_col]
        except KeyError:
            pass
        else:
            init_df[output_col] = df[output_col].fillna(fill)

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
    return _dispatch(
        _polars_add_delta,
        _pandas_add_delta,
        df, col, src_col, window, inplace
    )


def _pandas_add_delta(df, col, src_col, window, inplace):
    use_refit_index = window and not inplace

    if use_refit_index:
        df = df_refit_index(df, window=window)

    src = df[src_col] if src_col else df.index.to_series()
    delta = src.diff().shift(-1)

    # When use_refit_index=True, the last delta will already be sensible
    if not use_refit_index and window:
        _, end = window
        if end is not None:
            new_end = end - src.iloc[-1]
            new_end = new_end if new_end > 0 else 0
            delta.iloc[-1] = new_end

    if not inplace:
        df = df.copy()

    df[col] = delta

    return df


def _polars_add_delta(df, col, src_col, window, inplace):
    assert not inplace

    if window:
        df = df_refit_index(df, window=window)

    src_col = src_col if src_col is not None else _polars_index_col(df)

    return df.with_columns(
        pl.col(src_col).diff().shift(-1).alias(col)
    )


def series_combine(series_list, func, fill_value=None):
    """
    Same as :meth:`pandas.Series.combine` on a list of series rather than just
    two.
    """
    return _pandas_combine(series_list, func, fill_value)


def df_combine(series_list, func, fill_value=None):
    """
    Same as :meth:`pandas.DataFrame.combine` on a list of series rather than just
    two.
    """
    return _pandas_combine(series_list, func, fill_value)


def _pandas_combine(datas, func, fill_value=None):
    state = datas[0]
    for data in datas[1:]:
        state = state.combine(data, func=func, fill_value=fill_value)

    return state


def series_dereference(series, sources, inplace=False, method='ffill'):
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

    :param method: ``sources`` is reindexed so that it shares the same index
        as ``series``. ``method`` is forwarded to :meth:`pandas.Series.reindex`.
    :type method: str
    """
    def reindex(values):
        # Skip the reindex if they are in the same dataframe
        if values.index is not series.index:
            values = values.reindex(series.index, method=method)
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


def df_dereference(df, col, pointer_col=None, sources=None, inplace=False, **kwargs):
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

    :Variable keyword arguments: Forwarded to :func:`series_dereference`.
    """
    pointer_col = pointer_col or col
    sources = df if sources is None else sources
    df = df if inplace else df.copy(deep=False)
    df[col] = series_dereference(df[pointer_col], sources, inplace=inplace, **kwargs)
    return df


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
    @deprecate(msg='No new signals will be added to this list, use explicit signal description where appropriate in the Trace API', deprecated_in='3.0', removed_in='4.0')
    def from_event(cls, *args, **kwargs):
        return cls._from_event(*args, **kwargs)

    # Keep a warning-free private method for backward compat pandas code that
    # will one day be removed.
    @classmethod
    def _from_event(cls, event, fields=None):
        """
        Return list of :class:`SignalDesc` for the given event.

        The hand-coded list is used first, and then some generic heuristics are
        used to detect per-cpu and per-task signals.
        """

        # For backward compatibility, so that we still get signal descriptors
        # for traces before the events from the lisa module got renamed to
        # lisa__<event>
        from lisa.trace import _NamespaceTraceView
        events = _NamespaceTraceView._do_expand_namespaces(event, namespaces=('lisa__', None))

        for event in events:
            try:
                return cls._SIGNALS_MAP[event]
            except KeyError:
                continue

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


@SeriesAccessor.register_accessor
def series_convert(series, dtype, nullable=None):
    """
    Convert a :class:`pandas.Series` with a best effort strategy.

    Nullable types may be used if necessary and possible, otherwise ``object``
    dtype will be used.

    :param series: Series of another type than the target one. Strings are
        allowed.
    :type series: pandas.Series

    :param dtype: dtype to convert to. If it is a string (like ``"uint8"``), the
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

    :param nullable: If:

        - ``True``, use the nullable dtype equivalent of the requested dtype.
        - ``None``, use the equivalent nullable dtype if there is any missing
            data, otherwise a non-nullable dtype will be used for lower
            memory consumption.
    :type nullable: bool or None
    """

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

    if series.dtype.name == dtype and   \
            not (nullable and dtype in nullable_dtypes):
        # If there is a conversion to a nullable dtype, don't skip.
        return series

    def to_object(x):
        x = x.astype('object', copy=True)
        # If we had any pandas <NA> values, they need to be turned into None
        # first, otherwise pyarrow will choke on them
        x.loc[x.isna()] = None
        return x

    astype = lambda dtype: lambda x: x.astype(dtype, copy=False)
    make_convert = lambda dtype: lambda x: series_convert(x, dtype,
                                                            nullable=nullable)
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
            except Exception: # pylint: disable=broad-except
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
            lambda series: series.astype(object).apply(convert)
        )

    # Then try with a nullable type.
    # Floats are already nullable so we don't need to do anything
    elif is_bool or is_int:

        # Bare nullable dtype

        # Already nullable
        if dtype[0].isupper():
            nullable_type = dtype
        else:
            nullable_type = nullable_dtypes[dtype]
        to_nullable = astype(nullable_type)

        if nullable:
            # Only allow nullable dtype conversion.
            from_numeric = Alternative(
                to_nullable
            )
        elif nullable is None:
            # (nullable == None): default behaviour, try both.
            from_numeric = Alternative(
                basic,
                to_nullable
            )
        else:
            # Do not convert to nullable dtype unless the user specified one.
            from_numeric = Alternative(
                basic
            )

        if is_int:
            parse = Alternative(
                from_numeric,
                # Maybe we were trying to parse some strings that turned out to
                # need to go through the Python int constructor to be parsed,
                # so do that first
                Pipeline(
                    Alternative(
                        # Parse as integer
                        make_convert(int),
                        # Parse as hex int
                        make_convert(functools.partial(int, base=16))
                    ),
                    Alternative(
                        from_numeric,
                        # Or just leave the output as it is if nothing else can be
                        # done, as we already have 'object' of an integer type
                        to_object,
                        name='convert parser output',
                    ),
                    name='parse',
                ),
            )
        elif is_bool:
            parse = Alternative(
                Pipeline(
                    # Convert to int first, so that input like b'0' is
                    # converted to int before being interpreted as a bool,
                    # avoiding turning it into "True"
                    make_convert(int),
                    from_numeric,
                    name='parse as int',
                ),
                # If that failed, just feed the input to Python's bool()
                # builtin, and then convert to the right dtype to avoid ending
                # up with "object" dtype and bool values
                Pipeline(
                    make_convert(bool),
                    from_numeric,
                    name='parse as bool',
                )
            )

        else:
            assert False

        pipelines.append(parse)

    elif dtype == 'string':

        # Sadly, pandas==1.1.1 (and maybe later) series.astype('string') turns
        # b'hello' into "b'hello'" instead of "hello", so basic decoder becomes
        # unusable
        if (
            series.dtype.name == 'object' and
            series.astype(object).apply(isinstance, args=(bytes,)).any()
        ):
            string_basic = None
            # Handle mixed dtypes
            str_basic = lambda x : x.astype(object).apply(
                lambda x: x.decode('ascii') if isinstance(x, bytes) else str(x),
            )
        else:
            string_basic = basic
            str_basic = make_convert(str)

        # Faster than Series.str.decode()
        basic_decode = lambda x : x.astype(object).apply(bytes.decode, args=('ascii',))

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
            x = x.cat.rename_categories(new_cat)
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


@DataFrameAccessor.register_accessor
def df_convert_to_nullable(df):
    """
    Convert the columns of the dataframe to their equivalent nullable dtype,
    when possible.

    :param df: The dataframe to convert.
    :type df: pandas.DataFrame

    :returns: The dataframe with converted columns.
    """
    def _series_convert(column):
        return series_convert(column, str(column.dtype), nullable=True)

    return df.apply(_series_convert, raw=False)


@DataFrameAccessor.register_accessor
def df_find_redundant_cols(df, col, cols=None):
    """
    Find the columns that are redundant to ``col``, i.e. that can be computed
    as ``df[x] = df[col].map(dict(...))``.

    :param df: Dataframe to analyse.
    :type df: pandas.DataFrame

    :param col: Reference column
    :type col: str

    :param cols: Columns to restrict the analysis to. If ``None``, all columns
        are used.
    :type cols: str or None
    """
    grouped = df.groupby(col, observed=True, group_keys=False)
    cols = cols or (set(df.columns) - {col})
    return {
        _col: dict(map(
            lambda x: (x[0], x[1][0]),
            series.items()
        ))
        for _col, series in (
            (
                _col,
                grouped[_col].unique()
            )
            for _col in cols
            if (grouped[_col].nunique() == 1).all()
        )
    }


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
    SignalDesc('sched_cpu_capacity', ['cpu']),
    SignalDesc('cpu_frequency', ['cpu_id']),
    SignalDesc('userspace@cpu_frequency_devlib', ['cpu_id']),
    SignalDesc('sched_compute_energy', ['comm', 'pid']),

    SignalDesc('clk_set_rate', ['name']),
    SignalDesc('clk_enable', ['name']),
    SignalDesc('clk_disable', ['name']),

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

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
