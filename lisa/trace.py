# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

""" Trace Parser Module """

import shutil
import uuid
import sys
import re
import gc
import math
import abc
import copy
import io
import os
import os.path
import json
import warnings
import inspect
import shlex
import contextlib
import tempfile
from functools import reduce, wraps
from collections.abc import Iterable, Set, Mapping, Sequence
from collections import namedtuple
from operator import itemgetter
from numbers import Number, Integral, Real
import multiprocessing
import textwrap

import numpy as np
import pandas as pd
import pyarrow.lib

import trappy
import devlib
from devlib.target import KernelVersion

import lisa.utils
from lisa.utils import Loggable, HideExekallID, memoized, deduplicate, deprecate, nullcontext, measure_time, checksum, newtype
from lisa.conf import SimpleMultiSrcConf, KeyDesc, TopLevelKeyDesc, TypedList, Configurable
from lisa.datautils import df_split_signals, df_window, df_window_signals, SignalDesc, df_add_delta
from lisa.version import VERSION_TOKEN
from lisa.typeclass import FromString, IntListFromStringInstance


class TaskID(namedtuple('TaskID', ('pid', 'comm'))):
    """
    Unique identifier of a logical task in a :class:`Trace`.

    :param pid: PID of the task. ``None`` indicates the PID is not important.
    :type pid: int

    :param comm: Name of the task. ``None`` indicates the name is not important.
        This is useful to describe tasks like PID0, which can have multiple
        names associated.
    :type comm: str
    """

    # Prevent creation of a __dict__. This allows a more compact representation
    __slots__ = []

    def __init__(self, *args, **kwargs):
        # This happens when the number of saved PID/comms entries in the trace
        # is too low
        if self.comm == '<...>':
            raise ValueError('Invalid comm name "<...>", please increase saved_cmdlines_nr value on FtraceCollector')

    def __str__(self):
        if self.pid is not None and self.comm is not None:
            out = '{}:{}'.format(self.pid, self.comm)
        else:
            out = str(self.comm if self.comm is not None else self.pid)

        return '[{}]'.format(out)

    _STR_PARSE_REGEX = re.compile(r'\[?([0-9]+):([a-zA-Z0-9_-]+)\]?')


class TaskIDFromStringInstance(FromString, types=TaskID):
    """
    Instance of :class:`lisa.typeclass.FromString` for :class:`TaskID` type.
    """
    @classmethod
    def from_str(cls, string):
        try:
            pid = int(string)
            comm = None
        except ValueError:
            match = cls._STR_PARSE_REGEX.match(string)
            if match:
                pid = int(match.group(1))
                comm = match.group(2)
            else:
                pid = None
                comm = string

        return cls(pid=pid, comm=comm)

    @classmethod
    def get_format_description(cls, short):
        if short:
            return 'task ID'
        else:
            return textwrap.dedent("""
            Can be any of:
               * a PID
               * a task name
               * a PID (first) and a name (second): pid:name
            """).strip()


class TaskIDListFromStringInstance(FromString, types=TypedList[TaskID]):
    """
    Instance of :class:`lisa.typeclass.FromString` for lists :class:`TaskID` type.
    """
    @classmethod
    def from_str(cls, string):
        """
        The format is a comma-separated list of :class:`TaskID`.
        """
        from_str = FromString(TaskID).from_str
        return [
            from_str(string.strip())
            for string in string.split(',')
        ]

    @classmethod
    def get_format_description(cls, short):
        return 'comma-separated TaskIDs'


CPU = newtype(int, 'CPU')


class CPUListFromStringInstance(FromString, types=TypedList[CPU]):
    # Use the same implementation as for TypedList[int]
    from_str = IntListFromStringInstance.from_str

    @classmethod
    def get_format_description(cls, short):
        return FromString(TypedList[int]).get_format_description(short=short)


class TraceBase(abc.ABC):
    """
    Base class for common functionalities between :class:`Trace` and :class:`TraceView`
    """

    def __init__(self):
        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base
        from lisa.analysis.proxy import AnalysisProxy
        self.analysis = AnalysisProxy(self)

    @property
    def trace_state(self):
        """
        State of the trace object that might impact the output of dataframe
        getter functions like :meth:`Trace.df_events`.

        It must be hashable and serializable to JSON, so that it can be
        recorded when analysis methods results are cached to the swap.
        """
        return None

    @property
    def time_range(self):
        """
        Duration of that trace.
        """
        return self.end - self.start

    @property
    def window(self):
        """
        Same as ``(trace.start, trace.end)``.

        This is handy to pass to functions expecting a window tuple.
        """
        return (self.start, self.end)

    @abc.abstractmethod
    def get_view(self, window, **kwargs):
        """
        Get a view on a trace cropped time-wise to fit in ``window``

        :Variable keyword arguments: Forwarded to the contructor of the view.
        """
        pass

    def __getitem__(self, window):
        if not isinstance(window, slice):
            raise TypeError("Cropping window must be an instance of slice")

        if window.step is not None:
            raise ValueError("Slice step is not supported")

        return self.get_view((window.start, window.stop))

    @deprecate('Prefer adding delta once signals have been extracted from the event dataframe for correctness',
        deprecated_in='2.0',
        removed_in='2.1',
        replaced_by=df_add_delta,
    )
    def add_events_deltas(self, df, col_name='delta', inplace=True):
        """
        Store the time between each event in a new dataframe column

        This function assumes that at time [n] the event starts and at [n+1]
        the event stops, so the formula for the returned value is::

                  |        |            |
                  |        |            |
                  |        |            |
            ------+--------+------------+------
                [n-1]     [n]         [n+1]

            delta[n] = index[n+1] - index[n]

        :param df: The DataFrame to operate one
        :type df: pandas.DataFrame

        :param col_name: The name of the column to add
        :type col_name: str

        :param inplace: Whether to operate on the passed DataFrame, or to use
          a copy of it
        :type inplace: bool

        This method only really makes sense for events tracking an on/off state
        (e.g. overutilized, idle)
        """

        if col_name in df.columns:
            raise RuntimeError("Column {} is already present in the dataframe".
                               format(col_name))

        return df_add_delta(df, col=col_name, inplace=inplace, window=self.window)

    def df_all_events(self, events=None):
        """
        Provide a dataframe with an ``info`` column containing the textual
        human-readable representation of the events fields.

        :param events: List of events to include. If ``None``, all parsed
            events will be used.
        :type events: list(str) or None
        """
        if events is None:
            events = sorted(self.available_events)

        if not events:
            return pd.DataFrame({'info': []})

        max_event_name_len = max(len(event) for event in events)

        def make_info_row(row, event):
            fields = ' '.join(
                '{}={}'.format(key, value)
                for key, value in row.iteritems()
            )

            return '{:<{event_name_len}}: {}'.format(event, fields, event_name_len=max_event_name_len)

        def make_info_series(event):
            df = self.df_events(event)
            info = df.apply(make_info_row, axis=1, event=event)
            info.name = 'info'
            return info

        series_list = [
            make_info_series(event)
            for event in events
        ]

        series = pd.concat(series_list)
        series.sort_index(inplace=True)
        return pd.DataFrame({'info': series})


class TraceView(Loggable, TraceBase):
    """
    A view on a :class:`Trace`

    :param trace: The trace to trim
    :type trace: Trace

    :param clear_base_cache: Clear the cache of the base ``trace`` for non-raw
        data. This can release memory if the base trace is not going to be used
        anymore, apart as a data server for the view.
    :type clear_base_cache: bool

    :param window: The time window to base this view on
    :type window: tuple(float, float)

    :ivar base_trace: The original :class:`Trace` this view is based on.
    :ivar analysis: The analysis proxy on the trimmed down :class:`Trace`.

    :ivar start: The timestamp of the first trace event in the view (>= ``window[0]``)
    :ivar end: The timestamp of the last trace event in the view (<= ``window[1]``)

    You can substitute an instance of :class:`Trace` with an instance of
    :class:`TraceView`. This means you can create a view of a trimmed down trace
    and run analysis code/plots that will only use data within that window, e.g.::

      trace = Trace(...)
      view = trace.get_view((2, 4))

      # Alias for the above
      view = trace[2:4]

      # This will only use events in the (2, 4) time window
      df = view.analysis.tasks.df_tasks_runtime()

    **Design notes:**

      * :meth:`df_events` uses the underlying :meth:`lisa.trace.Trace.df_events`
        and trims the dataframe according to the given ``window`` before
        returning it.
      * ``self.start`` and ``self.end`` mimic the :class:`Trace` attributes but
        they are adjusted to match the given window. On top of this, this class
        mimics a regular :class:`Trace` using :func:`getattr`.
    """

    def __init__(self, trace, window, clear_base_cache=False):
        super().__init__()
        self.base_trace = trace

        # evict all the non-raw dataframes from the base cache, as they are
        # unlikely to be used anymore.
        if clear_base_cache:
            self.base_trace._cache.clear_all_events(raw=False)

        t_min, t_max = window
        self.start = t_min if t_min is not None else self.base_trace.start
        self.end = t_max if t_max is not None else self.base_trace.end

    @property
    def trace_state(self):
        return (self.start, self.end, self.base_trace.trace_state)

    def __getattr__(self, name):
        return getattr(self.base_trace, name)

    def df_events(self, event, **kwargs):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the sliced trace.

        :param event: Trace event name
        :type event: str

        :Variable keyword arguments: Forwarded to
            :meth:`lisa.trace.Trace.df_events`.
        """
        try:
            window = kwargs['window']
        except KeyError:
            window = (self.start, self.end)
        kwargs['window'] = window

        return self.base_trace.df_events(event, **kwargs)

    def get_view(self, window, **kwargs):
        start = self.start
        end = self.end

        if window[0]:
            start = max(start, window[0])

        if window[1]:
            end = min(end, window[1])

        return self.base_trace.get_view(window=(start, end), **kwargs)

# One might be tempted to make that a subclass of collections.abc.Set: don't.
# The problem is that Set expects new instances to be created by passing an
# iterable to __init__(), but that container cannot be randomly instanciated
# with values, it is tied to a Trace.
class _AvailableTraceEventsSet:
    """
    Smart container that uses demand event loading on the trace to check
    whether an event is present or not.

    This container can be iterated over to get the current available events,
    and supports membership tests using ``in``.
    """
    def __init__(self, trace):
        self._trace = trace

    def __contains__(self, event):
        if self._trace._strict_events:
            return self._trace._parsed_events.setdefault(event, False)

        # Try to parse the event in case it was not parsed already
        if event not in self._trace._parsed_events:
            # If the trace file is not accessible anymore, we will get an OSError
            with contextlib.suppress(MissingTraceEventError, OSError):
                self._trace.df_events(event=event, raw=True)

        return self._trace._parsed_events.setdefault(event, False)

    @property
    def _available_events(self):
        return {
            event
            for event, available in self._trace._parsed_events.items()
            if available
        }

    def __iter__(self):
        return iter(self._available_events)

    def __len__(self):
        return len(self._available_events)

    def __str__(self):
        return str(self._available_events)


class PandasDataDesc(Mapping):
    """
    Pandas data descriptor.

    :param spec: Specification of the data as a key/value mapping.

    This holds all the information needed to uniquely identify a
    :class:`pandas.DataFrame` or :class:`pandas.Series`. It is used to manage
    the cache and swap.

    It implements the :class:`collections.abc.Mapping` interface, so
    specification keys can be accessed directly like from a dict.

    .. note:: Once introduced in a container, instances must not be modified,
        directly or indirectly.

    :ivar normal_form: Normal form of the descriptor. Equality is implemented
        by comparing this attribute.
    :vartype normal_form: PandasDataDescNF
    """

    def __init__(self, spec):
        self.spec = spec
        self.normal_form = PandasDataDescNF.from_spec(self.spec)

    def __getitem__(self, key):
        return self.spec[key]

    def __iter__(self):
        return iter(self.spec)

    def __len__(self):
        return len(self.spec)

    @classmethod
    def from_kwargs(cls, **kwargs):
        """
        Build a :class:`PandasDataDesc` with the specifications as keyword
        arguments.
        """
        return cls(spec=kwargs)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(
                '{}={!r}'.format(key, val)
                for key, val in self.__dict__.items()
            )
        )

    def __eq__(self, other):
        return self.normal_form == other.normal_form

    def __hash__(self):
        return hash(self.normal_form)


class PandasDataDescNF:
    """
    Normal form of :class:`PandasDataDesc`.

    The normal form of the descriptor allows removing any possible differences
    in shape of values, and is serializable to JSON. The serialization is
    allowed to destroy some information (type mainly), as long as it does make
    two descriptors wrongly equal.
    """
    def __init__(self, nf):
        self._nf = nf
        # Since it's going to be inserted in dict for sure, precompute the hash
        # once and for all.
        self._hash = hash(self._nf)

    @classmethod
    def from_spec(cls, spec):
        """
        Build from a spec that can include any kind of Python objects.
        """
        nf = tuple(sorted(
            (key, cls._coerce(val))
            for key, val in spec.items()
        ))
        return cls(nf=nf)

    @classmethod
    def _coerce(cls, val):
        "Coerce data to a normal form that must be hashable"
        if isinstance(val, Integral):
            val = int(val)
        elif isinstance(val, Real):
            val = float(val)
        elif isinstance(val, (type(None), str, Number)):
            pass
        elif isinstance(val, Mapping):
            val = tuple(
                (cls._coerce(key), cls._coerce(val))
                for key, val in sorted(val.items())
            )
            val = ('mapping', val)
        elif isinstance(val, Set):
            val = tuple(map(cls._coerce, sorted(val)))
            val = ('set', val)
        elif isinstance(val, Sequence):
            val = tuple(map(cls._coerce, val))
            val = ('sequence', val)
        # In other cases save the name of the type along the value to make
        # sure we are not going to compare apple and oranges in the future
        else:
            type_name = '{}.{}'.format(
                val.__class__.__module__,
                val.__class__.__qualname__
            )
            val = (type_name, val)

        return val

    def __str__(self):
        return str(self._nf)

    def __eq__(self, other):
        return self._nf == other._nf

    def __hash__(self):
        return self._hash

    def to_json_map(self):
        return dict(self._nf)

    @classmethod
    def _coerce_json(cls, x):
        """
        JSON converts the original tuples into lists, so we need to convert it
        back.
        """
        if isinstance(x, str):
            return x
        elif isinstance(x, Sequence):
            return tuple(map(cls._coerce_json, x))
        else:
            return x

    @classmethod
    def from_json_map(cls, mapping):
        """
        Build from a mapping that was created using :meth:`to_json_map`.

        JSON does not preserve tuples for example, so they need to be converted
        back.
        """
        nf = tuple(sorted(
            (key, cls._coerce_json(val))
            for key, val in mapping.items()
        ))
        return cls(nf=nf)

class PandasDataSwapEntry:
    """
    Entry in the pandas data swap area of :class:`Trace`.

    :param pd_desc_nf: Normal form descriptor describing what the entry
        contains.
    :type pd_desc_nf: PandasDataDescNF

    :param name: Name of the entry. If ``None``, a random UUID will be
        generated.
    :type name: str or None
    """

    META_EXTENSION = '.meta'
    """
    Extension used by the metadata file of the swap entry in the swap.
    """

    def __init__(self, pd_desc_nf, name=None):
        self.pd_desc_nf = pd_desc_nf
        self.name = name or uuid.uuid4().hex

    @property
    def meta_filename(self):
        """
        Filename of the metadata file in the swap.
        """
        return '{}{}'.format(self.name, self.META_EXTENSION)

    @property
    def data_filename(self):
        """
        Filename of the pandas data file in the swap.
        """
        return '{}{}'.format(self.name, TraceCache.DATAFRAME_SWAP_EXTENSION)

    def to_json_map(self):
        """
        Return a mapping suitable for JSON serialization.
        """
        return {
            'version-token': VERSION_TOKEN,
            'name': self.name,
            'desc': self.pd_desc_nf.to_json_map(),
        }

    @classmethod
    def from_json_map(cls, mapping):
        """
        Create an instance with a mapping created using :meth:`to_json_map`.
        """
        if mapping['version-token'] != VERSION_TOKEN:
            raise TraceCacheSwapVersionError('Version token differ')

        pd_desc_nf = PandasDataDescNF.from_json_map(mapping['desc'])
        name = mapping['name']
        return cls(pd_desc_nf=pd_desc_nf, name=name)

    def to_path(self, path):
        """
        Save the swap entry metadata to the given ``path``.
        """
        data = self.to_json_map()
        with open(path, 'w') as f:
            json.dump(data, f)
            f.write('\n')

    @classmethod
    def from_path(cls, path):
        """
        Load the swap entry metadata from the given ``path``.
        """
        with open(path) as f:
            mapping = json.load(f)

        return cls.from_json_map(mapping)


class TraceCacheSwapVersionError(ValueError):
    """
    Exception raised when the swap entry was created by another version of LISA
    than the one loading it.
    """
    pass


class TraceCache:
    """
    Cache of a :class:`Trace`.

    :param max_mem_size: Maximum amount of memory to use in bytes. If the data
        hold in memory exceed that size, they will be evicted to the swap if
        possible and if it would be faster to reload them from swap rather than
        recomputing them. If there is no swap area, the data is just discarded.
    :type max_mem_size: int or None

    :param max_swap_size: Maximum amount of swap to use in bytes. When the
        amount of data saved to the swap exceeds that threshold, older files
        are discarded.
    :type max_swap_size: int or None

    :param swap_dir: Folder to use as swap area.
    :type swap_dir: str or None

    :param trace_path: Absolute path of the trace file.
    :type trace_path: str or None

    :param trace_md5: MD5 checksum of the trace file, to invalidate the cache
        if the file changed.
    :type trace_md5: str or None

    :param metadata: Metadata mapping to store in the swap area.
    :type metadata: dict or None

    :param swap_content: Initial content of the swap area.
    :type swap_content: dict(PandasDataDescNF, PandasDataSwapEntry) or None

    The cache manages both the :class:`pandas.DataFrame` and
    :class:`pandas.Series` generated in memory and a swap area used to evict
    them, and to reload them quickly.
    """

    INIT_SWAP_COST = 1e-7
    """
    Somewhat arbitrary number, must be small enough so that we write at
    least one dataset to the cache, which will allow us getting a better
    estimation. If the value is too high from the start, we will never
    write anything, and the value will never have a chance to re-adjust.
    """

    TRACE_META_FILENAME = 'trace.meta'
    """
    Name of the trace metadata file in the swap area.
    """

    DATAFRAME_SWAP_FORMAT = 'parquet'
    """
    Data storage format used to swap.
    """

    DATAFRAME_SWAP_EXTENSION = '.{}'.format(DATAFRAME_SWAP_FORMAT)
    """
    File extension of the data swap format.
    """

    def __init__(self, max_mem_size=None, trace_path=None, trace_md5=None, swap_dir=None, max_swap_size=None, swap_content=None, metadata=None):
        self._cache = {}
        self._data_cost = {}
        self._swap_content = swap_content or {}
        self._pd_desc_swap_filename = {}
        self.swap_cost = self.INIT_SWAP_COST
        self.swap_dir = swap_dir
        self.max_swap_size = max_swap_size if max_swap_size is not None else math.inf
        self._swap_size = self._get_swap_size()

        self.max_mem_size = max_mem_size if max_mem_size is not None else math.inf
        self._data_mem_swap_ratio = 7
        self._metadata = metadata or {}

        self.trace_path = os.path.abspath(trace_path)
        self._trace_md5 = trace_md5

    @property
    @memoized
    def _swap_size_overhead(self):
        def make_df(nr_col):
            return pd.DataFrame({
                str(x): []
                for x in range(nr_col)
            })

        def get_size(nr_col):
            df = make_df(nr_col)
            buffer = io.BytesIO()
            self._write_data(df, buffer)
            return buffer.getbuffer().nbytes

        size1 = get_size(1)
        size2 = get_size(2)

        col_overhead = size2 - size1
        # Since parquet seems to fail serializing of a dataframe with 0 columns
        # in some cases, we use the dataframe with one column and remove the
        # overhead of the column. It gives almost the same result.
        file_overhead = size1 - col_overhead
        assert col_overhead > 0

        return (file_overhead, col_overhead)

    def _unbias_swap_size(self, data, size):
        """
        Remove the fixed size overhead of the file format being used, assuming
        a non-compressible overhead per file and per column.

        .. note:: This model seems to work pretty well for parquet format.
        """
        file_overhead, col_overhead = self._swap_size_overhead
        # DataFrame
        try:
            nr_columns = data.shape[1]
        # Series
        except IndexError:
            nr_columns = 1

        size = size - file_overhead - nr_columns * col_overhead
        return size

    @property
    def trace_md5(self):
        md5 = self._trace_md5
        if md5 is None:
            with open(self.trace_path, 'rb') as f:
                md5 = checksum(f, 'md5')
            self._trace_md5 = md5

        return md5

    def update_metadata(self, metadata):
        """
        Update the metadata mapping with the given ``metadata`` mapping and
        write it back to the swap area.
        """
        self._metadata.update(metadata)
        self.to_swap_dir()

    def get_metadata(self, key):
        """
        Get the value of the given metadata ``key``.
        """
        return self._metadata[key]

    def to_json_map(self):
        """
        Returns a dictionary suitable for JSON serialization.
        """

        if self.swap_dir:
            trace_path = os.path.relpath(self.trace_path, self.swap_dir)
        else:
            trace_path = os.path.abspath(self.trace_path)

        return {
            'version-token': VERSION_TOKEN,
            'metadata': self._metadata,
            'trace-path': trace_path,
            'trace-md5': self.trace_md5,
        }

    def to_path(self, path):
        """
        Write the persistent state to the given ``path``.
        """
        mapping = self.to_json_map()
        with open(path, 'w') as f:
            json.dump(mapping, f)
            f.write('\n')

    @classmethod
    def _from_swap_dir(cls, swap_dir, trace_path=None, metadata=None, **kwargs):
        metapath = os.path.join(swap_dir, cls.TRACE_META_FILENAME)

        with open(metapath) as f:
            mapping = json.load(f)

        if mapping['version-token'] != VERSION_TOKEN:
            raise TraceCacheSwapVersionError('Version token differ')

        swap_trace_path = mapping['trace-path']
        swap_trace_path = os.path.join(swap_dir, swap_trace_path)

        metadata = metadata or {}

        try:
            with open(swap_trace_path, 'rb') as f:
                new_md5 = checksum(f, 'md5')
        except FileNotFoundError:
            new_md5 = None

        if trace_path and not os.path.samefile(swap_trace_path, trace_path):
            invalid_swap = True
        else:
            if new_md5 is None:
                invalid_swap = True
            else:
                old_md5 = mapping['trace-md5']
                invalid_swap = (old_md5 != new_md5)

        if invalid_swap:
            # Remove the invalid swap and create a fresh directory
            shutil.rmtree(swap_dir)
            os.makedirs(swap_dir)
            swap_content = None
        else:
            def load_swap_content(swap_dir):
                swap_entry_filenames = {
                    filename
                    for filename in os.listdir(swap_dir)
                    if filename.endswith(PandasDataSwapEntry.META_EXTENSION)
                }

                for filename in swap_entry_filenames:
                    path = os.path.join(swap_dir, filename)
                    try:
                        swap_entry = PandasDataSwapEntry.from_path(path)
                    # If there is any issue with that entry, just ignore it
                    except Exception:
                        continue
                    else:
                        yield (swap_entry.pd_desc_nf, swap_entry)

            swap_content = dict(load_swap_content(swap_dir))

            metadata_ = mapping['metadata']
            metadata = {**metadata_, **metadata}

        return cls(swap_content=swap_content, swap_dir=swap_dir, metadata=metadata, trace_path=trace_path, trace_md5=new_md5, **kwargs)

    def to_swap_dir(self):
        """
        Write the persistent state to the swap area if any, no-op otherwise.
        """
        if self.swap_dir:
            path = os.path.join(self.swap_dir, self.TRACE_META_FILENAME)
            self.to_path(path)

    @classmethod
    def from_swap_dir(cls, swap_dir, **kwargs):
        """
        Reload the persistent state from the given ``swap_dir``.

        :Variable keyword arguments: Forwarded to :class:`TraceCache`.
        """
        if swap_dir:
            try:
                return cls._from_swap_dir(swap_dir=swap_dir, **kwargs)
            except (FileNotFoundError, TraceCacheSwapVersionError, json.decoder.JSONDecodeError):
                pass

        return cls(swap_dir=swap_dir, **kwargs)

    def _estimate_data_swap_cost(self, data):
        return self._estimate_data_swap_size(data) * self.swap_cost

    def _estimate_data_swap_size(self, data):
        return self._data_mem_usage(data) * self._data_mem_swap_ratio

    def _update_ewma(self, attr, new, alpha=0.25, override=False):
        old = getattr(self, attr)
        if override:
            updated = new
        else:
            updated = (1 - alpha) * old + alpha * new

        setattr(self, attr, updated)

    def _update_data_swap_size_estimation(self, data, size):
        size = self._unbias_swap_size(data, size)

        # If size < 0, the dataframe is so small that it's basically just noise
        if size > 0:
            mem_usage = self._data_mem_usage(data)
            if mem_usage:
                self._update_ewma('_data_mem_swap_ratio', size / mem_usage)

    def _data_mem_usage(self, data):
        mem = data.memory_usage()
        try:
            return mem.sum()
        except AttributeError:
            return mem

    def _should_evict_to_swap(self, pd_desc, data):
        # If we don't have any cost info, assume it is expensive to compute
        compute_cost = self._data_cost.get(pd_desc, math.inf)
        swap_cost = self._estimate_data_swap_cost(data)
        return swap_cost <= compute_cost

    def _swap_path_of(self, pd_desc):
        if self.swap_dir:
            pd_desc_nf = pd_desc.normal_form
            swap_entry = self._swap_content[pd_desc_nf]
            filename = swap_entry.data_filename
            return os.path.join(self.swap_dir, filename)
        else:
            raise ValueError('Swap dir is not setup')

    def _update_swap_cost(self, data, swap_cost, mem_usage, swap_size):
        unbiased_swap_size = self._unbias_swap_size(data, swap_size)
        # Take out from the swap cost the time it took to write the overhead
        # that comes with the file format, assuming the cost is
        # proportional to amount of data written in the swap.
        swap_cost *= unbiased_swap_size / swap_size

        new_cost = swap_cost / mem_usage

        override = self.swap_cost == self.INIT_SWAP_COST
        # EWMA to keep a relatively stable cost
        self._update_ewma('swap_cost', new_cost, override=override)

    def _is_written_to_swap(self, pd_desc):
        return pd_desc.normal_form in self._swap_content

    @classmethod
    def _write_data(cls, data, path):
        if cls.DATAFRAME_SWAP_FORMAT == 'parquet':
            # Snappy compression seems very fast
            data.to_parquet(path, compression='snappy', index=True)
        else:
            raise ValueError('Dataframe swap format "{}" not handled'.format(cls.DATAFRAME_SWAP_FORMAT))

    def _write_swap(self, pd_desc, data):
        if not self.swap_dir:
            return
        else:
            if self._is_written_to_swap(pd_desc):
                return

            pd_desc_nf = pd_desc.normal_form
            swap_entry = PandasDataSwapEntry(pd_desc_nf)

            df_path = os.path.join(self.swap_dir, swap_entry.data_filename)

            # If that would make the swap dir too large, try to do some cleanup
            if self._estimate_data_swap_size(data) + self._swap_size > self.max_swap_size:
                self.scrub_swap()

            # Write the Parquet file and update the write speed
            with measure_time() as measure:
                self._write_data(data, df_path)

            # Update the swap
            swap_entry_path = os.path.join(self.swap_dir, swap_entry.meta_filename)
            swap_entry.to_path(swap_entry_path)
            self._swap_content[swap_entry.pd_desc_nf] = swap_entry

            # Assume that reading from the swap will take as much time as
            # writing to it. We cannot do better anyway, but that should
            # mostly bias to keeping things in memory if possible.
            swap_cost = measure.exclusive_delta
            data_swapped_size = os.stat(df_path).st_size

            mem_usage = self._data_mem_usage(data)
            if mem_usage:
                self._update_swap_cost(data, swap_cost, mem_usage, data_swapped_size)
            self._swap_size += data_swapped_size
            self._update_data_swap_size_estimation(data, data_swapped_size)
            self.scrub_swap()

    def _get_swap_size(self):
        if self.swap_dir:
            return sum(
                dir_entry.stat().st_size
                for dir_entry in os.scandir(self.swap_dir)
            )
        else:
            return 0

    def scrub_swap(self):
        """
        Scrub the swap area to remove old files if the storage size limit is exceeded.
        """
        # TODO: Load the file information from __init__ by discovering the swap
        # area's content to avoid doing it each time here
        if self._swap_size > self.max_swap_size and self.swap_dir:
            stats = {
                dir_entry.name: dir_entry.stat()
                for dir_entry in os.scandir(self.swap_dir)
            }

            data_files = {
                swap_entry.data_filename: swap_entry
                for swap_entry in self._swap_content.values()
            }

            # Get rid of stale files that are not referenced by any swap entry
            metadata_files = {
                swap_entry.meta_filename
                for swap_entry in self._swap_content.values()
            }
            metadata_files.add(self.TRACE_META_FILENAME)
            non_stale_files = data_files.keys() | metadata_files
            stale_files = stats.keys() - non_stale_files
            for filename in stale_files:
                del stats[filename]
                path = os.path.join(self.swap_dir, filename)
                os.unlink(path)

            def by_mtime(path_stat):
                path, stat = path_stat
                return stat.st_mtime

            # Sort by modification time, so we discard the oldest caches
            total_size = 0
            discarded_swap_entries = set()
            for filename, stat in sorted(stats.items(), key=by_mtime):
                total_size += stat.st_size
                if total_size > self.max_swap_size:
                    try:
                        swap_entry = data_files[filename]
                    # That was not a data file
                    except KeyError:
                        continue
                    else:
                        discarded_swap_entries.add(swap_entry)

            # Update the swap content
            for swap_entry in discarded_swap_entries:
                del self._swap_content[swap_entry.pd_desc_nf]
                del stats[swap_entry.data_filename]

                for filename in (swap_entry.meta_filename, swap_entry.data_filename):
                    path = os.path.join(self.swap_dir, filename)
                    os.unlink(path)

            self._swap_size = sum(
                stats[swap_entry.data_filename].st_size
                for swap_entry in self._swap_content.values()
            )

    def fetch(self, pd_desc, insert=True):
        """
        Fetch an entry from the cache or the swap.

        :param pd_desc: Descriptor to look for.
        :type pd_desc: PandasDataDesc

        :param insert: If ``True`` and if the fetch succeeds by loading the
            swap, the data is inserted in the cache.
        :type insert: bool
        """
        try:
            return self._cache[pd_desc]
        except KeyError as e:
            try:
                path = self._swap_path_of(pd_desc)
            # If there is no swap, bail out
            except (ValueError, KeyError):
                raise e
            else:
                # Try to load the dataframe from that path
                try:
                    if self.DATAFRAME_SWAP_FORMAT == 'parquet':
                        data = pd.read_parquet(path, memory_map=True)
                    else:
                        raise ValueError('Dataframe swap format "{}" not handled'.format(self.DATAFRAME_SWAP_FORMAT))
                except (OSError, pyarrow.lib.ArrowIOError):
                    raise e
                else:
                    if insert:
                        # We have no idea of the cost of something coming from
                        # the cache
                        self.insert(pd_desc, data, write_swap=False, compute_cost=None)

                    return data

    def insert(self, pd_desc, data, compute_cost=None, write_swap=False, force_write_swap=False):
        """
        Insert an entry in the cache.

        :param pd_desc: Descriptor of the data to insert.
        :type pd_desc: PandasDataDesc

        :param data: Pandas data to insert.
        :type data: pandas.DataFrame or pandas.Series

        :param compute_cost: Time spent to compute the data in seconds.
        :type compute_cost: float or None

        :param write_swap: If ``True``, the data will be written to the swap as
            well so it can be quickly reloaded. Note that it will be subject to
            cost evaluation, so it might not result in anything actually
            written.
        :type write_swap: bool

        :param force_write_swap: If ``True``, bypass the computation vs swap
            cost comparison.
        :type force_write_swap: bool
        """
        self._cache[pd_desc] = data
        if compute_cost is not None:
            self._data_cost[pd_desc] = compute_cost

        if write_swap:
            self.write_swap(pd_desc, force=force_write_swap)

        self._scrub_mem()

    def _scrub_mem(self):
        if self.max_mem_size == math.inf:
            return

        mem_usage = sum(
            self._data_mem_usage(data)
            for data in self._cache.values()
        )

        if mem_usage > self.max_mem_size:

            # Make sure garbage collection occurred recently, to get the most
            # accurate refcount possible
            gc.collect()
            refcounts = {
                pd_desc: sys.getrefcount(data)
                for pd_desc, data in self._cache.items()
            }
            min_refcount = min(refcounts.values())

            # Low retention score means it's more likely to be evicted
            def retention_score(pd_desc_and_data):
                pd_desc, data = pd_desc_and_data

                # If we don't know the computation cost, assume it can be evicted cheaply
                compute_cost = self._data_cost.get(pd_desc, 0)

                if not compute_cost:
                    score = 0
                else:
                    swap_cost = self._estimate_data_swap_cost(data)
                    # If it's already written back, make it cheaper to evict since
                    # the eviction itself is going to be cheap
                    if self._is_written_to_swap(pd_desc):
                        swap_cost /= 2

                    if swap_cost:
                        score = compute_cost / swap_cost
                    else:
                        score = 0

                # Assume that more references to an object implies it will
                # stay around for longer. Therefore, it's less interesting to
                # remove it from this cache and pay the cost of reading/writing it to
                # swap, since the memory will not be freed anyway.
                #
                # Normalize to the minimum refcount, so that the _cache and other
                # structures where references are stored are discounted for sure.
                return (refcounts[pd_desc] - min_refcount + 1) * score

            new_mem_usage = 0
            for pd_desc, data in sorted(self._cache.items(), key=retention_score):
                new_mem_usage += self._data_mem_usage(data)
                if new_mem_usage > self.max_mem_size:
                    self.evict(pd_desc)

    def evict(self, pd_desc):
        """
        Evict the given descriptor from memory.

        :param pd_desc: Descriptor to evict.
        :type pd_desc: PandasDataDesc

        If it would be cheaper to reload the data than to recompute them, they
        will be written to the swap area.
        """
        self.write_swap(pd_desc)

        try:
            del self._cache[pd_desc]
        except KeyError:
            pass

    def write_swap(self, pd_desc, force=False):
        """
        Write the given descriptor to the swap area if that would be faster to
        reload the data rather than recomputing it. If the descriptor is not in
        the cache or if there is no swap area, ignore it.

        :param pd_desc: Descriptor of the data to write to swap.
        :type pd_desc: PandasDataDesc

        :param force: If ``True``, bypass the compute vs swap cost comparison.
        :type force: bool
        """
        try:
            data = self._cache[pd_desc]
        except KeyError:
            pass
        else:
            if force or self._should_evict_to_swap(pd_desc, data):
                self._write_swap(pd_desc, data)

    def write_swap_all(self):
        """
        Attempt to write all cached data to the swap.
        """
        for pd_desc in self._cache.keys():
            self.write_swap(pd_desc)

    def clear_event(self, event, raw=None):
        """
        Clear cache entries referencing a given event.

        :param event: Event to clear.
        :type event: str

        :param raw: If ``True``, only clear entries that refer to raw data. If
            ``False``, only clear entries that refer to non-raw data. If
            ``None``, ignore whether the descriptor is about raw data or not.
        :type raw: bool or None
        """
        self._cache = {
            pd_desc: data
            for pd_desc, data in self._cache.items()
            if not (
                pd_desc.get('event') == event
                and (
                    raw is None
                    or pd_desc.get('raw') == raw
                )
            )
        }

    def clear_all_events(self, raw=None):
        """
        Same as :meth:`clear_event` but works on all events at once.
        """
        self._cache = {
            pd_desc: data
            for pd_desc, data in self._cache.items()
            if (
                # Cache entries can be associated to something else than events
                'event' not in pd_desc or
                # Either we care about raw and we check, or blanket clear
                raw is None or
                pd_desc.get('raw') == raw
            )
        }


class Trace(Loggable, TraceBase):
    """
    The Trace object is the LISA trace events parser.

    :param trace_path: File containing the trace
    :type trace_path: str

    :param events: events to be parsed. Since events can be loaded on-demand,
        that is optional but still recommended to improve trace parsing speed.
    :type events: list(str) or None

    :param strict_events: When ``True``, all the events specified in ``events``
        have to be present, and any other events will be assumed to not be
        present. This allows early failure and avoid the cost of lazy detection
        of events in very large traces.
    :type strict_events: bool

    :param plat_info: Platform info describing the target that this trace was
        collected on.
    :type plat_info: lisa.platforms.platinfo.PlatformInfo

    :param normalize_time: Make the first timestamp in the trace 0 instead
        of the system timestamp that was captured when tracing.
    :type normalize_time: bool

    :param trace_format: format of the trace. Possible values are:
        - FTrace
        - SysTrace
    :type trace_format: str or None

    :param plots_dir: directory where to save plots
    :type plots_dir: str

    :param sanitization_functions: Mapping of event name to sanitization
        function. Each function takes:

            * the trace instance
            * the name of the event
            * a dataframe of the raw event
            * a dictionary of aspects to sanitize
    :type sanitization_functions: dict(str, collections.abc.Callable) or None

    :param max_mem_size: Maximum memory usage to be used for dataframe cache.
        Note that the peak memory usage can exceed that, as the cache can not
        forcefully evict an object from memory (it can only drop references to it).
        When ``None``, use illimited amount of memory.
    :type max_mem_size: int or None

    :param swap_dir: Swap directory used to store dataframes evicted from the
        cache. When ``None``, a hidden directory along the trace file is used.
    :type swap_dir: str or None

    :param enable_swap: If ``True``, the on-disk swap is enabled.
    :type enable_swap: bool

    :param max_swap_size: Maximum size of the swap directory. When ``None``,
        the max size is the size of the trace file.
    :type max_swap_size: int or None

    :param write_swap: Default value used for :meth:`df_events` ``write_swap``
        parameter.
    :type write_swap: bool

    :ivar start: The timestamp of the first trace event in the trace
    :ivar end: The timestamp of the last trace event in the trace
    :ivar time_range: Maximum timespan for all collected events
    :ivar window: Conveniency tuple of ``(start, end)``.
    :ivar available_events: Events available in the parsed trace, exposed as
        some kind of set-ish smart container. Querying for event might trigger
        the parsing of it.
    """

    def __init__(self,
        trace_path,
        plat_info=None,
        events=None,
        strict_events=False,
        normalize_time=False,
        trace_format=None,
        plots_dir=None,
        sanitization_functions=None,

        max_mem_size=None,
        swap_dir=None,
        enable_swap=True,
        max_swap_size=None,
        write_swap=True,
    ):
        super().__init__()

        sanitization_functions = sanitization_functions or {}
        self._sanitization_functions = {
            **self._SANITIZATION_FUNCTIONS,
            **sanitization_functions,
        }

        if enable_swap:
            if swap_dir is None:
                basename = os.path.basename(trace_path)
                swap_dir = os.path.join(
                    os.path.dirname(trace_path),
                    '.{}.lisa-swap'.format(basename)
                )
                try:
                    os.makedirs(swap_dir, exist_ok=True)
                except OSError:
                    swap_dir = None

            if max_swap_size is None:
                trace_size = os.stat(trace_path).st_size
                max_swap_size = trace_size
        else:
            swap_dir = None
            max_swap_size = None

        self._cache = TraceCache.from_swap_dir(
            trace_path=trace_path,
            swap_dir=swap_dir,
            max_swap_size=max_swap_size,
            max_mem_size=max_mem_size,
        )
        # Initial scrub of the swap to discard unwanted data, honoring the
        # max_swap_size right from the beginning
        self._cache.scrub_swap()
        self._cache.to_swap_dir()

        self._write_swap = write_swap
        self.normalize_time = normalize_time
        self.trace_path = trace_path
        self._trace_format = trace_format

        # The platform information used to run the experiments
        if plat_info is None:
            # Delay import to avoid circular dependency
            from lisa.platforms.platinfo import PlatformInfo
            plat_info = PlatformInfo()
        self.plat_info = plat_info

        self._strict_events = strict_events
        self.available_events = _AvailableTraceEventsSet(self)
        self.plots_dir = plots_dir if plots_dir else os.path.dirname(trace_path)

        try:
            self._parsed_events = self._cache.get_metadata('parsed-events')
        except KeyError:
            self._parsed_events = {}

        if isinstance(events, str):
            raise ValueError('Events passed to Trace(events=...) must be a list of strings, not a string.')

        events = events if events is not None else []
        self.events = events
        # Pre-load the selected events
        if events:
            self._load_raw_df_map(events, write_swap=True, allow_missing_events=not self._strict_events)

    @property
    def trace_state(self):
        return (self.normalize_time,)

    @property
    @memoized
    def cpus_count(self):
        try:
            return self.plat_info['cpus-count']
        # If we don't know the number of CPUs, check the trace for the
        # highest-numbered CPU that traced an event.
        except KeyError:
            max_cpu = max(int(self.df_events(e)['__cpu'].max())
                          for e in self.available_events)
            count = max_cpu + 1
            self.get_logger().info("Estimated CPU count from trace: {}".format(count))
            return count

    @classmethod
    @contextlib.contextmanager
    def from_target(cls, target, events=None, buffer_size=10240, filepath=None, **kwargs):
        """
        Context manager that can be used to collect a :class:`Trace` directly
        from a :class:`lisa.target.Target` without needing to setup an
        :class:`FtraceCollector`.

        **Example**::

            from lisa.trace import Trace
            from lisa.target import Target

            target = Target.from_default_conf()

            with Trace.from_target(target, events=['sched_switch', 'sched_wakeup']) as trace:
                target.execute('echo hello world')
                # DO NOT USE trace object inside the `with` statement

            trace.analysis.tasks.plot_tasks_total_residency(filepath='plot.png')


        :param target: Target to connect to.
        :type target: Target

        :param events: ftrace events to collect and parse in the trace.
        :type events: list(str)

        :param buffer_size: Size of the ftrace ring buffer.
        :type buffer_size: int

        :param filepath: If set, the trace file will be saved at that location.
            Otherwise, a temporary file is created and removed as soon as the
            parsing is finished.
        :type filepath: str or None

        :Variable keyword arguments: Forwarded to :class:`Trace`.
        """
        ftrace_coll = FtraceCollector(target, events=events, buffer_size=buffer_size)
        plat_info = target.plat_info

        class TraceProxy(TraceBase):
            def get_view(self, *args, **kwargs):
                return self.base_trace.get_view(*args, **kwargs)

            def __getattr__(self, attr):
                try:
                    base_trace = self.__dict__['base_trace']
                except KeyError:
                    raise RuntimeError('The trace instance can only be used outside its "with" statement.')
                else:
                    return getattr(base_trace, attr)

        proxy = TraceProxy()

        with ftrace_coll:
            yield proxy

        if filepath:
            cm = nullcontext(filepath)
        else:
            @contextlib.contextmanager
            def cm_func():
                with tempfile.NamedTemporaryFile(suffix='.dat', delete=True) as temp:
                    yield temp.name

            cm = cm_func()

        with cm as path:
            ftrace_coll.get_data(path)
            trace = cls(
                path,
                events=events,
                strict_events=True,
                plat_info=plat_info,
                # Disable swap if the folder is going to disappear
                enable_swap=True if filepath else False,
                **kwargs
            )

        proxy.base_trace = trace

    def _get_trace(self, events):
        logger = self.get_logger()
        path = self.trace_path
        events = set(events)

        if self._trace_format is None:
            if path.endswith('html'):
                trace_format = 'SySTrace'
            else:
                trace_format = 'FTrace'
        else:
            trace_format = self._trace_format

        # Trappy chokes on some events for some reason, so make the user aware
        # of it and carry on
        mishandled_events = {'thermal_power_cpu_limit'}
        mishandled_events &= events
        if mishandled_events:
            logger.debug('A bug in Trappy prevents from loading these events: {}'.format(sorted(mishandled_events)))
            events -= mishandled_events


        logger.debug('Parsing {} events from {}: {}'.format(trace_format, path, sorted(events)))
        if trace_format == 'Systems':
            trace_class = trappy.SysTrace
        elif trace_format == 'FTrace':
            trace_class = trappy.FTrace
        else:
            raise ValueError('Unknown trace format: {}'.format(trace_format))

        # Since we handle the cache in lisa.trace.Trace, we do not need to duplicate it
        trace_class.disable_cache = True
        internal_trace = trace_class(
            path,
            scope="custom",
            events=sorted(events),
            normalize_time=False,
        )

        # trappy sometimes decides to be "clever" and overrules the path to be
        # used, even though it was specifically asked for a given file path
        assert path == internal_trace.trace_path

        # Since we got a trace here, use it to get basetime/endtime as well
        self._get_time_range(internal_trace=internal_trace)
        return internal_trace

    @property
    @memoized
    def basetime(self):
        """
        First absolute timestamp available in the trace.
        """
        return self._get_time_range()[0]

    @property
    @memoized
    def endtime(self):
        """
        Timestamp of when the tracing stopped.

        .. note:: With some parsers, that might be the timestamp of the last
            recorded event instead if the trace end timestamp was not recorded.
        """
        return self._get_time_range()[1]

    def _get_time_range(self, internal_trace=None):
        try:
            basetime = self._cache.get_metadata('basetime')
            endtime = self._cache.get_metadata('endtime')
        except KeyError:
            if internal_trace is None:
                internal_trace = self._get_trace(events=[])

            basetime = internal_trace.basetime
            endtime = internal_trace.endtime
            self._cache.update_metadata({
                'basetime': basetime,
                'endtime': endtime,
            })

        return (basetime, endtime)

    def df_events(self, event, raw=None, rename_cols=True, window=None, signals=None, signals_init=True, compress_signals_init=False, write_swap=None):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name
        :type event: str

        :param rename_cols: If ``True``, some columns will be renamed for
            consistency.
        :type rename_cols: bool

        :param window: Return a dataframe sliced to fit the given window (in
            seconds). Note that ``signals_init=True`` will result in including
            more rows than what you might expect.
        :type window: tuple(float, float)

        :param signals: List of signals to fixup if ``signals_init == True``.
            If left to ``None``, :meth:`lisa.datautils.SignalDesc.from_event`
            will be used to infer a list of default signals.
        :type signals: list(SignalDesc)

        :param signals_init: If ``True``, an initial value is provided for each
            signal that is multiplexed in that dataframe.
            .. seealso::

                :class:`lisa.datautils.SignalDesc` and
                :func:`lisa.datautils.df_window_signals`.
        :type signals_init: bool

        :param compress_signals_init: Give a timestamp very close to the
            beginning of the sliced dataframe to rows that are added by
            ``signals_init``. This allows keeping a very close time span
            without introducing duplicate indices.
        :type compress_signals_init: bool

        :param write_swap: If ``True``, the dataframe will be written to the
            swap area when meeting the following conditions:

                * This trace has a swap directory
                * Computing the dataframe takes more time than the estimated
                  time it takes to write it to the cache.
        :type write_swap: bool
        """

        sanitization_f = self._sanitization_functions.get(event)

        # Make sure no `None` value flies around in the cache, since it's
        # not uniquely identifying a dataframe
        orig_raw = raw
        if raw is None:
            if sanitization_f:
                raw = False
            else:
                raw = True

        if raw:
            sanitization_f = None
        elif orig_raw == False and not sanitization_f:
            raise ValueError('Sanitized dataframe for {} does not exist, please pass raw=True or raw=None'.format(event))

        if raw:
            # Make sure all raw descriptors are made the same way, to avoid
            # missed sharing opportunities
            spec = self._make_raw_pd_desc_spec(event)
        else:
            spec = dict(
                event=event,
                raw=raw,
                trace_state=self.trace_state,
            )

        if window is not None:
            signals = signals if signals else SignalDesc.from_event(event)
            cols_list = [
                signal_desc.fields
                for signal_desc in signals
            ]
            spec.update(
                window=window,
                signals=cols_list,
                signals_init=signals_init,
                compress_signals_init=compress_signals_init,
            )

        if not raw:
            spec.update(
                rename_cols=rename_cols,
                sanitization=sanitization_f.__qualname__ if sanitization_f else None,
            )

        pd_desc = PandasDataDesc(spec=spec)

        try:
            df = self._cache.fetch(pd_desc, insert=True)
        except KeyError:
            df = self._load_df(pd_desc, sanitization_f=sanitization_f, write_swap=write_swap)

        if df.empty:
            raise MissingTraceEventError(
                TraceEventChecker(event),
                available_events=self.available_events,
            )

        df.name = event
        return df

    def _make_raw_pd_desc(self, event):
        spec = self._make_raw_pd_desc_spec(event)
        return PandasDataDesc(spec=spec)

    def _make_raw_pd_desc_spec(self, event):
        return dict(
            event=event,
            raw=True,
            trace_state=self.trace_state,
        )

    def _load_df(self, pd_desc, sanitization_f=None, write_swap=None):
        raw = pd_desc['raw']
        event = pd_desc['event']

        # Do not even bother loading the event if we know it cannot be
        # there. This avoids some OSError in case the trace file has
        # disappeared
        if self._strict_events and event not in self.available_events:
            raise MissingTraceEventError(event, available_events=self.available_events)

        if write_swap is None:
            write_swap = self._write_swap

        df = self._load_raw_df_map([event], write_swap=True)[event]

        if sanitization_f:
            # Evict the raw dataframe once we got the sanitized version, since
            # we are unlikely to reuse it again
            self._cache.evict(self._make_raw_pd_desc(event))

            # We can ask to sanitize various aspects of the dataframe.
            # Adding a new aspect can be done without modifying existing
            # sanitization functions, as long as the default is the
            # previous behavior
            aspects = dict(
                rename_cols=pd_desc['rename_cols'],
            )
            with measure_time() as measure:
                df = sanitization_f(self, event, df, aspects=aspects)
            sanitization_time = measure.exclusive_delta
        else:
            sanitization_time = 0

        window = pd_desc.get('window')
        if window is not None:
            signals_init = pd_desc['signals_init']
            compress_signals_init = pd_desc['compress_signals_init']
            cols_list = pd_desc['signals']
            signals = [SignalDesc(event, cols) for cols in cols_list]

            with measure_time() as measure:
                if signals_init and signals:
                    df = df_window_signals(df, window, signals, compress_init=compress_signals_init)
                else:
                    df = df_window(df, window, method='pre')

            windowing_time = measure.exclusive_delta
        else:
            windowing_time = 0

        compute_cost = sanitization_time + windowing_time
        self._cache.insert(pd_desc, df, compute_cost=compute_cost, write_swap=write_swap)
        return df

    def _load_raw_df_map(self, events, write_swap, allow_missing_events=False):
        insert_kwargs = dict(
            write_swap=write_swap,
            # For raw dataframe, always write in the swap area if asked for
            # since parsing cost is known to be high
            force_write_swap=True,
        )

        # Get the raw dataframe from the cache if possible
        def try_from_cache(event):
            pd_desc = self._make_raw_pd_desc(event)
            try:
                # The caller is responsible of inserting in the cache if
                # necessary
                df = self._cache.fetch(pd_desc, insert=False)
            except KeyError:
                return None
            else:
                self._cache.insert(pd_desc, df, **insert_kwargs)
                return df

        from_cache = {
            event: try_from_cache(event)
            for event in events
        }

        from_cache = {
            event: df
            for event, df in from_cache.items()
            if df is not None
        }

        # Load the remaining events from the trace directly
        events_to_load = sorted(set(events) - from_cache.keys())
        from_trace = self._parse_raw_events(events_to_load)

        for event, df in from_trace.items():
            pd_desc = self._make_raw_pd_desc(event)
            self._cache.insert(pd_desc, df, **insert_kwargs)

        df_map = {**from_cache, **from_trace}
        missing_events = set(events) - df_map.keys()
        if missing_events:
            if allow_missing_events:
                self.get_logger().warning('Events {} not found in the trace: {}'.format(
                    ', '.join(sorted(missing_events)),
                    self.trace_path,
                ))
            else:
                raise MissingTraceEventError(missing_events)

        return df_map

    def _parse_raw_events_df(self, events):
        internal_trace = self._get_trace(events)

        mapping = {}
        for event in events:
            try:
                df = getattr(internal_trace, event).data_frame
            # If some events could not be parsed
            except AttributeError:
                continue
            else:
                # The dataframe cache will service future requests as needed, so we
                # can release the memory here
                delattr(internal_trace, event)

                # If the dataframe is empty, that event may not even exist at
                # all
                if df.empty:
                    continue
                else:
                    if self.normalize_time:
                        df.index -= self.basetime

                    mapping[event] = df

        return mapping

    def _parse_raw_events(self, events):
        if not events:
            return {}

        def chunk_list(l, nr_chunks):
            l_len = len(l)
            n = l_len // nr_chunks
            if not n:
                return l
            else:
                return [
                    l[i:i + n]
                    for i in range(0, l_len, n)
                ]

        nr_processes = os.cpu_count()
        parallel_parse = len(events) > 2
        # Parallel parsing with Trappy just slows things down at the moment so
        # disable it until we can experiment with other parsers which might
        # exhibit different behaviors
        parallel_parse = False
        if parallel_parse:
            chunked_events = chunk_list(events, nr_processes)
            df_map = {}
            with multiprocessing.Pool(processes=nr_processes) as pool:
                for df_map_ in pool.map(self._parse_raw_events_df, chunked_events):
                    df_map.update({
                        event: df
                        for event, df in df_map_.items()
                    })
        else:
            df_map = self._parse_raw_events_df(events)

        # remember the events that we tried to parse and that turned out to not be available
        self._parsed_events.update({
            event: not df.empty
            for event, df in df_map.items()
            # Only update the state if the event was not there, since it could
            # have been made available by a sanitization function
            if event not in self._parsed_events
        })

        # If for one reason or another we end up not having a dataframe at all
        self._parsed_events.update({
            event: False
            for event in (set(events) - df_map.keys())
            if event not in self._parsed_events
        })

        self._cache.update_metadata({
            'parsed-events': self._parsed_events,
        })

        return df_map

    @memoized
    def _get_task_maps(self):
        """
        Give the mapping from PID to task names, and the opposite.

        The names or PIDs are listed in appearance order.
        """


        # Keep only the values, in appearance order according to the timestamp
        # index
        def finalize(df, key_col, value_col, key_type, value_type):
            # Aggregate the values for each key and convert to python types
            mapping = {}
            grouped = df.groupby([key_col])
            for key, index in grouped.groups.items():
                values = df.loc[index][value_col].apply(value_type)
                values = list(values)
                key = key_type(key)
                mapping[key] = values

            return mapping

        mapping_df_list = []
        def _load(event, name_col, pid_col):
            df = self.df_events(event)

            # Get a Time column
            df = df.reset_index()
            grouped = df.groupby([name_col, pid_col])

            # Get timestamp of first occurrences of each key/value combinations
            mapping_df = grouped.first()
            mapping_df = mapping_df[['Time']]
            mapping_df.rename_axis(index={name_col: 'name', pid_col: 'pid'}, inplace=True)
            mapping_df_list.append(mapping_df)

        def load(event, *args, **kwargs):
            # All events have a __comm and __pid columns, so use it as well
            _load(event, '__comm', '__pid')
            _load(event, *args, **kwargs)

        # Import here to avoid circular dependency
        from lisa.analysis.load_tracking import LoadTrackingAnalysis
        # All events with a "comm" and "pid" column
        events = {
            'sched_wakeup',
            'sched_wakeup_new',
            *LoadTrackingAnalysis._SCHED_PELT_SE_NAMES,
        }
        for event in events:
            # Test each event independently, to make sure they will be parsed
            # if necessary
            if event in self.available_events:
                load(event, 'comm', 'pid')

        if 'sched_switch' in self.available_events:
            load('sched_switch', 'prev_comm', 'prev_pid')
            load('sched_switch', 'next_comm', 'next_pid')

        if not mapping_df_list:
            raise MissingTraceEventError(sorted(events) + ['sched_switch'], available_events=self.available_events)

        df = pd.concat(mapping_df_list)
        # Sort by order of appearance
        df.sort_values(by=['Time'], inplace=True)
        # Remove duplicated name/pid mapping and only keep the first appearance
        df = df.loc[~df.index.duplicated(keep='first')]
        # explode the multindex into a "key" and "value" columns
        df.reset_index(inplace=True)

        forbidden_names = {
            # <idle> is invented by trace-cmd, no event field contain this
            # value, so it's useless (and actually harmful, since it will
            # introduce a task that cannot be found in that trace)
            '<idle>',
            # This name appears when trace-cmd could not resolve the task name.
            # Ignore it since it's not a valid name, and we probably managed
            # to resolve it by looking at more events anyway.
            '<...>',
            # sched entity PELT events for task groups will get a comm="(null)"
            '(null)',
        }
        df = df[~df['name'].isin(forbidden_names)]

        name_to_pid = finalize(df, 'name', 'pid', str, int)
        pid_to_name = finalize(df, 'pid', 'name', int, str)

        return (name_to_pid, pid_to_name)

    @property
    def _task_name_map(self):
        return self._get_task_maps()[0]

    @property
    def _task_pid_map(self):
        return self._get_task_maps()[1]

    def has_events(self, events):
        """
        Returns True if the specified event is present in the parsed trace,
        False otherwise.

        :param events: trace event name or list of trace events
        :type events: str or list(str) or TraceEventCheckerBase
        """
        if isinstance(events, str):
            return events in self.available_events
        elif isinstance(events, TraceEventCheckerBase):
            try:
                events.check_events(self.available_events)
            except MissingTraceEventError:
                return False
            else:
                return True
        else:
            # Check each event independently in case they have not been parsed
            # yet.
            return all(
                event in self.available_events
                for event in self.available_events
            )

    def get_view(self, window, **kwargs):
        return TraceView(self, window, **kwargs)

    @property
    def start(self):
        return 0 if self.normalize_time else self.basetime

    @property
    def end(self):
        time_range = self.endtime - self.basetime
        return self.start + time_range

    def get_task_name_pids(self, name, ignore_fork=True):
        """
        Get the PIDs of all tasks with the specified name.

        The same PID can have different task names, mainly because once a task
        is generated it inherits the parent name and then its name is updated
        to represent what the task really is.

        :param name: task name
        :type name: str

        :param ignore_fork: Hide the PIDs of tasks that initially had ``name``
            but were later renamed. This is common for shell processes for
            example, which fork a new task, inheriting the shell name, and then
            being renamed with the final "real" task name
        :type ignore_fork: bool

        :return: a list of PID for tasks which name matches the required one.
        """
        pids = self._task_name_map[name]

        if ignore_fork:
            pids = [
                pid
                for pid in pids
                # Only keep the PID if its last name was the name we are
                # looking for.
                if self._task_pid_map[pid][-1] == name
            ]

        return pids

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='2.0',
        removed_in='2.1',
        replaced_by=get_task_name_pids,
    )
    def get_task_by_name(self, name):
        return self.get_task_name_pids(name, ignore_fork=True)

    def get_task_pid_names(self, pid):
        """
        Get the all the names of the task(s) with the specified PID, in
        appearance order.

        The same PID can have different task names, mainly because once a task
        is generated it inherits the parent name and then its name is
        updated to represent what the task really is.

        :param name: task PID
        :type name: int

        :return: the name of the task which PID matches the required one,
                 the last time they ran in the current trace
        """
        return self._task_pid_map[pid]

    @deprecate('This function raises exceptions when faced with ambiguity instead of giving the choice to the user',
        deprecated_in='2.0',
        removed_in='2.1',
        replaced_by=get_task_pid_names,
    )
    def get_task_by_pid(self, pid):
        """
        Get the name of the task with the specified PID.

        The same PID can have different task names, mainly because once a task
        is generated it inherits the parent name and then its name is
        updated to represent what the task really is.

        This API works under the assumption that a task name is updated at
        most one time and it always report the name the task had the last time
        it has been scheduled for execution in the current trace.

        :param name: task PID
        :type name: int

        :return: the name of the task which PID matches the required one,
                 the last time they ran in the current trace
        """
        name_list = self.get_task_pid_names(pid)

        if len(name_list) > 2:
            raise RuntimeError('The PID {} had more than two names in its life: {}'.format(
                pid, name_list,
            ))

        return name_list[-1]

    def get_task_ids(self, task, update=True):
        """
        Similar to :meth:`get_task_id` but returns a list with all the
        combinations, instead of raising an exception.

        :param task: Either the task name, the task PID, or a tuple ``(pid, comm)``
        :type task: int or str or tuple(int, str)

        :param update: If a partially-filled :class:`TaskID` is passed (one of
            the fields set to ``None``), returns a complete :class:`TaskID`
            instead of leaving the ``None`` fields.
        :type update: bool
        """

        def comm_to_pid(comm):
            try:
                pid_list = self._task_name_map[comm]
            except IndexError:
                raise ValueError('trace does not have any task named "{}"'.format(comm))

            return pid_list

        def pid_to_comm(pid):
            try:
                comm_list = self._task_pid_map[pid]
            except IndexError:
                raise ValueError('trace does not have any task PID {}'.format(pid))

            return comm_list

        if isinstance(task, str):
            task_ids = [
                TaskID(pid=pid, comm=task)
                for pid in comm_to_pid(task)
            ]
        elif isinstance(task, Number):
            task_ids = [
                TaskID(pid=task, comm=comm)
                for comm in pid_to_comm(task)
            ]
        else:
            pid, comm = task
            if pid is None and comm is None:
                raise ValueError('TaskID needs to have at least one of PID or comm specified')

            if update:
                non_none = pid if comm is None else comm
                task_ids = self.get_task_ids(non_none)
            else:
                task_ids = [TaskID(pid=pid, comm=comm)]

        return task_ids

    def get_task_id(self, task, update=True):
        """
        Helper that resolves a task PID or name to a :class:`TaskID`.

        :param task: Either the task name, the task PID, or a tuple ``(pid, comm)``
        :type task: int or str or tuple(int, str)

        :param update: If a partially-filled :class:`TaskID` is passed (one of
            the fields set to ``None``), returns a complete :class:`TaskID`
            instead of leaving the ``None`` fields.
        :type update: bool

        :raises ValueError: If there the input matches multiple tasks in the trace.
            See :meth:`get_task_ids` to get all the ambiguous alternatives
            instead of an exception.
        """
        task_ids = self.get_task_ids(task, update=update)
        if len(task_ids) > 1:
            raise ValueError('More than one TaskID matching: {}'.format(task_ids))

        return task_ids[0]

    @deprecate(deprecated_in='2.0', removed_in='2.1', replaced_by=get_task_id)
    def get_task_pid(self, task):
        """
        Helper that takes either a name or a PID and always returns a PID

        :param task: Either the task name or the task PID
        :type task: int or str or tuple(int, str)
        """
        return self.get_task_id(task).pid

    def get_tasks(self):
        """
        Get a dictionary of all the tasks in the Trace.

        :return: a dictionary which maps each PID to the corresponding list of
                 task name
        """
        return self._task_pid_map

    @property
    @memoized
    def task_ids(self):
        """
        List of all the :class:`TaskID` in the trace, sorted by PID.
        """
        return [
            TaskID(pid=pid, comm=comm)
            for pid, comms in sorted(self._task_pid_map.items(), key=itemgetter(0))
            for comm in comms
        ]

    def show(self):
        """
        Open the parsed trace using the most appropriate native viewer.

        The native viewer depends on the specified trace format:
        - ftrace: open using kernelshark
        - systrace: open using a browser

        In both cases the native viewer is assumed to be available in the host
        machine.
        """
        path = self.trace_path
        if path.endswith('.dat'):
            cmd = 'kernelshark'
        else:
            cmd = 'xdg-open'

        return os.popen("{} {}".format(
            cmd,
            shlex.quote(path)
        ))

###############################################################################
# Trace Events Sanitize Methods
###############################################################################

    _SANITIZATION_FUNCTIONS = {}
    def _sanitize_event(event, mapping=_SANITIZATION_FUNCTIONS):
        def decorator(f):
            mapping[event] = f
            return f
        return decorator

    @_sanitize_event('sched_load_avg_cpu')
    def _sanitize_load_avg_cpu(self, event, df, aspects):
        """
        If necessary, rename certain signal names from v5.0 to v5.1 format.
        """
        if aspects['rename_cols'] and 'utilization' in df:
            df.rename(columns={'utilization': 'util_avg'}, inplace=True)
            df.rename(columns={'load': 'load_avg'}, inplace=True)

        return df

    @_sanitize_event('sched_load_avg_task')
    def _sanitize_load_avg_task(self, event, df, aspects):
        """
        If necessary, rename certain signal names from v5.0 to v5.1 format.
        """
        if aspects['rename_cols'] and 'utilization' in df:
            df.rename(columns={'utilization': 'util_avg'}, inplace=True)
            df.rename(columns={'load': 'load_avg'}, inplace=True)
            df.rename(columns={'avg_period': 'period_contrib'}, inplace=True)
            df.rename(columns={'runnable_avg_sum': 'load_sum'}, inplace=True)
            df.rename(columns={'running_avg_sum': 'util_sum'}, inplace=True)

        return df

    @_sanitize_event('sched_boost_cpu')
    def _sanitize_boost_cpu(self, event, df, aspects):
        """
        Add a boosted utilization signal as the sum of utilization and margin.

        Also, if necessary, rename certain signal names from v5.0 to v5.1
        format.
        """
        if aspects['rename_cols'] and 'usage' in df:
            df.rename(columns={'usage': 'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']
        return df

    @_sanitize_event('sched_boost_task')
    def _sanitize_boost_task(self, event, df, aspects):
        """
        Add a boosted utilization signal as the sum of utilization and margin.

        Also, if necessary, rename certain signal names from v5.0 to v5.1
        format.
        """
        if aspects['rename_cols'] and 'utilization' in df:
            # Convert signals name from to v5.1 format
            df.rename(columns={'utilization': 'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']
        return df

    @_sanitize_event('sched_energy_diff')
    def _sanitize_energy_diff(self, event, df, aspects):
        """
        Convert between existing field name formats for sched_energy_diff
        """
        if aspects['rename_cols']:
            translations = {'nrg_d': 'nrg_diff',
                            'utl_d': 'usage_delta',
                            'payoff': 'nrg_payoff'
            }
            df.rename(columns=translations, inplace=True)

        return df

    @_sanitize_event('thermal_power_cpu_limit')
    @_sanitize_event('thermal_power_cpu_get_power')
    def _sanitize_thermal_power_cpu(self, event, df, aspects):

        def f(mask):
            # Replace '00000000,0000000f' format in more usable int
            return int(mask.replace(',', ''), 16)

        df['cpus'] = df['cpus'].apply(f)
        return df

    @_sanitize_event('cpu_frequency')
    def _sanitize_cpu_frequency(self, event, df, aspects):
        if aspects['rename_cols']:
            names = {
                'cpu_id': 'cpu',
                'state': 'frequency'
            }
            df.rename(columns=names, inplace=True)

        return df

    @_sanitize_event('funcgraph_entry')
    @_sanitize_event('funcgraph_exit')
    def _sanitize_funcgraph(self, event, df, aspects):
        """
        Resolve the kernel function names.
        """

        try:
            addr_map = self.plat_info['kernel']['symbols-address']
        except KeyError as e:
            self.get_logger().warning('Missing symbol addresses, function names will not be resolved: {}'.format(e))
            return df
        else:
            df['func_name'] = df['func'].map(addr_map)
            return df


class TraceEventCheckerBase(abc.ABC, Loggable):
    """
    ABC for events checker classes.

    Event checking can be achieved using a boolean expression on expected
    events.

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated
        method is applied.  If no such attribute is found, no check will be
        done.
    :type check: bool
    """

    def __init__(self, check=True):
        self.check = check

    @abc.abstractmethod
    def check_events(self, event_set):
        """
        Check that certain trace events are available in the given set of
        events.

        :raises: MissingTraceEventError if some events are not available
        """
        pass

    @abc.abstractmethod
    def get_all_events(self):
        """
        Return a set of all events that are checked by this checker.

        That may be a superset of events that are strictly required, when the
        checker checks a logical OR combination of events for example.
        """
        pass

    def __call__(self, f):
        """
        Decorator for methods that require some given trace events

        :param events: The list of required events
        :type events: list(str or TraceEventCheckerBase)

        The decorated method must operate on instances that have a ``self.trace``
        attribute.

        If some event requirements have already been defined for it (it has a
        `used_events` attribute, i.e. it has already been decorated), these
        will be combined with the new requirements using an
        :class`AndTraceEventChecker`.
        """
        def unwrap_down_to(obj):
            return hasattr(obj, 'used_events')

        try:
            # we want to see through all other kinds of wrappers, down to the
            # one that matters to us
            unwrapped_f = inspect.unwrap(f, stop=unwrap_down_to)
            used_events = unwrapped_f.used_events
        except AttributeError:
            checker = self
        else:
            # Update the existing checker inplace to avoid adding an extra
            # level of wrappers.
            checker = AndTraceEventChecker([self, used_events])
            unwrapped_f.used_events = checker
            return f

        sig = inspect.signature(f)
        if self.check and sig.parameters:
            @wraps(f)
            def wrapper(self, *args, **kwargs):
                try:
                    trace = self.trace
                # If there is no "trace" attribute, silently skip the check. This
                # allows using the decorator for documentation and chaining purpose
                # without having an actual trace to work on.
                except AttributeError:
                    pass
                else:
                    checker.check_events(trace.available_events)

                return f(self, *args, **kwargs)

        # If the decorated object takes no parameters, we cannot check anything
        else:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

        # Set an attribute on the wrapper itself, so it can be e.g. added
        # to the method documentation
        wrapper.used_events = checker
        return wrapper

    def __and__(self, other):
        """
        Combine two event checkers into one that checks the presence of both.

        .. seealso:: :class:`AndTraceEventChecker`
        """
        return AndTraceEventChecker([self, other])

    def __or__(self, other):
        """
        Combine two event checkers into one that checks the presence of either
        of them.

        .. seealso:: :class:`OrTraceEventChecker`
        """
        return OrTraceEventChecker([self, other])

    def __matmul__(self, other):
        """
        Combine two event checkers into an optional one.

        .. seealso:: :class:`OptionalTraceEventChecker`
        """
        return OptionalTraceEventChecker([self, other])

    @abc.abstractmethod
    def _str_internal(self, style=None, wrapped=True):
        """
        Format the boolean expression that this checker represents.

        :param style: When 'rst', a reStructuredText output is expected
        :type style: str

        :param wrapped: When True, the expression should be wrapped with
            parenthesis so it can be composed with other expressions.
        :type wrapped: bool
        """

        pass

    def doc_str(self):
        """
        Top-level function called by Sphinx's autodoc extension to augment
        docstrings of the functions.
        """
        return '\n    * {}'.format(self._str_internal(style='rst', wrapped=False))

    def __str__(self):
        return self._str_internal()


class TraceEventChecker(TraceEventCheckerBase):
    """
    Check for one single event.

    :param event: Name of the event to check for.
    :type event: str

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated
        method is applied.  If no such attribute is found, no check will be
        done.
    :type check: bool
    """

    def __init__(self, event, check=True):
        super().__init__(check=check)
        self.event = event

    def get_all_events(self):
        return {self.event}

    def check_events(self, event_set):
        if self.event not in event_set:
            raise MissingTraceEventError(self, available_events=event_set)

    def _str_internal(self, style=None, wrapped=True):
        template = '``{}``' if style == 'rst' else '{}'
        return template.format(self.event)


class AssociativeTraceEventChecker(TraceEventCheckerBase):
    """
    Base class for associative operators like `and` and `or`
    """

    def __init__(self, op_str, event_checkers, check=True, prefix_str=''):
        super().__init__(check=check)
        checker_list = []
        optional_checker_list = []
        for checker in event_checkers:
            # "unwrap" checkers of the same type, to avoid useless levels of
            # nesting. This is valid since the operator is known to be
            # associative. We don't use isinstance to avoid merging checkers
            # that may have different semantics.
            if type(checker) is type(self):
                checker_list.extend(checker.checkers)
            # Aggregate them separately to avoid having multiple of them
            elif isinstance(checker, OptionalTraceEventChecker):
                optional_checker_list.append(checker)
            else:
                checker_list.append(checker)

        if optional_checker_list:
            checker_list.append(OptionalTraceEventChecker(optional_checker_list))

        # Avoid having the same event twice at the same level
        def key(checker):
            if isinstance(checker, TraceEventChecker):
                return checker.event
            else:
                return checker
        checker_list = deduplicate(checker_list, key=key)

        self.checkers = checker_list
        self.op_str = op_str
        self.prefix_str = prefix_str

    def get_all_events(self):
        events = set()
        for checker in self.checkers:
            events.update(checker.get_all_events())
        return events

    @classmethod
    def from_events(cls, events, **kwargs):
        """
        Build an instance of the class, converting ``str`` to
        ``TraceEventChecker``.

        :param events: Iterable of events
        :type events: list(str or TraceEventCheckerBase)
        """
        def make_event(e):
            if isinstance(e, TraceEventCheckerBase):
                return e
            else:
                return TraceEventChecker(e)

        return cls({
            make_event(e)
            for e in events
        }, **kwargs)

    def _str_internal(self, style=None, wrapped=True):
        op_str = ' {} '.format(self.op_str)
        # Sort for stable output
        checker_list = sorted(self.checkers, key=lambda c: str(c))
        unwrapped_str = self.prefix_str + op_str.join(
            c._str_internal(style=style, wrapped=True)
            for c in checker_list
        )

        template = '({})' if len(self.checkers) > 1 and wrapped else '{}'
        return template.format(unwrapped_str)


class OrTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that one of the given event checkers is satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """

    def __init__(self, event_checkers, **kwargs):
        super().__init__('or', event_checkers, **kwargs)

    def check_events(self, event_set):
        if not self.checkers:
            return

        failed_checker_set = set()
        for checker in self.checkers:
            try:
                checker.check_events(event_set)
            except MissingTraceEventError as e:
                failed_checker_set.add(e.missing_events)
            else:
                break
        else:
            cls = type(self)
            raise MissingTraceEventError(
                cls(failed_checker_set),
                available_events=event_set,
            )


class OptionalTraceEventChecker(AssociativeTraceEventChecker):
    """
    Do not check anything, but exposes the information that the events may be
    used if present.

    :param event_checkers: Event checkers that may be used
    :type event_checkers: list(TraceEventCheckerBase)
    """

    def __init__(self, event_checkers, **kwargs):
        super().__init__(',', event_checkers, prefix_str='optional: ', **kwargs)

    def check_events(self, event_set):
        return


class AndTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that all the given event checkers are satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """

    def __init__(self, event_checkers, **kwargs):
        super().__init__('and', event_checkers, **kwargs)

    def check_events(self, event_set):
        if not self.checkers:
            return

        failed_checker_set = set()
        for checker in self.checkers:
            try:
                checker.check_events(event_set)
            except MissingTraceEventError as e:
                failed_checker_set.add(e.missing_events)

        if failed_checker_set:
            cls = type(self)
            raise MissingTraceEventError(
                cls(failed_checker_set),
                available_events=event_set,
            )

    def doc_str(self):
        joiner = '\n' + '    '
        rst = joiner + joiner.join(
            '* {}'.format(c._str_internal(style='rst', wrapped=False))
            # Sort for stable output
            for c in sorted(self.checkers, key=lambda c: str(c))
        )
        return rst


def requires_events(*events, **kwargs):
    """
    Decorator for methods that require some given trace events.

    :param events: The list of required events
    :type events: list(str or TraceEventCheckerBase)

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated method
        is applied.  If no such attribute is found, no check will be done.
    :type check: bool

    """
    return AndTraceEventChecker.from_events(events, **kwargs)


def requires_one_event_of(*events, **kwargs):
    """
    Same as :func:`requires_events` with logical `OR` semantic.
    """
    return OrTraceEventChecker.from_events(events, **kwargs)


def may_use_events(*events, **kwargs):
    """
    Same as :func:`requires_events` but just exposes some events that may be used
    if presents.
    """
    return OptionalTraceEventChecker.from_events(events, **kwargs)


class MissingTraceEventError(RuntimeError, ValueError):
    """
    :param missing_events: The missing trace events
    :type missing_events: TraceEventCheckerBase
    """

    def __init__(self, missing_events, available_events=None):
        msg = "Trace is missing the following required events: {}".format(missing_events)
        if available_events:
            msg += '. Available events are: {}'.format(
                ', '.join(available_events))

        super().__init__(msg)
        self.missing_events = missing_events


class FtraceConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class of :class:`FtraceCollector`

    Available keys:
    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('ftrace-conf', 'FTrace configuration', (
        KeyDesc('events', 'FTrace events to trace', [TypedList[str]]),
        KeyDesc('functions', 'FTrace functions to trace', [TypedList[str]]),
        KeyDesc('buffer-size', 'FTrace buffer size', [int]),
        KeyDesc('trace-clock', 'Clock used while tracing (see "trace_clock" in ftrace.txt kernel doc)', [str, None]),
        KeyDesc('saved-cmdlines-nr', 'Number of saved cmdlines with associated PID while tracing', [int]),
        KeyDesc('tracer', 'FTrace tracer to use', [str, None]),
    ))

    def add_merged_src(self, src, conf, **kwargs):
        """
        Merge-in a configuration source.

        :param src: Name of the merged source
        :type src: str

        :param conf: Conf to merge in
        :type conf: FtraceConf
        """
        def merge_conf(key, val):

            def non_mergeable(key):
                if self.get(key, val) == val:
                    return val
                else:
                    raise KeyError('Cannot merge key "{}": incompatible values specified: {} != {}'.format(
                        key, self[key], val,
                    ))

            if key in ('events', 'functions'):
                return sorted(set(val) | set(self.get(key, [])))
            elif key == 'buffer-size':
                return max(val, self.get(key, 0))
            elif key == 'trace-clock':
                return non_mergeable(key)
            elif key == 'saved-cmdlines-nr':
                return max(val, self.get(key, 0))
            elif key == 'tracer':
                return non_mergeable(key)
            else:
                raise KeyError('Cannot merge key "{}"'.format(key))

        merged = {
            key: merge_conf(key, val)
            for key, val in conf.items()
        }

        def is_modified(key, val):
            try:
                existing_val = self[key]
            except KeyError:
                return True
            else:
                return val != existing_val

        # We merge some keys with their current value in the conf
        return self.add_src(src,
            conf={
                key: val
                for key, val in merged.items()
                # Only add to the source if the result is different than what is
                # already set
                if is_modified(key, val)
            },
            **kwargs,
        )


class CollectorBase(Loggable):
    """
    Base class for :class:`devlib.collector.CollectorBase`-based collectors
    using composition.
    """

    TOOLS = []
    """
    Sequence of tools to install on the target when using the collector.
    """

    def __init__(self, target, collector):
        self._collector = collector
        if self.TOOLS:
            self.target.install_tools(self.TOOLS)

    def __getattr__(self, attr):
        return getattr(self._collector, attr)

    def __enter__(self):
        return self._collector.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._collector.__exit__(*args, **kwargs)

    def get_data(self, path):
        """
        Similar to :meth:`devlib.collector.CollectorBase.get_data` but takes
        the path directly as a parameter in order to disallow representing an
        invalid state where no path has been set.
        """
        coll = self._collector
        coll.set_output(path)
        return coll.get_data()

    @deprecate(replaced_by=get_data, deprecated_in='2.0', removed_in='2.1')
    def get_trace(self, path):
        """
        Deprecated alias for :meth:`get_data`.
        """
        return self.get_data(path)


class FtraceCollector(CollectorBase, Configurable):
    """
    Thin wrapper around :class:`devlib.FtraceCollector`.

    {configurable_params}
    """

    CONF_CLASS = FtraceConf
    TOOLS = ['trace-cmd']

    def __init__(self, target, events=None, functions=None, buffer_size=10240, autoreport=False, trace_clock=None, saved_cmdlines_nr=8192, tracer=None, **kwargs):
        events = events or []
        functions = functions or []
        trace_clock = trace_clock or 'global'
        kwargs.update(
            target=target,
            events=events,
            functions=functions,
            buffer_size=buffer_size,
            autoreport=autoreport,
            trace_clock=trace_clock,
            saved_cmdlines_nr=saved_cmdlines_nr,
            tracer=tracer,
        )
        self.check_init_param(**kwargs)

        self.events = events
        kernel_events = [
            event for event in events
            if self._is_kernel_event(event)
        ]

        if 'funcgraph_entry' in events or 'funcgraph_exit' in events:
            tracer = 'function_graph' if tracer is None else tracer

        # Only pass true kernel events to devlib, as it will reject any other.
        kwargs.update(
            events=kernel_events,
            tracer=tracer,
        )

        collector = devlib.FtraceCollector(**kwargs)
        super().__init__(target, collector)

    def _is_kernel_event(self, event):
        """
        Return ``True`` if the event is a kernel event.

        This allows events to be passed around in the collector to be inspected
        and used by other entities and then ignored when actually setting up
        ``trace-cmd``. An example are the userspace events generated by
        ``rt-app``.
        """
        # Avoid circular dependency with lisa.analysis submodule with late
        # import
        from lisa.analysis.rta import RTAEventsAnalysis

        return not (
            event in RTAEventsAnalysis.RTAPP_USERSPACE_EVENTS
            # trace-cmd start complains if given these events, even though they
            # are actually present in the trace when function_graph is
            # enabled
            or event in ['funcgraph_entry', 'funcgraph_exit']
        )

    @classmethod
    def from_conf(cls, target, conf):
        """
        Build an :class:`FtraceCollector` from a :class:`FtraceConf`

        :param target: Target to use when collecting the trace
        :type target: lisa.target.Target

        :param conf: Configuration object
        :type conf: FtraceConf
        """
        cls.get_logger().info('Ftrace configuration:\n{}'.format(conf))
        kwargs = cls.conf_to_init_kwargs(conf)
        kwargs['target'] = target
        cls.check_init_param(**kwargs)
        return cls(**kwargs)

    @classmethod
    def from_user_conf(cls, target, base_conf=None, user_conf=None, merged_src='merged'):
        """
        Build an :class:`FtraceCollector` from two :class:`FtraceConf`.

        ``base_conf`` is expected to contain the minimal configuration, and
        ``user_conf`` some additional settings that are used to augment the
        base configuration.

        :param target: Target to use when collecting the trace
        :type target: lisa.target.Target

        :param base_conf: Base configuration object, merged with ``user_conf``.
        :type base_conf: FtraceConf

        :param user_conf: User configuration object
        :type user_conf: FtraceConf

        :param merged_src: Name of the configuration source created by merging
            ``base_conf`` and ``user_conf``
        :type merged_src: str
        """
        user_conf = user_conf or FtraceConf()
        base_conf = base_conf or FtraceConf()

        # Make a copy of the conf, since it may be shared by multiple classes
        conf = copy.copy(base_conf)

        # Merge user configuration with the test's configuration
        conf.add_merged_src(
            src=merged_src,
            conf=user_conf,
        )
        return cls.from_conf(target, conf)


class DmesgCollector(CollectorBase):
    """
    Wrapper around :class:`devlib.collector.dmesg.DmesgCollector`.

    It installs the ``dmesg`` tool automatically on the target upon creation,
    so we know what version is being is used.
    """

    TOOLS = ['dmesg']
    LOG_LEVELS = devlib.DmesgCollector.LOG_LEVELS

    def __init__(self, target, **kwargs):
        collector = devlib.DmesgCollector(target, **kwargs)
        super().__init__(target, collector)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
