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
import inspect
import shlex
import contextlib
import tempfile
from functools import wraps
from collections.abc import Set, Mapping, Sequence, Iterable
from operator import itemgetter, attrgetter
from numbers import Number, Integral, Real
import multiprocessing
import textwrap
import subprocess
import itertools
import functools
import fnmatch
import typing
from pathlib import Path
from difflib import get_close_matches
import weakref
import atexit
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import types
import stat
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
import pyarrow.lib
import pyarrow.parquet
import polars as pl
import polars.exceptions
import polars.selectors as cs

import devlib

from lisa.utils import Loggable, HideExekallID, memoized, lru_memoized, deduplicate, take, deprecate, nullcontext, measure_time, checksum, newtype, groupby, PartialInit, kwargs_forwarded_to, kwargs_dispatcher, ComposedContextManager, get_nested_key, set_nested_key, unzip_into, order_as, DirCache, DelegateToAttr
from lisa.conf import SimpleMultiSrcConf, LevelKeyDesc, KeyDesc, TopLevelKeyDesc, Configurable
from lisa.datautils import SignalDesc, df_add_delta, df_deduplicate, df_window, df_window_signals, series_convert, df_update_duplicates, _polars_duration_expr, _df_to, _polars_df_in_memory, Timestamp, _pandas_cleanup_df
from lisa.version import VERSION_TOKEN
from lisa._typeclass import FromString
from lisa._kmod import LISADynamicKmod
from lisa._assets import get_bin


def _deprecated_warn(msg, **kwargs):
    warnings.warn(msg, DeprecationWarning, **kwargs)


def __getattr__(name):
    if name == 'TaskID':
        _deprecated_warn(f'TaskID has been moved to lisa.analysis.tasks.TaskID, update your import accordingly')
        from lisa.analysis.tasks import TaskID
        return TaskID
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_DEALLOCATORS = weakref.WeakSet()
_DEALLOCATORS_LOCK = threading.RLock()

def _dealloc_all():
    with _DEALLOCATORS_LOCK:
        for deallocator in _DEALLOCATORS:
            deallocator.run()
atexit.register(_dealloc_all)

def _file_cleanup(paths):
    paths = [
        path
        for path in paths
        if path is not None
    ]
    for path in paths:
        try:
            shutil.rmtree(path)
        except Exception:
            pass


def _identity(x):
    return x
def _make_identity():
    return _identity


class _Deallocator:
    def __init__(self, f, on_del=True, at_exit=True):
        self.f = f
        self._on_del = on_del
        if at_exit:
            with _DEALLOCATORS_LOCK:
                _DEALLOCATORS.add(self)

        self._lock = threading.Lock()

    def run(self):
        with self._lock:
            if (f := self.f) is not None:
                self.f = None
                f()

    def __del__(self):
        if self._on_del:
            self.run()

    # We never make a new copy. This is required so that only a single
    # deallocator ever handles a given resource.
    def __copy__(self):
        return self
    def __deepcopy__(self, memo):
        return self


class _LazyFrameOnDelete(_Deallocator):
    @classmethod
    def _attach(cls, df, f):
        return df.map_batches(
            cls(f),
            streamable=True,
            validate_output_schema=False,
        )

    @classmethod
    def attach_file_cleanup(cls, df, paths):
        paths = sorted(set(paths))
        if paths:
            df = cls._attach(
                df,
                functools.partial(
                    _file_cleanup,
                    paths,
                )
            )
        return df

    def __call__(self, x):
        return x

    def __reduce__(self):
        # Replace itself with an identity function upon deserialization, so
        # that we don't accidentally end up with 2 objects owning the same set
        # of paths.
        return (_make_identity, tuple())


def _make_hardlink(src, dst):
    try:
        os.link(src, dst)
    except FileExistsError:
        pass


def _df_json_serialize(df):
    # TODO: revisit based on the outcome of:
    # https://github.com/pola-rs/polars/issues/18284
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        return df.serialize(format='json')


def _df_json_deserialize(plan):
    # TODO: revisit based on the outcome of:
    # https://github.com/pola-rs/polars/issues/18284
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        return pl.LazyFrame.deserialize(plan, format='json')


def _logical_plan_resolve_paths(cache, plan, kind):
    swap_dir = Path(cache.swap_dir).resolve()

    def normalize(url):
        url = str(url)
        _url = urlparse(url)
        scheme = _url.scheme or 'file'
        if scheme == 'file':
            url = _url.path

        return (scheme, url)

    hardlinks_base = Path(uuid.uuid4().hex)
    hardlinks = set()
    def update_path(path):
        if kind == 'dump':
            scheme, path = normalize(path)
            if scheme == 'file':
                path = Path(path)
                try:
                    path = path.relative_to(swap_dir)
                except ValueError:
                    return path
                else:
                    # Remove the "hardlinks" part of the path so we point at the file
                    # in the cache
                    if path.parts[0] == 'hardlinks':
                        path = Path(path.name)

                    assert not path.is_absolute()
                    path = 'PATH_IN_LISA_CACHE' / path
                    return path
            else:
                return path
        elif kind == 'load':
            scheme, path = normalize(path)
            if scheme == 'file':
                path = Path(path)
                if path.parent.name == 'PATH_IN_LISA_CACHE':
                    path = Path(path.name)

                    # Create a hardlink to the data so that the data backing the
                    # LazyFrame we are reloading is guaranteed to stay around long
                    # enough and will not be scrubbed away.
                    hardlink_base, hardlink_path = cache._hardlink_path(
                        hardlinks_base,
                        path.name,
                    )
                    _make_hardlink(swap_dir / path, hardlink_path)
                    hardlinks.add(hardlink_base)
                    return hardlink_path
                # This path comes from somewhere else on the system so do not
                # rewrite it
                else:
                    return path
            else:
                return path
        else:
            raise ValueError(f'Unknown kind {kind}')
    plan = _logical_plan_update_paths(plan, update_path=update_path)
    return (plan, hardlinks)


def _logical_plan_update_paths(plan, update_path):
    def fixup_scans(obj):
        if isinstance(obj, Mapping):
            try:
                scan = obj['Scan']
            except KeyError:
                return fixup_scans(obj.values())
            else:
                # Polars 1.2.0 has a slightly changed format where
                # scan['paths'] is no longer a plain list[str]. It is a
                # list[list[str] | bool]. This function handles both formats.
                def dispatch_update(paths):
                    if isinstance(paths, str):
                        path = paths
                        return str(update_path(path))
                    elif isinstance(paths, Mapping):
                        return {
                            key: dispatch_update(path)
                            for key, path in paths.items()
                        }
                    elif isinstance(paths, Iterable):
                        return [
                            dispatch_update(path)
                            for path in paths
                        ]
                    else:
                        return paths

                # The location is based on the version of polars, since the
                # JSON format is unstable.
                locs = [
                    ['paths'],
                    # 1.8.1
                    ['sources', 'Paths'],
                    # 1.7.0
                    ['sources', 'sources', 'Paths'],
                ]
                for loc in locs:
                    try:
                        paths = get_nested_key(scan, loc)
                    except KeyError as e:
                        excep = e
                    else:
                        excep = None
                        set_nested_key(
                            scan,
                            loc,
                            [
                                dispatch_update(_paths)
                                for _paths in paths
                            ]
                        )
                        break
                if excep:
                    raise excep

        elif isinstance(obj, str):
            return
        elif isinstance(obj, Iterable):
            for value in obj:
                fixup_scans(value)

    plan = plan.copy()
    fixup_scans(plan)
    return plan


def _convert_df_from_parser(df, parser, cache):
    def to_polars(df):
        index = 'Time'
        if isinstance(df, pd.DataFrame):
            df.index.name = index
        df = _df_to(df, index=index, fmt='polars-lazyframe')
        return df

    def move_to_cache(df, cache):
        """
        Move data files backing a :class:`polars.LazyFrame` into the trace
        cache, so that freeing the temporary storage used by the trace parser
        does not break the returned objects.
        """
        def update_path(path, hardlinks_base, hardlinks):
            cache_path = cache.insert_disk_only(
                spec=dict(
                    # This unique key ensures we will never accidentally re-use
                    # that path for anything else.
                    key=uuid.uuid4().hex,
                ),
                compute_cost=None,
            )
            hardlink_base, hardlink_path = cache._hardlink_path(hardlinks_base, cache_path.name)

            # Move the original file under the hardlinks/ folder
            shutil.move(path, hardlink_path)

            # Hardlink the new path to a normal swap entry
            _make_hardlink(hardlink_path, cache_path)

            # Use the path under hardlinks/ so that if the swap entry gets
            # scrubed away, the existing LazyFrame will not break as they
            # will refer to the ones maintained by the trace.
            hardlinks.add(hardlink_base)
            return hardlink_path

        def fixup(df):
            if isinstance(df, pl.LazyFrame) and parser._STEAL_FILES:
                # We can only steal files if we have a swap to put it into
                try:
                    cache.swap_dir
                except (AttributeError, ValueError):
                    # If we cannot move the backing data to the swap folder, we
                    # are forced to just load the data eagerly as backing
                    # storage (e.g.  tmp folder) will probably disappear
                    df = df.collect()
                else:
                    hardlinks = set()
                    df = _lazyframe_rewrite(
                        df=df,
                        update_plan=functools.partial(
                            _logical_plan_update_paths,
                            update_path=functools.partial(
                                update_path,
                                hardlinks_base=uuid.uuid4().hex,
                                hardlinks=hardlinks
                            )
                        )
                    )

                    # TODO: Ideally we should only attach that to the LazyFrame
                    # right before leaking it in df_event, otherwise we end up
                    # serializing in the cache the deallocator as well. This is
                    # not a huge problem though, but it does mean that we will
                    # end up with 2 layers of "map_batches(identity)" on
                    # LazyFrames reloaded from the cache.
                    df = _LazyFrameOnDelete.attach_file_cleanup(df, hardlinks)

            return to_polars(df)
        return fixup(df)

    df = _ParsedDataFrame.from_df(df)
    return df.with_df(
        move_to_cache(df.df, cache=cache)
    )


def _lazyframe_rewrite(df, update_plan):
    assert isinstance(df, pl.LazyFrame)

    # TODO: once this is solved, we can just inspect the plan rather than
    # serialize()/deserialize() in JSON
    # https://github.com/pola-rs/polars/issues/9771
    plan = _df_json_serialize(df)
    plan = json.loads(plan)
    plan = update_plan(plan)
    plan = json.dumps(plan)
    plan = io.StringIO(plan)
    df = _df_json_deserialize(plan)
    return df


class _CacheDataDescEncodable(abc.ABC):
    """
    Inheriting from this class allows encoding a value in JSON for a cache
    desc.
    """

    @abc.abstractmethod
    def json_encode(self):
        """
        Returns a more basic object that can readily be encoded by an
        unmodified json serializer.
        """
        pass


CPU = newtype(int, 'CPU', doc='Alias to ``int`` used for CPU IDs')


class _CPUSeqFromStringInstance(FromString, types=(typing.List[CPU], typing.Sequence[CPU])):
    from_str = FromString(typing.Sequence[int]).from_str

    @classmethod
    def get_format_description(cls, short):
        return FromString(typing.Sequence[int]).get_format_description(short=short)


class MissingMetadataError(KeyError):
    """
    Raised when a given metadata is not available.
    """
    def __init__(self, metadata):
        # pylint: disable=super-init-not-called
        self.metadata = metadata

    def __str__(self):
        return f'Missing metadata: {self.metadata}'


class _AllEvents(Iterable):
    def __iter__(self):
        return iter([])

_ALL_EVENTS = _AllEvents()


class _ParsedDataFrame:
    def __init__(self, df, **meta):
        self.df = df
        self.meta = {
            'mem_cacheable': True,
            'swap_cacheable': True,
            **meta,
        }

    def with_df(self, df):
        return self.__class__(
            df=df,
            **self.meta,
        )

    @classmethod
    def from_df(cls, df, **meta):
        if isinstance(df, cls):
            meta = {
                **df.meta,
                **meta,
            }
            df = df.df

        return cls(df, **meta)


class TraceParserBase(abc.ABC, Loggable, PartialInit):
    """
    Abstract Base Class for trace parsers.

    :param events: Iterable of events to parse. An empty iterable can be
        passed, in which case some metadata may still be available. If
        ``_ALL_EVENTS`` is passed, the caller may subsequently call
        :meth:`parse_all_events`.
    :param events: collections.abc.Iterable(str)

    :param needed_metadata: Set of metadata name to gather in the parser.
    :type needed_metadata: collections.abc.Iterable(str)

    The parser will be used as a context manager whenever it is queried for
    either events dataframes. Querying for metadata could happen immediately
    after object creation, but without expectation of success. Expensive
    metadata should only be computed when the object is used as a context
    manager. Note that the same parser object might be used as a context
    manager multiple times in its lifetime.


    .. attention:: This class and its base class is not subject to the normal
        backward compatibility guarantees. It is considered somewhat internal
        and will be modified if necessary, with backward compatibility being
        offered on a best-effort basis.
    """

    _STEAL_FILES = False
    """
    If ``True``, files backing a :class:`polars.LazyFrame` will  be stolen by
    :class:`~lisa.trace.Trace` to add them to the cache.
    """

    METADATA_KEYS = [
        'time-range',
        'symbols-address',
        'cpus-count',
        'available-events',
        'trace-id',
    ]
    """
    Possible metadata keys
    """

    def __init__(self, events, temp_dir, needed_metadata=None, path=None):
        # pylint: disable=unused-argument
        self._requested_metadata = set(needed_metadata or [])
        self._requested_events = events if events is _ALL_EVENTS else set(events)
        self._temp_dir = Path(temp_dir)

    def get_parser_id(self):
        """
        Get the unique ID of that parser. Any parameter affecting the output
        dataframes or metadata must be somehow part of that ID, so that the
        cache is not accidentally hit with stale data.
        """
        # We rely on having a PartialInit object here, since TraceParserBase
        # inherits from it.
        assert isinstance(self, PartialInit)

        cls = type(self)
        id_ = '+'.join(
            f'{attr}={val!r}'
            for attr, val in sorted(self.__dict__['_kwargs'].items())
        )

        id_ = id_.encode('utf-8')
        id_ = checksum(io.BytesIO(id_), 'md5')
        return f'{cls.__module__}.{cls.__qualname__}-{id_}'

    def get_metadata(self, key):
        """
        Return the metadata value.

        :param key: Name of the metadata. Can be one of:

            * ``time-range``: tuple ``(start, end)`` of the timestamps in the
              trace. This must be the first timestamp to appear in the trace,
              regardless of what events is being parsed. Otherwise, it would be
              impossible to use the time range of a parser in the mother
              :class:`TraceBase` when requesting specific events.

            * ``symbols-address``: Dictionnary of address (int) to symbol names
              in the kernel (str) that was used to create the trace. This
              allows resolving the fields of events that recorded addresses
              rather than function names.

            * ``cpus-count``: Number of CPUs on the system the trace was
              collected on.

            * ``available-events``: List of all available events stored in the
              trace. The list must be exhaustive, not limited to the events
              that were requested. If an exhaustive list cannot be gathered,
              this metadata should not be implemented.

            * ``trace-id``: Unique identifier for that trace file used to
                validate the cache. If not available, a checksum will be used.

        :type key: str

        :raises: :exc:`MissingMetadataError` if the metadata is not available
            on that parser.

        .. note:: A given metadata can only be expected to be available if
            asked for in the constructor, but bear in mind that there is no
            promise on the availability of any except for the following that
            *must* be provided if asked for:

            * ``time-range``

            Metadata may still be made available if not asked for, but only if
            it's a very cheap byproduct of parsing that incurs no extra cost.
        """
        # pylint: disable=no-self-use
        raise MissingMetadataError(key)

    def get_all_metadata(self):
        """
        Collect all available metadata.
        """
        metadata = {}
        for key in self.METADATA_KEYS:
            try:
                val = self.get_metadata(key)
            except MissingMetadataError:
                pass
            else:
                metadata[key] = val

        return metadata

    @abc.abstractmethod
    def parse_event(self, event):
        """
        Parse the given event from the trace and return a
        :class:`pandas.DataFrame` with the following columns:

        * ``Time`` index: floating point absolute timestamp in seconds. The
          index *must not* have any duplicated values.
        * One column per event field, with the appropriate dtype.
        * Columns prefixed with ``__``: Header of each event, usually
          containing the following fields:

          * ``__cpu``: CPU number the event was emitted from
          * ``__pid``: PID of the current process scheduled at the time the event was emitted
          * ``__comm``: Task command name going with ``__pid`` at the point the
            event was emitted

        :param event: name of the event to parse
        :type event: str

        :raises MissingTraceEventError: If the event cannot be parsed.

        .. note:: The caller is free to modify the index of the data, and it
            must not affect other dataframes.
        """

    def parse_all_events(self):
        """
        Parse all available events.

        .. note:: A parser that does not support querying the
            ``available-events`` metadata may raise an exception. This might
            also lead to multilple scans of the trace in some implementations.
        """
        try:
            events = self.get_metadata('available-events')
            return self.parse_events(events)
        except Exception:
            raise NotImplementedError(f'{self.__class__.__qualname__} parser does not support parsing all events')


    def parse_events(self, events, best_effort=False, **kwargs):
        """
        Same as :meth:`parse_event` but taking a list of events as input, and
        returning a mapping of event names to :class:`pandas.DataFrame` for
        each.

        :param events: Event names to parse.
        :type events: list(str)

        :param best_effort: If ``True``, do not raise
            :exc:`MissingTraceEventError`, silently skip the event.
            Must default to ``False``.
        :type best_effort: bool

        :Variable keyword arguments: Forwarded to :meth:`parse_event`
        """

        def parse(event):
            try:
                return self.parse_event(event, **kwargs)
            except MissingTraceEventError:
                if best_effort:
                    return None
                else:
                    raise
        return {
            event: df
            for event, df in (
                (event, parse(event))
                for event in events
            )
            if df is not None
        }

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return


class MockTraceParser(TraceParserBase):
    """
    Mock parser that just returns the dataframes it was given.

    :param dfs: Dictionary of :class:`pandas.DataFrame` for each event that the
        parser can handle.
    :type dfs: dict(str, pandas.DataFrame)

    :param path: Useless for now, but it's part of the Trace API, and it will
        be used for the dataframe cache as well.
    :type path: str or None

    :param events: Unused.
    :param events: collections.abc.Iterable(str)

    :param time_range: Time range of the trace in seconds. If not specified,
        the min and max timestamp of all ``dfs`` will be extracted, but it can
        lead to wrong analysis results (especially for signals that are not
        updated very often).
    :type time_range: tuple(float, float)

    :Variable keyword arguments: Forwarded to :class:`TraceParserBase`


    As a subclass of :class:`lisa.utils.PartialInit`, its constructor supports
    being applied to a partial set of parameters, leaving the rest to the
    internals of :class:`lisa.trace.Trace`::

        dfs = {
            'sched_wakeup': pd.DataFrame.from_records(
                [
                    (0, 1, 1, 'task1', 'task1', 1, 1, 1),
                    (1, 2, 1, 'task1', 'task1', 1, 1, 2),
                    (2, 4, 2, 'task2', 'task2', 2, 1, 4),
                ],
                columns=('Time', '__cpu', '__pid', '__comm', 'comm', 'pid', 'prio', 'target_cpu'),
                index='Time',
            ),
        }
        trace = Trace(parser=MockTraceParser(dfs))
        print(trace.df_event('sched_wakeup'))
    """
    @kwargs_forwarded_to(TraceParserBase.__init__)
    def __init__(self, dfs, time_range=None, events=None, path=None, **kwargs):
        self.dfs = {
            event: _df_to(df, index='Time', fmt='polars-lazyframe')
            for event, df in dfs.items()
        }

        # "path" is useless for now, but it's part of the Trace API, and it will
        # be used for the dataframe cache as well.
        #
        # "events" is
        super().__init__(events=events, **kwargs)
        self._time_range = time_range

    @property
    def _available_events(self):
        return set(self.dfs.keys())

    def parse_event(self, event):
        try:
            return self.dfs[event]
        except KeyError as e:
            raise MissingTraceEventError(
                [event],
                available_events=self._available_events
            )

    def get_metadata(self, key):
        if key == 'time-range':
            if self._time_range:
                return self._time_range
            elif (dfs := self.dfs):
                df = pl.concat(
                    df.select('Time')
                    for df in dfs.values()
                )
                df = df.with_columns(
                    Time=pl.col('Time').cast(pl.Duration('ns')).cast(pl.Int64)
                )

                start, end = df.select(
                    (pl.min('Time').alias('min'), pl.max('Time').alias('max'))
                ).collect().row(0)

                start = Timestamp(start, unit='ns', rounding='down')
                end = Timestamp(end, unit='ns', rounding='up')
                return (start, end)
            else:
                return (0, 0)

        elif key == 'available-events':
            return set(self._available_events)
        else:
            super().get_metadata(key=key)


class PerfettoTraceParser(TraceParserBase):
    """
    Bridge to parsing Perfetto traces.

    :param path: Path to the Perfetto trace.
    :type path: str

    :param events: Events to parse eagerly. This is currently unused.
    :param events: collections.abc.Iterable(str)

    :param trace_processor_path: Path or URL to Perfetto's ``trace_processor``
        binary. If a URL is passed, it will be downloaded and cached
        automatically.
    :type trace_processor_path: str

    :Variable keyword arguments: Forwarded to :class:`TraceParserBase`

    """
    @kwargs_forwarded_to(TraceParserBase.__init__)
    def __init__(self, path=None, events=None, trace_processor_path='https://get.perfetto.dev/trace_processor', needed_metadata=None, **kwargs):

        if urllib.parse.urlparse(trace_processor_path).scheme:
            bin_path = self._download_trace_processor(url=trace_processor_path)
        else:
            bin_path = trace_processor_path

        self._bin_path = bin_path
        self._trace_path = path

        meta = {}
        needed = needed_metadata or set()

        if 'available-events' in needed:
            meta['available-events'] = set(
                row.name
                for row in self._query("SELECT DISTINCT name FROM raw")
            )

        if 'time-range' in needed:
            time_range, = self._query("SELECT MIN(ts), MAX(ts) FROM raw")
            meta['time-range'] = (
                time_range.__dict__['MIN(ts)'] / 1e9,
                time_range.__dict__['MAX(ts)'] / 1e9,
            )

        self._metadata = meta
        super().__init__(events=events, needed_metadata=needed_metadata, **kwargs)

    @property
    @memoized
    def _tp(self):
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

        config = TraceProcessorConfig(
            # Without that, perfetto will disallow querying for most events in
            # the raw table
            ingest_ftrace_in_raw=True,
            bin_path=self._bin_path,
        )
        tp = TraceProcessor(trace=self._trace_path, config=config)
        return tp

    @classmethod
    def _download_trace_processor(cls, url):
        def populate(key, path):
            url, = key
            dst = path / 'trace_processor'
            cls.get_logger().info(f"Downloading Perfetto's trace_processor at {dst}")
            urllib.request.urlretrieve(url, dst)
            os.chmod(dst, dst.stat().st_mode | stat.S_IEXEC)

        dir_cache = DirCache(
            category='perfetto_trace_processor',
            populate=populate,
            # FIXME: is that really what we want ? The URL does not have any
            # version number in it, so we will never invalidate the cache as it
            # is.
            fmt_version='1',
        )

        cache_path = dir_cache.get_entry([url])
        return cache_path / 'trace_processor'

    def _query(self, query):
        return self._tp.query(query)

    def _df_event(self, event):
        # List the fields of the event so we know their types and can query them
        query = f"SELECT arg_set_id FROM raw WHERE name = '{event}' LIMIT 1"
        arg_set_id, = map(attrgetter('arg_set_id'), self._query(query))

        query = f"SELECT * FROM args WHERE arg_set_id = {arg_set_id}"
        arg_fields = [
            (field.key, field.value_type)
            for field in self._query(query)
        ]

        def pick_col(typ):
            return {
                'uint': 'int_value',
                'int': 'int_value',
                'string': 'string_value',
            }[typ]

        arg_cols = [
            f"(SELECT {pick_col(typ)} FROM args WHERE key = '{key}' AND args.arg_set_id = raw.arg_set_id)"
            for (key, typ) in arg_fields
        ]
        common_cols = ['ts as Time', 'cpu as __cpu', 'utid as __pid', '(SELECT name from thread where thread.utid = raw.utid) as __comm']

        def translate_type(typ):
            return {
                'uint': pl.UInt64,
                'int': pl.Int64,
                'string': pl.Categorical(),
            }[typ]

        schema = {
            'Time': pl.UInt64,
            '__cpu': pl.UInt32,
            '__pid': pl.UInt32,
            '__comm': pl.Categorical,
            **{
                query: translate_type(typ)
                for (_, typ), query in zip(arg_fields, arg_cols)
            }
        }

        extract = ', '.join(common_cols + arg_cols)
        query = f"SELECT {extract} FROM raw WHERE name = '{event}'"

        df = pl.LazyFrame(
            (
                row.__dict__
                for row in self._query(query)
            ),
            orient='row',
            schema=schema,
        )
        df = df.rename({
            query: name
            for (name, _), query in zip(arg_fields, arg_cols)
        })
        df = df.select(order_as(df.columns, ['Time', '__cpu', '__pid', '__comm']))

        # Collect the data to expose to the caller that everything sits in
        # memory
        df = df.collect()
        return df

    def parse_event(self, event):
        try:
            return self._df_event(event)
        except Exception as e:
            raise MissingTraceEventError(
                [event],
                available_events=self._metadata.get('available-events'),
            ) from e

    def get_metadata(self, key):
        try:
            return self._metadata[key]
        except KeyError:
            return super().get_metadata(key=key)


class TraceDumpError(Exception):
    """
    Exception containing errors forwarded from the trace-dump parser
    """
    def __init__(self, errors, event=None, cmd=None):
        self.errors = sorted(set(errors))
        self.event = event
        self.cmd = cmd

    def __str__(self):
        event = self.event
        errors = self.errors
        nr_errors = len(errors)
        cmd = self.cmd
        cmd = ' '.join(map(shlex.quote, map(str, cmd))) if cmd else ''

        try:
            errors, = errors
        except ValueError:
            errors = '\n'.join(map(str, errors))
        else:
            errors = str(errors)

        if event:
            if nr_errors > 1:
                sep = '\n  '
                errors = errors.replace('\n', sep)
                return f'{cmd}{event}:{sep}{errors}'
            else:
                return f'{cmd}{event}: {errors}'
        else:
            return errors


class TraceDumpTraceParser(TraceParserBase):
    """
    trace.dat parser shipped by LISA
    """

    _STEAL_FILES = True
    _MAX_ERRORS = 256

    @kwargs_forwarded_to(TraceParserBase.__init__)
    def __init__(self, path, events, needed_metadata=None, **kwargs):
        super().__init__(events=events, needed_metadata=needed_metadata, **kwargs)
        self._trace_path = str(Path(path).resolve())
        self._metadata = {}
        self._trace_format = 'tracedat'

    def __enter__(self):
        temp_dir = self._temp_dir
        events = self._requested_events
        path = self._trace_path
        needed_metadata = (
            self._requested_metadata -
            self._metadata.keys()
        )
        meta = {}

        # time-range will not be available in the basic metadata, this requires
        # a full parse
        if events or events is _ALL_EVENTS:
            meta = self._make_parquets(
                path=path,
                trace_format=self._trace_format,
                events=events,
                temp_dir=temp_dir
            )
            for desc in meta['events-info']:
                _path = desc.get('path')
                if _path is not None:
                    _path = temp_dir / _path
                    event = desc['event']

        elif needed_metadata:
            meta = self._make_metadata(
                path=path,
                trace_format=self._trace_format,
                temp_dir=temp_dir,
                needed_metadata=needed_metadata,
            )

        self._metadata.update(meta)
        return self

    def __exit__(self, *args, **kwargs):
        pass

    @classmethod
    def from_dat(cls, path, events, **kwargs):
        return cls(path=path, events=events, **kwargs)

    @classmethod
    def _run(cls, cli_args, temp_dir):
        logger = cls.get_logger()

        trace_dump = get_bin('trace-dump')
        errors_path = 'errors.json'
        cmd = (
            trace_dump,
            '--errors-json', errors_path,
            *cli_args,
        )
        pretty_cmd = ' '.join(map(shlex.quote, map(str, cmd)))

        def log(stdout, stderr):
            stderr = stderr.decode()
            logger.debug(f'{pretty_cmd}:\n{stderr}')

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                cwd=temp_dir,
            )
        except subprocess.CalledProcessError as e:
            log(e.stdout, e.stderr)
            raise
        else:
            stdout = completed.stdout
            log(stdout, completed.stderr)
            return stdout
        finally:
            try:
                with open(temp_dir / errors_path) as f:
                    errors = json.load(f)
            except FileNotFoundError:
                pass
            else:
                # Die on general errors that are not associated with a specific
                # event
                if errors := errors['errors']:
                    raise TraceDumpError(errors, cmd=cmd)

    @classmethod
    def _process_metadata(cls, meta):
        try:
            meta['trace-id'] = f'trace.dat-{meta["trace-id"]}'
        except KeyError:
            pass

        try:
            start, end = meta['time-range']
        except KeyError:
            pass
        else:
            start = Timestamp(start, unit='ns', rounding='down')
            end = Timestamp(end, unit='ns', rounding='up')
            meta['time-range'] = (start, end)

        try:
            meta['pid-comms'] = dict(meta['pid-comms'])
        except KeyError:
            pass

        # symbols-address is stored as a list of items in JSON, since JSON
        # objects can only have string keys.
        try:
            meta['symbols-address'] = dict(meta['symbols-address'])
        except KeyError:
            pass

        try:
            events_info = meta['events-info']
        except KeyError:
            pass
        else:
            # The parser reports all the events that appeared in the trace even
            # if they were not collected.
            meta['available-events'] = {
                desc['event']
                for desc in events_info
                if not desc.get('errors')
            }

        return meta

    @classmethod
    def _make_metadata(cls, path, trace_format, temp_dir, needed_metadata):
        stdout = cls._run(
            cli_args=(
                'metadata',
                '--trace', path,
                '--trace-format', trace_format,
                '--max-errors', str(cls._MAX_ERRORS),
                *(
                    arg
                    for key in sorted(set(needed_metadata or []))
                    for arg in ('--key', key)
                )
            ),
            temp_dir=temp_dir,
        )
        return cls._process_metadata(json.loads(stdout))

    @classmethod
    def _make_parquets(cls, events, path, trace_format, temp_dir):
        # Let the parser parse all available events
        if events is _ALL_EVENTS:
            events = []
        else:
            events = [
                arg
                for event in sorted(set(events))
                # If we don't enable the source of the meta event, the
                # parser will reject the source and we will never get a
                # chance to get the meta event itself, so we need both.
                for _event in Trace.get_event_sources(event)
                for arg in ('--event', _event)
            ]

        cls._run(
            cli_args=(
                'parquet',
                '--trace', path,
                '--trace-format', trace_format,
                '--max-errors', str(cls._MAX_ERRORS),
                '--compression', 'lz4',
                '--unique-timestamps',
                *events,
            ),
            temp_dir=temp_dir,
        )

        with open(temp_dir / 'meta.json') as f:
            meta = json.load(f)

        meta = cls._process_metadata(meta)
        return meta

    @property
    @memoized
    def _event_descs(self):
        return {
            desc['event']: desc
            for desc in self._metadata['events-info']
        }

    @property
    @memoized
    def _pid_comms(self):
        pid_comms = {
            0: '<idle>',
            **self._metadata['pid-comms']
        }

        class Mapper(dict):
            # This avoids getting NaN and is consistent with trace-cmd display
            def __missing__(self, _):
                return '<...>'
        return Mapper(pid_comms)

    def parse_event(self, event):
        try:
            desc = self._event_descs[event]
        except KeyError:
            raise MissingTraceEventError([event])
        else:
            pid_comms = self._pid_comms
            temp_dir = Path(self._temp_dir)

            if errors := desc.get('errors', []):
                # Only raise a TraceDumpError if the event is contained within
                # the trace but we had problems parsing it out
                raise TraceDumpError(errors, event=event)
            else:
                if (path := desc.get('path')) is None:
                    raise FileNotFoundError(f'No parquet file for event "{event}"')
                else:
                    df = pl.scan_parquet(temp_dir / path)
                    df = self._fixup_df(
                        event=event,
                        df=df,
                        pid_comms=pid_comms,
                    )
                    return _ParsedDataFrame.from_df(
                        df=df,
                        swap_cacheable=True,
                        mem_cacheable=True,
                        nr_rows=desc['nr-rows'],
                    )

    def _fixup_df(self, event, df, pid_comms):
        df = (
            df
            .drop('Time', '__pid', '__cpu', strict=False)
            .rename({
                'common_ts': 'Time',
                'common_pid': '__pid',
                'common_cpu': '__cpu',
            })
        )
        df = df.with_columns([
            pl.col('Time').cast(pl.Duration("ns")),
            pl.col('__pid').replace_strict(pid_comms, default=None).alias('__comm')
        ])
        df = df.drop(('common_type', 'common_flags', 'common_preempt_count'), strict=False)

        monotonic_clocks = (
            'local',
            'global',
            'uptime',
            'x86-tsc',
            'mono',
            'mono_raw',
            'counter'
        )
        if self._metadata.get('trace-clock') in monotonic_clocks:
            df = df.set_sorted('Time')
        else:
            df = df.sort('Time')

        # Turn all string columns into categorical columns, since strings are
        # typically extremely repetitive
        df = df.with_columns((cs.string() | cs.binary()).cast(pl.Categorical))

        return df

    def get_metadata(self, key):
        try:
            return self._metadata[key]
        except KeyError:
            return super().get_metadata(key=key)


class EventParserBase:
    """
    Base class for trace event parser.

    Required attributes or properties:

    * ``event``: name of the event
    * ``regex``: full regex to parse a line of the event
    * ``fields``: mapping of field names to :mod:`pandas` dtype to use for
      the :class:`pandas.DataFrame` column.
    """

    PARSER_REGEX_TERMINALS = dict(
        # To be used with the re.ASCII regex flag
        identifier=r'\w+',
        integer=r'\d+',
        floating=r'\d+\.\d+',
        blank=r' +',
    )
    """
    Snippets of regex to be used in building more complex regexes in textual trace parsers.

    .. note:: Meant to be used with the :data:`re.ASCII` flags.
    """

    def __init__(self, event, fields):
        self.event = event
        self.fields = fields


class TxtEventParser(EventParserBase):
    """
    Trace event parser for raw output of ``trace-cmd report -R trace.dat``.

    :param event: name of the event
    :type event: str

    :param fields: mapping of field name to :class:`pandas.DataFrame` column
        dtype to use for each.
    :type fields: dict(str, str)

    :param positional_field: Name of the positional field. If ``None``, no
        positional field will be parsed.
    :type positional_field: str or None

    :param greedy_field: Name of a greedy field that will consume the
        remainder of the line, no matter what the content is. This allows
        parsing events with a field containing a string formatted itself as
        an event
    :type greedy_field: str or None

    :param raw: If ``True``, `trace-cmd report` will be used in raw mode. This
        usually ensures compliance with the format, but may sometimes be a problem.
        For exampe ``const char*`` are displayed as an hex pointer in raw mode,
        which is not helpful.
    :type raw: bool

    Parses events with the following format::

          devlib:     <idle>-0     [001]    76.214046: sched_wakeup: something here: comm=watchdog/1 pid=15 prio=0 success=1 target_cpu=1
          \\____/     \\___________________________________________/ \\____________/ \\_________________________________________________/
          buffer                          header                        positional                          fields
        (optional)

    """

    def __init__(self, event, fields, positional_field=None, greedy_field=None, raw=True):
        super().__init__(
            event=event,
            fields=fields,
        )
        regex = self._get_regex(event, fields, positional_field, greedy_field)
        self.regex = re.compile(regex, flags=re.ASCII)
        self.raw = raw

    @property
    def bytes_regex(self):
        """
        Same as ``regex`` but acting on :class:`bytes` instead of :class:`str`.
        """
        regex = self.regex
        return re.compile(
            self.regex.pattern.encode('ascii'),
            flags=regex.flags,
        )

    @classmethod
    def _get_fields_regex(cls, event, fields, positional_field, greedy_field):
        """
        Returns the regex to parse the fields part of the event line.

        :param event: name of the event
        :type event: str

        :param fields: Mapping of field names to dataframe column dtype
        :type fields: dict(str, str)

        :param positional_field: Name to give to the positional field column,
            or ``None`` if it should not be parsed.
        :type positional_field: str or None

        :param greedy_field: Name of a greedy field that will consume the
            remainder of the line, no matter what the content is. This allows
            parsing events with a field containing a string formatted itself as
            an event
        :type greedy_field: str or None
        """
        # pylint: disable=unused-argument
        fields = fields.keys() - {positional_field}

        if fields:
            def combine(fields):
                return r'(?:{})+'.format(
                    '|'.join(fields)
                )

            def make_regex(field):
                if field == greedy_field:
                    return r'{field}=(?P<{field}>.*)'.format(
                        field=re.escape(field),
                        **cls.PARSER_REGEX_TERMINALS
                    )
                else:
                    # The non-capturing group with positive lookahead is
                    # necessary to be able to correctly collect spaces in the
                    # values of fields
                    return r'{field}=(?P<{field}>.+?)(?:{blank}|$)'.format(
                        field=re.escape(field),
                        **cls.PARSER_REGEX_TERMINALS
                    )

            fields_regexes = list(map(make_regex, fields))

            # Catch-all field that will consume any unknown field, allowing for
            # partial parsing (both for performance/memory consumption and
            # forward compatibility)
            fields_regexes.append(r'{identifier}=\S+?(?:{blank}|$)'.format(
                other_fields='|'.join(fields),
                **cls.PARSER_REGEX_TERMINALS
            ))

            fields = combine(fields_regexes)
        else:
            fields = ''

        if positional_field:
            # If there are more fields to match, use the first ":" or spaces as
            # separator, otherwise just consume everything
            if fields:
                fields =  fr' *:? *{fields}'

            fields = r'(?P<{pos}>.*?){fields}$'.format(pos=positional_field, fields=fields, **cls.PARSER_REGEX_TERMINALS)

        return fields

    @classmethod
    def _get_header_regex(cls, event):
        """
        Return the regex for the header of the event.

        :param event: Name of the event
        :type event: str
        """
        blank = cls.PARSER_REGEX_TERMINALS['blank']
        regex_map = dict(
            __comm=r'.+',
            __pid=cls.PARSER_REGEX_TERMINALS['integer'],
            __cpu=cls.PARSER_REGEX_TERMINALS['integer'],
            __timestamp=cls.PARSER_REGEX_TERMINALS['floating'],
            __event=re.escape(event),
        )

        compos = {
            field: fr'(?P<{field}>{regex})'
            for field, regex in regex_map.items()
        }

        # We don't need to capture these ones as they have already been parsed
        # in the skeleton dataframe, and fixed up for __timestamp
        compos.update(
            (field, regex)
            for field, regex in regex_map.items()
            if field in ('__timestamp', '__event')
        )

        regex = r'(?:(?:^.*?:)|^){blank}{__comm}-{__pid}{blank}\[{__cpu}\]{blank}{__timestamp}:{blank}{__event}:'.format(**compos, blank=blank)
        return regex

    def _get_regex(self, event, fields, positional_field, greedy_field):
        """
        Return the full regex to parse the event line.

        This includes both the header and the fields part.
        """
        fields = self._get_fields_regex(event, fields, positional_field, greedy_field)
        header = self._get_header_regex(event)
        return r'{header} *{fields}'.format(header=header, fields=fields, **self.PARSER_REGEX_TERMINALS)


class PrintTxtEventParser(TxtEventParser):
    """
    Event parser for the folling events, displayed in non-raw format by
    ``trace-cmd``:

    * ``print``
    * ``bprint``
    * ``bputs``

    .. note:: ``bputs`` and ``print`` could be parsed in raw format, but that
        would make them harder to parse (function resolution needed), and
        ``bprint`` is just impossible to parse in raw format, since the data to
        interpolate the format string with are not displayed by ``trace-cmd``.
    """
    def __init__(self, event, func_field, content_field):
        fields = {
            func_field: 'string',
            # Use bytes so that we can easily parse meta events out of it
            # without conversion
            content_field: 'bytes',
        }
        self._func_field = func_field
        self._content_field = content_field
        super().__init__(event=event, fields=fields, raw=False)

    def _get_fields_regex(self, event, fields, positional_field, greedy_field):
        return fr'(?P<{self._func_field}>.*?): *(?P<{self._content_field}>.*)'


class CustomFieldsTxtEventParser(TxtEventParser):
    """
    Subclass of :class:`TxtEventParser` to be used for funky formats.

    When the format of the textual event does not respect at all the raw
    ``trace-cmd`` format, and if raw format cannot be used (e.g. because of
    ``const char*`` fields), this class provides a way out. For example, this
    event can be parsed with this class, but would be impossible to be parse
    using :class:`TxtEventParser`::

        # non-raw format lacks a field delimiter for the "reason"
        kworker/u16:6-262   [003]   177.417147: ipi_raise:            target_mask=00000000,00000020 (Function call interrupts)
        # raw format, even less usable because of the const char* pointer not being resolved
        kworker/u16:6-262   [003]   177.417147: ipi_raise:             target_cpus=ARRAY[20, 00, 00, 00, 00, 00, 00, 00] reason=0xffffff8c0774fe6b

    .. note:: Use :class:`TxtEventParser` if possible, since it provides a more
        optimized fields regex than what you are likely to come up with, and
        can deal with missing fields.

    :param event: Name of the event.
    :type event: str

    :param fields_regex: Regex to parse the fields part of the event occurence.
        Regex groups are used to delimit fields, e.g.
        ``r"field1=(?P<field1>[0-9]+)"`` would recognize ``"field1=42"`` as a
        ``field1`` column.
    :type fields_regex: str

    :param fields: Mapping of field names (group names in the regex) to dtype
        to use in the :class:`pandas.DataFrame`. This is passed to
        :func:`lisa.datautils.series_convert` so the accepted values are a bit
        wider than :mod:`pandas` dtypes.
    :type fields: dict(str, object)

    :param raw: If ``True``, the event will be parsed as raw by ``trace-cmd``.
        If you have ``const char*`` fields, this must be ``False`` in order to
        get the string instead of the pointer.
    :type raw: bool
    """
    def __init__(self, event, fields_regex, fields, raw):
        self._fields_regex = fields_regex
        super().__init__(event=event, fields=fields, raw=raw)

    def _get_fields_regex(self, event, fields, positional_field, greedy_field):
        return self._fields_regex


class TxtTraceParserBase(TraceParserBase):
    """
    Text trace parser base class.

    :param lines: Iterable of text lines as :class:`bytes`.
    :type lines: collections.abc.Iterable(bytes)

    :param events: List of events that will be available using
        :meth:`parse_event`. If not provided, all events will be considered.
        .. note:: Restricting the set of events can speed up some operations.
    :type events: list(str)

    :param event_parsers: Pre-built event parsers. Missing event parsers will
        be inferred from the fields parsed in the trace, which is costly and
        can lead to using larger dtypes than necessary (e.g. ``int64`` rather
        than ``uint16``).

        .. seealso:: :class:`TxtEventParser`
    :type event_parsers: list(EventParserBase)

    :param default_event_parser_cls: Class used to build event parsers inferred from the trace.
    :type default_event_parser_cls: type

    :param pre_filled_metadata: Metadata pre-filled by the caller of the
        constructor.
    :type pre_filled_metadata: dict(str, object) or None
    """

    _KERNEL_DTYPE = {
        'timestamp': 'uint64',
        'pointer': 'uint64',
        'symbol': 'string',
        'cpu': 'uint16',
        'pid': 'uint32',
        'signed_pid': 'int32',
        'comm': 'string',
        'cgroup_path': 'string',
        # prio in [-1, 140]
        'prio': 'int16',
        'util': 'uint16',
    }
    """
    Dtypes for various columns that occur frequently in multiple events.
    """

    HEADER_FIELDS = {
        '__comm': 'string',
        '__pid': 'uint32',
        '__cpu': 'uint32',
        '__timestamp': 'float64',
        '__event': 'string',
    }
    """
    Pandas dtype of the header fields.
    """

    DTYPE_INFERENCE_ORDER = ['int64', 'uint64', 'float64']
    """
    When the dtype of a field is not provided by a user-defined parser, these
    dtypes will be tried in order to convert the column from string to
    something more appropriate.

    .. note:: ``uint64`` allows testing for hexadecimal formatting of numbers.
    """

    DEFAULT_EVENT_PARSER_CLS = None
    """
    Class used to create event parsers when inferred from the trace.
    """

    EVENT_DESCS = {}
    """
    Mapping of event names to parser description as a dict.

    Each event description can include the constructor parameters of the class
    used as :attr:`DEFAULT_EVENT_PARSER_CLS`, which will be used to build event
    parsers from the descriptions.

    If an instance of :class:`EventParserBase` is passed instead of a dict, it
    will be used as is.
    """

    _RE_MATCH_CLS = re.Match

    @kwargs_forwarded_to(TraceParserBase.__init__)
    def __init__(self,
        lines,
        events=None,
        needed_metadata=None,
        event_parsers=None,
        default_event_parser_cls=None,
        pre_filled_metadata=None,
        **kwargs,
    ):
        needed_metadata = set(needed_metadata or [])
        super().__init__(events, needed_metadata=needed_metadata, **kwargs)
        self._pre_filled_metadata = pre_filled_metadata or {}
        events = set(events or [])

        if events or needed_metadata - {'trace-id'}:
            default_event_parser_cls, event_parsers = self._resolve_event_parsers(event_parsers, default_event_parser_cls)

            # Remove all the parsers that are unnecessary
            event_parsers = {
                event: parser
                for event, parser in event_parsers.items()
                if event in events
            }

            # If we don't need the fields in the skeleton df, avoid collecting them
            # to save memory and speed things up
            need_fields = (events != event_parsers.keys())
            skeleton_regex = self._get_skeleton_regex(need_fields)

            self.logger.debug(f'Scanning the trace for metadata {needed_metadata} and events: {events}')

            events_df, skeleton_df, time_range, available_events = self._eagerly_parse_lines(
                lines=lines,
                skeleton_regex=skeleton_regex,
                event_parsers=event_parsers,
                events=events,
            )
            self._events_df = events_df
            self._time_range = time_range
            self._skeleton_df = skeleton_df
            self._available_events = available_events

            inferred_event_descs = self._get_event_descs(skeleton_df, events, event_parsers)
            # We only needed the fields to infer the descriptors, so let's drop
            # them to lower peak memory usage
            with contextlib.suppress(KeyError):
                self._skeleton_df = self._skeleton_df.drop(['__fields'], strict=False)

            event_parsers = {
                **{
                    event: default_event_parser_cls(
                        event=event,
                        **desc,
                    )
                    for event, desc in inferred_event_descs.items()
                },
                # Existing parsers take precedence so the user can override
                # autodetected events
                **event_parsers,
            }
            self._event_parsers = event_parsers
        else:
            self._events_df = {}
            self._time_range = None
            self._skeleton_df = None
            self._available_events = None
            self._event_parsers = {}

    @classmethod
    def _resolve_event_parsers(cls, event_parsers, default_event_parser_cls):
        default_event_parser_cls = default_event_parser_cls or cls.DEFAULT_EVENT_PARSER_CLS
        event_descs = list(cls.EVENT_DESCS.items() or [])

        # Hack so that we use the format of the "lisa__<event>" event even for
        # non-LISA events. This allows compatibility with old traces where
        # events generated by LISA were not namespaced with the prefix.
        extra_descs = [
            (event[len('lisa__'):], desc)
            for event, desc in event_descs
            if event.startswith('lisa__')
        ]
        event_descs = event_descs + extra_descs

        event_parsers = {
            **{
                event: (
                    desc
                    if isinstance(desc, EventParserBase)
                    else default_event_parser_cls(event=event, **desc)
                )
                for event, desc in event_descs
            },
            **{
                parser.event: parser
                for parser in event_parsers or []
            }
        }

        return (default_event_parser_cls, event_parsers)


    @PartialInit.factory
    @kwargs_forwarded_to(__init__, ignore=['lines'])
    def from_string(cls, txt, path=None, **kwargs):
        """
        Build an instance from a single multiline string.

        :param txt: String containing the trace. It will be encoded as ASCII
            before being forwarded to the constructor if it's not already
            :class:`bytes`.
        :type txt: bytes or str

        :Variable keyword arguments: Forwarded to ``__init__``
        """
        # The text could already be bytes, in which case this will fail
        with contextlib.suppress(AttributeError):
            txt = txt.encode('ascii')

        return cls(lines=txt.splitlines(), **kwargs)

    @PartialInit.factory
    @kwargs_forwarded_to(__init__, ignore=['lines'])
    def from_txt_file(cls, path, **kwargs):
        """
        Build an instance from a path to a text file.

        :Variable keyword arguments: Forwarded to ``__init__``
        """
        with open(path, 'rb') as f:
            return cls(lines=f, **kwargs)

    @abc.abstractmethod
    def _get_skeleton_regex(self, need_fields):
        """
        Return a :class:`bytes` regex that provides the following groups:

            * ``__event``: name of the event
            * ``__timestamp``: timestamp of event occurence
            * ``__fields`` if ``need_fields == True``: a string containing all
              the named fields of each event occurence.

        .. note:: This regex is critical for performances as it will be used to
            scan the whole trace.
        """

    @staticmethod
    def _make_df_from_data(regex, data, extra_cols=None):
        extra_cols = extra_cols or []
        columns = sorted(
            regex.groupindex.keys(),
            # Order columns so that we can directly append the
            # groups() tuple of the regex match
            key=lambda field: regex.groupindex[field]
        )
        # Rename regex columns to avoid clashes that were explicitly added as
        # extra
        columns = [
            f'__parsed_{col}' if col in extra_cols else col
            for col in columns
        ]
        columns += extra_cols

        df = pl.DataFrame(
            data,
            orient='row',
            schema=columns,
            # Scan the entire dataset to figure out the best dtype
            infer_schema_length=None,
        ).lazy()

        df = df.with_columns(cs.binary().cast(pl.String))

        # Put the timestamp first so it's recognized as the index
        df = df.select(
            order_as(columns, ['__timestamp'])
        )
        return df

    def _eagerly_parse_lines(self, lines, skeleton_regex, event_parsers, events, time=None, time_unit='s'):
        """
        Filter the lines to select the ones with events.

        Also eagerly parse events from them to avoid the extra memory
        consumption from line storage, and to speed up parsing by acting as a
        pipeline on lazy lines stream.
        """

        # Recompile all regex so that they work on bytes rather than strings.
        # This simplifies the rest of the code while allowing the raw output
        # from a process to be fed
        def encode(string):
            return string.encode('ascii')

        events = list(map(encode, events))
        event_parsers = {
            encode(event): parser
            for event, parser in event_parsers.items()
        }

        # Only add an extra iterator and tuple unpacking if that is strictly
        # necessary, as it comes with a performance cost
        time_is_provided = time is not None
        skel_search = skeleton_regex.search
        if time_is_provided:
            lines = zip(time, lines)
            drop_filter = lambda line: not skel_search(line[1])
        else:
            drop_filter = lambda line: not skel_search(line)

        # First, get rid of all the lines coming before the trace
        lines = itertools.dropwhile(drop_filter, lines)

        # Appending to lists is amortized O(1). Inside the list, we store
        # tuples since they are:
        # 1) the most compact Python representation of a product type
        # 2) output directly by regex.search()
        skeleton_data = []
        events_data = {
            **{event: (None, None) for event in events},
            **{
                event: (parser.bytes_regex.search, [])
                for event, parser in event_parsers.items()
            },
        }
        available_events = set()

        begin_time = None
        end_time = None

        # THE FOLLOWING LOOP IS A THE MOST PERFORMANCE-SENSITIVE PART OF THAT
        # CLASS, APPLY EXTREME CARE AND BENCHMARK WHEN MODIFYING
        # Best practices:
        # - resolve all dotted names ahead of time
        # - minimize the amount of local variables. Prefer anonymous
        #   expressions
        # - Catch exceptions for exceptional cases rather than explicit check

        # Pre-lookup methods out of the loop to speed it up
        append = list.append
        group = self._RE_MATCH_CLS.group
        groups = self._RE_MATCH_CLS.groups
        inf = math.inf
        prev_time = Timestamp(0, unit='ns')
        parse_time = '__timestamp' in skeleton_regex.groupindex.keys()

        for line in lines:
            if time_is_provided:
                line_time, line = line

            match = skel_search(line)
            # Stop at the first non-matching line
            try:
                event = group(match, '__event')
            # The line did not match the skeleton regex, so skip it
            except TypeError:
                if b'EVENTS DROPPED' in line:
                    raise DroppedTraceEventError('The trace buffer got overridden by new data, increase the buffer size to ensure all events are recorded')
                # Unknown line, could be coming e.g. from stderr
                else:
                    continue
            else:
                if not time_is_provided:
                    line_time = Timestamp(
                        group(match, '__timestamp').decode('utf-8')
                    )
                    # Do a global deduplication of timestamps, across all
                    # events regardless of the one we will parse. This ensures
                    # stable results and joinable dataframes from multiple
                    # parser instance.
                    line_time = line_time.as_nanoseconds
                    if line_time <= prev_time:
                        line_time += prev_time - line_time + 2
                    prev_time = line_time

            if begin_time is None:
                begin_time = line_time

            # If we can parse it right away, let's do it now
            try:
                search, data = events_data[event]
                append(
                    data,
                    # Add the fixedup time
                    groups(search(line)) + (line_time,)
                )
            # If we don't have a parser for it yet (search == None),
            # just store the line so we can infer its parser later
            except TypeError:
                # Add the fixedup time and the full line for later
                # parsing as well
                append(
                    skeleton_data,
                    groups(match) + (line_time, line)
                )
            # We are not interested in that event, but we still remember the
            # pareseable events
            except KeyError:
                available_events.add(event)

        # This should have been set on the first line.
        # Note: we don't raise the exception if no events were asked for, to
        # allow creating dummy parsers without any line
        if begin_time is None and events:
            raise ValueError('No lines containing events have been found')

        available_events = {
            event.decode('ascii')
            for event in available_events
        }

        end_time = line_time
        available_events.update(
            event.decode('ascii')
            for event, (search, data) in events_data.items()
            if data
        )

        events_df = {}
        for event, parser in event_parsers.items():
            try:
                # Remove the tuple data from the dict as we go, to free memory
                # before proceeding to the next event to smooth the peak memory
                # consumption
                _, data = events_data.pop(event)
            except KeyError:
                pass
            else:
                decoded_event = event.decode('ascii')
                df = self._make_df_from_data(parser.regex, data, ['__timestamp'])
                # Post-process immediately to shorten the memory consumption
                # peak
                df = self._postprocess_df(
                    decoded_event,
                    parser,
                    df,
                )
                events_df[decoded_event] = df

        # Compute the skeleton dataframe for the events that have not been
        # parsed already. It contains the event name, the time, and potentially
        # the fields if they are needed
        skeleton_df = self._make_df_from_data(skeleton_regex, skeleton_data, ['__timestamp', 'line'])
        skeleton_df = skeleton_df.with_columns(
            pl.col('__event').cast(pl.Categorical)
        )

        # Drop unnecessary columns that might have been parsed by the regex
        to_keep = {'__timestamp', '__event', '__fields', 'line'}
        skeleton_df = skeleton_df.select(sorted(to_keep & set(skeleton_df.collect_schema().names())))
        # This is very fast on a category dtype
        available_events.update(
            skeleton_df.select(
                pl.col('__event').unique()
            ).collect()['__event']
        )

        if time_is_provided:
            begin_time = Timestamp(begin_time, unit=time_unit)
            end_time = Timestamp(end_time, unit=time_unit)
        else:
            begin_time = Timestamp(begin_time, unit='ns')
            end_time = Timestamp(end_time, unit='ns')

        return (events_df, skeleton_df, (begin_time, end_time), available_events)

    def _lazily_parse_event(self, event, parser, df):
        # Only parse the lines that have a chance to match
        df = df.filter(
            pl.col('__event') == event
        )

        pattern = parser.bytes_regex.pattern.decode('utf-8')

        new_df = df.select((
            pl.col('__timestamp'),
            pl.col('line').str.strip_chars().str.extract_groups(pattern),
        )).unnest('line')
        new_df = self._postprocess_df(event, parser, new_df)
        return new_df

    @staticmethod
    def _get_event_descs(df, events, event_parsers):
        user_supplied = event_parsers.keys()
        all_events = events is None

        if not all_events and set(events) == user_supplied:
            return {}
        else:
            def encode(string):
                return string.encode('ascii')

            # Find the field names only for the events we don't already know about,
            # since the inference is relatively expensive
            if all_events:
                df = df.filter(
                    pl.col('__event').is_in(user_supplied).not_()
                )
            else:
                events = set(events) - user_supplied
                df = df.filter(
                    pl.col('__event').is_in(events)
                )

            fields_regex = r'({identifier})='.format(
                **TxtEventParser.PARSER_REGEX_TERMINALS
            )

            df = df.with_columns(
                pl.col('__fields')
                .cast(pl.String)
                .str.extract_all(fields_regex)
                # We can't match groups and extract all occurences at the
                # moment, so we postprocess the field name instead:
                # https://github.com/pola-rs/polars/issues/11857
                .list.eval(
                    pl.element()
                    .str.strip_chars_end('=')
                )
            )

            df = df.group_by('__event').agg(
                pl.col('__fields')
                .list.explode()
                .unique()
            )

            def make_desc(event, fields):
                new = dict(
                    positional_field=None,
                    fields=dict.fromkeys(fields, None)
                )
                return new

            dct = {
                event: make_desc(event, fields)
                for event, fields in df.select((
                    pl.col('__event').cast(pl.String),
                    pl.col('__fields'),
                )).collect().iter_rows()
            }
            return dct

    def parse_event(self, event):
        try:
            parser = self._event_parsers[event]
        except KeyError as e:
            raise MissingTraceEventError([event])

        # Maybe it was eagerly parsed
        try:
            df = self._events_df[event]
        except KeyError:
            df = self._lazily_parse_event(event, parser, self._skeleton_df)

        # Everything resides in memory anyway, so make that clear to the _Trace
        # infrastructure so it does not accidentally tries to serialize a
        # LazyFrame as a huge JSON.
        df = df.collect()

        # Since there is no way to distinguish between no event entry and
        # non-collected events in text traces, map empty dataframe to missing
        # event
        if len(df):
            return df
        else:
            raise MissingTraceEventError([event])

    @classmethod
    def _postprocess_df(cls, event, parser, df):
        assert isinstance(df, pl.LazyFrame)

        def get_representative(df, col):
            df = (
                df
                .select(col)
                .filter(
                    pl.col(col).is_null().not_()
                )
                .head(1)
                .collect()
            )
            try:
                return df.item()
            except ValueError:
                return None

        def infer_schema(df):
            # Select a subset of the dataframe that has no null value in it, so
            # we can accurately infer the dtype.
            df = pl.DataFrame({
                col: [get_representative(df, col)]
                for col in df.collect_schema().names()
            })

            # Ugly hack: dump the first row to CSV to infer the schema of the
            # dataframe based on text data, since that's what we have.
            bytes_io = io.BytesIO()
            df.write_csv(bytes_io)
            df = pl.read_csv(bytes_io)
            schema = df.schema

            schema = {
                # Cast all strings as categorical since they are typically very
                # repetitive
                col: pl.Categorical if isinstance(dtype, (pl.String, pl.Binary)) else dtype
                for col, dtype in schema.items()
            }

            return schema

        # Polars will fail to convert strings with trailing spaces (e.g. "42  "
        # into 42), so ensure we clean that up before conversion
        df = df.with_columns(cs.string().str.strip_chars())

        df = df.with_columns(
            pl.col(name).cast(dtype)
            for name, dtype in infer_schema(df).items()
        )
        df = (
            df
            .drop('Time', strict=False)
            .rename({'__timestamp': 'Time'})
        )

        schema = df.collect_schema()
        if event == 'sched_switch':
            if isinstance(schema['prev_state'], (pl.String, pl.Categorical)):
                # Avoid circular dependency issue by importing at the last moment
                # pylint: disable=import-outside-toplevel
                from lisa.analysis.tasks import TaskState
                df = df.with_columns(
                    pl.col('prev_state').map_elements(
                        TaskState.from_sched_switch_str,
                        return_dtype=pl.UInt64,
                    ).cast(pl.UInt16)
                )

        elif event in ('sched_overutilized', 'lisa__sched_overutilized'):
            df = df.with_columns(
                pl.col('overutilized').cast(pl.Boolean)
            )

            if isinstance(schema.get('span'), (pl.String, pl.Binary, pl.Categorical)):
                df = df.with_columns(
                    pl.col('span')
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.strip_prefix('0x')
                    .str.to_integer(base=16)
                )

        elif event in ('thermal_power_cpu_limit', 'thermal_power_cpu_get_power'):

            # In-kernel name is "cpumask", "cpus" is just an artifact of the pretty
            # printing format string of ftrace, that happens to be used by a
            # specific parser.
            df = (
                df
                .drop('cpumask', strict=False)
                .rename({'cpus': 'cpumask'})
            )

            if event == 'thermal_power_cpu_get_power':
                if isinstance(schema['load'], (pl.String, pl.Binary, pl.Categorical)):
                    df = df.with_columns(
                        # Parse b'{2 3 2 8}'
                        pl.col('load')
                        .cast(pl.String)
                        .str.strip_chars('{}')
                        .str.split(' ')
                        .list.eval(
                            pl.element()
                            .str.to_integer()
                        )
                    )

        elif event in ('ipi_entry', 'ipi_exit'):
            df = df.with_columns(
                pl.col('reason').cast(pl.String).str.strip_chars('()').cast(pl.Categorical)
            )

        return df

    def get_metadata(self, key):
        time_range = self._time_range
        if key == 'time-range' and time_range:
            return time_range

        # If we filtered some events, we are not exhaustive anymore so we
        # cannot return the list
        if (
            key == 'available-events' and
            (
                (available_events := self._available_events)
                is not None
            )
        ):
            return available_events

        try:
            return self._pre_filled_metadata[key]
        except KeyError:
            return super().get_metadata(key)

class TxtTraceParser(TxtTraceParserBase):
    """
    Text trace parser for the raw output of ``trace-cmd report -R trace.dat``.

    :param lines: Iterable of text lines.
    :type lines: collections.abc.Iterable(str)

    :param events: List of events that will be available using
        :meth:`~TxtTraceParserBase.parse_event`. If not provided, all events will be considered.
        .. note:: Restricting the set of events can speed up some operations.
    :type events: list(str) or None

    :param event_parsers: Pre-built event parsers. Missing event parsers will
        be inferred from the fields parsed in the trace, which is costly and can
        lead to using larger dtypes than necessary (e.g. ``int64`` rather than
        ``uint16``).

        .. seealso:: :class:`TxtEventParser`
    :type event_parsers: list(EventParserBase)

    :param default_event_parser_cls: Class used to build event parsers inferred from the trace.
    """
    DEFAULT_EVENT_PARSER_CLS = TxtEventParser

    _KERNEL_DTYPE = TxtTraceParserBase._KERNEL_DTYPE
    EVENT_DESCS = {
        'print': PrintTxtEventParser(
            event='print',
            func_field='ip',
            content_field='buf',
        ),
        'bprint': PrintTxtEventParser(
            event='bprint',
            func_field='ip',
            content_field='buf',
        ),
        'bputs': PrintTxtEventParser(
            event='bputs',
            func_field='ip',
            content_field='str',
        ),
        'thermal_power_cpu_limit': dict(
            fields={
                'cpus': 'bytes',
                'freq': 'uint32',
                'cdev_state': 'uint64',
                'power': 'uint32',
            },
            # Allow parsing the cpus bitmask
            raw=False,
        ),
        'thermal_power_cpu_get_power': dict(
            fields={
                'cpus': 'bytes',
                'freq': 'uint32',
                'load': 'bytes',
                'dynamic_power': 'uint32',
            },
            # Allow parsing the cpus bitmask and load array
            raw=False,
        ),
        'cpuhp_enter': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'target': 'uint16',
                'idx': 'uint16',
                'fun': _KERNEL_DTYPE['symbol'],
            },
        ),
        'funcgraph_entry': dict(
            fields={
                'func': _KERNEL_DTYPE['pointer'],
                'depth': 'uint16',
            },
        ),
        'funcgraph_exit': dict(
            fields={
                'func': _KERNEL_DTYPE['pointer'],
                'depth': 'uint16',
                'overrun': 'bool',
                'calltime': 'uint64',
                'rettime': 'uint64',
            },
        ),
        'ipi_entry': dict(
            fields={
                'reason': 'string',
            },
            positional_field='reason',
            # const char* reason is not displayed properly in raw mode
            raw=False,
        ),
        'ipi_exit': dict(
            fields={
                'reason': 'string',
            },
            positional_field='reason',
            # const char* reason is not displayed properly in raw mode
            raw=False,
        ),
        'ipi_raise': CustomFieldsTxtEventParser(
            event='ipi_raise',
            fields_regex=r'target_mask=(?P<target_cpus>[0-9,]+) +\((?P<reason>[^)]+)\)',
            fields={
                'target_cpus': 'bytes',
                'reason': 'string',
            },
            raw=False,
        ),
        'sched_switch': dict(
            fields={
                'prev_comm': _KERNEL_DTYPE['comm'],
                'prev_pid': _KERNEL_DTYPE['pid'],
                'prev_prio': _KERNEL_DTYPE['prio'],
                'prev_state': 'uint16',
                'next_comm': _KERNEL_DTYPE['comm'],
                'next_pid': _KERNEL_DTYPE['pid'],
                'next_prio': _KERNEL_DTYPE['prio'],
            },
        ),
        'sched_wakeup': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'pid': _KERNEL_DTYPE['pid'],
                'prio': _KERNEL_DTYPE['prio'],
                'target_cpu': _KERNEL_DTYPE['cpu'],
                # This field does exist but it's useless nowadays as it has a
                # constant value of 1, so save some memory by just not parsing
                # it
                # 'success': 'bool',
            },
        ),
        'task_rename': dict(
            fields={
                'pid': _KERNEL_DTYPE['pid'],
                'oldcomm': _KERNEL_DTYPE['comm'],
                'newcomm': _KERNEL_DTYPE['comm'],
            },
        ),
        'cpu_frequency': dict(
            fields={
                'cpu_id': _KERNEL_DTYPE['cpu'],
                'state': 'uint32',
            },
        ),
        'cpu_idle': dict(
            fields={
                'cpu_id': _KERNEL_DTYPE['cpu'],
                # Technically, the value used in the kernel can go down to -1,
                # but it ends up stored in an unsigned type (why ? ...), which
                # means we actually get 4294967295 in the event, as an unsigned
                # value. Therefore, it's up to sanitization to fix that up ...
                'state': 'int64',
            },
        ),
        'sched_compute_energy': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'dst_cpu': _KERNEL_DTYPE['cpu'],
                'energy': 'uint64',
                'pid': _KERNEL_DTYPE['pid'],
                'prev_cpu': _KERNEL_DTYPE['cpu'],
            },
        ),
        'lisa__sched_cpu_capacity': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'capacity': _KERNEL_DTYPE['util'],
                'capacity_orig': _KERNEL_DTYPE['util'],
                'capacity_curr': _KERNEL_DTYPE['util'],
            },
        ),
        'lisa__sched_pelt_cfs': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'load': _KERNEL_DTYPE['util'],
                'path': _KERNEL_DTYPE['cgroup_path'],
                'rbl_load': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
                'update_time': _KERNEL_DTYPE['timestamp'],
            },
        ),
        'lisa__sched_pelt_se': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'cpu': _KERNEL_DTYPE['cpu'],
                'load': _KERNEL_DTYPE['util'],
                'path': _KERNEL_DTYPE['cgroup_path'],
                'pid': _KERNEL_DTYPE['signed_pid'],
                'rbl_load': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
                'update_time': _KERNEL_DTYPE['timestamp'],
            },
        ),
        'sched_migrate_task': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'dest_cpu': _KERNEL_DTYPE['cpu'],
                'orig_cpu': _KERNEL_DTYPE['cpu'],
                'pid': _KERNEL_DTYPE['pid'],
                'prio': _KERNEL_DTYPE['prio'],
            },
        ),
        'lisa__sched_overutilized': dict(
            fields={
                'overutilized': 'bool',
                'span': 'string',
            },
        ),
        'lisa__sched_pelt_dl': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'load': _KERNEL_DTYPE['util'],
                'rbl_load': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
            },
        ),
        'lisa__sched_pelt_irq': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'load': _KERNEL_DTYPE['util'],
                'rbl_load': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
            },
        ),
        'lisa__sched_pelt_rt': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'load': _KERNEL_DTYPE['util'],
                'rbl_load': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
            },
        ),
        'sched_process_wait': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'pid': _KERNEL_DTYPE['pid'],
                'prio': _KERNEL_DTYPE['prio'],
            },
        ),
        'lisa__sched_util_est_cfs': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'path': _KERNEL_DTYPE['cgroup_path'],
                'enqueued': _KERNEL_DTYPE['util'],
                'ewma': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
            },
        ),
        'lisa__sched_util_est_se': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'comm': _KERNEL_DTYPE['comm'],
                'pid': _KERNEL_DTYPE['signed_pid'],
                'path': _KERNEL_DTYPE['cgroup_path'],
                'enqueued': _KERNEL_DTYPE['util'],
                'ewma': _KERNEL_DTYPE['util'],
                'util': _KERNEL_DTYPE['util'],
            },
        ),
        'sched_wakeup_new': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'pid': _KERNEL_DTYPE['pid'],
                'prio': _KERNEL_DTYPE['prio'],
                'success': 'bool',
                'target_cpu': _KERNEL_DTYPE['cpu'],
            },
        ),
        'sched_waking': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'pid': _KERNEL_DTYPE['pid'],
                'prio': _KERNEL_DTYPE['prio'],
                'success': 'bool',
                'target_cpu': _KERNEL_DTYPE['cpu'],
            },
        ),
        'lisa__uclamp_util_cfs': dict(
            fields={
                'cpu': _KERNEL_DTYPE['cpu'],
                'uclamp_avg': _KERNEL_DTYPE['util'],
                'uclamp_max': _KERNEL_DTYPE['util'],
                'uclamp_min': _KERNEL_DTYPE['util'],
                'util_avg': _KERNEL_DTYPE['util'],
            },
        ),
        'lisa__uclamp_util_se': dict(
            fields={
                'comm': _KERNEL_DTYPE['comm'],
                'cpu': _KERNEL_DTYPE['cpu'],
                'pid': _KERNEL_DTYPE['pid'],
                'uclamp_avg': _KERNEL_DTYPE['util'],
                'uclamp_max': _KERNEL_DTYPE['util'],
                'uclamp_min': _KERNEL_DTYPE['util'],
                'util_avg': _KERNEL_DTYPE['util'],
            },
        ),
    }

    @PartialInit.factory
    @kwargs_forwarded_to(TxtTraceParserBase.__init__, ignore=['lines'])
    def from_dat(cls, path, events, needed_metadata=None, event_parsers=None, default_event_parser_cls=None, **kwargs):
        """
        Build an instance from a path to a trace.dat file created with
        ``trace-cmd``.

        :Variable keyword arguments: Forwarded to ``__init__``

        .. note:: We unfortunately cannot use ``-F`` filter option to
            pre-filter on some events, since global timestamp deduplication has
            to happen. The returned dataframe must be stable, because it could
            be reused in another context (cached on disk), and the set of
            events in a :class:`Trace` object can be expanded dynamically.
        """
        bin_ = get_bin('trace-cmd')

        needed_metadata = set(needed_metadata or [])
        default_event_parser_cls, event_parsers = cls._resolve_event_parsers(event_parsers, default_event_parser_cls)
        event_parsers = event_parsers.values()

        pre_filled_metadata = {}

        if 'symbols-address' in needed_metadata:
            # Get the symbol addresses in that trace
            def parse(line):
                addr, sym = line.split(' ', 1)
                addr = int(addr, base=16)
                sym = sym.strip()
                return (addr, sym)

            symbols_address = dict(
                parse(line)
                for line in subprocess.check_output(
                    [bin_, 'report', '-N', '-f', '--', path],
                    stderr=subprocess.DEVNULL,
                    universal_newlines=True,
                ).splitlines()
            )

            # If we get only "0" as a key, that means kptr_restrict was in use and
            # no useable address is available
            if symbols_address.keys() != {0}:
                pre_filled_metadata['symbols-address'] = symbols_address

        if 'cpus-count' in needed_metadata:
            regex = re.compile(rb'cpus=(?P<cpus>\d+)')
            with subprocess.Popen(
                [bin_, 'report', '-N', '--', path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ) as p:
                try:
                    match, _ = take(1, itertools.filterfalse(None, map(regex.search, p.stdout)))
                except ValueError:
                    pass
                else:
                    pre_filled_metadata['cpus-count'] = match.group('cpus')

        kwargs.update(
            events=events,
            needed_metadata=needed_metadata,
            event_parsers=event_parsers,
            default_event_parser_cls=default_event_parser_cls,
            pre_filled_metadata=pre_filled_metadata,
        )

        cmd = cls._tracecmd_report(
            bin_=bin_,
            path=path,
            events=events,
            event_parsers=event_parsers,
            default_event_parser_cls=default_event_parser_cls,
            # We unfortunately need to parse every single line in order to
            # ensure each event has a unique timestamp in the trace, as pandas
            # cannot deal with duplicated timestamps. Having unique timestamps
            # inside an event dataframe is not enough as dataframes of
            # different events can be combined.
            filter_events=False,
        )

        # A fairly large buffer reduces the interaction overhead
        bufsize = 10 * 1024 * 1024
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=bufsize) as p:
            # Consume the lines as they come straight from the stdout object to
            # avoid the memory overhead of storing the whole output in one
            # gigantic string
            return cls(lines=p.stdout, **kwargs)

    @classmethod
    def _tracecmd_report(cls, bin_, path, events, event_parsers=None, default_event_parser_cls=None, filter_events=True):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Unable to locate specified trace file: {path}')

        # Make sure we only ask to trace-cmd events that can exist, otherwise
        # it might bail out and give nothing at all, especially with -F
        kernel_events = {
            event.split(':', 1)[1]
            for event in subprocess.check_output(
                [bin_, 'report', '-N', '-E', '--', path],
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ).splitlines()
            if not event.startswith('version =')
        }
        events = kernel_events if events is None else set(events)
        events &= kernel_events

        filter_events &= (events != kernel_events)

        default_event_parser_cls, event_parsers = cls._resolve_event_parsers(event_parsers, default_event_parser_cls)
        def use_raw(event):
            try:
                parser = event_parsers[event]
            except KeyError:
                # If we don't have a known parser, use the raw output by
                # default, since it will be either the same as human readable,
                # or unparseable without a dedicated parser.
                return True
            else:
                return parser.raw

        if filter_events:
            filter_ = list(itertools.chain.from_iterable(
                ('-F', event)
                for event in events
            ))
        else:
            filter_ = []

        raw_events = list(itertools.chain.from_iterable(
            ('-r', event) if use_raw(event) else []
            for event in events
        ))
        cmd = [
            bin_,
            'report',
            # Do not load any plugin, so that we get fully reproducible results
            '-N',
            # Full accuracy on timestamp
            '-t',
            # Event filter
            *filter_,
            # All events in raw format
            *raw_events,
            '--', path
        ]
        return cmd

    def _get_skeleton_regex(self, need_fields):
        regex = r'\] +(?P<__timestamp>{floating}): *(?P<__event>{identifier}):'.format(**TxtEventParser.PARSER_REGEX_TERMINALS)
        if need_fields:
            regex += r' *(?P<__fields>.*)'

        return re.compile(regex.encode('ascii'), flags=re.ASCII)


class SimpleTxtTraceParser(TxtTraceParserBase):
    """
    Simple text trace parser (base) class.

    :param lines: Lines of the text to parse.
    :type lines: collections.abc.Iterable(str)

    :param events: List of events that will be potentially parsed by
        :meth:`~TraceParserBase.parse_events`. If ``None``, all available
        will be considered but that may increase the initial parsing stage.
    :type events: list(str)

    :param event_parsers: Optional list of :class:`TxtTraceParserBase` to
        provide fully customized event regex.

        .. note:: See :data:`EVENT_DESCS` for class-provided special case
            handling.
    :type event_parsers: list(TxtTraceParserBase)

    :param header_regex: Regex used to parse the header of each event. See
        :data:`HEADER_REGEX` documentation.
    :type header_regex: str

    .. note:: This class is easier to customize than :class:`TxtTraceParser`
        but may have higher processing time and peak memory usage.
    """

    EVENT_DESCS = {}
    """
    Mapping of event names to parser description as a dict.

    Each event description can include the following dict keys:

    * ``header_regex``: Regex to parse the event header. If not set, the
      header regex from the trace parser will be used.

    * ``fields_regex``: Regex to parse the fields part of the event (i.e.
      the part after the header). This is the most commonly modified
      setting to take into account special cases in event formatting.

    * ``fields``: Mapping of field names to :class:`pandas.DataFrame`
      column dtype. This allows using a smaller dtype or the use of a
      non-inferred dtype like ``boolean``.

    * ``positional_field``: Name of the positional field (comming before
      the named fields). If ``None``, the column will be suppressed in the
      parsed dataframe.
    """

    HEADER_REGEX = None
    """
    Default regex to use to parse event header.
    It must parse the following groups:

    * ``__timestamp``: the timestamp of the event
    * ``__event``: the name of the event
    * ``__cpu`` (optional): the CPU by which the event was emitted
    * ``__pid`` (optional): the currently scheduled PID at the point the event was emitted
    * ``__comm`` (optional): the currently scheduled task's name at the point the event was emitted

    .. note:: It must *not* capture the event fields, as it will be
        concatenated with the field regex of each event to parse full lines.
    """
    # Removing default_event_parser_cls restricts the API of __init__, so that
    # means that inherited alternative constructors such as from_txt_file would
    # need to be overridden to be strictly accurate too.
    @kwargs_forwarded_to(TxtTraceParserBase.__init__, ignore=['default_event_parser_cls'])
    def __init__(self, header_regex=None, **kwargs):
        header_regex = header_regex or self.HEADER_REGEX
        self.header_regex = header_regex

        # Do not parse for each event unnecessary columns that are already
        # parsed in the skeleton dataframe
        regex = header_regex
        for field in ('__timestamp', '__event'):
            regex = regex.replace(fr'(?P<{field}>', r'(?:')
        event_parser_header_regex = regex

        class SimpleTxtEventParser(TxtEventParser):
            def __init__(self, header_regex=event_parser_header_regex, fields_regex=None, **kwargs):
                self.fields_regex = fields_regex
                self.header_regex = header_regex
                super().__init__(**kwargs)

            def _get_header_regex(self, event):
                return self.header_regex

            def _get_fields_regex(self, *args, **kwargs):
                if self.fields_regex:
                    return self.fields_regex
                else:
                    return super()._get_fields_regex(*args, **kwargs)

        super().__init__(
            default_event_parser_cls=SimpleTxtEventParser,
            **kwargs,
        )

    def _get_skeleton_regex(self, need_fields):
        # Parse the whole header, which is wasteful but provides a simpler interface
        regex = self.header_regex + r' *(?P<__fields>.*)'
        return re.compile(regex.encode('ascii'), flags=re.ASCII)


class MetaTxtTraceParser(SimpleTxtTraceParser):
    """
    Textual trace parser to parse meta-events.

    :param time: Iterable of timestamps matching ``lines``.
    :type time: collections.abc.Iterable(float)

    Meta events are events "embedded" as a string inside the field of another
    event. They are expected to comply with the raw format as output by
    ``trace-cmd report -R``.

    """
    HEADER_REGEX = r'(?P<__event>[\w@]+):?'

    class DEFAULT_EVENT_PARSER_CLS(TxtEventParser):
        @classmethod
        def _get_header_regex(cls, event):
            regex = r'^ *{__event}:?'.format(
                __event=re.escape(event),
                **cls.PARSER_REGEX_TERMINALS
            )
            return regex

    EVENT_DESCS = {
        **SimpleTxtTraceParser.EVENT_DESCS,
        # Provide the description of rtapp userspace events to speed up parsing
        # of these
        'userspace@rtapp_loop': dict(
            fields={
                'phase': 'uint32',
                'phase_loop': 'uint32',
                'thread_loop': 'uint32',
                'event': 'string',
            },
        ),
        'userspace@rtapp_main': dict(
            fields={
                'event': 'string',
                'data': None,
            },
        ),
        'userspace@rtapp_task': dict(
            fields={
                'event': 'string',
            },
        ),
        'userspace@rtapp_event': dict(
            fields={
                'desc': 'string',
                # Not sure about the meaning or real size of these
                'id': 'uint32',
                'type': 'uint32',
            },
        ),
        'userspace@rtapp_stats': dict(
            fields={
                'period': 'uint64',
                'run': 'uint64',
                'wa_lat': 'uint64',
                'slack': 'uint64',
                'c_period': 'uint64',
                'c_run': 'uint64',
            },
        ),
    }

    @kwargs_forwarded_to(SimpleTxtTraceParser.__init__)
    def __init__(self, *args, time, time_unit='s', **kwargs):
        self._time = time
        self._time_unit = time_unit
        super().__init__(*args, **kwargs)

    def _eagerly_parse_lines(self, *args, **kwargs):
        # Use the iloc as "time", and we fix it up manually afterwards
        return super()._eagerly_parse_lines(
            *args, **kwargs, time=self._time, time_unit=self._time_unit,
        )


class HRTxtTraceParser(SimpleTxtTraceParser):
    """
    Parse text trace in their human readable format (as opposed to the raw
    format).

    The accepted format is the one produced by the kernel, after formatting
    event records with their format string. This means that format strings
    deviating from the classic ``field=value`` format need a custom regex.

    .. note:: This parser is provided for convenience but is probably not
        complete. More specifically, it does not contain custom regex for all the
        events which format deviates from the raw format as output by ``trace-cmd
        report -R``.

        For a better supported format, see :class:`TxtTraceParser`.
    """

    _KERNEL_DTYPE = SimpleTxtTraceParser._KERNEL_DTYPE
    EVENT_DESCS = {
        'sched_switch': dict(
            fields_regex=r'prev_comm=(?P<prev_comm>.+?) +prev_pid=(?P<prev_pid>\d+) +prev_prio=(?P<prev_prio>\d+) +prev_state=(?P<prev_state>[^ ]+) ==> next_comm=(?P<next_comm>.+?) +next_pid=(?P<next_pid>\d+) +next_prio=(?P<next_prio>\d+)',
            fields={
                'prev_comm': _KERNEL_DTYPE['comm'],
                'prev_pid': _KERNEL_DTYPE['pid'],
                'prev_prio': _KERNEL_DTYPE['prio'],
                'prev_state': 'string',
                'next_comm': _KERNEL_DTYPE['comm'],
                'next_pid': _KERNEL_DTYPE['pid'],
                'next_prio': _KERNEL_DTYPE['prio'],
            },
        ),
        'tracing_mark_write': dict(
            fields={
                'buf': 'bytes',
            },
            positional_field='buf',
        ),
    }

    HEADER_REGEX = r'\s*(?P<__comm>.+)-(?P<__pid>\d+)[^[]*\[(?P<__cpu>\d*)\][^\d]+(?P<__timestamp>\d+\.\d+): +(?P<__event>\w+):'


class SysTraceParser(HRTxtTraceParser):
    """
    Parse Google's systrace format.

    .. note:: This parser is based on :class:`HRTxtTraceParser` and is
        therefore provided for convenience but may lack some events custom
        field regex.
    """

    @PartialInit.factory
    @wraps(HRTxtTraceParser.from_txt_file)
    def from_html(cls, *args, **kwargs):
        return super().from_txt_file(*args, **kwargs)


class _InternalTraceBase(abc.ABC):
    """
    Base class for common functionalities between :class:`_Trace` and
    :class:`_TraceViewBase`.
    """

    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def start(self):
        '''
        The timestamp of the first trace event.
        '''

    @property
    @abc.abstractmethod
    def end(self):
        '''
        The timestamp of the last trace event.
        '''

    @property
    @abc.abstractmethod
    def basetime(self):
        '''
        Absolute timestamp when the tracing started.

        This might differ from :attr:`start` as the latter can be affected
        by various normalization or windowing features.
        '''

    @property
    @abc.abstractmethod
    def endtime(self):
        '''
        Absolute timestamp when the tracing stopped.

        This might differ from :attr:`end` as the latter can be affected by
        various normalization or windowing features.

        .. note:: With some parsers, that might be the timestamp of the last
           recorded event instead if the trace end timestamp was not recorded.
        '''

    @property
    def trace_state(self):
        """
        State of the trace object that might impact the output of dataframe
        getter functions like :meth:`lisa.trace.TraceBase.df_event`.

        It must be hashable and serializable to JSON, so that it can be
        recorded when analysis methods results are cached to the swap.
        """
        return None

    @property
    def time_range(self):
        """
        Duration of that trace (difference between :attr:`start` and :attr:`end`).
        """
        return self.end - self.start

    @property
    def window(self):
        """
        Same as ``(trace.start, trace.end)``.

        This is handy to pass to functions expecting a window tuple.
        """
        return (self.start, self.end)

    @property
    def available_events(self):
        """
        Set of available events on that trace.

        .. warning:: The set of events can change as new events are parsed. Not
            all trace parsers are able to provide the list of events that could
            be parsed upfront, so do not rely on this set to be stable.
            However, using ``event in trace.available_events`` will always
            return ``True`` if the event can be parsed, possibly at the cost of
            actually parsing the event to check if that works.
        """
        return _AvailableTraceEventsSet(self)

    def get_view(self, **kwargs):
        """
        Get a view on a trace.

        Various aspects of the trace can be altered depending on the
        parameters, such as cropping time-wise to fit in ``window``.

        :param window: Crop the dataframe to include events that are inside the
            given window. This includes the event immediately preceding the
            left boundary if there is no exact timestamp match. This can also
            include more rows before the beginning of the window based on the
            ``signals`` required by the user. A ``None`` boundary will extend
            to the beginning/end of the trace.
        :type window: tuple(float or None, float or None) or None

        :param signals: List of :class:`lisa.datautils.SignalDesc` to use when
            selecting rows before the beginning of the ``window``. This allows
            ensuring that all the given signals have a known value at the beginning
            of the window.
        :type signals: list(lisa.datautils.SignalDesc) or None

        :param compress_signals_init: If ``True``, the timestamp of the events
            before the beginning of the ``window`` will be compressed to be
            either right before the beginning of the window, or at the exact
            timestamp of the beginning of the window (depending on the
            dataframe library chosen, since pandas cannot cope with more than
            one row for each timestamp).
        :type compress_signals_init: bool or None

        :param normalize_time: If ``True``, the beginning of the ``window``
            will become timestamp 0. If no ``window`` is used, the beginning of
            the trace is taken as T=0. This allows easier comparison of traces
            that were generated with absolute timestamps (e.g. timestamp
            related to the uptime of the system). It also allows comparing
            various slices of the same trace.
        :type normalize_time: bool or None

        :param events_namespaces: List of namespaces of the requested events.
            Each namespace will be tried in order until the event is found. The
            ``None`` namespace can be used to specify no namespace. The full
            event name is formed with ``<namespace><event>``.
        :type events_namespaces: list(str or None)

        :param events: Preload the given events when creating the view. This
            can be advantageous as a single instance of the parser will be
            spawned, so if the parser supports it, multiple events will be
            parsed in one trace traversal.
        :type events: list(str) or lisa.trace.TraceEventCheckerBase or None

        :param strict_events: If ``True``, will raise an exception if the
            ``events`` specified cannot be loaded from the trace. This allows
            failing early in trace processing.
        :param strict_events: bool or None

        :param process_df: Function called on each dataframe returned by
            :meth:`lisa.trace.TraceBase.df_event`. The parameters are as follow:

            1. Name of the event being queried.
            2. A :class:`polars.LazyFrame` of the event.

            It is expected to return a :class:`polars.LazyFrame` as well.

        :type process_df: typing.Callable[[str, polars.LazyFrame], polars.LazyFrame] or None

        :param df_fmt: Format of the dataframes returned by
            :meth:`lisa.trace.TraceBase.df_events`. One of:

            * ``"pandas"``: :class:`pandas.DataFrame`.
            * ``"polars-lazyframe"``: :class:`polars.LazyFrame`.
            * ``None``: defaults to ``"pandas"`` for
              backward-compatibility.

        :type df_fmt: str or None

        :Variable arguments: Forwarded to the contructor of the view.
        """
        view = _TraceViewBase._make_view(self, **kwargs)
        assert isinstance(view, _TraceViewBase)
        return view

    @abc.abstractmethod
    def _internal_df_event(self, event, **kwargs):
        """
        Internal function creating the :class:`polars.LazyFrame` for the given
        ``event`` and returning associated metadata.

        Unrecognized keyword arguments should be passed down to the parent
        trace view when that's relevant or simply ignored.
        """
        pass

    @abc.abstractmethod
    def _preload_events(self, events):
        """
        Preload the given events by parsing them if necessary.

        This can be more efficient than requesting events one by one as the
        parser might be able to parse multiple events in one pass.
        """
        pass

    def __getitem__(self, window):
        """
        Slice the trace with the given time range.
        """
        if not isinstance(window, slice):
            raise TypeError("Cropping window must be an instance of slice")

        if window.step is not None:
            raise ValueError("Slice step is not supported")

        return self.get_view(window=(window.start, window.stop))

    @deprecate('Prefer adding delta once signals have been extracted from the event dataframe for correctness',
        deprecated_in='2.0',
        removed_in='4.0',
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
            raise RuntimeError(f"Column {col_name} is already present in the dataframe")

        return df_add_delta(df, col=col_name, inplace=inplace, window=self.window)

    @deprecate('This method has been deprecated and is an alias for "trace.ana.notebook.df_all_events()"',
        deprecated_in='2.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.notebook.NotebookAnalysis.df_all_event',
    )
    def df_all_events(self, *args, **kwargs):
        return self.ana.notebook.df_all_events(*args, **kwargs)

    @abc.abstractmethod
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, *args):
        pass


# User-facing
class TraceBase(_InternalTraceBase):
    """
    Base class for all public trace classes.

    This :class:`abc.ABC` class defines the API available on trace-like
    objects, and is suitable to use with ``isinstance`` and ``issubclass``.
    """
    @abc.abstractmethod
    def df_event(self, event, **kwargs):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name.

            In addition to actual events, the following formats for meta events
            are supported:

            * ``trace_printk@``: The event format is described by the
              ``bprint`` event format string, and the field values are decoded
              from the variable arguments buffer. Note that:

              * The field values *must* be in the buffer, i.e. the format
                string is only used as the event format, no "literal value"
                will be extracted from it.

              * The event *must* have fields. If not, ``trace_printk()``
                will emit a bputs event that will be ignored at the moment.
                We need to get a bprint event.

              * Field names *must* be unique.

              .. code-block:: C

                  // trace.df_event('trace_printk@myevent')
                  void foo(void) {
                      trace_printk("myevent: field1=%s field2=%i", "foo", 42);
                  }

            * ``userspace@``: the event is generated by userspace:

              .. code-block:: shell

                  # trace.df_event('userspace@myevent')
                  echo "myevent: field1=foo field2=42" > /sys/kernel/debug/tracing/trace_marker

              Note that the field names must be unique.

            .. note:: All meta event names are expected to be valid C language
                identifiers. Usage of other characters will prevent correct
                parsing.

        :type event: str

        :param signals: List of signals to fixup if ``signals_init == True``.
            If left to ``None``, :meth:`lisa.datautils.SignalDesc.from_event`
            will be used to infer a list of default signals.
        :type signals: list(SignalDesc)

        :param compress_signals_init: Give a timestamp very close to the
            beginning of the sliced dataframe to rows that are added by
            ``signals_init``. This allows keeping a very close time span
            without introducing duplicate indices.
        :type compress_signals_init: bool
        """
        pass

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='2.0',
        removed_in='4.0',
        replaced_by='df_event',
    )
    def df_events(self, *args, **kwargs):
        return self.df_event(*args, **kwargs)

    @property
    @abc.abstractmethod
    def ana(self):
        """
        Allows calling an analysis method on the trace, sharing the dataframe cache.

        **Example**

        Call lisa.analysis.LoadTrackingAnalysis.df_task_signal() on a trace::

            df = trace.ana.load_tracking.df_task_signal(task='foo', signal='util')

        The ``trace.ana`` proxy can also be called like a function to define default
        values for analysis methods::

            ana = trace.ana(task='big_0-3')
            ana.load_tracking.df_task_signal(signal='util')

            # Equivalent to:
            ana.load_tracking.df_task_signal(task='big_0-3', signal='util')

            # The proxy can be called again to override the value given to some
            # parameters, and the the value can also be overridden when calling the
            # method:
            ana(task='foo').df_task_signal(signal='util')
            ana.df_task_signal(task='foo', signal='util')
        """
        pass

    @property
    @abc.abstractmethod
    @deprecate(replaced_by=ana, deprecated_in='3.0', removed_in='4.0')
    def analysis(self):
        pass

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
        if not path:
            raise ValueError('No trace file is backing this Trace instance')

        if path.endswith('.dat'):
            cmd = 'kernelshark'
        else:
            cmd = 'xdg-open'

        return os.popen(f"{cmd} {shlex.quote(path)}")


    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_task_name_pids',
    )
    def get_task_name_pids(self, *args, **kwargs):
        return self.ana.tasks.get_task_name_pids(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='2.0',
        removed_in='4.0',
        replaced_by=get_task_name_pids,
    )
    def get_task_by_name(self, name):
        return self.get_task_name_pids(name, ignore_fork=True)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_task_pid_names',
    )
    def get_task_pid_names(self, *args, **kwargs):
        return self.ana.tasks.get_task_pid_names(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_task_by_pid',
    )
    def get_task_by_pid(self, *args, **kwargs):
        return self.ana.tasks.get_task_by_pid(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_task_ids',
    )
    def get_task_ids(self, *args, **kwargs):
        return self.ana.tasks.get_task_ids(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_task_id',
    )
    def get_task_id(self, *args, **kwargs):
        return self.ana.tasks.get_task_id(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_task_pid',
    )
    def get_task_pid(self, *args, **kwargs):
        return self.ana.tasks.get_task_pid(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.get_tasks',
    )
    def get_tasks(self, *args, **kwargs):
        return self.ana.tasks.get_tasks(*args, **kwargs)

    @deprecate('This property has been deprecated and is an alias',
        deprecated_in='3.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.tasks.TasksAnalysis.task_ids',
    )
    @property
    def task_ids(self):
        """
        List of all the :class:`lisa.analysis.tasks.TaskID` in the trace,
        sorted by PID.
        """
        return self.ana.tasks.task_ids


class _TraceViewBase(
    DelegateToAttr(
        'base_trace',
        [_InternalTraceBase],
    ),
    _InternalTraceBase
):
    def __init__(self, trace):
        self.base_trace = trace
        """
        The original :class:`TraceBase` this view is based on.
        """
        super().__init__()

    def __enter__(self):
        self.base_trace.__enter__()
        return self

    def __exit__(self, *args):
        return self.base_trace.__exit__(*args)

    @classmethod
    def _make_view(cls, trace, *, window=None, signals=None, compress_signals_init=None, normalize_time=False, events_namespaces=None, events=None, strict_events=False, process_df=None, df_fmt=None, clear_base_cache=None):
        if clear_base_cache is not None:
            _deprecated_warn(f'"clear_base_cache" parameter has no effect anymore')

        view = trace

        # Most views are stacked regardless whether they were needed right away
        # in order to have them ready to handle the parameters they add to
        # df_event() (via _internal_df_event())

        view = _NamespaceTraceView(view, namespaces=events_namespaces)

        # Preload events as early as possible in case other views need to
        # access some of the costly metadata that would be gathered when
        # preloading.
        view = _PreloadEventsTraceView(
            view,
            events=events,
            strict_events=strict_events
        )

        window_cls = _NormalizedTimeTraceView if normalize_time else _WindowTraceView
        view = window_cls(
            view,
            window=window,
            signals=signals,
            compress_signals_init=compress_signals_init,
        )

        if process_df:
            view = _ProcessTraceView(view, process_df)

        return view

    @property
    def trace_state(self):
        return (
            self.__class__.__qualname__,
            self.base_trace.trace_state,
        )

    def _preload_events(self, *args, **kwargs):
        return self.base_trace._preload_events(*args, **kwargs)

    def _internal_df_event(self, *args, **kwargs):
        return self.base_trace._internal_df_event(*args, **kwargs)

    @property
    def basetime(self):
        return self.base_trace.basetime

    @property
    def endtime(self):
        return self.base_trace.endtime

    @property
    def start(self):
        return self.base_trace.start

    @property
    def end(self):
        return self.base_trace.end


class _WindowTraceViewBase(_TraceViewBase, abc.ABC):
    @property
    @abc.abstractmethod
    def normalize_time(self):
        """
        ``True`` if the trace timestamps were normalized to start at ``0``.
        """


class _WindowTraceView(_WindowTraceViewBase):
    """
    A view on a :class:`Trace`.

    :param trace: The base trace.
    :type trace: Trace

    :param window: The time window to base this view on. If ``None``, the whole
        base trace will be selected.
    :type window: tuple(float, float) or None

    :param process_df: Function used to post process the event dataframes
        returned by :meth:`TraceBase.df_event`.
    :type process_df: typing.Callable[[str, pandas.DataFrame], pandas.DataFrame] or None

    You can substitute an instance of :class:`Trace` with an instance of
    :class:`_WindowTraceView`. This means you can create a view of a trimmed down trace
    and run analysis code/plots that will only use data within that window, e.g.::

      trace = Trace(...)
      view = trace.get_view((2, 4))

      # Alias for the above
      view = trace[2:4]

      # This will only use events in the (2, 4) time window
      df = view.ana.tasks.df_tasks_runtime()
    """

    def __init__(
        self,
        trace,
        *,
        window=None,
        signals=None,
        compress_signals_init=None,
    ):
        super().__init__(trace)

        def fixup_window(window):
            if window:
                start, end = window
                if start is not None:
                    start = Timestamp(start, rounding='down')

                if end is not None:
                    end = Timestamp(end, rounding='up')

                return (start, end)
            else:
                return None

        self._window = fixup_window(window)
        self._signals = set(signals or [])
        self._compress_signals_init = compress_signals_init

    # These attributes are absolutely critical for the performance of the view
    # stack, but pre-computing them is very harmful as it might trigger a
    # trace-parse to find the boundary timestamps before we need to parse a
    # dataframe, which can be very costly on a big trace.
    @property
    @memoized
    def start(self):
        """
        The timestamp of the first trace event in the view (>= :attr:`start`)
        """
        t_min, _ = self._window or (None, None)
        if t_min is None:
            return self.base_trace.start
        else:
            return t_min

    @property
    @memoized
    def end(self):
        """
        The timestamp of the last trace event in the view (<= :attr:`end`)
        """
        _, t_max = self._window or (None, None)
        if t_max is None:
            return self.base_trace.end
        else:
            end = t_max

        # Ensure we never end up with end < start
        end = max(end, self.start)
        return end

    @property
    def normalize_time(self):
        return False

    def _fixup_window(self, window):
        _start = self.start
        _end = self.end
        # Add missing window and fill-in None values
        if window is None:
            start = _start
            end = _end
        else:
            start, end = window

        # Clip the window to our own window
        start = _start if start is None else max(start, _start)
        end = _end if end is None else min(end, _end)

        # If we are not restricting more than the base trace, we don't want to
        # apply any windowing
        if start == self.base_trace.start:
            start = None

        if end == self.base_trace.end:
            end = None

        window = (start, end)

        if window == (None, None):
            window = None

        return window

    @property
    def trace_state(self):
        return (
            super().trace_state,
            self.window,
            self._signals,
            self._compress_signals_init,
        )

    def _internal_df_event(self, event, *, df_fmt=None, _legacy_signals=None, _inner_window=False, window=None, compress_signals_init=None, signals_init=None, signals=None, **kwargs):
        compress_signals_init = self._compress_signals_init if compress_signals_init is None else compress_signals_init
        compress_signals_init = False if compress_signals_init is None else compress_signals_init
        _legacy_signals = df_fmt == 'pandas' if _legacy_signals is None else _legacy_signals

        if window is not None:
            _deprecated_warn('"window", "signals" and "signals_init" are deprecated parameter of Trace.df_event(). Instead, use the matching parameters on trace.get_view(...).df_event(...)')

        def fixup_signals(events):
            return {
                signal_desc
                for event in events
                for signal_desc in self._fixup_signals(
                    event=event,
                    signals=signals,
                    signals_init=signals_init,
                    legacy_signals=_legacy_signals,
                    inner_window=_inner_window,
                )
            }

        _window = self._fixup_window(window)
        _signals = self._fixup_signals(
                event=event,
                signals=signals,
                signals_init=signals_init,
                legacy_signals=_legacy_signals,
                inner_window=_inner_window,
            )

        df, meta = self.base_trace._internal_df_event(
            event,
            _inner_window=True,
            df_fmt=df_fmt,
            # We propagate the list signals to base traces. This will let some
            # events "leak into the window" if a window was set, but that is
            # necessary to reliably implement self-contained analysis.
            # Otherwise, an analysis function would have no way of asking for
            # some specific signals it needs to work. Even if the user manually
            # specified signals to be friendly to that function, it could still
            # break the day the function makes use of a new signal (e.g. using
            # a new equivalent signal the legacy user code could not know about
            # at the time of writing).
            signals=_signals,
            # This needs to be passed to the base trace, in case we are not the
            # view applying a windowing. This can happen if there is e.g. a
            # no-op _WindowTraceView() layer.
            compress_signals_init=compress_signals_init,
            **kwargs,
        )

        df = self._apply_window(
            df=df,
            window=_window,
            signals=_signals,
            compress_signals_init=compress_signals_init,
        )

        return (df, meta)

    def _fixup_signals(self, event, signals, signals_init, legacy_signals, inner_window):
        if legacy_signals:
            # Warning-less backward compat for the default parameter value, so
            # that existing code will keep running without extra output
            if signals_init is None:
                signals_init = True
            # If the user specified manually, they will get the warning and be
            # asked to make a change that will ease transition to polars.
            else:
                _deprecated_warn('signals_init should not be used anymore. Instead, use trace.get_view(signals=[...]) with the list of signals you need', stacklevel=2)

            # Legacy behavior where signals are inferred from the event. We
            # only keep this in pandas for backward commpat. New code based
            # on polars has to define its own signals, so we don't have to
            # maintain an ever-growing list of signals in LISA itself, and
            # possibly breaking user code when we add a new one.
            signals = list(SignalDesc._from_event(event) if signals is None else signals)
            signals = signals if signals_init else []
        else:
            signals = list(signals or [])
            if signals_init is not None:
                raise ValueError(f'"signals_init" parameter is only supported for df_fmt="pandas". Use the TraceBase.get_view(signals=...) parameter.')

        # If we are in an inner window, it is pointless to add our
        # self._signals since they will be discarded by the outer window layer
        # (e.g. if the outer layer asked for no signals, it will simply apply a
        # normal window and not make use of any extra data we would have
        # included)
        if inner_window:
            pass
        # Otherwise, the signals are coming from the user and we should add our
        # own, since the user will be able to see their effect
        else:
            signals = {*signals, *self._signals}

        signals = {
            signal_desc
            for signal_desc in signals
            if signal_desc.event == event and signal_desc.fields
        }
        return signals

    def _apply_window(self, df, window, signals, compress_signals_init):
        if window is None or window == (None, None):
            return df
        else:
            if signals:
                df = df_window_signals(
                    df,
                    window=window,
                    signals=signals,
                    compress_init=compress_signals_init,
                )
            else:
                df = df_window(df, window, method='pre')

            return df


class _ProcessTraceView(_TraceViewBase):
    def __init__(self, trace, process_df):
        super().__init__(trace)
        self._process_df = process_df

    @property
    def trace_state(self):
        f = self._process_df

        return (
            super().trace_state,
            # This likely will be a value that cannot be serialized to JSON if
            # it was user-provided. This will prevent caching as it should.
            f,
        )

    def _internal_df_event(self, event, **kwargs):
        df, meta = self.base_trace._internal_df_event(event, **kwargs)

        if (process := self._process_df):
            df = process(event, df)

        return (df, meta)


class _PreloadEventsTraceView(_TraceViewBase):
    def __init__(self, trace, events=None, strict_events=False):
        super().__init__(trace)
        trace = self.base_trace

        if isinstance(events, str):
            raise ValueError('Events passed to Trace(events=...) must be a list of strings, not a string.')
        elif events is _ALL_EVENTS:
            pass
        else:
            events = set(events or [])
            events = AndTraceEventChecker.from_events(events)

        if events or events is _ALL_EVENTS:
            preloaded = trace._preload_events(events)

            if events is _ALL_EVENTS:
                events = AndTraceEventChecker.from_events(set(trace.available_events))

            if strict_events:
                events.check_events(preloaded)

        self._events = events

    @property
    def events(self):
        """
        Preloaded events as a :class:`TraceEventCheckerBase`.
        """
        try:
            base_events = self.base_trace.events
        except AttributeError:
            base_events = set()
        else:
            base_events = base_events.get_all_events()

        events = sorted({*self._events, *base_events})
        return AndTraceEventChecker.from_events(events)


class _NamespaceTraceView(_TraceViewBase):
    def __init__(self, trace, namespaces):
        # Allow using self.base_trace.events
        trace = _PreloadEventsTraceView(trace)
        super().__init__(trace)

        self._events_namespaces = namespaces or []
        self._preload_events(self.base_trace.events)

    def _preload_events(self, events):
        # It is critical to not enter that path if we don't have any actual
        # namespaces, otherwise the overhead of the loop will get multiplicated
        # at every layer and we end up with large overheads.
        if self._events_namespaces:
            if events is _ALL_EVENTS:
                preloaded = super()._preload_events(events)
                if None in self._events_namespaces:
                    return preloaded
                else:
                    return set()
            else:
                mapping = {
                    _event: event
                    for event in events
                    for _event in self._expand_namespaces(event)
                }

                # Preload the events in a single batch. This is critical for
                # performance, otherwise we will spin up the parser multiple times.
                preloaded = self.base_trace._preload_events(mapping.keys())

                return {
                    # Return the requested name instead of the actual event
                    # preloaded so that _PreloadEventsTraceView() can check for
                    # what it actually asked for
                    mapping.get(_event, _event)
                    for _event in preloaded
                }
        else:
            return super()._preload_events(events)

    @property
    def trace_state(self):
        return (
            super().trace_state,
            self._events_namespaces,
        )

    @property
    @memoized
    def events_namespaces(self):
        """
        Namespaces events will be looked up in.
        """
        try:
            base_namespaces = self.base_trace.events_namespaces
        except AttributeError:
            base_namespaces = (None,)

        return deduplicate(
            [
                *self._events_namespaces,
                *base_namespaces,
            ],
            keep_last=False,
        )

    @classmethod
    def _resolve_namespaces(cls, namespaces):
        return namespaces or (None,)

    def _expand_namespaces(self, event, namespaces=None):
        return self._do_expand_namespaces(
            event,
            self._resolve_namespaces(
                namespaces or self._events_namespaces
            )
        )

    @classmethod
    def _do_expand_namespaces(cls, event, namespaces):
        namespaces = cls._resolve_namespaces(namespaces)

        def expand_namespace(event, namespace):
            if namespace:
                if event.startswith(namespace):
                    return event
                else:
                    return f'{namespace}{event}'
            else:
                return event

        def expand(event, namespaces):
            if _Trace._is_meta_event(event):
                return [event]
            else:
                return deduplicate(
                    [
                        expand_namespace(event, namespace)
                        for namespace in namespaces
                    ],
                    keep_last=False,
                )

        return expand(event, namespaces)

    def _internal_df_event(self, event, namespaces=None, **kwargs):
        if namespaces is not None:
            _deprecated_warn('"namespaces" is a deprecated parameter of Trace.df_event(). Instead, use trace.get_view(events_namespaces=...).df_event(...)')

        if namespaces or self._events_namespaces:
            events = self._expand_namespaces(event, namespaces)
            events = events or [event]
            trace = self.base_trace

            # Preload all the events as we likely will only get one match
            # anyway, so we can avoid spinning up a parser several times for
            # nothing.
            events = sorted(trace._preload_events(events))

            last_excep = MissingTraceEventError([event])
            for _event in events:
                try:
                    return trace._internal_df_event(_event, **kwargs)
                except MissingTraceEventError as e:
                    last_excep = e
            else:
                raise last_excep
        else:
            return super()._internal_df_event(event, **kwargs)


class _TimeOffsetter(_CacheDataDescEncodable):
    def __init__(self, offset):
        assert isinstance(offset, Timestamp)
        offset_ns = offset.as_nanoseconds
        self._offset_ns = offset_ns
        self._offset_polars = _polars_duration_expr(offset_ns, unit='ns', rounding='down')

    def json_encode(self):
        return self._offset_ns

    def __call__(self, event, df):
        return df.with_columns(
            pl.col('Time') - self._offset_polars
        )


class _NormalizedTimeTraceView(_WindowTraceViewBase):
    def __init__(self, trace, window, **kwargs):
        window = window or (trace.start, None)
        try:
            start, end = window
        except ValueError:
            raise ValueError('A window with non-None left bound must be provided with normalize_time=True')
        else:
            if start is None:
                raise ValueError('A window with non-None left bound must be provided with normalize_time=True')
            else:
                self._offset = start

                view = trace.get_view(window=window)
                view = self._with_time_offset(view, start)
                view = view.get_view(**kwargs)
                super().__init__(view)

    @classmethod
    def _with_time_offset(cls, trace, start):
        # Round down to avoid ending up with negative Time for anything that
        # does not actually happen before the start
        start = Timestamp(start, rounding='down')

        return trace.get_view(
            process_df=_TimeOffsetter(start)
        )

    @property
    def basetime(self):
        return self.base_trace.basetime - self._offset

    @property
    def endtime(self):
        return self.base_trace.endtime - self._offset

    @property
    def start(self):
        return 0

    @property
    def end(self):
        return self.base_trace.end - self._offset

    @property
    def trace_state(self):
        return (
            super().trace_state,
            self._offset,
        )

    @property
    def normalize_time(self):
        return True


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
        return self.contains(event)

    def contains(self, event, namespaces=None):
        trace = self._trace
        view = trace.get_view(events_namespaces=namespaces)

        try:
            trace._internal_df_event(event=event)
        except MissingTraceEventError:
            return False
        else:
            return True

    @property
    def _available_events(self):
        trace = self._trace

        # Prime the trace metadata with what the events the parser reports.
        # This will not include meta events.
        try:
            available = trace.get_metadata('available-events')
        except MissingMetadataError:
            available = []

        available = set(available)

        # List all the events we have succesfully parsed in the past. This will
        # be updated in various places, so the result will not be stable.
        try:
            parseable = trace._cache.get_metadata('parseable-events')
        except MissingMetadataError:
            # This won't include meta-events, but better than nothing
            return available
        else:
            return {
                _event
                for _event, _parseable in parseable.items()
                if _parseable
            }

    def __iter__(self):
        return iter(self._available_events)

    def __len__(self):
        return len(self._available_events)

    def __str__(self):
        return str(self._available_events)


class _CacheDataDesc(Mapping):
    """
    Cached data descriptor.

    :param spec: Specification of the data as a key/value mapping.

    This holds all the information needed to uniquely identify a
    :class:`pandas.DataFrame` or :class:`pandas.Series` or
    :class:`polars.LazyFrame` or :class:`polars.DataFrame`. It is used to
    manage the cache and swap.

    It implements the :class:`collections.abc.Mapping` interface, so
    specification keys can be accessed directly like from a dict.

    .. note:: Once introduced in a container, instances must not be modified,
        directly or indirectly.
    """

    def __init__(self, spec, fmt):
        if fmt == 'polars-lazyframe':
            spec = {
                'polars-version': pl.__version__,
                **spec
            }

        self.fmt = fmt
        self.spec = spec
        self.normal_form = _CacheDataDescNF.from_spec(self.spec, fmt)
        """
        Normal form of the descriptor. Equality is implemented by comparing
        this attribute.
        """

    def __getitem__(self, key):
        return self.spec[key]

    def __iter__(self):
        return iter(self.spec)

    def __len__(self):
        return len(self.spec)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(
                f'{key}={val!r}'
                for key, val in self.__dict__.items()
            )
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.normal_form == other.normal_form
        else:
            return False

    def __hash__(self):
        return hash(self.normal_form)


class _CacheDataDescNF:
    """
    Normal form of :class:`_CacheDataDesc`.

    The normal form of the descriptor allows removing any possible differences
    in shape of values, and is serializable to JSON. The serialization is
    allowed to destroy some information (type mainly), as long as it does make
    two descriptors wrongly equal.
    """
    def __init__(self, nf, fmt):
        assert fmt != _CacheDataSwapEntry.META_EXTENSION
        self._fmt = fmt
        self._nf = nf
        # Since it's going to be inserted in dict for sure, precompute the hash
        # once and for all.
        self._hash = hash(self._nf)

    @classmethod
    def from_spec(cls, spec, fmt):
        """
        Build from a spec that can include any kind of Python objects.
        """
        nf = tuple(sorted(
            (key, cls._coerce(val))
            for key, val in spec.items()
        ))
        return cls(nf=nf, fmt=fmt)

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
            type_name = f'{val.__class__.__module__}.{val.__class__.__qualname__}'
            val = (type_name, val)

        return val

    def __str__(self):
        return str(self._nf)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._fmt == other._fmt and self._nf == other._nf
        else:
            return False

    def __hash__(self):
        return self._hash

    def to_json_map(self):
        return dict(
            fmt=self._fmt,
            data=self._nf,
        )

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
        fmt = mapping['fmt']
        data = dict(mapping['data'])
        nf = tuple(sorted(
            (key, cls._coerce_json(val))
            for key, val in data.items()
        ))
        return cls(nf=nf, fmt=fmt)


class _CannotWriteSwapEntry(Exception):
    pass


class _CacheDataSwapEntry:
    """
    Entry in the data swap area of :class:`Trace`.

    :param cache_desc_nf: Normal form descriptor describing what the entry
        contains.
    :type cache_desc_nf: _CacheDataDescNF

    :param name: Name of the entry. If ``None``, a random UUID will be
        generated.
    :type name: str or None

    :param written: ``True`` if the swap entry is already written on-disk.
    :type written: bool
    """

    META_EXTENSION = 'meta'
    """
    Extension used by the metadata file of the swap entry in the swap.
    """

    def __init__(self, cache_desc_nf, name=None, written=False):
        self.cache_desc_nf = cache_desc_nf
        self.name = name or uuid.uuid4().hex
        self.written = written

    @property
    def meta_filename(self):
        """
        Filename of the metadata file in the swap.
        """
        return f'{self.name}.{self.META_EXTENSION}'

    @property
    def fmt(self):
        """
        Format of the swap entry.
        """
        return self.cache_desc_nf._fmt

    @property
    def data_filename(self):
        """
        Filename of the data file in the swap.
        """
        return f'{self.name}.{self.fmt}'

    def to_json_map(self):
        """
        Return a mapping suitable for JSON serialization.
        """
        desc = self.cache_desc_nf.to_json_map()

        class Encoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, _CacheDataDescEncodable):
                    cls = o.__class__
                    return {
                        'module': cls.__module__,
                        'cls': cls.__qualname__,
                        'value': o.json_encode(),
                    }
                else:
                   return super().default(o)

        try:
            # Use json.dumps() here to fail early if the descriptor cannot be
            # dumped to JSON
            desc = Encoder().encode(desc)
        except TypeError as e:
            raise _CannotWriteSwapEntry(e)

        return {
            'version-token': VERSION_TOKEN,
            'name': self.name,
            'encoded_desc': desc,
        }

    @classmethod
    def from_json_map(cls, mapping, written=False):
        """
        Create an instance with a mapping created using :meth:`to_json_map`.
        """
        if mapping['version-token'] != VERSION_TOKEN:
            raise _TraceCacheSwapVersionError('Version token differ')

        desc = json.loads(mapping['encoded_desc'])
        cache_desc_nf = _CacheDataDescNF.from_json_map(desc)
        name = mapping['name']
        return cls(cache_desc_nf=cache_desc_nf, name=name, written=written)

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

        return cls.from_json_map(mapping, written=True)


class _TraceCacheSwapVersionError(ValueError):
    """
    Exception raised when the swap entry was created by another version of LISA
    than the one loading it.
    """


class _TraceCache(Loggable):
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

    :param trace_id: Unique id of the trace file, to invalidate the cache
        if the file changed.
    :type trace_id: str or None

    :param metadata: Metadata mapping to store in the swap area.
    :type metadata: dict or None

    :param swap_content: Initial content of the swap area.
    :type swap_content: dict(_CacheDataDescNF, _CacheDataSwapEntry) or None

    The cache manages both the :class:`pandas.DataFrame` and
    :class:`pandas.Series` generated in memory and a swap area used to evict
    them, and to reload them quickly. Some other data (typically JSON) can also
    be stored in the cache by analysis method.
    """

    TRACE_META_FILENAME = 'trace.meta'
    """
    Name of the trace metadata file in the swap area.
    """

    DATAFRAME_SWAP_FORMAT = 'polars-lazyframe'
    """
    Data storage format used to swap.
    """

    def __init__(self, max_mem_size=None, trace_path=None, trace_id=None, swap_dir=None, max_swap_size=None, swap_content=None, metadata=None):
        self._lock = threading.RLock()
        self._cache = {}
        self._swappable = {}
        self._data_cost = {}
        self._swap_content = swap_content or {}
        self._cache_desc_swap_filename = {}
        self._swap_dir = swap_dir
        self.max_swap_size = max_swap_size if max_swap_size is not None else math.inf
        self._swap_size = self._get_swap_size()

        self.max_mem_size = max_mem_size if max_mem_size is not None else math.inf
        self._metadata = metadata or {}

        self.trace_path = os.path.abspath(trace_path) if trace_path else trace_path
        self._trace_id = trace_id
        self._unique_id = uuid.uuid4().hex

        # Limit to one worker, as we will likely take the self._lock anyway
        self._thread_executor = ThreadPoolExecutor(max_workers=1)

        self.__deallocator_callbacks = [
            # Ensure we block until all workers are finished. Otherwise, e.g.
            # removing the swap area might fail because an worker is still creating
            # the metadata file in there.
            lambda: self._thread_executor.shutdown()
        ]


    @property
    def swap_dir(self):
        if (swap_dir := self._swap_dir) is None:
            raise ValueError(f'swap_dir is not set')
        else:
            return swap_dir

    @property
    @memoized
    def _hardlinks_base(self):
        path = Path(self.swap_dir) / 'hardlinks' / str(self._unique_id)
        path = path.resolve()
        def cleanup():
            # Only try with rmdir first, so that we don't sabbotage existing
            # LazyFrame that might still be alive.
            try:
                os.rmdir(path)
            except Exception:
                pass
        with self._lock:
            self.__deallocator_callbacks.append(cleanup)
        return path

    def _hardlink_path(self, base, name):
        path = self._hardlinks_base / base
        path.mkdir(parents=True, exist_ok=True)

        return (
            path,
            path / name,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for cb in self.__deallocator_callbacks:
            cb()

    def _parser_temp_path(self):
        try:
            swap_dir = self.swap_dir
        except ValueError:
            path = None
        else:
            path = Path(swap_dir) / 'temp'
            path.mkdir(parents=True, exist_ok=True)

        return tempfile.TemporaryDirectory(dir=path)

    def update_metadata(self, metadata, blocking=True):
        """
        Update the metadata mapping with the given ``metadata`` mapping and
        write it back to the swap area.
        """
        if metadata:
            with self._lock:
                self._metadata.update(metadata)
            self.to_swap_dir(blocking=blocking)

    def get_metadata(self, key):
        """
        Get the value of the given metadata ``key``.
        """
        try:
            with self._lock:
                return self._metadata[key]
        except KeyError as e:
            raise MissingMetadataError(key) from e

    def to_json_map(self):
        """
        Returns a dictionary suitable for JSON serialization.
        """
        with self._lock:
            trace_path = self.trace_path

            if trace_path:
                try:
                    swap_dir = self.swap_dir
                except ValueError:
                    trace_path = os.path.abspath(trace_path)
                else:
                    trace_path = os.path.relpath(trace_path, swap_dir)

            return ({
                'version-token': VERSION_TOKEN,
                'metadata': self._metadata,
                'trace-path': trace_path,
                'trace-id': self._trace_id,
            })

    def to_path(self, path, blocking=True):
        """
        Write the persistent state to the given ``path``.
        """
        def f():
            mapping = self.to_json_map()
            with open(path, 'w') as f:
                json.dump(mapping, f)
                f.write('\n')

        with measure_time() as m:
            if blocking:
                f()
            else:
                self._thread_executor.submit(f)

    @classmethod
    def _from_swap_dir(cls, swap_dir, trace_id, trace_path=None, metadata=None, **kwargs):
        metapath = os.path.join(swap_dir, cls.TRACE_META_FILENAME)

        with open(metapath) as f:
            mapping = json.load(f)

        if mapping['version-token'] != VERSION_TOKEN:
            raise _TraceCacheSwapVersionError('Version token differ')

        swap_trace_path = mapping['trace-path']
        swap_trace_path = os.path.join(swap_dir, swap_trace_path) if swap_trace_path else None

        metadata = metadata or {}

        if trace_path and not os.path.samefile(swap_trace_path, trace_path):
            invalid_swap = True
        else:
            old_trace_id = mapping['trace-id']
            invalid_swap = (old_trace_id != trace_id)

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
                    if filename.endswith(f'.{_CacheDataSwapEntry.META_EXTENSION}')
                }

                for filename in swap_entry_filenames:
                    path = os.path.join(swap_dir, filename)
                    try:
                        swap_entry = _CacheDataSwapEntry.from_path(path)
                    # If there is any issue with that entry, just ignore it
                    # pylint: disable=broad-except
                    except Exception:
                        continue
                    else:
                        yield (swap_entry.cache_desc_nf, swap_entry)

            swap_content = dict(load_swap_content(swap_dir))

            metadata_ = mapping['metadata']
            metadata = {**metadata_, **metadata}

        return cls(swap_content=swap_content, swap_dir=swap_dir, metadata=metadata, trace_path=trace_path, trace_id=trace_id, **kwargs)

    def to_swap_dir(self, blocking=True):
        """
        Write the persistent state to the swap area if any, no-op otherwise.
        """
        try:
            swap_dir = self.swap_dir
        except ValueError:
            pass
        else:
            path = os.path.join(swap_dir, self.TRACE_META_FILENAME)
            self.to_path(path, blocking=blocking)

    @classmethod
    def from_swap_dir(cls, swap_dir, **kwargs):
        """
        Reload the persistent state from the given ``swap_dir``.

        :Variable keyword arguments: Forwarded to :class:`_TraceCache`.
        """
        if swap_dir:
            try:
                return cls._from_swap_dir(swap_dir=swap_dir, **kwargs)
            except (FileNotFoundError, _TraceCacheSwapVersionError, json.decoder.JSONDecodeError):
                pass

        return cls(swap_dir=swap_dir, **kwargs)

    def _estimate_data_swap_size(self, data):
        return self._data_mem_usage(data)

    @staticmethod
    def _data_mem_usage(data):
        if data is None:
            return 1
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            mem = data.memory_usage()
            try:
                return mem.sum()
            except AttributeError:
                return mem
        else:
            return sys.getsizeof(data)

    def _should_evict_to_swap(self, cache_desc, data):
        with self._lock:
            compute_cost = self._data_cost.get(
                cache_desc,
                # If we don't have any cost info, assume it is expensive to
                # compute
                math.inf,
            )
            return compute_cost >= 100e-6

    def _path_of_swap_entry(self, swap_entry):
        return os.path.join(self.swap_dir, swap_entry.meta_filename)

    def _cache_desc_swap_path(self, cache_desc, create=False):
        cache_desc_nf = cache_desc.normal_form

        if create and not self._is_written_to_swap(cache_desc):
            self.insert(
                cache_desc,
                data=None,
                compute_cost=None,
                write_swap=True,
                ignore_cost=True,
                # We do not write the swap_entry meta file, so that the
                # user can write the data file before the swap entry is
                # added. This way, another process will not be tricked into
                # believing the data is available whereas in fact it's in
                # the process of being populated.
                write_meta=False,
            )

        with self._lock:
            swap_entry = self._swap_content[cache_desc_nf]
        filename = swap_entry.data_filename
        return os.path.join(self.swap_dir, filename)

    def _is_written_to_swap(self, cache_desc):
        try:
            with self._lock:
                swap_entry = self._swap_content[cache_desc.normal_form]
        except KeyError:
            return False
        else:
            return swap_entry.written

    @staticmethod
    def _data_to_parquet(data, path, compression='lz4', **kwargs):
        kwargs['compression'] = compression
        if isinstance(data, pd.DataFrame):
            data = _pandas_cleanup_df(data)

            # Data must be convertible to bytes so we dump them as JSON
            attrs = json.dumps(data.attrs)
            table = pyarrow.Table.from_pandas(data)
            updated_metadata = dict(
                table.schema.metadata or {},
                lisa=attrs,
            )
            table = table.replace_schema_metadata(updated_metadata)
            pyarrow.parquet.write_table(table, path, **kwargs)
        elif isinstance(data, pl.DataFrame):
            data.write_parquet(path, **kwargs)
        elif isinstance(data, pl.LazyFrame):
            with pl.StringCache():
                try:
                    # TOOD: revisit when polars streaming engine is complete
                    # and it does not raise a DeprecationWarning anymore.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=DeprecationWarning)
                        data.sink_parquet(path, **kwargs)
                # The streaming engine may have issues with some LazyFrames, so
                # fall back on collecting.
                except Exception:
                    path.unlink(missing_ok=True)
                    data.collect().write_parquet(path, **kwargs)
        else:
            data.to_parquet(path, **kwargs)

    def _write_data(self, fmt, data, path):
        if fmt == 'disk-only':
            return
        elif fmt == 'parquet':
            self._data_to_parquet(data, path)
        elif fmt == 'polars-lazyframe':
            assert isinstance(data, pl.LazyFrame)

            def to_parquet():
                self._data_to_parquet(data, path)

            def to_json(plan):
                plan = json.loads(plan)
                plan, _ = _logical_plan_resolve_paths(
                    self,
                    plan=plan,
                    kind='dump',
                )
                with open(path, 'wt') as f:
                    json.dump(plan, f)

            # If the LazyFrame is actually backed by an in-memory DataFrame, we
            # just execute it and dump it to parquet. Otherwise, that would
            # force rendering the entire dataframe to JSON first which would
            # not scale well at all.
            if _polars_df_in_memory(data):
                to_parquet()
            else:
                try:
                    plan = _df_json_serialize(data)
                # We failed to serialize the logical plan. This could happen
                # because it contains references to UDF (e.g. a lambda passed
                # to Expr.map_elements())
                except ValueError:
                    to_parquet()
                else:
                    to_json(plan)

        elif fmt == 'json':
            with open(path, 'wt') as f:
                try:
                    json.dump(data, f, separators=(',', ':'))
                except Exception as e:
                    raise ValueError(f'Does not know how to write data type {data.__class__} to the cache: {e}') from e
        else:
            raise ValueError(f'Does not know how to dump to disk format: {fmt}')

    def _load_data(self, fmt, path):
        def load_parquet(path):
            path = Path(path)
            hardlink_base, hardlink_path = self._hardlink_path(
                uuid.uuid4().hex,
                path.name
            )
            # We make a hardlink to the cache entry that will not be
            # subject to scrubbing. This way we ensure that we can keep the
            # LazyFrame working even if another process decides to scrub
            # the swap area.
            _make_hardlink(path, hardlink_path)
            try:
                df = pl.scan_parquet(hardlink_path)
            except BaseException:
                try:
                    shutil.rmtree(hardlink_base)
                except Exception:
                    pass
                raise
            else:
                # Ensure we actually trigger a file read, in case we are trying
                # to interpret as parquet something that is not parquet
                df.clear().collect()

                df = _LazyFrameOnDelete.attach_file_cleanup(df, [hardlink_base])

                parquet_meta = pyarrow.parquet.read_metadata(hardlink_path)
                parquet_meta = parquet_meta.metadata
                try:
                    pandas_meta = parquet_meta[b'pandas']
                except KeyError:
                    pass
                else:
                    # Load the pandas metadata and put the index column
                    # first, so that _polars_index_col() detects the index
                    # correctly.
                    pandas_meta = json.loads(pandas_meta.decode('utf-8'))
                    index_cols = pandas_meta['index_columns']
                    df = df.select(order_as(df.collect_schema().names(), index_cols))

                return df

        if fmt == 'disk-only':
            data = None
        elif fmt == 'parquet':
            data = load_parquet(path)
        elif fmt == 'json':
            with open(path, 'rt') as f:
                data = json.load(f)
        elif fmt == 'polars-lazyframe':
            try:
                data = load_parquet(path)
            except polars.exceptions.ComputeError:
                with open(path, 'r') as f:
                    plan = json.load(f)

                plan, hardlinks = _logical_plan_resolve_paths(
                    self,
                    plan=plan,
                    kind='load',
                )
                plan = json.dumps(plan)
                plan = io.StringIO(plan)
                data = _df_json_deserialize(plan)
                data = _LazyFrameOnDelete.attach_file_cleanup(data, hardlinks)
        else:
            raise ValueError(f'File format not supported "{fmt}" at path: {path}')

        return data

    def _write_swap(self, cache_desc, data, write_meta=True, best_effort=False):
        try:
            swap_dir = self.swap_dir
        except ValueError:
            pass
        else:
            if self._is_written_to_swap(cache_desc):
                return

            cache_desc_nf = cache_desc.normal_form
            # We may already have a swap entry if we used the None data
            # placeholder. This would have allowed the user to reserve the swap
            # data file in advance so they can write to it directly, instead of
            # managing the data in the memory cache.
            try:
                with self._lock:
                    swap_entry = self._swap_content[cache_desc_nf]
            except KeyError:
                swap_entry = _CacheDataSwapEntry(cache_desc_nf)

            data_path = Path(swap_dir, swap_entry.data_filename)

            # If that would make the swap dir too large, try to do some cleanup
            if self._estimate_data_swap_size(data) + self._swap_size > self.max_swap_size:
                self.scrub_swap()

            # Write the Parquet file and update the write speed
            try:
                self._write_data(cache_desc.fmt, data, data_path)
            except Exception as e:
                if best_effort:
                    # Do not log the error, as it could be an expected one
                    # (e.g. we have an object column in a dataframe that cannot
                    # be converted to arrow.
                    pass
                else:
                    raise e
            else:
                # Update the swap entry on disk
                if write_meta:
                    try:
                        swap_entry.to_path(
                            self._path_of_swap_entry(swap_entry)
                        )
                    # We have a swap entry that cannot be written to the swap,
                    # probably because the descriptor includes something that
                    # cannot be serialized to JSON. This may happen under
                    # normal operations, e.g. with a user-defined process_df
                    # function passed to _ProcessTraceView.
                    except _CannotWriteSwapEntry as e:
                        self.logger.debug(f'Could not write {cache_desc} to swap: {e}')
                        swap_entry.written = False
                        return
                    else:
                        swap_entry.written = True

                with self._lock:
                    self._swap_content[swap_entry.cache_desc_nf] = swap_entry

                try:
                    data_swapped_size = data_path.stat().st_size
                except FileNotFoundError:
                    data_swapped_size = 0

                mem_usage = self._data_mem_usage(data)
                with self._lock:
                    self._swap_size += data_swapped_size
                self.scrub_swap()

    def _get_swap_size(self):
        try:
            swap_dir = self.swap_dir
        except ValueError:
            return 1
        else:
            return sum(
                dir_entry.stat().st_size
                for dir_entry in os.scandir(swap_dir)
            )

    def scrub_swap(self):
        """
        Scrub the swap area to remove old files if the storage size limit is exceeded.
        """
        try:
            swap_dir = self.swap_dir
        except ValueError:
            pass
        else:
            # TODO: Load the file information from __init__ by discovering the
            # swap area's content to avoid doing it each time here
            if self._swap_size > self.max_swap_size:
                self._scrub_swap(swap_dir)

    def _scrub_swap(self, swap_dir):
        with self._lock:
            def get_stat(dir_entry):
                try:
                    return dir_entry.stat()
                except FileNotFoundError:
                    return None

            stats = {
                dir_entry.name: stat
                for dir_entry in os.scandir(swap_dir)
                if (stat := get_stat(dir_entry)) is not None
            }

            swap_content = list(self._swap_content.values())

        data_files = {
            swap_entry.data_filename: swap_entry
            for swap_entry in swap_content
        }

        # Get rid of stale files that are not referenced by any swap entry
        metadata_files = {
            swap_entry.meta_filename
            for swap_entry in swap_content
        }
        metadata_files.add(self.TRACE_META_FILENAME)
        non_stale_files = data_files.keys() | metadata_files | {'hardlinks', 'temp'}
        stale_files = stats.keys() - non_stale_files
        for filename in stale_files:
            stats.pop(filename, None)
            path = os.path.join(swap_dir, filename)
            try:
                os.unlink(path)
            except Exception:
                pass

        def by_mtime(path_stat):
            _, stat = path_stat
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
            stats.pop(swap_entry.data_filename, None)
            self._clear_swap_entry(swap_entry)

        with self._lock:
            self._swap_size = sum(
                stats[swap_entry.data_filename].st_size
                for swap_entry in swap_content
                if swap_entry.data_filename in stats
            )

    def _clear_cache_desc_swap(self, cache_desc):
        with self._lock:
            try:
                swap_entry = self._swap_content[cache_desc.normal_form]
            except KeyError:
                pass
            else:
                self._clear_swap_entry(swap_entry)

    def _clear_swap_entry(self, swap_entry):
        try:
            swap_dir = self.swap_dir
        except ValueError:
            pass
        else:
            with self._lock:
                self._swap_content.pop(swap_entry.cache_desc_nf, None)

            for filename in (swap_entry.meta_filename, swap_entry.data_filename):
                path = os.path.join(swap_dir, filename)
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def fetch(self, cache_desc, insert=True):
        """
        Fetch an entry from the cache or the swap.

        :param cache_desc: Descriptor to look for.
        :type cache_desc: _CacheDataDesc

        :param insert: If ``True`` and if the fetch succeeds by loading the
            swap, the data is inserted in the cache.
        :type insert: bool
        """
        try:
            with self._lock:
                return self._cache[cache_desc]
        except KeyError as e:
            # pylint: disable=raise-missing-from
            try:
                path = self._cache_desc_swap_path(cache_desc)
            # If there is no swap, bail out
            except (ValueError, KeyError):
                raise KeyError('Could not find swap entry for cache_desc')
            else:
                try:
                    data = self._load_data(cache_desc.fmt, path)
                except Exception as e:
                    # Rotten swap entry, we clear it. This may happen e.g. when
                    # reloading a LazyFrame that depended on another cache
                    # entry that has been scrubbed
                    self._clear_cache_desc_swap(cache_desc)
                    raise KeyError('Could not load swap entry content of cache_desc')

                if insert:
                    # We have no idea of the cost of something coming from
                    # the cache
                    self.insert(cache_desc, data, write_swap=False, compute_cost=None)

                return data

    def insert(self, cache_desc, data, compute_cost=None, write_swap=True, ignore_cost=False, write_meta=True, swappable=None):
        """
        Insert an entry in the cache.

        :param cache_desc: Descriptor of the data to insert.
        :type cache_desc: _CacheDataDesc

        :param data: data to insert.
        :type data: object

        :param compute_cost: Time spent to compute the data in seconds.
        :type compute_cost: float or None

        :param write_swap: If ``True``, the data will be written to the swap as
            well so it can be quickly reloaded. Note that it will be subject to
            cost evaluation, so it might not result in anything actually
            written. If ``"best-effort"`` is passed, writing will be attempted
            and any exception suppressed.
        :type write_swap: bool or str

        :param ignore_cost: If ``True``, bypass the computation vs swap
            cost comparison.
        :type ignore_cost: bool

        :param write_meta: If ``True``, the swap entry metadata will be written
            on disk if the data are. Otherwise, no swap entry is written to disk.
        :type write_meta: bool

        :param swappable: If ``False``, the data will never be written to the
            swap and will only be kept in memory. If ``None``, the swappability
            will not change if it was already set, otherwise it will be set to
            ``True``.
        :type swappable: bool or None
        """
        with self._lock:
            self._cache[cache_desc] = data
            if swappable is not None:
                self._swappable[cache_desc] = swappable
            if compute_cost is not None:
                self._data_cost[cache_desc] = compute_cost

        if write_swap:
            best_effort = (write_swap == 'best-effort')
            self.write_swap(
                cache_desc,
                ignore_cost=ignore_cost,
                write_meta=write_meta,
                best_effort=best_effort,
            )

        self._scrub_mem()

    def insert_disk_only(self, spec, compute_cost=None):
        cache_desc = _CacheDataDesc(spec=spec, fmt='disk-only')
        self.insert(cache_desc, data=None, compute_cost=compute_cost, write_swap=True)
        path = self._cache_desc_swap_path(cache_desc, create=True)
        return Path(path).resolve()

    def _scrub_mem(self):
        if self.max_mem_size == math.inf:
            return

        mem_usage = sum(
            self._data_mem_usage(data)
            for data in self._cache.values()
        )

        if mem_usage > self.max_mem_size:
            with self._lock:
                # Make sure garbage collection occurred recently, to get the most
                # accurate refcount possible
                gc.collect()
                refcounts = {
                    cache_desc: sys.getrefcount(data)
                    for cache_desc, data in self._cache.items()
                }
                min_refcount = min(refcounts.values())

                # Low retention score means it's more likely to be evicted
                def retention_score(cache_desc_and_data):
                    cache_desc, data = cache_desc_and_data

                    # If we don't know the computation cost, assume it can be evicted cheaply
                    compute_cost = self._data_cost.get(cache_desc, 0)

                    # Assume that more references to an object implies it will
                    # stay around for longer. Therefore, it's less interesting to
                    # remove it from this cache and pay the cost of reading/writing it to
                    # swap, since the memory will not be freed anyway.
                    #
                    # Normalize to the minimum refcount, so that the _cache and other
                    # structures where references are stored are discounted for sure.
                    return (refcounts[cache_desc] - min_refcount + 1) * compute_cost

                new_mem_usage = 0
                for cache_desc, data in sorted(self._cache.items(), key=retention_score):
                    new_mem_usage += self._data_mem_usage(data)
                    if new_mem_usage > self.max_mem_size:
                        self.evict(cache_desc)

    def evict(self, cache_desc):
        """
        Evict the given descriptor from memory.

        :param cache_desc: Descriptor to evict.
        :type cache_desc: _CacheDataDesc

        If it would be cheaper to reload the data than to recompute them, they
        will be written to the swap area.
        """
        self.write_swap(cache_desc, best_effort=True)

        try:
            with self._lock:
                del self._cache[cache_desc]
        except KeyError:
            pass

    def write_swap(self, cache_desc, ignore_cost=False, write_meta=True, best_effort=False):
        """
        Write the given descriptor to the swap area if that would be faster to
        reload the data rather than recomputing it. If the descriptor is not in
        the cache or if there is no swap area, ignore it.

        :param cache_desc: Descriptor of the data to write to swap.
        :type cache_desc: _CacheDataDesc

        :param ignore_cost: If ``True``, bypass the compute vs swap cost comparison.
        :type ignore_cost: bool

        :param write_meta: If ``True``, the swap entry metadata will be written
            on disk if the data are. Otherwise, no swap entry is written to disk.
        :type write_meta: bool

        :param best_effort: If ``True``, attempt to write to the swap and
            simply log an error rather than raising an exception in case of
            failure.
        :type best_effort: bool
        """
        try:
            with self._lock:
                data = self._cache[cache_desc]
                swappable = self._swappable.get(cache_desc, True)
        except KeyError:
            pass
        else:
            if (
                swappable and
                (
                    ignore_cost or
                    self._should_evict_to_swap(cache_desc, data)
                )
            ):
                self._write_swap(
                    cache_desc=cache_desc,
                    data=data,
                    write_meta=write_meta,
                    best_effort=best_effort,
                )

    def write_swap_all(self, **kwargs):
        """
        Attempt to write all cached data to the swap.
        """
        with self._lock:
            cache_descs = list(self._cache.keys())

        for cache_desc in cache_descs:
            self.write_swap(cache_desc, **kwargs)

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
        with self._lock:
            self._cache = {
                cache_desc: data
                for cache_desc, data in self._cache.items()
                if not (
                    cache_desc.get('event') == event
                    and (
                        raw is None
                        or cache_desc.get('raw') == raw
                    )
                )
            }

    def clear_all_events(self, raw=None):
        """
        Same as :meth:`clear_event` but works on all events at once.
        """
        with self._lock:
            self._cache = {
                cache_desc: data
                for cache_desc, data in self._cache.items()
                if (
                    # Cache entries can be associated to something else than events
                    'event' not in cache_desc or
                    # Either we care about raw and we check, or blanket clear
                    raw is None or
                    cache_desc.get('raw') == raw
                )
            }


class _Trace(Loggable, _InternalTraceBase):
    """
    Object at the bottom of a :class:`_TraceViewBase` stack.

    It drives the actual trace parser and caches.
    """
    def _select_userspace(self, source_event, meta_event, df):
        # pylint: disable=unused-argument,no-self-use

        # tracing_mark_write is the name of the kernel function invoked when
        # writing to: /sys/kernel/debug/tracing/trace_marker
        # That said, it's not the end of the world if we don't filter on that
        # as the meta event name is supposed to be unique anyway
        if isinstance(df.collect_schema()['ip'], (pl.String, pl.Binary, pl.Categorical)):
            df = df.filter(pl.col('ip').cast(pl.String).str.starts_with('tracing_mark_write'))
        return (df, 'buf')

    def _select_trace_printk(self, source_event, meta_event, df):
        content_col = {
            'bprint': 'buf',
        }[source_event]
        return (df, content_col)

    _META_EVENT_SOURCE = {
        'userspace': {
            'print': _select_userspace,
        },
        'trace_printk': {
            'bprint': _select_trace_printk,
        },
    }
    """
    Define the source of each meta event.

    Meta events are events derived by parsing a field of another given real event.

    Meta events are event with a name formed as ``prefix:event``.
    This mapping has the format::

        prefix:
            source event: getter_func
            ... # Multiple sources are allowed
        ...

    ``getter_func`` takes the :class:`pandas.DataFrame` of the source event as
    input and returns a tuple of ``(dataframe, line_column)`` where:

        * ``dataframe`` is the source event dataframe, potentially filtered.
        * ``line_column`` is the name of the column containing the lines to
          parse.
    """

    def __init__(self,
        trace_path=None,
        plat_info=None,
        parser=None,
        plots_dir=None,

        max_mem_size=None,
        swap_dir=None,
        enable_swap=True,
        max_swap_size=None,
    ):
        super().__init__()
        self._lock = threading.RLock()
        self._parseable_events = {}

        stack = contextlib.ExitStack()
        self._cm_stack = stack
        self.__deallocator = _Deallocator(
            f=lambda: self._cm_stack.__exit__(None, None, None),
            on_del=True,
            at_exit=True,
        )

        # Make sure that we always operate with an active StringCache when
        # manipulating a trace object. This prevents issues with LazyFrame
        # built out of a DataFrame containing Categorical data, in places where
        # the user does not control the creation of the DataFrame.
        stack.enter_context(pl.StringCache())

        trace_path = str(trace_path) if trace_path else None
        self.trace_path = trace_path

        if parser is None:
            if not trace_path:
                raise ValueError('A trace path must be provided')

            url = urlparse(trace_path)
            scheme = url.scheme
            if scheme == 'lisatrace':
                parser = ClientTraceParser.from_trace_url(trace_path)
            elif scheme in ('file', ''):
                _, extension = os.path.splitext(url.path)

                if extension == '.html':
                    parser = SysTraceParser.from_html
                elif extension == '.txt':
                    parser = HRTxtTraceParser.from_txt_file
                elif extension == '.perfetto-trace':
                    parser = PerfettoTraceParser
                else:
                    parser = TraceDumpTraceParser.from_dat
        self._parser = parser

        # No-op cache so that the cacheable metadata machinery does not fall
        # over when querying the trace-id.
        self._cache = _TraceCache()
        trace_id = self._get_trace_id()

        if enable_swap:
            if trace_path and os.path.exists(trace_path):
                if swap_dir is None:
                    basename = os.path.basename(trace_path)
                    swap_dir = os.path.join(
                        os.path.dirname(trace_path),
                        f'.{basename}.lisa-swap'
                    )
                    try:
                        os.makedirs(swap_dir, exist_ok=True)
                    except OSError:
                        dir_cache = DirCache(
                            category='trace_swap',
                            fmt_version='1',
                        )
                        swap_dir = str(dir_cache.get_entry(trace_id))

                if max_swap_size is None:
                    trace_size = os.stat(trace_path).st_size
                    # Use 10 times the size of the trace so that there is
                    # enough room to store large artifacts like a JSON dump of
                    # the trace
                    max_swap_size = trace_size * 10
        else:
            swap_dir = None
            max_swap_size = None

        # The platform information used to run the experiments
        if plat_info is None:
            # Delay import to avoid circular dependency
            # pylint: disable=import-outside-toplevel
            from lisa.platforms.platinfo import PlatformInfo
            plat_info = PlatformInfo()
        else:
            # Make a shallow copy so we can update it
            plat_info = copy.copy(plat_info)

        self._source_events_known = False

        if plots_dir:
            _deprecated_warn('Trace(plots_dir=...) parameter is deprecated', stacklevel=2)
        elif not plots_dir and trace_path:
            plots_dir = os.path.dirname(trace_path)
        self.plots_dir = plots_dir

        cache = _TraceCache.from_swap_dir(
            trace_path=trace_path,
            swap_dir=swap_dir,
            max_swap_size=max_swap_size,
            max_mem_size=max_mem_size,
            trace_id=trace_id,
            metadata=self._cache._metadata,
        )
        stack.enter_context(cache)
        self._cache = cache

        # Initial scrub of the swap to discard unwanted data, honoring the
        # max_swap_size right from the beginning
        self._cache.scrub_swap()
        self._cache.to_swap_dir(blocking=False)

        try:
            self._parseable_events = self._cache.get_metadata('parseable-events')
        except MissingMetadataError:
            pass

        # Preload metadata from the cache, to trigger any side effect when
        # processing them. For example, this can help avoiding spinning the
        # parser if the set of available events is already known.
        self._preload_metadata_cache()

        # Register what we currently have
        self.plat_info = plat_info
        # Update the platform info with the data available from the trace once
        # the Trace is almost fully initialized
        self.plat_info = plat_info.add_trace_src(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.__deallocator.run()

    def _preload_metadata_cache(self):
        def fail():
            raise ValueError('Fake metadata value')

        def load(key):
            try:
                return self._process_metadata(key=key, get=fail)
            except ValueError:
                return None

        return {
            key: value
            for key in TraceParserBase.METADATA_KEYS
            if (value := load(key)) is not None
        }

    def _preload_events(self, events):
        if events is _ALL_EVENTS:
            return self._preload_all_events()
        else:
            events = OptionalTraceEventChecker.from_events(events)

            # This creates a polars LazyFrame from the cache if available, so it's
            # cheap since we always cache the data coming from the parser.
            df_map = self._load_cache_raw_df(
                events,
                allow_missing_events=True,
            )

            return set(df_map.keys())

    @memoized
    def _preload_all_events(self):
        meta, df_map = self._parse_all()
        return set(df_map.keys())

    def _parse_all(self):

        def convert(parser, df_map):
            return {
                event: _convert_df_from_parser(
                    df,
                    parser=parser,
                    cache=self._cache,
                )
                for event, df in df_map.items()
            }

        def finalize(preloaded_df_map, preloaded_meta, parser, df_map):
            parsed_df_map = convert(parser, df_map)
            self._insert_events({
                event: df
                for event, df in parsed_df_map.items()
                if event not in preloaded_df_map
            })
            df_map = {
                **preloaded_df_map,
                **parsed_df_map,
            }

            if set(preloaded_meta.keys()) < set(TraceParserBase.METADATA_KEYS):
                parsed_meta = parser.get_all_metadata()
                parsed_meta = {
                    k: self._process_metadata(key=k, get=lambda: v)
                    for k, v in parsed_meta.items()
                    if k not in preloaded_meta.keys()
                }
            else:
                parsed_meta = {}

            meta = {
                **preloaded_meta,
                **parsed_meta
            }
            return (meta, df_map)

        def load_from_cache(event):
            cache_desc = self._make_raw_cache_desc(event)
            try:
                df = self._cache.fetch(cache_desc, insert=True)
            except KeyError:
                return None
            else:
                return _ParsedDataFrame.from_df(df)

        def parse(preloaded_df_map, preloaded_meta, events):
            needed_metadata = set(TraceParserBase.METADATA_KEYS) - set(preloaded_meta.keys())

            if (needed_metadata or events or events is _ALL_EVENTS):
                _finalize = functools.partial(
                    finalize,
                    preloaded_df_map=preloaded_df_map,
                    preloaded_meta=preloaded_meta,
                )

                if events is _ALL_EVENTS:
                    with self._get_parser(events=events, needed_metadata=needed_metadata) as parser:
                        try:
                            return _finalize(parser=parser, df_map=parser.parse_all_events())
                        except NotImplementedError:
                            events = parser.get_metadata('available-events')

                with self._get_parser(events=events, needed_metadata=needed_metadata) as parser:

                    # Use custom best-effort implementation that allows
                    # filtering out _all_ exceptions. We are here to parse as
                    # many events as we can, regardless of any error.
                    def parse_best_effort(event):
                        try:
                            return parser.parse_event(event)
                        except Exception:
                            return None

                    return _finalize(
                        parser=parser,
                        df_map={
                            event: df
                            for event in events
                            if (df := parse_best_effort(event)) is not None
                        }
                    )
            else:
                return (
                    preloaded_meta,
                    preloaded_df_map,
                )

        preloaded_meta = self._preload_metadata_cache()

        try:
            events = preloaded_meta['available-events']
        except KeyError:
            events = _ALL_EVENTS
            preloaded_df_map = {}
        else:
            preloaded_df_map = {
                event: df
                for event in events
                if (df := load_from_cache(event)) is not None
            }
            events = {
                event
                for event in events
                if event not in preloaded_df_map
            }

        meta, df_map = parse(preloaded_df_map, preloaded_meta, events)

        df_map = {
            event: df
            for event, df in df_map.items()
        }
        return (meta, df_map)

    def get_metadata(self, key):
        """
        Get metadata from the underlying trace parser.

        .. seealso:: :meth:`TraceParserBase.get_metadata`
        """
        return self._get_metadata(key=key)

    def _get_metadata(self, key, parser=None, try_hard=False):
        def get():
            if parser is None:
                @contextlib.contextmanager
                def _cm():
                    with self._get_parser(needed_metadata={key}) as parser:
                        yield parser

                cm = _cm()
            else:
                # If we got passed a parser, we leave the decision to use it as a
                # context manager or not to the caller.
                cm = parser if try_hard else nullcontext(parser)

            with cm as _parser:
                value = _parser.get_metadata(key)

            return value

        return self._process_metadata(get=get, key=key)

    @staticmethod
    def _meta_to_json(meta):
        def process(key, value):
            if key == 'available-events':
                # Ensure we have a list so that it can be dumped to JSON
                value = sorted(set(value))

            elif key == 'trace-id':
                value = str(value)

            elif key == 'time-range':
                start, end = value
                # Ensure we round the float value of the window boundaries
                # correctly so that all events are within that window.
                # Otherwise we could have the first/last event out of that
                # window due to wrong rounding.
                start = Timestamp(start, rounding='down')
                end = Timestamp(end, rounding='up')

                # Return integer number of nanoseconds. This ensures we cache
                # the nanosecond exact value rather than a float that might be
                # imprecise.
                value = (start.as_nanoseconds, end.as_nanoseconds)

            elif key == 'symbols-address':
                # Unzip the dict into a list of keys and list of values, since
                # that will be cheaper to deserialize and serialize to JSON
                addrs, names = tuple(unzip_into(
                    2,
                    # Turn to a list to allow JSON caching, since JSON cannot
                    # have non-string keys
                    dict(value).items()
                ))
                names = [
                    [name] if isinstance(name, str) else name
                    for name in names
                ]
                value = (addrs, names)

            return value

        return {
            key: process(key, value)
            for key, value in meta.items()
        }

    @staticmethod
    def _meta_from_json(meta):
        def process(key, value):
            """
            Process a value prepared with :meth:`_meta_to_json`
            """
            if key == 'available-events':
                value = set(value)
            elif key == 'time-range':
                start, end = value
                assert isinstance(start, int)
                assert isinstance(end, int)

                # Ensure we round the float value of the window boundaries
                # correctly so that all events are within that window.
                # Otherwise we could have the first/last event out of that
                # window due to wrong rounding.
                start = Timestamp(start, unit='ns', rounding='down')
                end = Timestamp(end, unit='ns', rounding='up')
                value = (start, end)

            elif key == 'symbols-address':
                # Turn the list of items (to allow JSON storage) back to a dict.
                addr, names = value
                value = dict(zip(
                    map(int, addr),
                    names
                ))

            return value

        return {
            key: process(key, value)
            for key, value in meta.items()
        }

    def _process_metadata(self, key, get):
        def process(key, value):
            """
            Process a value prepared with :meth:`_meta_to_json`
            """
            value = self._meta_from_json({key: value})[key]

            if key == 'available-events':
                with self._lock:
                    # Populate the list of available events, and inform the
                    # rest of the code that this list is definitive.
                    self._update_parseable_events({
                        event: True
                        for event in value
                    })
                    # Note that this will not take into account meta-events.
                    self._source_events_known = True

            elif key == 'symbols-address':
                # The parsers can provide a list of names for each address, but
                # our API only exposes a single name, so we pick the best one
                def pick(names):
                    # If we can, we choose from the names that are valid
                    # identifiers. This will weed-out strange symbols like arm
                    # mapping symbols and the likes.
                    best_names = sorted(
                        (
                            name
                            for name in names
                            if name.isidentifier()
                        ),
                        key=len,
                    )
                    if best_names:
                        # Return the longest name, as it's less likely to be a
                        # less descriptive section name or something like that.
                        return best_names[-1]
                    else:
                        return names[0]

                value = {
                    addr: pick(names)
                    for addr, names in value.items()
                }

            return value

        def _get(key):
            value = get()
            return self._meta_to_json({key: value})[key]

        def get_cacheable(key):
            try:
                value = self._cache.get_metadata(key)
            except MissingMetadataError:
                value = _get(key)
                self._cache.update_metadata({key: value}, blocking=False)
            return value

        value = get_cacheable(key)
        return process(key, value)

    @classmethod
    def _is_meta_event(cls, event):
        sources = cls.get_event_sources(event)
        return len(sources) > 1

    @classmethod
    def get_event_sources(cls, event):
        """
        Get the possible sources events of a given event.

        For normal events, this will just be a list with the event itself in
        it.

        For meta events, this will be the list of source events hosting that
        meta-event.
        """
        try:
            prefix, _ = event.split('@', 1)
        except ValueError:
            return [event]

        try:
            # It is capital that "event" is the first item so we allow the
            # parser to handle it directly.
            return (event, *sorted(cls._META_EVENT_SOURCE[prefix].keys()))
        except KeyError:
            return (event,)

    @property
    # Memoization is necessary to ensure the parser always gets the same name
    # on a given Trace instance when the parser is not a type.
    @lru_memoized(first_param_maxsize=None, other_params_maxsize=None)
    def trace_state(self):
        parser = self._parser
        # The parser type will potentially change the exact content in raw
        # dataframes
        def get_name(parser):
            if isinstance(parser, TraceParserBase):
                return parser.get_parser_id()
            else:
                # If the parser is an instance of something, we cannot safely track its
                # state so just make a unique name for it
                if (
                    isinstance(parser, (type, types.MethodType)) and
                    not any(
                        x in parser.__qualname__
                        for x in ('<locals>', '<lambda>')
                    )
                ):
                    return f'{parser.__module__}.{parser.__qualname__}'
                else:
                    cls = type(parser)
                    return f'{cls.__module__}.{cls.__qualname__}-instance:{uuid.uuid4().hex}'

        return (
            self.__class__.__qualname__,
            get_name(parser),
        )

    @property
    @memoized
    def cpus_count(self):
        """
        Number of CPUs on which data was gathered in that trace.

        This will typically be the number of CPUs on the target, but might
        sometimes differ depending on the file format of the trace.
        """
        try:
            return self.plat_info['cpus-count']
        except KeyError:
            try:
                count = self.get_metadata('cpus-count')
            # If we don't know the number of CPUs, check the trace for the
            # highest-numbered CPU that traced an event.
            except MissingMetadataError:
                # Sched_switch should be enough if it's available
                if 'sched_switch' in self.available_events:
                    checked_events = ['sched_switch']
                # This is pretty costly, as it will trigger parsing of all
                # events in the trace, so only do it as last resort
                else:
                    checked_events = self.available_events

                max_cpu = max(
                    int(df.select(pl.max('__cpu')).collect().item())
                    for df, meta in (
                        self._internal_df_event(event)
                        for event in checked_events
                    )
                    if '__cpu' in df.collect_schema().names()
                )
                count = max_cpu + 1
                self.logger.debug(f"Estimated CPU count from trace: {count}")

            return count

    def _get_parser(self, events=tuple(), needed_metadata=None):
        cache = self._cache
        path = self.trace_path
        events = events if events is _ALL_EVENTS else set(events)
        needed_metadata = set(needed_metadata or [])

        @contextlib.contextmanager
        def cm():
            with pl.StringCache(), self._cache._parser_temp_path() as temp_dir:
                self.logger.debug(f'Spinning up trace parser {self._parser}: path={path}, events={events}, needed_metadata={needed_metadata}')
                parser = self._parser(
                    path=path,
                    events=events,
                    needed_metadata=needed_metadata,
                    temp_dir=temp_dir,
                )

                with parser as parser:
                    yield parser

                # While we are at it, gather a bunch of metadata. Since we did not
                # explicitly asked for it, the parser will only give
                # it if it was a cheap byproduct.
                self._update_metadata(parser)

        return cm()

    def _update_metadata(self, parser):
        # Tentatively get the metadata value, in case they are available
        for key in TraceParserBase.METADATA_KEYS:
            with contextlib.suppress(MissingMetadataError):
                self._get_metadata(key, parser=parser)

    def _update_parseable_events(self, mapping):
        with self._lock:
            update = {
                k: v
                for k, v in mapping.items()
                if v != self._parseable_events.get(k)
            }
            if update:
                self._parseable_events.update(update)
                self._cache.update_metadata(
                    {
                        'parseable-events': self._parseable_events,
                    },
                    blocking=False,
                )
            return self._parseable_events

    @property
    def basetime(self):
        return self._get_time_range()[0]

    @property
    def endtime(self):
        return self._get_time_range()[1]

    @property
    def start(self):
        return self.basetime

    @property
    def end(self):
        return self.endtime

    @memoized
    def _get_time_range(self, parser=None):
        return self._get_metadata('time-range', parser=parser)

    @memoized
    def _get_trace_id(self):
        try:
            return self._get_metadata('trace-id', try_hard=True)
        except MissingMetadataError:
            if (path := self.trace_path):
                with open(path, 'rb') as f:
                    md5 = checksum(f, 'md5')
                id_ = f'md5sum-{md5}'
            else:
                id_ = '<unknown>'

            self._cache.update_metadata(
                {
                    'trace-id': id_,
                },
                blocking=False,
            )
            return id_

    @memoized
    def _make_raw_cache_desc(self, event):
        spec = dict(
            # This is used when clearing the cache to know if a given entry is
            # related to a raw event or e.g. an analysis.
            raw=True,
            event=event,
            trace_state=self.trace_state,
        )
        return _CacheDataDesc(spec=spec, fmt=_TraceCache.DATAFRAME_SWAP_FORMAT)

    def _internal_df_event(self, event, write_swap=None, raw=None, **kwargs):
        if write_swap is not None:
            _deprecated_warn('write_swap parameter has no effect anymore')

        if raw:
            raise ValueError(f'raw=True is not supported anymore, dataframes are always post processed by parsers to be as close as possible to the ftrace event format')

        # Make sure all raw descriptors are made the same way, to avoid
        # missed sharing opportunities
        cache_desc = self._make_raw_cache_desc(event)

        try:
            try:
                df = self._cache.fetch(cache_desc, insert=True)
            except KeyError:
                df = self._load_cache_raw_df(TraceEventChecker(event))[event]
            else:
                df = _ParsedDataFrame.from_df(df)
        except MissingTraceEventError as e:
            e.available_events = self.available_events
            raise e
        else:
            assert isinstance(df, _ParsedDataFrame)
            df = df.df
            # TODO: If and when this is solved, attach the name of the event to
            # the LazyFrames:
            # https://github.com/pola-rs/polars/issues/5117
            meta = dict(event=event)
            return df, meta

    def _load_cache_raw_df(self, event_checker, allow_missing_events=False):
        events = event_checker.get_all_events()

        # Get the raw dataframe from the cache if possible
        def try_from_cache(event):
            cache_desc = self._make_raw_cache_desc(event)
            try:
                df = self._cache.fetch(cache_desc, insert=True)
            except KeyError:
                return None
            else:
                return _ParsedDataFrame.from_df(df)

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
        events_to_load = sorted(events - from_cache.keys())

        # Cheap check to avoid spinning up the parser for nothing
        with self._lock:
            for event in list(events_to_load):
                # If we have discovered all the available events, there is no
                # point in trying again.
                default = (not self._source_events_known) or self._is_meta_event(event)
                if not self._parseable_events.get(event, default):
                    if allow_missing_events:
                        events_to_load.remove(event)
                    else:
                        raise MissingTraceEventError([event])

        df_from_trace = self._load_raw_df(events_to_load)
        self._insert_events(df_from_trace)

        df_map = {**from_cache, **df_from_trace}
        try:
            event_checker.check_events(df_map.keys())
        except MissingTraceEventError as e:
            if not allow_missing_events:
                raise MissingTraceEventError(
                    e.missing_events,
                    msg='Events not found in the trace: {missing_events}{available}'
                ) from e

        return df_map

    def _insert_events(self, df_map):
        for event, df in df_map.items():
            assert isinstance(df, _ParsedDataFrame)
            if df.meta['mem_cacheable']:
                cache_desc = self._make_raw_cache_desc(event)
                self._cache.insert(
                    cache_desc,
                    df.df,
                    write_swap=True,
                    swappable=df.meta['swap_cacheable'],
                    # For raw dataframe, always write in the swap area if asked for
                    # since parsing cost is known to be high
                    ignore_cost=True,
                )

    def _parse_raw_events(self, events):
        if not events:
            return {}

        def parse(parser, event):
            try:
                return parser.parse_event(event)
            # We only catch MissingTraceEventError, so that we know the parser
            # made a voluntary decision to state the event was not available,
            # rather than just crashing in an unintended way. However, we will
            # display the exceptions's __cause__ later on, which the parser can
            # set to inform the user of any actual problem it encountered while
            # parsing the data.
            except MissingTraceEventError as e:
                return None


        with self._get_parser(events) as parser:
            df_map = {
                event: _convert_df_from_parser(
                    df=df,
                    parser=parser,
                    cache=self._cache,
                )
                for event in events
                if (df := parse(parser, event)) is not None
            }

        return df_map

    def _parse_meta_events(self, meta_events):
        if not meta_events:
            return {}

        # Gather the information to parse the meta event
        def make_spec(meta_event):
            prefix, event = meta_event.split('@', 1)
            data_getters = self._META_EVENT_SOURCE[prefix]
            return (meta_event, event, data_getters)

        meta_specs = list(map(make_spec, meta_events))

        # Map each trimmed event name back to its meta event name
        events_map = {
            event: meta_event
            for meta_event, event, _ in meta_specs
        }

        # Explode per source event
        meta_specs = [
            (meta_event, event, source_event, source_getter)
            for (meta_event, event, data_getters) in meta_specs
            for source_event, source_getter in data_getters.items()
        ]

        def get_missing(df_map):
            return events_map.keys() - df_map.keys()

        df_map = {}
        # Group all the meta events by their source event, so we process source
        # dataframes one by one
        for source_event, specs in groupby(meta_specs, key=itemgetter(2)):
            try:
                df, meta = self._internal_df_event(source_event)
            except MissingTraceEventError:
                pass
            else:
                assert isinstance(df, pl.LazyFrame)

                # Add all the header fields from the source dataframes
                extra_df = df.select(cs.by_name('Time') | cs.starts_with('__'))

                with tempfile.TemporaryDirectory() as temp_dir:
                    for (meta_event, event, _source_event, source_getter) in specs:  # pylint: disable=unused-variable
                        source_df, line_field = source_getter(self, _source_event, event, df)
                        schema = source_df.collect_schema()

                        # If the lines are in a dtype we won't be able to
                        # handle, we won't add an entry to df_map, leading to a
                        # missing event
                        if not isinstance(schema[line_field], (pl.String, pl.Categorical, pl.Binary)):
                            continue

                        # Ensure we have bytes and not str
                        source_df = source_df.with_columns(
                            pl.col(line_field).cast(pl.String).cast(pl.Binary)
                        )

                        source_df = source_df.select((
                            # Use nanoseconds, so that we don't end up with
                            # timedelta() values that are only accurate down to
                            # the microsecond.
                            pl.col('Time').dt.total_nanoseconds(),
                            pl.col(line_field),
                        )).collect()

                        try:
                            parser = MetaTxtTraceParser(
                                lines=source_df[line_field],
                                time=source_df['Time'],
                                time_unit='ns',
                                events=[event],
                                temp_dir=temp_dir,
                            )
                        # An empty dataframe would trigger an exception from
                        # MetaTxtTraceParser, so we just skip over it and if the
                        # event cannot be found anywhere, we will raise an
                        # exception.
                        #
                        # Also, some android kernels hide trace_printk() strings
                        # which prevents from parsing anything at all ...
                        except ValueError:
                            continue
                        else:
                            try:
                                with parser as parser:
                                    _df = parser.parse_event(event)
                            except MissingTraceEventError:
                                continue
                                # In case a meta-event is spread among multiple
                                # events, we get all the dataframes and concatenate
                                # them together
                            else:
                                _df = _convert_df_from_parser(
                                    _df,
                                    parser=parser,
                                    cache=self._cache,
                                )
                                _df = _df.df

                                _df = _df.join(
                                    extra_df,
                                    on='Time',
                                    how='left',
                                    coalesce=True,
                                )

                                _df = _df.select(
                                    order_as(
                                        sorted(
                                            _df.collect_schema().names(),
                                            key=lambda col: 0 if col.startswith('__') else 1
                                        ),
                                        ['Time']
                                    )
                                )

                                df_map.setdefault(event, []).append(_df)

                        if not get_missing(df_map):
                            break

        def concat(df_list):
            try:
                df, = df_list
            except ValueError:
                df = pl.concat(df_list, how='diagonal_relaxed')
                df = df.sort('Time')

            return df

        df_map = {
            event: concat(df_list)
            for event, df_list in df_map.items()
        }

        # On some parsers, meta events are treated as regular events so attempt
        # to load them from there as well
        for event in get_missing(df_map):
            with contextlib.suppress(MissingTraceEventError):
                df, meta = self._internal_df_event(event)
                df_map[event] = df

        return {
            events_map[event]: _ParsedDataFrame.from_df(df)
            for event, df in df_map.items()
        }

    def _load_raw_df(self, events):
        events = set(events)
        if not events:
            return {}

        meta_events = set(filter(self._is_meta_event, events))

        # Pass the entire set of events to _parse_raw() first, in case the
        # parser can handle meta events natively.
        df_map = self._parse_raw_events(events)
        meta_df_map = self._parse_meta_events(meta_events - df_map.keys())
        df_map.update(meta_df_map)

        # remember the events that we tried to parse and that turned out to not be available
        self._update_parseable_events({
            event: (event in df_map)
            for event in events
        })

        return df_map

    def has_events(self, events, namespaces=None):
        """
        Returns True if the specified event is present in the parsed trace,
        False otherwise.

        :param events: trace event name or list of trace events
        :type events: str or list(str) or TraceEventCheckerBase
        """

        if isinstance(events, str):
            events = [events]

        if not isinstance(events, TraceEventCheckerBase):
            events = AndTraceEventChecker.from_events(events=events)

        try:
            events.check_events(self.available_events, namespaces=namespaces)
        except MissingTraceEventError:
            return False
        else:
            return True


class Trace(
    DelegateToAttr(
        '_Trace__view',
        [_InternalTraceBase],
    ),
    TraceBase,
):
    """
    This class provides a way to access event dataframes and ties
    together various low-level moving pieces to make that happen.

    :param trace_path: File containing the trace
    :type trace_path: str or None

    :param events: events to be parsed. Since events can be loaded on-demand,
        that is optional but still recommended to improve trace parsing speed.

        .. seealso:: :meth:`lisa.trace.TraceBase.df_event` for event formats accepted.
    :type events: TraceEventCheckerBase or list(str) or None

    :param events_namespaces: List of namespaces of the requested events. Each
        namespace will be tried in order until the event is found. The ``None``
        namespace can be used to specify no namespace. The full event name is
        formed with ``<namespace>__<event>``.
    :type events_namespaces: list(str or None)

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

    :param parser: Optional trace parser to use as a backend. It must
        implement the API defined by :class:`TraceParserBase`, and will be
        called as ``parser(path=trace_path, events=events,
        needed_metadata={'time-range', ...})`` with the events that should be
        parsed. Other parameters must either have default values, or be
        pre-assigned using partially-applied constructor (for subclasses of
        :class:`lisa.utils.PartialInit`) or :func:`functools.partial`. By
        default, ``.txt`` files will be assumed to be in human-readable format
        output directly by the kernel (or ``trace-cmd report`` without ``-R``).
        Support for this format is limited and some events might not be parsed
        correctly (or at least without custom event parsers).
    :type parser: object or None

    :param plots_dir: directory where to save plots
    :type plots_dir: str

    :param sanitization_functions: This parameter is only for backward
        compatibility with existing code, use
        :meth:`lisa.trace.TraceBase.get_view` with ``process_df`` parameter
        instead.
    :type sanitization_functions: object

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

    :Supporting more events in text parsers:

        .. note:: ``trace.dat`` parser can now fully infer the dataframe schema
            from the binary trace.dat and does not require (nor allow) any
            manual setting.

        Subclasses of :class:`TraceParserBase` can usually auto-detect the
        event format, but there may be a number of reasons to pass a custom
        event parser:

        * The event format produced by a given kernel differs from the
          description bundled with the parser, leading to incorrect parse
          (missing field).

        * The event cannot be parsed in raw format in case text output of
          ``trace-cmd`` is used, because of a ``const char*`` field displayed
          as a pointer for example.

          .. seealso:: For events not following the regular field syntax,
            use :class:`CustomFieldsTxtEventParser`

        * Automatic detection can take a heavy performance toll. This is
          why parsers needing descriptions will come with pre-defined
          descritption of most used events.

        Custom event parsers can be passed as extra parameters to the parser,
        which can be set manually::

            # Event parsers provided by TxtTraceParser can also be overridden
            # with user-provided ones in case they fail to parse what a given
            # kernel produced
            event_parsers = [
                TxtEventParser('foobar', fields={'foo': int, 'bar': 'string'}, raw=True)
            ]

            # Pre-fill the "event_parsers" parameter of
            # TxtEventParser.from_dat() using a partial application.
            #
            # Note: you need to choose the parser appropriately for the
            # format of the trace, since the automatic filetype detection is
            # bypassed.
            parser = TxtTraceParser.from_dat(event_parsers=event_parsers)
            trace = Trace('foobar.dat', parser=parser)

        .. warning:: Custom event parsers that are not types or functions (such
            as partially-applied values) will tie the on-disc swap entries to
            the parser :class:`Trace` instance, incurring higher
            :class:`pandas.DataFrame` load time when a new :class:`Trace`
            object is created.
    """

    def _init(self, view, df_fmt):
        self.__view = view
        self._df_fmt = df_fmt or 'pandas'

    def __enter__(self):
        self.__view.__enter__()
        return self

    def __exit__(self, *args):
        return self.__view.__exit__(*args)

    def __init__(self, *args, df_fmt=None, **kwargs):
        view = self._view_from_user_kwargs(*args, **kwargs)
        self._init(view, df_fmt=df_fmt)

    def _with_view(self, view, df_fmt):
        new = super().__new__(self.__class__)
        new._init(
            view=view,
            df_fmt=df_fmt or self._df_fmt
        )
        return new

    @classmethod
    def _view_from_user_kwargs(cls,
        *args,
        normalize_time=False,
        strict_events=False,
        events=None,
        events_namespaces=('lisa__', None),

        sanitization_functions=None,
        write_swap=None,
        **kwargs,
    ):
        view_kwargs = {}

        if write_swap is not None:
            _deprecated_warn('write_swap parameter has no effect anymore, you can stop using it')

        if sanitization_functions is not None:
            _deprecated_warn('Custom sanitization functions are not supported anymore, use trace.get_view(process_df=...) instead.')

            def process_df(event, df):
                try:
                    f = sanitization_functions[event]
                except KeyError:
                    return df
                else:
                    return f(view, event, df, dict())

            view_kwargs.update(process_df=process_df)

        view_kwargs.update(
            normalize_time=normalize_time,
            events_namespaces=events_namespaces,
            strict_events=strict_events,
            events=events,
        )

        trace = _Trace(*args, **kwargs)
        view = trace.get_view(**view_kwargs)

        assert isinstance(view, _TraceViewBase)
        return view

    @property
    def _default_ana_params(self):
        # In the user-visible analysis, we want to change some defaults that
        # will improve the immediate experience, at the expense of good
        # composition. For example, using ui=None means that a user calling a
        # plot method twice will get 2 toolbars. but it can still be disabled
        # manually. Since composition can sometimes suffer, the internal
        # analysis proxy and the default values on plot methods are set to less
        # friendly but more predictable defaults.
        return dict(
            # Default to displaying a toolbar in notebooks
            output=None,
            df_fmt=self._df_fmt,
        )

    # Memoize the analysis as the analysis themselves might memoize some
    # things, so we don't want to trash that.
    @property
    @memoized
    def ana(self):
        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base
        # pylint: disable=import-outside-toplevel
        from lisa.analysis._proxy import AnalysisProxy
        return AnalysisProxy(self, params=self._default_ana_params)

    @property
    @memoized
    @deprecate(replaced_by=ana, deprecated_in='3.0', removed_in='4.0')
    def analysis(self):
        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base
        # pylint: disable=import-outside-toplevel
        from lisa.analysis._proxy import _DeprecatedAnalysisProxy

        # self.analysis is deprecated so we can transition to using holoviews
        # in all situations, even when the backend is matplotlib
        return _DeprecatedAnalysisProxy(self, params=self._default_ana_params)

    # Allow positional parameter for "window" for backward compat
    def get_view(self, window=None, *, df_fmt=None, **kwargs):
        kwargs['window'] = window
        view = self.__view.get_view(**kwargs)
        # Always preserve the same user-visible type so that view types are
        # 100% an implementation detail that does not leak.
        return self._with_view(
            view,
            df_fmt=df_fmt
        )

    @property
    def trace_state(self):
        return (
            self.__view.trace_state,
            self._df_fmt,
        )

    def df_event(self, event, *, df_fmt=None, **kwargs):
        df_fmt = df_fmt or self._df_fmt

        df, meta = self.__view._internal_df_event(
            event,
            # Provide the information to the stack so that we can apply the
            # legacy signals in _WindowTraceView for pandas format
            df_fmt=df_fmt,
            **kwargs
        )

        df = _df_to(df, index='Time', fmt=df_fmt)
        return df

    @classmethod
    @contextlib.contextmanager
    def from_target(cls, target, events=None, buffer_size=10240, filepath=None, **kwargs):
        """
        Context manager that can be used to collect a
        :class:`lisa.trace.TraceBase` directly from a
        :class:`lisa.target.Target` without needing to setup an
        :class:`FtraceCollector`.

        **Example**::

            from lisa.trace import Trace
            from lisa.target import Target

            target = Target.from_default_conf()

            with Trace.from_target(target, events=['sched_switch', 'sched_wakeup']) as trace:
                target.execute('echo hello world')
                # DO NOT USE trace object inside the `with` statement

            trace.ana.tasks.plot_tasks_total_residency(filepath='plot.png')


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
        plat_info = target.plat_info
        needs_temp = filepath is None

        if needs_temp:
            @contextlib.contextmanager
            def cm_func():
                with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as temp:
                    yield temp.name

            cm = cm_func()
        else:
            cm = nullcontext(filepath)

        with cm as path:
            proxy = _TraceProxy(path if needs_temp else None)
            ftrace_coll = FtraceCollector(target, events=events, buffer_size=buffer_size, output_path=path)
            with ftrace_coll:
                yield proxy

            trace = cls(
                path,
                events=events,
                strict_events=True,
                plat_info=plat_info,
                **kwargs
            )

        proxy._set_trace(trace)

    @classmethod
    def get_event_sources(cls, *args, **kwargs):
        return _Trace.get_event_sources(*args, **kwargs)

    def _internal_df_event(self, *args, **kwargs):
        return self.__view._internal_df_event(*args, **kwargs)

    def _preload_events(self, *args, **kwargs):
        return self.__view._preload_events(*args, **kwargs)

    @property
    def basetime(self):
        return self.__view.basetime

    @property
    def endtime(self):
        return self.__view.endtime

    @property
    def start(self):
        return self.__view.start

    @property
    def end(self):
        return self.__view.end


class _TraceProxy(
    DelegateToAttr(
        '_TraceProxy__base_trace',
        [Trace],
    ),
    TraceBase,
):
    class _TraceNotSet:
        def __getattribute__(self, attr):
            raise RuntimeError('The trace instance can only be used after the end of the "with" statement.')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def __init__(self, path):
        self.__base_trace = self._TraceNotSet()
        self.__path = path
        self.__deallocator = _Deallocator(
            # Delete the file once we are done accessing it
            f=functools.partial(_file_cleanup, paths=[path]),
            on_del=True,
            at_exit=True,
        )

    def _set_trace(self, trace):
        self.__base_trace = trace

    def __enter__(self):
        self.__base_trace.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self.__base_trace.__exit__(*args)
        finally:
            self.__deallocator.run()

    @property
    def ana(self):
        return self.__base_trace.ana

    @property
    @deprecate(replaced_by=ana, deprecated_in='3.0', removed_in='4.0')
    def analysis(self):
        return self.__base_trace.analysis

    def df_event(self, *args, **kwargs):
        return self.__base_trace.df_event(*args, **kwargs)

    def _internal_df_event(self, *args, **kwargs):
        return self.__base_trace._internal_df_event(*args, **kwargs)

    def _preload_events(self, *args, **kwargs):
        return self.__base_trace._preload_events(*args, **kwargs)

    @property
    def basetime(self):
        return self.__base_trace.basetime

    @property
    def endtime(self):
        return self.__base_trace.endtime

    @property
    def start(self):
        return self.__base_trace.start

    @property
    def end(self):
        return self.__base_trace.end


class TraceEventCheckerBase(abc.ABC, Loggable, Sequence):
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

    def __init__(self, check=True, checkers=None):
        self.check = check
        self.checkers = checkers or []

    @abc.abstractmethod
    def _select_events(self, check, event_set):
        return set()

    def check_events(self, event_set, check_optional=False, namespaces=None):
        """
        Check that certain trace events are available in the given set of
        events.

        :return: Set of events selected by the checker in ``event_set``.

        :raises: MissingTraceEventError if some events are not available. The
            exception must be raised after inspecting children node and combine
            their missing events so that the resulting exception is accurate.
        """
        if check_optional:
            def rewrite(checker):
                if isinstance(checker, _OptionalTraceEventCheckerBase):
                    return AndTraceEventChecker(checker.checkers)
                else:
                    return checker
            checker = self.map(rewrite)
        else:
            checker = self


        if isinstance(event_set, _AvailableTraceEventsSet):
            # Note that this will lead to selecting an event if one of its
            # namespaced names is available. This is expected, but the
            # check_events() returned set will also contain that non-namespaced
            # name, rather than the actual namespaced name that exists.
            trace = event_set._trace
            view = trace.get_view(events_namespaces=namespaces)
            event_set = view.available_events
        else:
            checker = checker.expand_namespaces(namespaces)

        def check(event):
            return event in event_set
        return checker._select_events(check=check, event_set=event_set)

    @abc.abstractmethod
    def get_all_events(self):
        """
        Return a set of all events that are checked by this checker.

        That may be a superset of events that are strictly required, when the
        checker checks a logical OR combination of events for example.
        """

    @abc.abstractmethod
    def map(self, f):
        """
        Apply the given function to all the children and rebuild a new object
        with the result.
        """

    def expand_namespaces(self, namespaces=None):
        """
        Build a :class:`TraceEventCheckerBase` that will fixup the event names
        to take into account the given namespaces.
        """
        expand = lambda event: _NamespaceTraceView._do_expand_namespaces(event, namespaces=namespaces)
        return self._expand_namespaces(expand)

    def _expand_namespaces(self, expand):
        def fixup(checker):
            if isinstance(checker, TraceEventChecker):
                event = checker.event
                namespaced = expand(event)

                # fnmatch patterns need to be AND-ed so that we can collect all
                # the events matching that pattern, in all namespaces.
                if checker._is_pattern:
                    return AndTraceEventChecker.from_events(namespaced)
                else:
                    return OrTraceEventChecker.from_events(namespaced)
            else:
                return checker
        return self.map(fixup)

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
        :class:`AndTraceEventChecker`.
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
                    # Preload all the events that can be useful for that
                    # function. This will allow loading them in parallel,
                    # speeding up individual trace.df_event() calls
                    view = trace.get_view(events=checker.get_all_events())
                    checker.check_events(view.available_events)

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

    @abc.abstractmethod
    def __bool__(self):
        pass

    def __iter__(self):
        return iter(sorted(self.get_all_events()))

    def __getitem__(self, i):
        return sorted(self.get_all_events())[i]

    def __contains__(self, event):
        return event in self.get_all_events()

    def __len__(self, event):
        return len(self.get_all_events())

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

    def doc_str(self):
        """
        Top-level function called by Sphinx's autodoc extension to augment
        docstrings of the functions.
        """
        return f"\n    * {self._str_internal(style='rst', wrapped=False)}"

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

    def __bool__(self):
        return True

    def get_all_events(self):
        return {self.event}

    def _select_events(self, check, event_set):
        event = self.event
        if check(event):
            return {event}
        else:
            raise MissingTraceEventError(self, available_events=event_set)

    def _str_internal(self, style=None, wrapped=True):
        template = '``{}``' if style == 'rst' else '{}'
        return template.format(self.event)

    @property
    def _is_pattern(self):
        s = self.event
        return '*' in s or '?' in s or '[' in s or ']' in s

    def map(self, f):
        return f(self)


class EmptyTraceEventChecker(TraceEventCheckerBase):
    """
    Check for no event at all.

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated
        method is applied.  If no such attribute is found, no check will be
        done.
    :type check: bool
    """

    def __init__(self, check=True):
        super().__init__(check=check)

    def __bool__(self):
        return False

    def get_all_events(self):
        return set()

    def _select_events(self, check, event_set):
        return set()

    def _str_internal(self, style=None, wrapped=True):
        return '<no event>'

    def map(self, f):
        return f(self)


class AssociativeTraceEventChecker(TraceEventCheckerBase):
    """
    Base class for associative operators like `and` and `or`
    """

    def __init__(self, op_str, event_checkers, check=True, prefix_str=''):
        super().__init__(check=check)
        checker_list = []
        event_checkers = event_checkers or []
        for checker in event_checkers:
            # "unwrap" checkers of the same type, to avoid useless levels of
            # nesting. This is valid since the operator is known to be
            # associative. We don't use isinstance to avoid merging checkers
            # that may have different semantics.
            if type(checker) is type(self):
                checker_list.extend(checker.checkers)
            else:
                checker_list.append(checker)

        # Aggregate them separately to avoid having multiple of them, since
        # they can appear anywhere in the expression tree with the exact same
        # overall effect
        optional_checker_list = [
            checker
            for checker in checker_list
            if isinstance(checker, OptionalTraceEventChecker)
        ]
        checker_list = [
            checker
            for checker in checker_list
            if checker not in optional_checker_list
        ]

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

    def __bool__(self):
        return any(map(bool, self.checkers))

    def map(self, f):
        new = copy.copy(self)
        new = f(new)
        new.checkers = [
            checker.map(f)
            for checker in new.checkers
        ]
        return new

    @memoized
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

        return cls(map(make_event, events), **kwargs)

    def _str_internal(self, style=None, wrapped=True):
        op_str = f' {self.op_str} '
        unwrapped_str = self.prefix_str + op_str.join(
            c._str_internal(style=style, wrapped=True)
            # Sort for stable output
            for c in self.checkers
        )

        template = '({})' if len(self.checkers) > 1 and wrapped else '{}'
        return template.format(unwrapped_str)


class OrTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that one of the given event checkers is satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """

    def __init__(self, event_checkers=None, **kwargs):
        super().__init__('or', event_checkers, **kwargs)

    def _select_events(self, check, event_set):
        if self.checkers:
            failed_checkers = []
            for checker in self.checkers:
                try:
                    return checker._select_events(check=check, event_set=event_set)
                except MissingTraceEventError as e:
                    failed_checkers.extend(e.missing_events)

            cls = type(self)
            raise MissingTraceEventError(
                cls.from_events(failed_checkers),
                available_events=event_set,
            )
        else:
            return set()


class _OptionalTraceEventCheckerBase(AssociativeTraceEventChecker):
    _PREFIX_STR = None
    def __init__(self, event_checkers=None, **kwargs):
        super().__init__(',', event_checkers, prefix_str=self._PREFIX_STR, **kwargs)

    def _select_events(self, check, event_set):
        selected = set()
        for checker in self.checkers:
            try:
                selected.update(
                    checker._select_events(check=check, event_set=event_set)
                )
            except MissingTraceEventError as e:
                pass

        return selected


class OptionalTraceEventChecker(_OptionalTraceEventCheckerBase):
    """
    Do not check anything, but exposes the information that the events may be
    used if present.

    :param event_checkers: Event checkers that may be used
    :type event_checkers: list(TraceEventCheckerBase)
    """
    _PREFIX_STR = 'optional: '


class DynamicTraceEventChecker(_OptionalTraceEventCheckerBase):
    """
    Do not check anything, but exposes the information that one of the group of
    events will be used.

    This allows an API to manually decide which group is chosen based on its
    parameters, but will still convey the information that they are not really
    optional.

    :param event_checkers: Event checkers that may be used
    :type event_checkers: list(TraceEventCheckerBase)
    """
    _PREFIX_STR = 'one group of: '


class AndTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that all the given event checkers are satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """

    def __init__(self, event_checkers=None, **kwargs):
        super().__init__('and', event_checkers, **kwargs)

    def _select_events(self, check, event_set):
        if self.checkers:
            failed_checkers = []
            selected = set()
            for checker in self.checkers:
                try:
                    selected.update(
                        checker._select_events(check=check, event_set=event_set)
                    )
                except MissingTraceEventError as e:
                    failed_checkers.extend(e.missing_events)

            if failed_checkers:
                cls = type(self)
                raise MissingTraceEventError(
                    cls.from_events(failed_checkers),
                    available_events=event_set,
                )
            else:
                return selected
        else:

            return set()

    def doc_str(self):
        joiner = '\n' + '    '
        rst = joiner + joiner.join(
            f"* {c._str_internal(style='rst', wrapped=False)}"
            # Sort for stable output
            for c in self.checkers
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


def will_use_events_from(*events, **kwargs):
    """
    Same as :func:`requires_events` but just exposes some events groups that
    will be used, depending on some dynamic factor.
    """
    return DynamicTraceEventChecker.from_events(events, **kwargs)


class DroppedTraceEventError(Exception):
    """
    Raised when some events were dropped from the ftrace ring buffer because of
    lack of space.
    """


class MissingTraceEventError(RuntimeError, ValueError):
    """
    :param missing_events: The missing trace events
    :type missing_events: TraceEventCheckerBase or list(str)
    """

    def __init__(self, missing_events, available_events=None, msg='Trace is missing the following required events: {missing_events}{available}'):
        # pylint: disable=super-init-not-called
        if not isinstance(missing_events, TraceEventCheckerBase):
            missing = sorted(missing_events)
            try:
                event, = missing
            except ValueError:
                missing_events = AndTraceEventChecker.from_events(
                    missing
                )
            else:
                missing_events = TraceEventChecker(event)

        self._template = msg
        self.missing_events = missing_events
        # Forcibly turn into a list, to avoid carrying around an
        # _AvailableTraceEventsSet with its Trace instance
        self.available_events = sorted(available_events or [])

    def __str__(self):
        available = self.available_events
        missing = self.missing_events

        available_str = ''
        if available:
            closest = {
                matches[0]
                for matches in (
                    get_close_matches(event, available, n=1)
                    for event in missing
                )
                if matches
            }
            if closest:
                available_str = '. Closest available matches are: {}'.format(
                    ', '.join(sorted(closest)),
                )

        return self._template.format(missing_events=missing, available=available_str)


class FtraceConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class of :class:`FtraceCollector`

    Available keys:
    {generated_help}
    {yaml_example}
    """
    STRUCTURE = TopLevelKeyDesc('ftrace-conf', 'FTrace configuration', (
        KeyDesc('events', 'FTrace events to trace', [typing.Sequence[str], TraceEventCheckerBase]),
        KeyDesc('events-namespaces', 'FTrace events namespaces to use. See Trace namespace constructor parameter.', [typing.Sequence[typing.Union[str, None]], None]),
        KeyDesc('functions', 'FTrace functions to trace', [typing.Sequence[str]]),
        KeyDesc('buffer-size', 'FTrace buffer size', [int]),
        KeyDesc('trace-clock', 'Clock used while tracing (see "trace_clock" in ftrace.txt kernel doc)', [str, None]),
        KeyDesc('saved-cmdlines-nr', 'Number of saved cmdlines with associated PID while tracing', [int]),
        KeyDesc('tracer', 'FTrace tracer to use', [str, None]),
        LevelKeyDesc('modules', 'Kernel modules settings', (
            KeyDesc('auto-load', 'Compile kernel modules and load them automatically based on the events that are needed.', [bool]),
        )),
    ))

    def add_merged_src(self, src, conf, optional_events=False, **kwargs):
        """
        Merge-in a configuration source.

        :param src: Name of the merged source
        :type src: str

        :param conf: Conf to merge in
        :type conf: FtraceConf or dict(str, object)

        :param optional_events: If ``True``, the events brought by ``conf``
            will be wrapped in :class:`OptionalTraceEventChecker`. This avoids
            failing just because the user asked for extra events that are not
            present in the kernel.
        :type optional_events: bool
        """

        def get_key(conf, key):
            return conf.get_key(key, quiet=True)

        def get_key_default(conf, key, default):
            try:
                return get_key(conf, key)
            except KeyError:
                return default

        if not isinstance(conf, self.__class__):
            conf = self.__class__(conf=conf)

        def merge_conf(key, val, path):
            new = _merge_conf(key, val, path)
            try:
                existing = get_nested_key(self, path + [key], getitem=get_key)
            except KeyError:
                return (True, new)
            else:
                # Only add to the source if the result is different than what
                # is already set
                return (existing != new, new)

        def _merge_conf(key, val, path):

            def non_mergeable(key):
                if get_key_default(self, key, val) == val:
                    return val
                else:
                    raise KeyError(f'Cannot merge key "{key}": incompatible values specified: {self[key]} != {val}')

            if key == 'functions':
                return sorted(set(val) | set(get_key_default(self, key, [])))
            elif key == 'events-namespaces':
                # We already applied the namespaces to the events so the result
                # can be cleanly merged according to the original meaning.
                return []
            elif key == 'events':
                if not isinstance(val, TraceEventCheckerBase):
                    val = AndTraceEventChecker.from_events(val)
                if optional_events:
                    val = OptionalTraceEventChecker([val])

                # Merging has to take into account defaults, as we will then
                # set the namespace to be empty (None, )
                def get(conf, key):
                    try:
                        return get_key(conf, key)
                    except KeyError:
                        return conf.DEFAULT_SRC.get(key)

                val = val.expand_namespaces(
                    namespaces=get(conf, 'events-namespaces')
                )

                self_val = get_key_default(self, key, [])
                if not isinstance(self_val, TraceEventCheckerBase):
                    self_val = AndTraceEventChecker.from_events(self_val)

                self_val = self_val.expand_namespaces(
                    namespaces=get(self, 'events-namespaces')
                )

                return AndTraceEventChecker([val, self_val])
            elif key == 'buffer-size':
                return max(val, get_key_default(self, key, 0))
            elif key == 'trace-clock':
                return non_mergeable(key)
            elif key == 'saved-cmdlines-nr':
                return max(val, get_key_default(self, key, 0))
            elif key == 'tracer':
                return non_mergeable(key)
            elif key == 'modules':
                return merge_level(val, path + [key])
            elif key == 'auto-load':
                return non_mergeable(key)
            else:
                raise KeyError(f'Cannot merge key "{key}"')

        def merge_level(conf, path=[]):
            return {
                k: v
                for k, (keep, v) in (
                    (k, merge_conf(k, v, path))
                    for k, v in conf.items()
                )
                if keep
            }

        merged = merge_level(conf)
        # Namespaces were expanded in events directly so we want to ensure they
        # will not undergo expansion again if there are cascading merges.
        merged['events-namespaces'] = []

        # We merge some keys with their current value in the conf
        return self.add_src(src, conf=merged, **kwargs)


class CollectorBase(DelegateToAttr('_collector'), Loggable):
    """
    Base class for :class:`devlib.collector.CollectorBase`-based collectors
    using composition.
    """

    TOOLS = []
    """
    Sequence of tools to install on the target when using the collector.
    """

    NAME = None
    """
    Name of the collector class.
    """

    _COMPOSITION_ORDER = 0
    """
    Order in which context managers are composed. ``0`` will be used as the
    innermost context manager.
    """

    def __init__(self, collector, output_path=None):
        self._collector = collector
        self._output_path = output_path
        self._install_tools(collector.target)

    def _install_tools(self, target):
        target.install_tools(self.TOOLS)

    def __enter__(self):
        self._collector.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        # How did we get some coconuts ?
        swallow = self._collector.__exit__(*args, **kwargs)

        try:
            self.get_data()
        except ValueError:
            pass

        return swallow

    def get_data(self, path=None):
        """
        Similar to :meth:`devlib.collector.CollectorBase.get_data` but takes
        the path directly as a parameter in order to disallow representing an
        invalid state where no path has been set.
        """
        coll = self._collector
        path = path or self._output_path

        if path is None:
            raise ValueError('Path cannot be None')

        coll.set_output(path)
        return coll.get_data()

    @deprecate(replaced_by=get_data, deprecated_in='2.0', removed_in='4.0')
    def get_trace(self, path):
        """
        Deprecated alias for :meth:`get_data`.
        """
        return self.get_data(path)


class ComposedCollector(Mapping):
    """
    Compose multiple :class:`lisa.trace.CollectorBase` together.

    When used as a context manager, collectors will be nested. Individual
    collectors can be retrieved by using the instance as a mapping, using the
    collectors' ``NAME`` attribute as key.

    .. note:: Only one collector of each type is allowed. This allows:

        * Getting back the collector instance using a fixed name.
        * Some collectors like :class:`lisa.trace.DmesgCollector` are not
          re-entrant
    """

    _COMPOSITION_ORDER = 100

    def __init__(self, collectors):
        collectors = list(collectors)
        if len(set(map(attrgetter('NAME'), collectors))) != len(collectors):
            raise ValueError('Collectors of the same type cannot be composed together')

        _collectors = {
            c.NAME: c
            for c in collectors
            if hasattr(c, 'NAME')
        }
        self._collectors = _collectors
        self._cm = ComposedContextManager(collectors)

    def __enter__(self):
        self._cm.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self._cm.__exit__(*args, **kwargs)

    def __getitem__(self, key):
        return self._collectors[key]

    def __iter__(self):
        return iter(self._collectors)

    def __len__(self):
        return len(self._collectors)


class FtraceCollector(CollectorBase, Configurable):
    """
    Thin wrapper around :class:`devlib.collector.ftrace.FtraceCollector`.

    .. note:: Events are expected to be provided by the target's kernel, but if
        they are not :class:`lisa._kmod.LISADynamicKmod` will build a
        kernel module to attempt to satisfy the missing events. This will
        typically require correct target setup, see
        :class:`lisa.target.TargetConf` ``kernel/src`` configurations.

    {configurable_params}
    """

    NAME = 'ftrace'
    CONF_CLASS = FtraceConf
    INIT_KWARGS_KEY_MAP = {
        'kmod_auto_load': ['modules', 'auto-load'],
    }
    TOOLS = ['trace-cmd']
    _COMPOSITION_ORDER = 0

    def __init__(self, target, *, events=None, functions=None, buffer_size=10240, output_path=None, autoreport=False, trace_clock=None, saved_cmdlines_nr=8192, tracer=None, kmod_auto_load=True, events_namespaces=('lisa__', None), **kwargs):

        kconfig = target.plat_info['kernel']['config']
        if not kconfig.get('FTRACE'):
            raise ValueError("The target's kernel needs CONFIG_FTRACE=y kconfig enabled")

        tracing_path = devlib.FtraceCollector.find_tracing_path(target)
        target_available_events, avoided = self._target_available_events(target, tracing_path)

        # We always exclude the events expected to come
        # from the module as we need to load the module and query it to
        # know if they are actually available. If the module is already
        # loaded, ftrace will report the event as being available but there
        # is nothing guaranteeing that the module can actually emit it
        # (e.g. that it has all its probe succesfully setup)
        target_available_events = {
            event
           for event in target_available_events
           if not event.startswith('lisa__')
       }

        kmod = target.get_kmod(LISADynamicKmod)
        # Get the events possibly defined in the module. Note that it's a
        # superset of the events actually defined as this is based on pretty
        # crude filtering of the source files, rather than analyzing the events
        # actually defined in the .ko after it has been compiled.
        kmod_available_events = set(kmod.possible_events)
        available_events = target_available_events | kmod_available_events

        if events is None:
            events_checker = EmptyTraceEventChecker()
        elif isinstance(events, TraceEventCheckerBase):
            events_checker = events
        else:
            events_checker = AndTraceEventChecker.from_events(sorted(set(events)))

        def rewrite(checker):
            if isinstance(checker, TraceEventChecker):
                # Turn meta events into their source event
                return OrTraceEventChecker.from_events(
                    Trace.get_event_sources(checker.event)
                )
            else:
                return checker

        def wildcard(checker):
            if isinstance(checker, TraceEventChecker) and checker._is_pattern:
                # Make fnmatch pattern optional
                return OptionalTraceEventChecker.from_events(
                    fnmatch.filter(
                        available_events,
                        checker.event
                    )
                )
            else:
                return checker

        meta_events = {
            event
            for event in events_checker.get_all_events()
            if _Trace._is_meta_event(event)
        }

        events_checker = events_checker.map(rewrite)
        events_checker = events_checker.expand_namespaces(namespaces=events_namespaces)
        # Expand the wildcards after having expanded the namespaces.
        events_checker = events_checker.map(wildcard)
        self.logger.debug(f'Will try to collect events: {events_checker}')

        # Select the events, after having expanded the namespaces
        events = events_checker.check_events(available_events)

        functions = functions or []
        trace_clock = trace_clock or 'global'
        kwargs.update(
            target=target,
            events=sorted(events),
            functions=functions,
            buffer_size=buffer_size,
            autoreport=autoreport,
            trace_clock=trace_clock,
            saved_cmdlines_nr=saved_cmdlines_nr,
            tracer=tracer,
        )
        self.check_init_param(**kwargs)

        if functions and not kconfig.get('FUNCTION_TRACER'):
            raise ValueError(f"The target's kernel needs CONFIG_FUNCTION_TRACER=y kconfig enabled in order to trace functions: {functions}")

        if 'funcgraph_entry' in events or 'funcgraph_exit' in events:
            tracer = 'function_graph' if tracer is None else tracer

        needed_from_kmod = (
            # If we ask for anything that the module provides, we need to
            # load the module in order to check whether the event is indeed
            # provided (i.e. it will be emitted at runtime). If we don't do
            # that and if the module is already loaded, ftrace will show an
            # event as "existing" but the kernel module may not necessarily
            # be able to emit it if e.g. it failed to register a probe
            # somewhere. We can only clarify that by talking to the loaded
            # module.
            {
                event
                for event in events
                if event.startswith('lisa__')
            } |
            # If some events are not already available on that kernel, look
            # them up in custom modules
            (kmod_available_events & events)
        )

        kmod_defined_events = set()
        kmod_cm = None
        if needed_from_kmod:
            # If anything wrong happens, we will be restricted to the events
            # already available.
            events = events & target_available_events

            if kmod_auto_load:
                self.logger.info(f'Building kernel module to try to provide the following events that are not currently available on the target: {", ".join(sorted(needed_from_kmod))}')
                try:
                    kmod_defined_events, provided, kmod_cm = self._get_kmod(
                        target,
                        target_available_events=target_available_events,
                        needed_events=needed_from_kmod,
                    )
                except Exception as e:
                    try:
                        events_checker.check_events(events)
                    except MissingTraceEventError as e:
                        raise MissingTraceEventError(
                            e.missing_events,
                            available_events=target_available_events,
                            msg='Ftrace events are missing in the kernel: {missing_events}{available}. Kernel module build was attempted to provide missing events but failed'
                        ) from e

                    try:
                        events_checker.check_events(events, check_optional=True)
                    except MissingTraceEventError as e:
                        self.logger.error(f'Could not build the kernel module to provide optional events: {e}. Use kmod_auto_load=False to disable automatic module build.')
                else:
                    # Module build went well, we add the events effectively
                    # provided by the module
                    events = events | provided
            else:
                try:
                    events_checker.check_events(events)
                except MissingTraceEventError as e:
                    raise MissingTraceEventError(
                        e.missing_events,
                        available_events=target_available_events,
                        msg='Ftrace events are missing in the kernel: {missing_events}{available}. Use kmod_auto_load=True to enable automatic module build to provide these events.',
                    )

        self._kmod_cm = kmod_cm

        ############################################
        # Final checks after we enabled all we could
        ############################################

        try:
            events_checker.check_events(events)
        except MissingTraceEventError as e:
            raise MissingTraceEventError(
                e.missing_events,
                available_events=target_available_events | kmod_defined_events,
                msg='Ftrace events are missing: {missing_events}{available}',
            )

        try:
            events_checker.check_events(events, check_optional=True)
        except MissingTraceEventError as e:
            e = MissingTraceEventError(
                e.missing_events,
                available_events=target_available_events | kmod_defined_events,
                msg='{missing_events}{available}',
            )
            self.logger.info(f'Optional events missing: {e}')

        if not events:
            raise ValueError('No ftrace events selected')

        self.events = sorted(events | meta_events)

        # Some events are "special" and cannot be disabled or enabled. We
        # therefore cannot pass them to trace-cmd.
        events -= avoided
        # trace-cmd fails if passed no events, which is an issue since we
        # cannot pass e.g. "print" event.
        if not events:
            events |= {'sched_switch'}

        events = sorted(events)

        self._cm = None

        # Install the tools before creating the collector, as devlib will check
        # for it
        self._install_tools(target)

        # Only pass true kernel events to devlib, as it will reject any other.
        kwargs.update(
            events=events,
            tracer=tracer,
        )

        # We need to install the kmod when initializing the devlib object, so
        # that the available events are accurate.
        with self._make_cm(record=False):
            collector = devlib.FtraceCollector(
                # Prevent devlib from pushing its own trace-cmd since we provide
                # our own binary
                no_install=True,
                tracing_path=tracing_path,
                strict=True,
                **kwargs
            )
        super().__init__(collector, output_path=output_path)

    @classmethod
    def _get_kmod(cls, target, target_available_events, needed_events):
        logger = cls.get_logger()
        kmod = target.get_kmod(LISADynamicKmod)
        defined_events = set(kmod._defined_events)
        logger.debug(f'Kernel module defined events: {defined_events}')

        needed = needed_events & defined_events
        if needed:
            overlapping = defined_events & target_available_events
            if overlapping:
                raise ValueError(f'Events defined in {kmod.src.mod_name} ({", ".join(needed)}) are needed but some events overlap with the ones already provided by the kernel: {", ".join(overlapping)}')
            else:
                @contextlib.contextmanager
                def cm():
                    with kmod.run() as _kmod:
                        config = _kmod._event_features_conf(needed)
                        with _kmod._reconfigure(configs=[config]):
                            yield

                return (
                    defined_events,
                    needed,
                    cm,
                )
        else:
            return (defined_events, set(), None)

    @contextlib.contextmanager
    def _make_cm(self, record=True):
        with contextlib.ExitStack() as stack:
            kmod_cm = self._kmod_cm
            if kmod_cm is not None:
                stack.enter_context(kmod_cm())

            if record:
                proxy = super()
                class RecordCM:
                    def __enter__(self):
                        return proxy.__enter__()
                    def __exit__(self, *args, **kwargs):
                        return proxy.__exit__(*args, **kwargs)

                stack.enter_context(RecordCM())

            yield

    def __enter__(self):
        self._cm = self._make_cm()
        return self._cm.__enter__()

    def __exit__(self, *args, **kwargs):
        try:
            x = self._cm.__exit__(*args, **kwargs)
        finally:
            self._cm = None
        return x

    def get_data(self, *args, **kwargs):
        if self._kmod_cm and not self._cm:
            raise ValueError('FtraceCollector.get_data() cannot be called after the kernel module was unloaded.')
        else:
            return super().get_data(*args, **kwargs)

    @staticmethod
    def _target_available_events(target, tracing_path):
        events = target.read_value(target.path.join(tracing_path, 'available_events'))

        # trace-cmd start complains if given these events, even though they are
        # valid
        avoided = set(target.list_directory(target.path.join(tracing_path, 'events', 'ftrace'), as_root=True))

        available = set(
            event.split(':', 1)[1]
            for event in events.splitlines()
        )
        # These events are available, but we still cannot pass them to trace-cmd record
        available.update(avoided)

        return (available, avoided)

    @classmethod
    @kwargs_forwarded_to(__init__)
    def from_conf(cls, target, conf, **kwargs):
        """
        Build an :class:`FtraceCollector` from a :class:`FtraceConf`

        :param target: Target to use when collecting the trace
        :type target: lisa.target.Target

        :param conf: Configuration object
        :type conf: FtraceConf

        :Variable keyword arguments: Forwarded to ``__init__``.
        """
        cls.get_logger().info(f'Ftrace configuration:\n{conf}')
        _kwargs = cls.conf_to_init_kwargs(conf)
        _kwargs['target'] = target
        _kwargs.update(kwargs)
        cls.check_init_param(**_kwargs)
        return cls(**_kwargs)

    @classmethod
    @kwargs_forwarded_to(
        from_conf,
        ignore=['conf']
    )
    def from_user_conf(cls, target, base_conf=None, user_conf=None, merged_src='merged', **kwargs):
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

        :Other keyword arguments: Forwarded to :meth:`from_conf`.
        """
        user_conf = user_conf or FtraceConf()
        base_conf = base_conf or FtraceConf()

        # Make a copy of the conf, since it may be shared by multiple classes
        conf = copy.copy(base_conf)

        # Merge user configuration with the test's configuration
        conf.add_merged_src(
            src=merged_src,
            conf=user_conf,
            optional_events=True,
        )
        return cls.from_conf(target, conf, **kwargs)


class DmesgCollector(CollectorBase):
    """
    Wrapper around :class:`devlib.collector.dmesg.DmesgCollector`.

    It installs the ``dmesg`` tool automatically on the target upon creation,
    so we know what version is being is used.
    """

    NAME = 'dmesg'
    TOOLS = ['dmesg']
    LOG_LEVELS = devlib.DmesgCollector.LOG_LEVELS

    _COMPOSITION_ORDER = 10

    @kwargs_dispatcher(
        {
            CollectorBase: 'init_kwargs',
            devlib.DmesgCollector: 'devlib_kwargs',
        },
        ignore=['collector'],
    )
    def __init__(self, target, init_kwargs, devlib_kwargs):
        collector = devlib.DmesgCollector(**devlib_kwargs)
        super().__init__(collector=collector, **init_kwargs)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
