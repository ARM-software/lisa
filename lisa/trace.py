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
import stat
import json
import inspect
import shlex
import contextlib
import tempfile
from functools import wraps
from collections.abc import Set, Mapping, Sequence
from collections import namedtuple
from operator import itemgetter, attrgetter
from numbers import Number, Integral, Real
import multiprocessing
import textwrap
import subprocess
import itertools
import functools
import fnmatch
import typing
from difflib import get_close_matches
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pyarrow.lib
import pyarrow.parquet

import devlib

from lisa.utils import Loggable, HideExekallID, memoized, lru_memoized, deduplicate, take, deprecate, nullcontext, measure_time, checksum, newtype, groupby, PartialInit, kwargs_forwarded_to, kwargs_dispatcher, ComposedContextManager, get_nested_key, bothmethod, DirCache
from lisa.conf import SimpleMultiSrcConf, LevelKeyDesc, KeyDesc, TopLevelKeyDesc, Configurable
from lisa.datautils import SignalDesc, df_add_delta, df_deduplicate, df_window, df_window_signals, series_convert, df_update_duplicates
from lisa.version import VERSION_TOKEN
from lisa._typeclass import FromString
from lisa._kmod import LISAFtraceDynamicKmod
from lisa._assets import get_bin


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
        # pylint: disable=unused-argument
        super().__init__()
        # This happens when the number of saved PID/comms entries in the trace
        # is too low
        if self.comm == '<...>':
            raise ValueError('Invalid comm name "<...>", please increase saved_cmdlines_nr value on FtraceCollector')

    def __str__(self):
        if self.pid is not None and self.comm is not None:
            out = f'{self.pid}:{self.comm}'
        else:
            out = str(self.comm if self.comm is not None else self.pid)

        return f'[{out}]'

    _STR_PARSE_REGEX = re.compile(r'\[?([0-9]+):([a-zA-Z0-9_-]+)\]?')


class _TaskIDFromStringInstance(FromString, types=TaskID):
    """
    Instance of :class:`lisa._typeclass.FromString` for :class:`TaskID` type.
    """
    @classmethod
    def from_str(cls, string):
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
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


class _TaskIDSeqFromStringInstance(FromString, types=(typing.List[TaskID], typing.Sequence[TaskID])):
    """
    Instance of :class:`lisa._typeclass.FromString` for lists :class:`TaskID` type.
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


class TraceParserBase(abc.ABC, Loggable, PartialInit):
    """
    Abstract Base Class for trace parsers.

    :param events: Iterable of events to parse. An empty iterable can be
        passed, in which case some metadata may still be available.
    :param events: collections.abc.Iterable(str)

    :param needed_metadata: Set of metadata name to gather in the parser.
    :type needed_metadata: collections.abc.Iterable(str)
    """

    METADATA_KEYS = [
        'time-range',
        'symbols-address',
        'cpus-count',
        'available-events',
    ]
    """
    Possible metadata keys
    """

    def __init__(self, events, needed_metadata=None):
        # pylint: disable=unused-argument
        self._needed_metadata = set(needed_metadata or [])

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
              that were requested.
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
            * Columns prefixed with ``__``: Header of each event, usually containing the following fields:

                * ``__cpu``: CPU number the event was emitted from
                * ``__pid``: PID of the current process scheduled at the time the event was emitted
                * ``__comm``: Task command name going with ``__pid`` at the point the event was emitted

        :param event: name of the event to parse
        :type event: str

        :raises MissingTraceEventError: If the event cannot be parsed.

        .. note:: The caller is free to modify the index of the data, and it
            must not affect other dataframes.
        """

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
        self.dfs = dfs
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
            return self.dfs[event].copy(deep=True)
        except KeyError as e:
            raise MissingTraceEventError(
                [event],
                available_events=self._available_events
            ) from e

    def get_metadata(self, key):
        if key == 'time-range':
            if self._time_range:
                return self._time_range
            elif self.dfs:
                indices = [
                    df.index
                    for df in self.dfs.values()
                    if not df.empty
                ]

                return (
                    float(min(map(itemgetter(0), indices))),
                    float(max(map(itemgetter(-1), indices))),
                )
            else:
                return (0, 0)

        elif key == 'available-events':
            return sorted(self._available_events)
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
    def __init__(self, path=None, events=None, trace_processor_path='https://get.perfetto.dev/trace_processor', **kwargs):

        if urllib.parse.urlparse(trace_processor_path).scheme:
            bin_path = self._download_trace_processor(url=trace_processor_path)
        else:
            bin_path = trace_processor_path

        self._bin_path = bin_path
        tp = self._make_tp(path)

        time_range, = tp.query("SELECT MIN(ts), MAX(ts) FROM raw")
        time_range = (
            time_range.__dict__['MIN(ts)'] / 1e9,
            time_range.__dict__['MAX(ts)'] / 1e9,
        )
        available_events = set(
            row.name
            for row in tp.query("SELECT DISTINCT name FROM raw")
        )

        self._metadata = {
            'available-events': available_events,
            'time-range': time_range,
        }

        self._tp = tp
        super().__init__(events=events, **kwargs)

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
        )

        cache_path = dir_cache.get_entry([url])
        return cache_path / 'trace_processor'

    def _make_tp(self, path):
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

        config = TraceProcessorConfig(
            # Without that, perfetto will disallow querying for most events in
            # the raw table
            ingest_ftrace_in_raw=True,
            bin_path=self._bin_path,
        )
        tp = TraceProcessor(trace=path, config=config)
        return tp

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
        arg_cols = [
            f"(SELECT display_value FROM args WHERE key = '{key}' AND args.arg_set_id = raw.arg_set_id)"
            for (key, typ) in arg_fields
        ]
        common_cols = ['ts as Time', 'cpu as __cpu', 'utid as __pid', '(SELECT name from thread where thread.utid = raw.utid) as __comm']

        extract = ', '.join(common_cols + arg_cols)
        query = f"SELECT {extract} FROM raw WHERE name = '{event}'"
        df = pd.DataFrame.from_records(
            row.__dict__
            for row in self._query(query)
        )

        dtype_map = {
            'uint': 'uint64',
            'int': 'int64',
            'string': 'category',
        }

        df = df.rename(columns={
            query: name
            for (name, _), query in zip(arg_fields, arg_cols)
        })

        df = df.astype(
            {
                **{
                    '__pid': 'uint32',
                    '__cpu': 'uint32',
                    '__comm': 'category',
                    'Time': 'uint64',
                },
                **{
                    field: dtype_map[typ]
                    for (field, typ) in arg_fields
                    if typ in dtype_map
                },
            }
        )
        df['Time'] /= 1e9

        # Attempt to avoid getting duplicated index issues. We may still end up
        # with the same timestamp for different events accross 2 dataframes,
        # which could create an issue if someone somewhat joins them, and it
        # can happen in practice but there isn't really any good solution here.
        # Fixing that would require adding a new fixedup column to the DB.
        df_update_duplicates(df, col='Time', inplace=True)
        df = df.set_index('Time')

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
                    return r'{field}=(?P<{field}>.+?)(?:{blank}(?={identifier}=)|$)'.format(
                        field=re.escape(field),
                        **cls.PARSER_REGEX_TERMINALS
                    )

            fields_regexes = list(map(make_regex, fields))

            # Catch-all field that will consume any unknown field, allowing for
            # partial parsing (both for performance/memory consumption and
            # forward compatibility)
            fields_regexes.append(r'{identifier}=.*?(?=(?:{other_fields})=)'.format(
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

        regex = r'^.*:?{blank}{__comm}-{__pid}{blank}\[{__cpu}\]{blank}{__timestamp}:{blank}{__event}:'.format(**compos, blank=blank)
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

    def __init__(self,
        lines,
        events=None,
        needed_metadata=None,
        event_parsers=None,
        default_event_parser_cls=None,
        pre_filled_metadata=None,
    ):
        super().__init__(events, needed_metadata=needed_metadata)
        self._pre_filled_metadata = pre_filled_metadata or {}
        events = set(events or [])

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
            del self._skeleton_df['__fields']

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
    def from_string(cls, txt, **kwargs):
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

        index = '__timestamp' if data and '__timestamp' in columns else None
        return pd.DataFrame.from_records(
            data,
            columns=columns,
            index=index,
        )

    def _eagerly_parse_lines(self, lines, skeleton_regex, event_parsers, events, time=None):
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
        time_type = getattr(np, self.HEADER_FIELDS['__timestamp'])

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
        nextafter = np.nextafter
        inf = math.inf
        line_time = 0
        parse_time = '__timestamp' in skeleton_regex.groupindex.keys()

        for line in lines:
            prev_time = line_time
            if time_is_provided:
                line_time, line = line

            match = skel_search(line)
            # Stop at the first non-matching line
            try:
                event = group(match, '__event')
                line_time = time_type(group(match, '__timestamp'))
            # Assume only "time" is not in the regex. Keep that out of the hot
            # path since it's only needed in rare cases (like nesting parsers)
            except IndexError:
                # If we are supposed to parse time, let's re-raise the
                # exception
                if parse_time:
                    raise
                else:
                    # Otherwise, make sure "event" is defined so that we only
                    # go a match failure on "time"
                    event # pylint: disable=pointless-statement
            # The line did not match the skeleton regex, so skip it
            except TypeError:
                if b'EVENTS DROPPED' in line:
                    raise DroppedTraceEventError('The trace buffer got overridden by new data, increase the buffer size to ensure all events are recorded')
                # Unknown line, could be coming e.g. from stderr
                else:
                    continue

            # Do a global deduplication of timestamps, across all
            # events regardless of the one we will parse. This ensures
            # stable results and joinable dataframes from multiple
            # parser instance.
            if line_time <= prev_time:
                line_time = nextafter(prev_time, inf)

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

        end_time = line_time
        available_events.update(
            event
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
                df = self._postprocess_df(decoded_event, parser, df)
                events_df[decoded_event] = df

        # Compute the skeleton dataframe for the events that have not been
        # parsed already. It contains the event name, the time, and potentially
        # the fields if they are needed
        skeleton_df = self._make_df_from_data(skeleton_regex, skeleton_data, ['__timestamp', 'line'])
        # Drop unnecessary columns that might have been parsed by the regex
        to_keep = {'__event', '__fields', 'line'}
        skeleton_df = skeleton_df[sorted(to_keep & set(skeleton_df.columns))]
        # Make the event column more compact
        skeleton_df['__event'] = skeleton_df['__event'].astype('category', copy=False)
        # This is very fast on a category dtype
        available_events.update(skeleton_df['__event'].unique())

        available_events = {event.decode('ascii') for event in available_events}
        return (events_df, skeleton_df, (begin_time, end_time), available_events)

    def _lazyily_parse_event(self, event, parser, df):
        # Only parse the lines that have a chance to match
        df = df[df['__event'] == event.encode('ascii')]
        index = df.index
        regex = parser.bytes_regex

        # Resolve names outside of the comprehension
        search = regex.search
        groups = self._RE_MATCH_CLS.groups
        # Parse the lines with the regex.
        # note: we cannot use Series.str.extract(expand=True) since it does
        # not work on bytes
        data = [
            groups(search(line))
            for line in df['line']
        ]
        df = self._make_df_from_data(regex, data)
        df.index = index
        df = self._postprocess_df(event, parser, df)
        return df

    @staticmethod
    def _get_event_descs(df, events, event_parsers):
        user_supplied = event_parsers.keys()
        all_events = events is None

        if not all_events and set(events) == user_supplied:
            return {}
        else:
            def encode(string):
                return string.encode('ascii')

            user_supplied = set(map(encode, user_supplied))
            events = set(map(encode, events))

            # Since we make a shallow copy:
            # DO NOT MODIFY ANY EXISTING COLUMN
            df = df.copy(deep=False)

            # Find the field names only for the events we don't already know about,
            # since the inference is relatively expensive
            if all_events:
                df = df[~df['__event'].isin(user_supplied)]
            else:
                events = set(events) - user_supplied
                df = df[df['__event'].isin(events)]

            def apply_regex(series, func, regex):
                regex = encode(regex.format(**TxtEventParser.PARSER_REGEX_TERMINALS))
                regex = re.compile(regex, flags=re.ASCII)
                return [
                    func(regex, x)
                    for x in series
                ]

            def is_match(pat, x):
                return bool(pat.search(x))

            # We cannot use Series.str.(count|finall) since they don't work
            # with bytes
            df['split_fields'] = apply_regex(df['__fields'], re.findall, r'({identifier})=')
            # Look for any value before the first named field
            df['pos_fields'] = apply_regex(df['__fields'], is_match, r'(?:^ *(?:[^ =]+ [^ =]+)+=)|(?:^[^=]*$)')

            def infer_fields(x):
                field_alternatives = x.transform(tuple).unique()
                if len(field_alternatives) == 1:
                    fields = field_alternatives[0]
                else:
                    # Use the union of all the fields, and the order of appearance is not guaranteed
                    fields = sorted(set(itertools.chain.from_iterable(field_alternatives)))

                return fields

            # For each event we don't already know, get the list of all fields available
            inferred = df.groupby('__event', observed=True, group_keys=False)['split_fields'].apply(infer_fields).to_frame()
            inferred['positional_field'] = df.groupby('__event', observed=True, group_keys=False)['pos_fields'].any()

            def update_desc(desc):
                new = dict(
                    positional_field='__positional' if desc['positional_field'] else None,
                    fields={
                        field.decode('ascii'): None
                        for field in desc['split_fields']
                    },
                )
                return new

            dct = {
                event.decode('ascii'): update_desc(desc)
                for event, desc in inferred.T.to_dict().items()
            }
            return dct

    def parse_event(self, event):
        try:
            parser = self._event_parsers[event]
        except KeyError as e:
            raise MissingTraceEventError([event]) from e

        # Maybe it was eagerly parsed
        try:
            df = self._events_df[event]
        except KeyError:
            df = self._lazyily_parse_event(event, parser, self._skeleton_df)

        # Since there is no way to distinguish between no event entry and
        # non-collected events in text traces, map empty dataframe to missing
        # event
        if df.empty:
            raise MissingTraceEventError([event])
        else:
            return df

    @classmethod
    def _postprocess_df(cls, event, parser, df):
        """
        ALL THE PROCESSING MUST HAPPEN INPLACE on the dataframe
        """
        # pylint: disable=unused-argument

        # Convert fields from extracted strings to appropriate dtype
        all_fields = {
            **parser.fields,
            **cls.HEADER_FIELDS,
        }

        def default_converter(x):
            first_success = None

            for dtype in cls.DTYPE_INFERENCE_ORDER:
                convert = make_converter(dtype)
                with contextlib.suppress(ValueError, TypeError):
                    converted = convert(x)
                    # If we got the dtype we wanted, use it immediately.
                    # Otherwise, record the first conversion (i.e. the most
                    # specific) that did no completely fail so we can reuse it
                    # instead of "string"
                    if converted.dtype == dtype:
                        return converted
                    elif first_success is None:
                        first_success = converted

            # If we got no perfect conversion, return the most specific one
            # that gave a result, otherwise bailout to just strings
            if first_success is None:
                try:
                    return make_converter('string')(x)
                except (ValueError, TypeError):
                    return x
            else:
                return first_success

        def make_converter(dtype):
            # If the dtype is already known, just use that
            if dtype:
                return lambda x: series_convert(x, dtype)
            else:
                # Otherwise, infer it from the data we have
                return default_converter

        converters = {
            field: make_converter(dtype)
            for field, dtype in all_fields.items()
            if field in df.columns
        }
        # DataFrame.apply() can lead to recursion error when a conversion
        # fails, so use an explicit loop instead
        for col in set(df.columns) & converters.keys():
            df[col] = converters[col](df[col])
        return df

    def get_metadata(self, key):
        if key == 'time-range':
            return self._time_range

        # If we filtered some events, we are not exhaustive anymore so we
        # cannot return the list
        if key == 'available-events':
            return self._available_events

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

    :param merged_df: Dataframe to merge into the parsed ones, to add
        pre-computed fields.
    :type merged_df: pandas.DataFrame

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
    def __init__(self, *args, time, merged_df=None, **kwargs):
        self._time = time
        self._merged_df = merged_df
        super().__init__(*args, **kwargs)

    def _eagerly_parse_lines(self, *args, **kwargs):
        # Use the iloc as "time", and we fix it up manually afterwards
        return super()._eagerly_parse_lines(
            *args, **kwargs, time=self._time,
        )

    def _postprocess_df(self, event, parser, df):
        df = super()._postprocess_df(event, parser, df)
        merged_df = self._merged_df
        if merged_df is not None:
            df = df.merge(merged_df, left_index=True, right_index=True, copy=False)
        return df


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


class TrappyTraceParser(TraceParserBase):
    """
    Glue with :mod:`trappy` trace parsers.

    .. note:: This class is deprecated and the use of :class:`TxtTraceParser`
        is recommended instead. This class is only provided in case
        compatibility with :mod:`trappy` is needed for one reason or another.
    """

    @kwargs_forwarded_to(TraceParserBase.__init__)
    def __init__(self, path, events, trace_format=None, **kwargs):
        super().__init__(events=events, **kwargs)

        # Lazy import so that it's not a required dependency
        import trappy # pylint: disable=import-outside-toplevel,import-error

        events = set(events)
        # Make sure we won't attempt parsing 'print' events, so that this
        # parser is not used to resolve meta events.
        events -= {'print'}
        self._events = events

        events = set(events)
        if trace_format is None:
            if path.endswith('html'):
                trace_format = 'SySTrace'
            else:
                trace_format = 'FTrace'

        self.logger.debug(f'Parsing {trace_format} events from {path}: {sorted(events)}')
        if trace_format == 'SysTrace':
            trace_class = trappy.SysTrace
        elif trace_format == 'FTrace':
            trace_class = trappy.FTrace
        else:
            raise ValueError(f'Unknown trace format: {trace_format}')

        # Since we handle the cache in lisa.trace.Trace, we do not need to duplicate it
        trace_class.disable_cache = True
        trace = trace_class(
            path,
            scope="custom",
            events=sorted(events),
            normalize_time=False,
        )
        self._trace = trace

        # trappy sometimes decides to be "clever" and overrules the path to be
        # used, even though it was specifically asked for a given file path
        assert path == trace.trace_path

    def get_metadata(self, key):
        if key == 'time-range':
            trace = self._trace
            return (trace.basetime, trace.endtime)
        else:
            return super().get_metadata(key)

    def parse_event(self, event):
        trace = self._trace

        if event not in self._events:
            raise MissingTraceEventError([event])

        df = getattr(trace, event).data_frame
        if df.empty:
            raise MissingTraceEventError([event])

        return df


class TraceBase(abc.ABC):
    """
    Base class for common functionalities between :class:`Trace` and :class:`TraceView`
    """

    def __init__(self):
        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base

        # pylint: disable=import-outside-toplevel
        from lisa.analysis._proxy import AnalysisProxy, _DeprecatedAnalysisProxy
        # self.analysis is deprecated so we can transition to using holoviews
        # in all situations, even when the backend is matplotlib

        # In the user-visible analysis, we want to change some defaults that
        # will improve the immediate experience, at the expense of good
        # composition. For example, using ui=None means that a user calling a
        # plot method twice will get 2 toolbars. but it can still be disabled
        # manually. Since composition can sometimes suffer, the internal
        # analysis proxy and the default values on plot methods are set to less
        # friendly but more predictable defaults.
        params = dict(
            # Default to displaying a toolbar in notebooks
            output=None,
        )
        self.analysis = _DeprecatedAnalysisProxy(self, params=params)
        self.ana = AnalysisProxy(self, params=params)

    @property
    def trace_state(self):
        """
        State of the trace object that might impact the output of dataframe
        getter functions like :meth:`Trace.df_event`.

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

    def __getitem__(self, window):
        if not isinstance(window, slice):
            raise TypeError("Cropping window must be an instance of slice")

        if window.step is not None:
            raise ValueError("Slice step is not supported")

        return self.get_view((window.start, window.stop))

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

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='2.0',
        removed_in='4.0',
        replaced_by='df_event',
    )
    def df_events(self, *args, **kwargs):
        return self.df_event(*args, **kwargs)

    @deprecate('This method has been deprecated and is an alias for "trace.ana.notebook.df_all_events()"',
        deprecated_in='2.0',
        removed_in='4.0',
        replaced_by='lisa.analysis.notebook.NotebookAnalysis.df_all_event',
    )
    def df_all_events(self, *args, **kwargs):
        return self.ana.notebook.df_all_events(*args, **kwargs)


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

    :Attributes:
        * ``base_trace``: The original :class`:`Trace` this view is based on.
        * ``ana``: The analysis proxy on the trimmed down :class`:`Trace`.
        * ``start``: The timestamp of the first trace event in the view (>=
          ``window[0]``)
        * ``end``: The timestamp of the last trace event in the view (<=
          ``window[1]``)

    You can substitute an instance of :class:`Trace` with an instance of
    :class:`TraceView`. This means you can create a view of a trimmed down trace
    and run analysis code/plots that will only use data within that window, e.g.::

      trace = Trace(...)
      view = trace.get_view((2, 4))

      # Alias for the above
      view = trace[2:4]

      # This will only use events in the (2, 4) time window
      df = view.ana.tasks.df_tasks_runtime()

    **Design notes:**

      * :meth:`df_event` uses the underlying :meth:`lisa.trace.Trace.df_event`
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

    def df_event(self, event, **kwargs):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the sliced trace.

        :param event: Trace event name
        :type event: str

        :Variable keyword arguments: Forwarded to
            :meth:`lisa.trace.Trace.df_event`.
        """
        try:
            window = kwargs['window']
        except KeyError:
            window = self.window
        kwargs['window'] = window

        return self.base_trace.df_event(event, **kwargs)

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
        return self.contains(event)

    def contains(self, event, namespaces=None):
        return any(map(self._contains, self._trace._expand_namespaces(event, namespaces)))

    def _contains(self, event):
        trace = self._trace

        if trace._strict_events and not trace._is_meta_event(event):
            return trace._parseable_events.setdefault(event, False)

        # Try to parse the event in case it was not parsed already
        if event not in trace._parseable_events:
            # If the trace file is not accessible anymore, we will get an OSError
            with contextlib.suppress(MissingTraceEventError, OSError):
                trace.df_event(event=event, raw=True, namespaces=[])

        return trace._parseable_events.setdefault(event, False)

    @property
    def _available_events(self):
        parseable = lambda: self._trace._parseable_events

        # Get the available events, which will populate the parseable events if
        # they were empty, which happens when a trace has just been created
        if not parseable():
            try:
                self._trace.get_metadata('available-events')
            except MissingMetadataError:
                pass

        return {
            event
            for event, available in parseable().items()
            if available
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
    :class:`pandas.DataFrame` or :class:`pandas.Series`. It is used to manage
    the cache and swap.

    It implements the :class:`collections.abc.Mapping` interface, so
    specification keys can be accessed directly like from a dict.

    .. note:: Once introduced in a container, instances must not be modified,
        directly or indirectly.

    :Attributes:
        * ``normal_form``: Normal form of the descriptor. Equality is
          implemented by comparing this attribute.
    """

    def __init__(self, spec, fmt):
        self.fmt = fmt
        self.spec = spec
        self.normal_form = _CacheDataDescNF.from_spec(self.spec, fmt)

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
    def data_filename(self):
        """
        Filename of the data file in the swap.
        """
        return f'{self.name}.{self.cache_desc_nf._fmt}'

    def to_json_map(self):
        """
        Return a mapping suitable for JSON serialization.
        """
        return {
            'version-token': VERSION_TOKEN,
            'name': self.name,
            'desc': self.cache_desc_nf.to_json_map(),
        }

    @classmethod
    def from_json_map(cls, mapping, written=False):
        """
        Create an instance with a mapping created using :meth:`to_json_map`.
        """
        if mapping['version-token'] != VERSION_TOKEN:
            raise TraceCacheSwapVersionError('Version token differ')

        cache_desc_nf = _CacheDataDescNF.from_json_map(mapping['desc'])
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


class TraceCacheSwapVersionError(ValueError):
    """
    Exception raised when the swap entry was created by another version of LISA
    than the one loading it.
    """


class TraceCache(Loggable):
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
    :type swap_content: dict(_CacheDataDescNF, _CacheDataSwapEntry) or None

    The cache manages both the :class:`pandas.DataFrame` and
    :class:`pandas.Series` generated in memory and a swap area used to evict
    them, and to reload them quickly. Some other data (typically JSON) can also
    be stored in the cache by analysis method.
    """

    INIT_SWAP_COST = 1e-8
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

    def __init__(self, max_mem_size=None, trace_path=None, trace_md5=None, swap_dir=None, max_swap_size=None, swap_content=None, metadata=None):
        self._cache = {}
        self._data_cost = {}
        self._swap_content = swap_content or {}
        self._cache_desc_swap_filename = {}
        self.swap_cost = self.INIT_SWAP_COST
        self.swap_dir = swap_dir
        self.max_swap_size = max_swap_size if max_swap_size is not None else math.inf
        self._swap_size = self._get_swap_size()

        self.max_mem_size = max_mem_size if max_mem_size is not None else math.inf
        self._data_mem_swap_ratio = 1
        self._metadata = metadata or {}

        self.trace_path = os.path.abspath(trace_path) if trace_path else trace_path
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
            self._write_data('parquet', df, buffer)
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
        if isinstance(data, (pd.DataFrame, pd.Series)):
            file_overhead, col_overhead = self._swap_size_overhead
            # DataFrame
            try:
                nr_columns = data.shape[1]
            # Series
            except IndexError:
                nr_columns = 1

            size = size - file_overhead - nr_columns * col_overhead
            return size
        else:
            return self._data_mem_usage(data)

    @property
    def trace_md5(self):
        md5 = self._trace_md5
        trace_path = self.trace_path
        if md5 is None and trace_path:
            with open(trace_path, 'rb') as f:
                md5 = checksum(f, 'md5')
            self._trace_md5 = md5

        return md5

    def update_metadata(self, metadata):
        """
        Update the metadata mapping with the given ``metadata`` mapping and
        write it back to the swap area.
        """
        if metadata:
            self._metadata.update(metadata)
            self.to_swap_dir()

    def get_metadata(self, key):
        """
        Get the value of the given metadata ``key``.
        """
        try:
            return self._metadata[key]
        except KeyError as e:
            raise MissingMetadataError(key) from e

    def to_json_map(self):
        """
        Returns a dictionary suitable for JSON serialization.
        """
        trace_path = self.trace_path

        if trace_path:
            if self.swap_dir:
                trace_path = os.path.relpath(trace_path, self.swap_dir)
            else:
                trace_path = os.path.abspath(trace_path)

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
        swap_trace_path = os.path.join(swap_dir, swap_trace_path) if swap_trace_path else None

        metadata = metadata or {}

        if swap_trace_path:
            try:
                with open(swap_trace_path, 'rb') as f:
                    new_md5 = checksum(f, 'md5')
            except FileNotFoundError:
                new_md5 = None
        else:
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
        # If we don't have any cost info, assume it is expensive to compute
        compute_cost = self._data_cost.get(cache_desc, math.inf)
        swap_cost = self._estimate_data_swap_cost(data)
        return swap_cost <= compute_cost

    def _path_of_swap_entry(self, swap_entry):
        return os.path.join(self.swap_dir, swap_entry.meta_filename)

    def _cache_desc_swap_path(self, cache_desc, create=False):
        if self.swap_dir:
            cache_desc_nf = cache_desc.normal_form

            if create and not self._is_written_to_swap(cache_desc):
                self.insert(
                    cache_desc,
                    data=None,
                    compute_cost=None,
                    write_swap=True,
                    force_write_swap=True,
                    # We do not write the swap_entry meta file, so that the
                    # user can write the data file before the swap entry is
                    # added. This way, another process will not be tricked into
                    # believing the data is available whereas in fact it's in
                    # the process of being populated.
                    write_meta=False,
                )

            swap_entry = self._swap_content[cache_desc_nf]
            filename = swap_entry.data_filename
            return os.path.join(self.swap_dir, filename)
        else:
            raise ValueError('Swap dir is not setup')

    def _update_swap_cost(self, data, swap_cost, mem_usage, swap_size):
        unbiased_swap_size = self._unbias_swap_size(data, swap_size)
        # Take out from the swap cost the time it took to write the overhead
        # that comes with the file format, assuming the cost is
        # proportional to amount of data written in the swap.
        if swap_size:
            swap_cost *= unbiased_swap_size / swap_size
        else:
            swap_cost = 0

        new_cost = swap_cost / mem_usage

        override = self.swap_cost == self.INIT_SWAP_COST
        # EWMA to keep a relatively stable cost
        self._update_ewma('swap_cost', new_cost, override=override)

    def _is_written_to_swap(self, cache_desc):
        try:
            swap_entry = self._swap_content[cache_desc.normal_form]
        except KeyError:
            return False
        else:
            return swap_entry.written

    @staticmethod
    def _data_to_parquet(data, path, **kwargs):
        """
        Equivalent to `df.to_parquet(...)` but workaround until pandas can save
        attrs to parquet on its own: ENH request on pandas:
        https://github.com/pandas-dev/pandas/issues/20521

        Workaround:
        https://github.com/pandas-dev/pandas/pull/20534#issuecomment-453236538
        """
        if isinstance(data, pd.DataFrame):
            # Data must be convertible to bytes so we dump them as JSON
            attrs = json.dumps(data.attrs)
            table = pyarrow.Table.from_pandas(data)
            updated_metadata = dict(
                table.schema.metadata or {},
                lisa=attrs,
            )
            table = table.replace_schema_metadata(updated_metadata)
            pyarrow.parquet.write_table(table, path, **kwargs)
        else:
            data.to_parquet(path, **kwargs)

    @staticmethod
    def _data_from_parquet(path):
        """
        Equivalent to `pd.read_parquet(...)` but also load the metadata back
        into dataframes's attrs
        """
        data = pd.read_parquet(path)

        # Load back LISA metadata into "df.attrs", as they were written in
        # _data_to_parquet()
        if isinstance(data, pd.DataFrame):
            schema = pyarrow.parquet.read_schema(path)
            attrs = schema.metadata.get(b'lisa', '{}')
            data.attrs = json.loads(attrs)

        return data

    @classmethod
    def _write_data(cls, fmt, data, path):
        if fmt == 'disk-only':
            return
        elif fmt == 'parquet':
            # Snappy compression seems very fast
            cls._data_to_parquet(data, path, compression='snappy')
        elif fmt == 'json':
            with open(path, 'wt') as f:
                try:
                    json.dump(data, f, separators=(',', ':'))
                except Exception as e:
                    raise ValueError(f'Does not know how to write data type {data.__class__} to the cache: {e}') from e
        else:
            raise ValueError(f'Does not know how to dump to disk format: {fmt}')


    @classmethod
    def _load_data(cls, fmt, path):
        if fmt == 'disk-only':
            data = None
        elif fmt == 'parquet':
            data = cls._data_from_parquet(path)
        elif fmt == 'json':
            with open(path, 'rt') as f:
                data = json.load(f)
        else:
            raise ValueError(f'File format not supported "{fmt}" at path: {path}')

        return data

    def _write_swap(self, cache_desc, data, write_meta=True):
        if not self.swap_dir:
            return
        else:
            # TODO: this is broken for disk-only format, as we have the swap
            # entry in _swap_content[] in order to match it again but the meta
            # file has not been written to the disk yet.
            if self._is_written_to_swap(cache_desc):
                return

            cache_desc_nf = cache_desc.normal_form
            # We may already have a swap entry if we used the None data
            # placeholder. This would have allowed the user to reserve the swap
            # data file in advance so they can write to it directly, instead of
            # managing the data in the memory cache.
            try:
                swap_entry = self._swap_content[cache_desc_nf]
            except KeyError:
                swap_entry = _CacheDataSwapEntry(cache_desc_nf)

            data_path = os.path.join(self.swap_dir, swap_entry.data_filename)

            # If that would make the swap dir too large, try to do some cleanup
            if self._estimate_data_swap_size(data) + self._swap_size > self.max_swap_size:
                self.scrub_swap()

            def log_error(e):
                self.logger.error(f'Could not write {cache_desc} to swap: {e}')

            # Write the Parquet file and update the write speed
            try:
                with measure_time() as measure:
                    self._write_data(cache_desc.fmt, data, data_path)
            # PyArrow fails to save dataframes containing integers > 64bits
            except OverflowError as e:
                log_error(e)
            else:
                # Update the swap entry on disk
                if write_meta:
                    swap_entry.to_path(
                        self._path_of_swap_entry(swap_entry)
                    )
                    swap_entry.written = True
                self._swap_content[swap_entry.cache_desc_nf] = swap_entry

                # Assume that reading from the swap will take as much time as
                # writing to it. We cannot do better anyway, but that should
                # mostly bias to keeping things in memory if possible.
                swap_cost = measure.exclusive_delta
                try:
                    data_swapped_size = os.stat(data_path).st_size
                except FileNotFoundError:
                    data_swapped_size = 0

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
            return 1

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
                stats.pop(filename, None)
                path = os.path.join(self.swap_dir, filename)
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
                del self._swap_content[swap_entry.cache_desc_nf]
                stats.pop(swap_entry.data_filename, None)

                for filename in (swap_entry.meta_filename, swap_entry.data_filename):
                    path = os.path.join(self.swap_dir, filename)
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

            self._swap_size = sum(
                stats[swap_entry.data_filename].st_size
                for swap_entry in self._swap_content.values()
                if swap_entry.data_filename in stats
            )

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
            return self._cache[cache_desc]
        except KeyError as e:
            # pylint: disable=raise-missing-from
            try:
                path = self._cache_desc_swap_path(cache_desc)
            # If there is no swap, bail out
            except (ValueError, KeyError):
                raise KeyError(f'Could not find swap entry for: {cache_desc}')
            else:
                data = self._load_data(cache_desc.fmt, path)
                if insert:
                    # We have no idea of the cost of something coming from
                    # the cache
                    self.insert(cache_desc, data, write_swap=False, compute_cost=None)

                return data

    def insert(self, cache_desc, data, compute_cost=None, write_swap=False, force_write_swap=False, write_meta=True):
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
            written.
        :type write_swap: bool

        :param force_write_swap: If ``True``, bypass the computation vs swap
            cost comparison.
        :type force_write_swap: bool

        :param write_meta: If ``True``, the swap entry metadata will be written
            on disk if the data are. Otherwise, no swap entry is written to disk.
        :type write_meta: bool
        """
        self._cache[cache_desc] = data
        if compute_cost is not None:
            self._data_cost[cache_desc] = compute_cost

        if write_swap:
            self.write_swap(
                cache_desc,
                force=force_write_swap,
                write_meta=write_meta
            )

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
                cache_desc: sys.getrefcount(data)
                for cache_desc, data in self._cache.items()
            }
            min_refcount = min(refcounts.values())

            # Low retention score means it's more likely to be evicted
            def retention_score(cache_desc_and_data):
                cache_desc, data = cache_desc_and_data

                # If we don't know the computation cost, assume it can be evicted cheaply
                compute_cost = self._data_cost.get(cache_desc, 0)

                if not compute_cost:
                    score = 0
                else:
                    swap_cost = self._estimate_data_swap_cost(data)
                    # If it's already written back, make it cheaper to evict since
                    # the eviction itself is going to be cheap
                    if self._is_written_to_swap(cache_desc):
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
                return (refcounts[cache_desc] - min_refcount + 1) * score

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
        self.write_swap(cache_desc)

        try:
            del self._cache[cache_desc]
        except KeyError:
            pass

    def write_swap(self, cache_desc, force=False, write_meta=True):
        """
        Write the given descriptor to the swap area if that would be faster to
        reload the data rather than recomputing it. If the descriptor is not in
        the cache or if there is no swap area, ignore it.

        :param cache_desc: Descriptor of the data to write to swap.
        :type cache_desc: _CacheDataDesc

        :param force: If ``True``, bypass the compute vs swap cost comparison.
        :type force: bool

        :param write_meta: If ``True``, the swap entry metadata will be written
            on disk if the data are. Otherwise, no swap entry is written to disk.
        :type write_meta: bool
        """
        try:
            data = self._cache[cache_desc]
        except KeyError:
            pass
        else:
            if force or self._should_evict_to_swap(cache_desc, data):
                self._write_swap(cache_desc, data, write_meta)

    def write_swap_all(self):
        """
        Attempt to write all cached data to the swap.
        """
        for cache_desc in self._cache.keys():
            self.write_swap(cache_desc)

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


class Trace(Loggable, TraceBase):
    """
    The Trace object is the LISA trace events parser.

    :param trace_path: File containing the trace
    :type trace_path: str or None

    :param events: events to be parsed. Since events can be loaded on-demand,
        that is optional but still recommended to improve trace parsing speed.

        .. seealso:: :meth:`df_event` for event formats accepted.
    :type events: TraceEventCheckerBase or list(str) or None

    :param events_namespaces: List of namespaces of the requested events. Each
        namespace will be tried in order until the event is found. The
        ``None`` namespace can be used to specify no namespace. An empty
        list is treated as ``[None]`` The full event name is formed with
        ``<namespace>__<event>``.
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

    :param sanitization_functions: Mapping of event name to sanitization
        function. Each function takes:

            * the trace instance
            * the name of the event
            * a dataframe of the raw event
            * a dictionary of aspects to sanitize

        These functions *must not* modify their input dataframe under any
        circumstances. They are required to make copies where appropriate.
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

    :param write_swap: Default value used for :meth:`df_event` ``write_swap``
        parameter.
    :type write_swap: bool

    :Attributes:
        * ``start``: The timestamp of the first trace event in the trace
        * ``end``: The timestamp of the last trace event in the trace
        * ``time_range``: Maximum timespan for all collected events
        * ``window``: Conveniency tuple of ``(start, end)``.
        * ``available_events``: Events available in the parsed trace, exposed
          as some kind of set-ish smart container. Querying for event might
          trigger the parsing of it.
        * ``ana``: The analysis proxy used as an entry point to run analysis
          methods on the trace. See :class:`lisa.analysis._proxy.AnalysisProxy`.

    :Supporting more events:
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

    def _select_userspace(self, source_event, meta_event, df):
        # pylint: disable=unused-argument,no-self-use

        # tracing_mark_write is the name of the kernel function invoked when
        # writing to: /sys/kernel/debug/tracing/trace_marker
        # That said, it's not the end of the world if we don't filter on that
        # as the meta event name is supposed to be unique anyway
        if not is_numeric_dtype(df['ip'].dtype):
            df = df[df['ip'] == 'tracing_mark_write']
        return (df, 'buf')

    def _select_trace_printk(self, source_event, meta_event, df):
        content_col = {
            'bprint': 'buf',
            'bputs': 'str',
        }[source_event]

        # Select on foobar function name with "trace_printk@func@foobar"
        func_prefix = 'func@'
        if meta_event.startswith(func_prefix):
            func_name = meta_event[len(func_prefix):]
            df = self.ana.functions.df_resolve_ksym(df, addr_col='ip', name_col='func_name', exact=False)
            df = df[df['func_name'] == func_name]
            df = df.copy(deep=False)
            # Prepend the meta event name so it will be matched
            fake_event = meta_event.encode('ascii') + b': '
            df[content_col] = fake_event + df[content_col]

        return (df, content_col)

    _META_EVENT_SOURCE = {
        'userspace': {
            'print': _select_userspace,
        },
        'trace_printk': {
            'bprint': _select_trace_printk,
            'bputs': _select_trace_printk,
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
        events=None,
        strict_events=False,
        normalize_time=False,
        parser=None,
        plots_dir=None,
        sanitization_functions=None,

        max_mem_size=None,
        swap_dir=None,
        enable_swap=True,
        max_swap_size=None,
        write_swap=True,
        events_namespaces=('lisa', None),
    ):
        super().__init__()
        trace_path = str(trace_path) if trace_path else None

        sanitization_functions = sanitization_functions or {}
        self._sanitization_functions = {
            **self._SANITIZATION_FUNCTIONS,
            **sanitization_functions,
        }

        if enable_swap:
            if trace_path:
                if swap_dir is None:
                    basename = os.path.basename(trace_path)
                    swap_dir = os.path.join(
                        os.path.dirname(trace_path),
                        f'.{basename}.lisa-swap'
                    )
                    try:
                        os.makedirs(swap_dir, exist_ok=True)
                    except OSError:
                        swap_dir = None

                if max_swap_size is None:
                    trace_size = os.stat(trace_path).st_size
                    # Use 10 times the size of the trace so that there is
                    # enough room to store large artifacts like a JSON dump of
                    # the trace
                    max_swap_size = trace_size * 10
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

        if parser is None:
            if not trace_path:
                raise ValueError('A trace path must be provided')

            _, extension = os.path.splitext(trace_path)
            if extension == '.html':
                parser = SysTraceParser.from_html
            elif extension == '.txt':
                parser = HRTxtTraceParser.from_txt_file
            elif extension == '.perfetto-trace':
                parser = PerfettoTraceParser
            else:
                parser = TxtTraceParser.from_dat
        self._parser = parser

        # The platform information used to run the experiments
        if plat_info is None:
            # Delay import to avoid circular dependency
            # pylint: disable=import-outside-toplevel
            from lisa.platforms.platinfo import PlatformInfo
            plat_info = PlatformInfo()
        else:
            # Make a shallow copy so we can update it
            plat_info = copy.copy(plat_info)

        self._strict_events = strict_events
        self.available_events = _AvailableTraceEventsSet(self)
        if not plots_dir and trace_path:
            plots_dir = os.path.dirname(trace_path)
        self.plots_dir = plots_dir

        try:
            self._parseable_events = self._cache.get_metadata('parseable-events')
        except MissingMetadataError:
            self._parseable_events = {}

        if isinstance(events, str):
            raise ValueError('Events passed to Trace(events=...) must be a list of strings, not a string.')
        elif events is None:
            events = AndTraceEventChecker()
        elif isinstance(events, TraceEventCheckerBase):
            pass
        else:
            events = AndTraceEventChecker.from_events(events)

        self.events_namespaces = events_namespaces
        self.events = events
        # Pre-load the selected events
        if events:
            preload_events = OptionalTraceEventChecker.from_events(
                event_
                for event in events
                for event_ in self._expand_namespaces(event, events_namespaces)
            )
            df_map = self._load_cache_raw_df(preload_events, write_swap=True, allow_missing_events=True)

            missing_events = {
                event
                for event in events
                if not (df_map.keys() & set(self._expand_namespaces(event, events_namespaces)))
            }

            if strict_events and missing_events:
                raise MissingTraceEventError(
                    missing_events,
                    available_events=self.available_events,
                )

        # Register what we currently have
        self.plat_info = plat_info
        # Update the platform info with the data available from the trace once
        # the Trace is almost fully initialized
        self.plat_info = plat_info.add_trace_src(self)

    @bothmethod
    def _resolve_namespaces(self_or_cls, namespaces=None):
        if not isinstance(self_or_cls, type):
            namespaces = self_or_cls.events_namespaces if namespaces is None else namespaces
        return namespaces or (None,)

    @bothmethod
    def _expand_namespaces(self_or_cls, event, namespaces=None):
        namespaces = self_or_cls._resolve_namespaces(namespaces)

        def expand(event, namespace):
            ns_prefix = f'{namespace}__'
            if not namespace:
                return [event]
            elif self_or_cls._is_meta_event(event):
                prefix, _ = event.split('@', 1)
                return [
                    f'{prefix}@{source_}'
                    for source in self_or_cls.get_event_sources(event)
                    for source_ in expand(source, namespace)
                ]
            elif event.startswith(ns_prefix):
                return [event]
            else:
                return [f'{ns_prefix}{event}']

        return [
            event_
            for namespace in namespaces
            for event_ in expand(event, namespace)
        ]

    _CACHEABLE_METADATA = {
        'time-range',
        'cpus-count',
        # Do not cache symbols-address as JSON is unable to store integer keys
        # in objects, so the data will wrongly have string keys when reloaded.
    }
    """
    Parser metadata that can safely be cached, i.e. that are serializable in
    the trace cache.
    """

    def get_metadata(self, key):
        """
        Get metadata from the underlying trace parser.

        .. seealso:: :meth:`TraceParserBase.get_metadata`
        """
        if key in self._CACHEABLE_METADATA:
            return self._get_cacheable_metadata(key)
        else:
            return self._get_metadata(key)

    def _get_metadata(self, key, parser=None):
        if parser is None:
            parser = self._get_parser(needed_metadata={key})

        return parser.get_metadata(key)

    def _get_cacheable_metadata(self, key, parser=None):
        try:
            value = self._cache.get_metadata(key)
        except MissingMetadataError:
            value = self._get_metadata(key, parser=parser)
            self._cache.update_metadata({key: value})

        return value

    @classmethod
    def _is_meta_event(cls, event):
        sources = cls.get_event_sources(event)
        # If an event is not its own source, this is a meta-event by definition
        return sources[0] != event

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
            return sorted(cls._META_EVENT_SOURCE[prefix].keys())
        except KeyError:
            return [event]

    @property
    # Memoization is necessary to ensure the parser always gets the same name
    # on a given Trace instance when the parser is not a type.
    @lru_memoized(first_param_maxsize=None, other_params_maxsize=None)
    def trace_state(self):
        parser = self._parser
        # The parser type will potentially change the exact content in raw
        # dataframes
        def get_name(parser):
            return f'{parser.__module__}.{parser.__qualname__}'

        try:
            parser_name = get_name(parser)
        # If the parser is an instance of something, we cannot safely track its
        # state so just make a unique name for it
        except AttributeError:
            parser_name = f'{get_name(parser.__class__)}-instance:{uuid.uuid4().hex}'

        return (self.normalize_time, parser_name)

    @property
    @memoized
    def cpus_count(self):
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
                    int(df['__cpu'].max())
                    for df in (
                        self.df_event(event, namespaces=[])
                        for event in checked_events
                    )
                    if '__cpu' in df.columns
                )
                count = max_cpu + 1
                self.logger.debug(f"Estimated CPU count from trace: {count}")

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

        class TraceProxy(TraceBase):
            def get_view(self, *args, **kwargs):
                return self.base_trace.get_view(*args, **kwargs)

            def __getattr__(self, attr):
                try:
                    base_trace = self.__dict__['base_trace']
                except KeyError:
                    # pylint: disable=raise-missing-from
                    raise RuntimeError('The trace instance can only be used outside its "with" statement.')
                else:
                    return getattr(base_trace, attr)

        proxy = TraceProxy()

        if filepath:
            cm = nullcontext(filepath)
        else:
            @contextlib.contextmanager
            def cm_func():
                with tempfile.NamedTemporaryFile(suffix='.dat', delete=True) as temp:
                    yield temp.name

            cm = cm_func()

        with cm as path:
            ftrace_coll = FtraceCollector(target, events=events, buffer_size=buffer_size, output_path=path)
            with ftrace_coll:
                yield proxy

            trace = cls(
                path,
                events=events,
                strict_events=True,
                plat_info=plat_info,
                # Disable swap if the folder is going to disappear
                enable_swap=bool(filepath),
                **kwargs
            )

        # pylint: disable=attribute-defined-outside-init
        proxy.base_trace = trace

    def _get_parser(self, events=tuple(), needed_metadata=None, update_metadata=True):
        path = self.trace_path
        events = set(events)
        needed_metadata = set(needed_metadata or [])
        parser = self._parser(path=path, events=events, needed_metadata=needed_metadata)

        # While we are at it, gather a bunch of metadata. Since we did not
        # explicitly asked for it, the parser will only give
        # it if it was a cheap byproduct.
        if update_metadata:

            # Since we got a parser here, use it to get basetime/endtime as well
            with contextlib.suppress(MissingMetadataError):
                self._get_time_range(parser=parser)

            # Populate the list of available events, and inform the rest of the
            # code that this list is definitive.
            try:
                available_events = self._get_metadata('available-events', parser=parser)
            except MissingMetadataError:
                pass
            else:
                self._update_parseable_events({
                    event: True
                    for event in available_events
                })
                self._strict_events = True

        return parser

    def _update_parseable_events(self, mapping):
        self._parseable_events.update(mapping)
        self._cache.update_metadata({
            'parseable-events': self._parseable_events,
        })
        return self._parseable_events

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

    def _get_time_range(self, parser=None):
        return self._get_cacheable_metadata('time-range', parser)

    def df_event(self, event, raw=None, window=None, signals=None, signals_init=True, compress_signals_init=False, write_swap=None, namespaces=None):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name.

            In addition to actual events, the following formats for meta events
            are supported:

            * ``trace_printk@``: the event will be assumed to be embedded in
              textual form inside the field of another event as a string,
              typically emitted using the ``trace_printk()`` kernel function:

              .. code-block:: C

                  // trace.df_event('trace_printk@event')
                  void foo(void) {
                      trace_printk("event: optional_positional_field field1=foo field2=42");
                  }

            * ``trace_printk@func@``: the event name will be the name of the
              function calling trace_printk:

              .. code-block:: C

                  // trace.df_event('trace_printk@func@foo')
                  void foo(void) {
                      trace_printk("optional_positional_field field1=foo field2=42")
                  }

            * ``userspace@``: the event is generated by userspace:

              .. code-block:: shell

                  # trace.df_event('userspace@event')
                  echo "event: optional_positional_field field1=foo field2=42" > /sys/kernel/debug/tracing/trace_marker

            .. note:: All meta event names are expected to be valid C language
                identifiers. Usage of other characters will prevent correct
                parsing.

        :type event: str

        :param namespaces: List of namespaces of the requested event. See
            :class:`lisa.trace.Trace` ``events_namespaces`` parameters for the
            format. A ``None`` value defaults to the trace's namespace.
        :type namespaces: list(str or None) or None

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
        call = functools.partial(
            self._df_event,
            raw=raw,
            window=window,
            signals=signals,
            signals_init=signals_init,
            compress_signals_init=compress_signals_init,
            write_swap=write_swap,
        )

        for event_ in self._expand_namespaces(event, namespaces):
            try:
                return call(event=event_)
            except MissingTraceEventError as e:
                last_excep = e

        raise last_excep


    def _df_event(self, event, raw, window, signals, signals_init, compress_signals_init, write_swap):
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
        elif not orig_raw and orig_raw is not None and not sanitization_f:
            raise ValueError(f'Sanitized dataframe for {event} does not exist, please pass raw=True or raw=None')

        if raw:
            # Make sure all raw descriptors are made the same way, to avoid
            # missed sharing opportunities
            spec = self._make_raw_cache_desc_spec(event)
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
                rename_cols=True,
                sanitization=sanitization_f.__qualname__ if sanitization_f else None,
            )

        cache_desc = _CacheDataDesc(spec=spec, fmt=TraceCache.DATAFRAME_SWAP_FORMAT)

        try:
            df = self._cache.fetch(cache_desc, insert=True)
        except KeyError:
            df = self._load_df(cache_desc, sanitization_f=sanitization_f, write_swap=write_swap)

        if df.empty:
            raise MissingTraceEventError(
                [event],
                available_events=self.available_events,
            )

        # We used to set a ".name" attribute, but:
        # 1. There is no central way of saving these metadata when serializing
        # 2. It conflicts with a "name" column in the dataframe.
        df.attrs['name'] = event
        return df

    def _make_raw_cache_desc(self, event):
        spec = self._make_raw_cache_desc_spec(event)
        return _CacheDataDesc(spec=spec, fmt=TraceCache.DATAFRAME_SWAP_FORMAT)

    def _make_raw_cache_desc_spec(self, event):
        return dict(
            event=event,
            raw=True,
            trace_state=self.trace_state,
        )

    def _load_df(self, cache_desc, sanitization_f=None, write_swap=None):
        event = cache_desc['event']

        # Do not even bother loading the event if we know it cannot be
        # there. This avoids some OSError in case the trace file has
        # disappeared
        if self._strict_events and not self._is_meta_event(event) and not self.available_events.contains(event, namespaces=[]):
            raise MissingTraceEventError([event], available_events=self.available_events)

        if write_swap is None:
            write_swap = self._write_swap

        df = self._load_cache_raw_df(TraceEventChecker(event), write_swap=True)[event]

        if sanitization_f:
            # Evict the raw dataframe once we got the sanitized version, since
            # we are unlikely to reuse it again
            self._cache.evict(self._make_raw_cache_desc(event))

            # We can ask to sanitize various aspects of the dataframe.
            # Adding a new aspect can be done without modifying existing
            # sanitization functions, as long as the default is the
            # previous behavior
            aspects = dict(
                rename_cols=cache_desc['rename_cols'],
            )
            with measure_time() as measure:
                df = sanitization_f(self, event, df, aspects=aspects)
            sanitization_time = measure.exclusive_delta
        else:
            sanitization_time = 0

        window = cache_desc.get('window')
        if window is not None:
            signals_init = cache_desc['signals_init']
            compress_signals_init = cache_desc['compress_signals_init']
            cols_list = cache_desc['signals']
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
        self._cache.insert(cache_desc, df, compute_cost=compute_cost, write_swap=write_swap)
        return df

    def _load_cache_raw_df(self, event_checker, write_swap, allow_missing_events=False):
        events = event_checker.get_all_events()
        insert_kwargs = dict(
            write_swap=write_swap,
            # For raw dataframe, always write in the swap area if asked for
            # since parsing cost is known to be high
            force_write_swap=True,
        )

        # Get the raw dataframe from the cache if possible
        def try_from_cache(event):
            cache_desc = self._make_raw_cache_desc(event)
            try:
                # The caller is responsible of inserting in the cache if
                # necessary
                df = self._cache.fetch(cache_desc, insert=False)
            except KeyError:
                return None
            else:
                self._cache.insert(cache_desc, df, **insert_kwargs)
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
        events_to_load = sorted(events - from_cache.keys())
        from_trace = self._load_raw_df(events_to_load)

        for event, df in from_trace.items():
            cache_desc = self._make_raw_cache_desc(event)
            self._cache.insert(cache_desc, df, **insert_kwargs)

        df_map = {**from_cache, **from_trace}
        try:
            event_checker.check_events(df_map.keys())
        except MissingTraceEventError as e:
            if allow_missing_events:
                self.logger.warning(f'Events not found in the trace {self.trace_path}: {e}')
            else:
                raise
        return df_map

    def _apply_normalize_time(self, df, inplace):
        df = df if inplace else df.copy(deep=False)

        if self.normalize_time:
            df.index -= self.basetime

        return df

    def _mp_parse_worker(self, event):
        # Do not update the metadata to avoid concurrency issues while updating
        # the cache. Instead, we return the metadata and let the main thread
        # deal with it.
        parser = self._get_parser([event], update_metadata=False)

        try:
            data = parser.parse_event(event)
        except MissingTraceEventError as e:
            data = e
        else:
            data = self._apply_normalize_time(data, inplace=True)

        metadata = parser.get_all_metadata()
        return (data, metadata)

    def _parse_raw_events(self, events):
        if not events:
            return {}

        nr_processes = min(
            len(events),
            multiprocessing.cpu_count(),
        )
        chunk_size = int(math.ceil(len(events) / nr_processes))

        # Only use multiprocessing if there is no memory limit, since the peak
        # consumption will increase.
        # Daemonic threads cannot have children, so we cannot create a Pool if
        # we are already executing from a Pool.
        use_mp = (
            self._cache.max_mem_size >= math.inf
            and nr_processes > 1
            and not multiprocessing.current_process().daemon
        )

        if use_mp:
            with multiprocessing.Pool(processes=nr_processes) as pool:
                res_list = pool.map(self._mp_parse_worker, events, chunksize=chunk_size)

            if res_list:
                data_list, metadata_list = zip(*res_list)
                metadata = functools.reduce(lambda d1, d2: {**d1, **d2}, metadata_list, {})
                metadata = {
                    key: val
                    for key, val in metadata.items()
                    if key in self._CACHEABLE_METADATA
                }
                self._cache.update_metadata(metadata)

                df_map = {
                    event: df
                    for event, df in zip(
                        events,
                        data_list,
                    )
                    # similar to best_effort=True
                    if not isinstance(df, BaseException)
                }
            else:
                df_map = {}
        else:
            parser = self._get_parser(events, update_metadata=True)
            df_map = parser.parse_events(events, best_effort=True)

            for df in df_map.values():
                self._apply_normalize_time(df, inplace=True)

        return df_map

    def _parse_meta_events(self, meta_events):
        if not meta_events:
            return {}

        # Gather the infor to parse the meta event
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
                df = self.df_event(source_event, namespaces=[])
            except MissingTraceEventError:
                pass
            else:
                # Add all the header fields from the source dataframes
                extra_fields = [x for x in df.columns if x.startswith('__')]
                merged_df = df[extra_fields]

                for (meta_event, event, _source_event, source_getter) in specs:  # pylint: disable=unused-variable
                    source_df, line_field = source_getter(self, _source_event, event, df)
                    try:
                        parser = MetaTxtTraceParser(
                            lines=source_df[line_field],
                            time=source_df.index,
                            merged_df=merged_df,
                            events=[event],
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
                            meta_event_df = parser.parse_event(event)
                        except MissingTraceEventError:
                            continue
                            # In case a meta-event is spread among multiple
                            # events, we get all the dataframes and concatenate
                            # them together
                        else:
                            df_map.setdefault(event, []).append(meta_event_df)

                    if not get_missing(df_map):
                        break

        def concat(df_list):
            if len(df_list) > 1:
                # As of pandas == 1.0.3, concatenating dataframe with nullable
                # columns will give an object dtype and both NaN and <NA> if
                # that column does not exist in all dataframes. This is quite
                # annoying but there is no straightforward way of working
                # around that.
                df = pd.concat(df_list, copy=False)
                df.sort_index(inplace=True)
                return df
            # Avoid creating a new dataframe and sorting the index
            # unnecessarily if there is only one
            else:
                return df_list[0]

        df_map = {
            event: concat(df_list)
            for event, df_list in df_map.items()
        }

        # On some parsers, meta events are treated as regular events so attempt
        # to load them from there as well
        for event in get_missing(df_map):
            with contextlib.suppress(MissingTraceEventError):
                df_map[event] = self.df_event(event, raw=True, namespaces=[])

        return {
            events_map[event]: df
            for event, df in df_map.items()
        }

    def _load_raw_df(self, events):
        events = set(events)
        if not events:
            return {}

        meta_events = set(filter(self._is_meta_event, events))
        regular_events = events - meta_events

        df_map = {
            **self._parse_raw_events(regular_events),
            **self._parse_meta_events(meta_events),
        }

        for event, df in df_map.items():
            df.attrs['name'] = event
            df.index.name = 'Time'

        # Save some memory by changing values of this column into an category
        categorical_fields = [
            '__comm',
            'comm',
        ]
        for df in df_map.values():
            for field in categorical_fields:
                with contextlib.suppress(KeyError):
                    df[field] = df[field].astype('category', copy=False)

        # remember the events that we tried to parse and that turned out to not be available
        self._update_parseable_events({
            event: (event in df_map)
            for event in events
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
            grouped = df.groupby(key_col, observed=True, sort=False, group_keys=False)
            for key, subdf in grouped:
                values = subdf[value_col].apply(value_type).to_list()
                key = key_type(key)
                mapping[key] = values

            return mapping

        mapping_df_list = []
        def _load(event, name_col, pid_col):
            df = self.df_event(event, namespaces=[])

            # Get a Time column
            df = df.reset_index()
            grouped = df.groupby([name_col, pid_col], observed=True, sort=False)

            # Get timestamp of first occurrences of each key/value combinations
            mapping_df = grouped.head(1)
            mapping_df = mapping_df[['Time', name_col, pid_col]]
            mapping_df.rename({name_col: 'name', pid_col: 'pid'}, axis=1, inplace=True)
            mapping_df_list.append(mapping_df)

        missing = []
        def load(event, *args, **kwargs):
            try:
                _load(event, *args, **kwargs)
            except MissingTraceEventError as e:
                missing.append(e.missing_events)

        load('task_rename', 'oldcomm', 'pid')
        load('task_rename', 'newcomm', 'pid')

        load('sched_switch', 'prev_comm', 'prev_pid')
        load('sched_switch', 'next_comm', 'next_pid')

        if not mapping_df_list:
            missing = OrTraceEventChecker.from_events(events=missing)
            raise MissingTraceEventError(missing, available_events=self.available_events)

        df = pd.concat(mapping_df_list)
        # Sort by order of appearance
        df.sort_values(by=['Time'], inplace=True)
        # Remove duplicated name/pid mapping and only keep the first appearance
        df = df_deduplicate(df, consecutives=False, keep='first', cols=['name', 'pid'])

        name_to_pid = finalize(df, 'name', 'pid', str, int)
        pid_to_name = finalize(df, 'pid', 'name', int, str)

        return (name_to_pid, pid_to_name)

    @property
    def _task_name_map(self):
        return self._get_task_maps()[0]

    @property
    def _task_pid_map(self):
        return self._get_task_maps()[1]

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
        removed_in='4.0',
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
        removed_in='4.0',
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
            raise RuntimeError(f'The PID {pid} had more than two names in its life: {name_list}')

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
                # pylint: disable=raise-missing-from
                raise ValueError(f'trace does not have any task named "{comm}"')

            return pid_list

        def pid_to_comm(pid):
            try:
                comm_list = self._task_pid_map[pid]
            except IndexError:
                # pylint: disable=raise-missing-from
                raise ValueError(f'trace does not have any task PID {pid}')

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

            if update and (pid is None or comm is None):
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
            raise ValueError(f'More than one TaskID matching: {task_ids}')

        return task_ids[0]

    @deprecate(deprecated_in='2.0', removed_in='4.0', replaced_by=get_task_id)
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
        if not path:
            raise ValueError('No trace file is backing this Trace instance')

        if path.endswith('.dat'):
            cmd = 'kernelshark'
        else:
            cmd = 'xdg-open'

        return os.popen(f"{cmd} {shlex.quote(path)}")

###############################################################################
# Trace Events Sanitize Methods
###############################################################################

    _SANITIZATION_FUNCTIONS = {}
    def _sanitize_event(event, mapping=_SANITIZATION_FUNCTIONS):
        """
        Sanitization functions must not modify their input.
        """
        # pylint: disable=dangerous-default-value,no-self-argument

        def decorator(f):
            mapping[event] = f
            return f
        return decorator

    @_sanitize_event('sched_switch')
    def _sanitize_sched_switch(self, event, df, aspects):
        """
        If ``prev_state`` is a string, turn it back into an integer state by
        parsing it.
        """
        # pylint: disable=unused-argument,no-self-use
        copied = False
        def copy_once(x):
            nonlocal copied
            if copied:
                return x
            else:
                copied = True
                return x.copy(deep=False)

        if df['prev_state'].dtype.name == 'string':
            # Avoid circular dependency issue by importing at the last moment
            # pylint: disable=import-outside-toplevel
            from lisa.analysis.tasks import TaskState
            df = copy_once(df)
            df['prev_state'] = df['prev_state'].apply(TaskState.from_sched_switch_str).astype('uint16', copy=False)

        # Save a lot of memory by using category for strings
        df = copy_once(df)
        for col in ('next_comm', 'prev_comm'):
            df[col] = df[col].astype('category', copy=False)

        return df

    @_sanitize_event('lisa__sched_overutilized')
    def _sanitize_sched_overutilized(self, event, df, aspects):
        # pylint: disable=unused-argument,no-self-use
        copied = False
        def copy_once(x):
            nonlocal copied
            if copied:
                return x
            else:
                copied = True
                return x.copy(deep=False)

        if not df['overutilized'].dtype.name == 'bool':
            df = copy_once(df)
            df['overutilized'] = df['overutilized'].astype(bool, copy=False)

        if 'span' in df.columns and df['span'].dtype.name == 'string':
            df = copy_once(df)
            df['span'] = df['span'].apply(lambda x: x if pd.isna(x) else int(x, base=16))

        return df

    @_sanitize_event('thermal_power_cpu_limit')
    @_sanitize_event('thermal_power_cpu_get_power')
    def _sanitize_thermal_power_cpu(self, event, df, aspects):
        # pylint: disable=unused-argument,no-self-use

        def parse_load(array):
            # Parse b'{2 3 2 8}'
            return tuple(map(int, array[1:-1].split()))

        # In-kernel name is "cpumask", "cpus" is just an artifact of the pretty
        # printing format string of ftrace, that happens to be used by a
        # specific parser.
        df = df.rename(columns={'cpus': 'cpumask'}, copy=False)
        df = df.copy(deep=False)

        if df['cpumask'].dtype.name == 'object':
            df['cpumask'] = df['cpumask'].apply(self._expand_bitmask_field)

        if event == 'thermal_power_cpu_get_power':
            if df['load'].dtype.name == 'object':
                df['load'] = df['load'].apply(parse_load)

        return df

    @_sanitize_event('print')
    @_sanitize_event('bprint')
    @_sanitize_event('bputs')
    def _sanitize_print(self, event, df, aspects):
        # pylint: disable=unused-argument,no-self-use

        df = df.copy(deep=False)

        # Only process string "ip" (function name), not if it is a numeric
        # address
        if not is_numeric_dtype(df['ip'].dtype):
            # Reduce memory usage and speedup selection based on function
            with contextlib.suppress(KeyError):
                df['ip'] = df['ip'].astype('category', copy=False)

        content_col = 'str' if event == 'bputs' else 'buf'

        # Ensure we have "bytes" values, since some parsers might give
        # str type.
        try:
            df[content_col] = df[content_col].str.encode('utf-8')
        except TypeError:
            pass

        if event == 'print':
            # Print event is mainly used through the trace_marker sysfs file.
            # Since userspace application typically end the write with a
            # newline char, strip it from the values, as some parsers will not
            # include that in the output.
            try:
                last_char = df['buf'].iat[0][-1]
            except (KeyError, IndexError):
                pass
            else:
                if last_char == ord(b'\n'):
                    df['buf'] = df['buf'].apply(lambda x: x.rstrip(b'\n'))

        return df

    @staticmethod
    def _expand_bitmask_field(mask):
        """
        Turn a bitmask (like cpu_mask) formated by trace-cmd in non-raw mode
        into a list of integers for each bitmask position that is set.

        ``mask`` is a string with comma-separated hex numbers like
        "000001,12345,..."
        """
        numbers = mask.split(b',')

        # hex number, so 4 bit per digit
        nr_bits = len(numbers[0]) * 4

        def bit_pos(number):
            # Little endian
            number = int(number, base=16)
            return (
                i
                for i in range(nr_bits)
                if number & (1 << i)
            )

        return tuple(
            i + (nr_bits * offset)
            for offset, positions in enumerate(
                # LSB is in the number at the end of the list so we reverse it
                map(bit_pos, reversed(numbers))
            )
            for i in positions
        )

    @_sanitize_event('ipi_raise')
    def _sanitize_ipi_raise(self, event, df, aspects):
        # pylint: disable=unused-argument,no-self-use

        df = df.copy(deep=False)
        df['target_cpus'] = df['target_cpus'].apply(self._expand_bitmask_field)
        return df

    @_sanitize_event('ipi_entry')
    @_sanitize_event('ipi_exit')
    def _sanitize_ipi_enty_exit(self, event, df, aspects):
        # pylint: disable=unused-argument,no-self-use

        df = df.copy(deep=False)
        df['reason'] = df['reason'].str.strip('()')
        return df


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
                if isinstance(checker, OptionalTraceEventChecker):
                    return AndTraceEventChecker(checker.checkers)
                else:
                    return checker
            checker = self.map(rewrite)
        else:
            checker = self

        if isinstance(event_set, _AvailableTraceEventsSet):
            namespaces = event_set._trace._resolve_namespaces(namespaces)
            def check(event):
                # We already expanded namespaces, so we don't want the
                # inclusion check to apply the default trace's namespace.
                return event_set.contains(event, namespaces=[])
        else:
            def check(event):
                return event in event_set

        checker = checker.expand_namespaces(namespaces=namespaces)

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
        def fixup(checker):
            if isinstance(checker, TraceEventChecker):
                event = checker.event
                namespaced = Trace._expand_namespaces(event, namespaces=namespaces)

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
            failed_checker_set = set()
            for checker in self.checkers:
                try:
                    return checker._select_events(check=check, event_set=event_set)
                except MissingTraceEventError as e:
                    failed_checker_set.add(e.missing_events)

            cls = type(self)
            raise MissingTraceEventError(
                cls(failed_checker_set),
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
            failed_checker_set = set()
            selected = set()
            for checker in self.checkers:
                try:
                    selected.update(
                        checker._select_events(check=check, event_set=event_set)
                    )
                except MissingTraceEventError as e:
                    failed_checker_set.add(e.missing_events)

            if failed_checker_set:
                cls = type(self)
                raise MissingTraceEventError(
                    cls(failed_checker_set),
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

        if available:
            closest = {
                matches[0]
                for matches in (
                    get_close_matches(event, available, n=1)
                    for event in missing
                )
                if matches
            }
            available = '. Closest available matches are: {}. Available events are: {}'.format(
                ', '.join(sorted(closest)),
                ', '.join(sorted(available))
            )
        else:
            available = ''

        return self._template.format(missing_events=missing, available=available)


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

        if not isinstance(conf, self.__class__):
            conf = self.__class__(conf=conf)

        def merge_conf(key, val, path):
            new = _merge_conf(key, val, path)
            try:
                existing = get_nested_key(self, path + [key])
            except KeyError:
                return (True, new)
            else:
                # Only add to the source if the result is different than what
                # is already set
                return (existing != new, new)

        def _merge_conf(key, val, path):

            def non_mergeable(key):
                if self.get(key, val) == val:
                    return val
                else:
                    raise KeyError(f'Cannot merge key "{key}": incompatible values specified: {self[key]} != {val}')

            if key == 'functions':
                return sorted(set(val) | set(self.get(key, [])))
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
                        return conf.get(key)
                    except KeyError:
                        return conf.DEFAULT_SRC.get(key)

                val = val.expand_namespaces(
                    namespaces=get(conf, 'events-namespaces')
                )

                self_val = self.get(key, [])
                if not isinstance(self_val, TraceEventCheckerBase):
                    self_val = AndTraceEventChecker.from_events(self_val)

                self_val = self_val.expand_namespaces(
                    namespaces=get(self, 'events-namespaces')
                )

                return AndTraceEventChecker([val, self_val])
            elif key == 'buffer-size':
                return max(val, self.get(key, 0))
            elif key == 'trace-clock':
                return non_mergeable(key)
            elif key == 'saved-cmdlines-nr':
                return max(val, self.get(key, 0))
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


class CollectorBase(Loggable):
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

    def __getattr__(self, attr):
        return getattr(self._collector, attr)

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
        they are not :class:`lisa._kmod.LISAFtraceDynamicKmod` will build a
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

    def __init__(self, target, *, events=None, functions=None, buffer_size=10240, output_path=None, autoreport=False, trace_clock=None, saved_cmdlines_nr=8192, tracer=None, kmod_auto_load=True, events_namespaces=('lisa', None), **kwargs):

        kconfig = target.plat_info['kernel']['config']
        if not kconfig.get('FTRACE'):
            raise ValueError("The target's kernel needs CONFIG_FTRACE=y kconfig enabled")

        tracing_path = devlib.FtraceCollector.find_tracing_path(target)
        target_available_events, avoided = self._target_available_events(target, tracing_path)

        kmod = target.get_kmod(LISAFtraceDynamicKmod)
        # Get the events possibly defined in the module. Note that it's a
        # superset of the events actually defined as this is based on pretty
        # crude filtering of the source files, rather than analyzing the events
        # actually defined in the .ko after it has been compiled.
        kmod_available_events = set(kmod.possible_events)

        # Events provided by the module are namespaced and therefore should
        # never overlap with the target, so we never want to provide it via the
        # module if it already exists on the target.
        kmod_available_events -= target_available_events

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
            elif isinstance(checker, DynamicTraceEventChecker):
                return AndTraceEventChecker(checker.checkers)
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
            if Trace._is_meta_event(event)
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

        # If some events are not already available on that kernel, look them up
        # in custom modules
        needed_from_kmod = kmod_available_events & events

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
            self.logger.info(f'Optional events missing: {str(e)}')

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
        kmod = target.get_kmod(LISAFtraceDynamicKmod)
        defined_events = set(kmod.defined_events)
        needed = needed_events & defined_events
        if needed:
            overlapping = defined_events & target_available_events
            if overlapping:
                raise ValueError(f'Events defined in {mod.src.mod_name} ({", ".join(needed)}) are needed but some events overlap with the ones already provided by the kernel: {", ".join(overlapping)}')
            else:
                return (
                    defined_events,
                    needed,
                    functools.partial(
                        kmod.run,
                        kmod_params={
                            'features': sorted(kmod._event_features(needed))
                        }
                    )
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
        avoided = set(target.list_directory(target.path.join(tracing_path, 'events', 'ftrace')))

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
