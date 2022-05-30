# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022, ARM Limited and contributors.
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

""" Rust Analysis Module """

import heapq
import operator
import multiprocessing
import threading
import subprocess
import functools
import contextlib
import itertools
import os
import time
import json
from pathlib import Path
import uuid
import shlex
import copy
from collections.abc import Mapping, Iterable
from collections import Counter
import inspect
import textwrap
from tempfile import TemporaryDirectory

import ujson
import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import TxtTraceParser, TraceView, TraceEventChecker, AndTraceEventChecker, OrTraceEventChecker, OptionalTraceEventChecker, DynamicTraceEventChecker, _CacheDataDesc, MissingTraceEventError, doc_events
from lisa.analysis._rust_fast_import import _json_line, _json_record
from lisa.utils import mp_spawn_pool, Loggable, nullcontext, measure_time, FrozenDict, get_nested_key, group_by_value, optional_kwargs
from lisa._assets import HOST_BINARIES
from lisa.version import VERSION_TOKEN
from lisa.datautils import series_update_duplicates
from lisa._feather import load_feather


_RUST_ANALYSIS_PATH = HOST_BINARIES['lisa-rust-analysis']


class _PipeSafePopen:
    def __init__(self, popen):
        self.popen = popen

    def __enter__(self):
        return self.popen.__enter__()

    def __exit__(self, *args, **kwargs):
        try:
            return self.popen.__exit__(*args, **kwargs)
        except BrokenPipeError:
            pass


def _map_best_effort(f, xs):
    exceps = []
    for x in xs:
        try:
            f(x)
        except Exception as e:
            exceps.append(e)
        else:
            exceps.append(None)

    if all(exceps):
        raise exceps[0]


class _TeeFile:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, data) :
        _map_best_effort(lambda f: f.write(data), self.fs)

    def flush(self) :
        _map_best_effort(lambda f: f.flush(), self.fs)

    def close(self) :
        _map_best_effort(lambda f: f.close(), self.fs)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _event_checker_from_json(mapping):
    k, v = mapping.popitem()
    if k == 'single':
        return TraceEventChecker(v)
    else:
        classes = {
            'and': AndTraceEventChecker,
            'or': OrTraceEventChecker,
            'optional': OptionalTraceEventChecker,
            'dynamic': DynamicTraceEventChecker,
        }
        try:
            cls = classes[k]
        except KeyError:
            raise ValueError(f'Unknown trace event checker type: {k}')
        else:
            return cls(
                list(map(_event_checker_from_json, v))
            )


class Type:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return str(self)


class NewType(Type):
    def __init__(self, name, typ):
        super().__init__(name)
        self.typ = typ

    def __str__(self):
        name = self.name
        typ = self.typ
        if name:
            return f'{name}({typ})'
        else:
            return typ


class BasicType(Type):
    def __init__(self, typ):
        super().__init__(typ)
        self.typ = typ

    def __str__(self):
        return self.typ


class ArrayType(Type):
    def __init__(self, name, typ):
        super().__init__(name)
        self.typ = typ

    def __str__(self):
        return f'[{self.typ}]'


class UnitType(Type):
    def __init__(self):
        super().__init__('()')

    def __str__(self):
        return self.name


class SumType(Type):
    def __init__(self, name, ctors):
        name = name or '<anonymous>'
        super().__init__(name)

        self.ctors = dict(ctors)

    def __str__(self):
        ctors = ', '.join(
            (
                f'{name}({param})'
                if not isinstance(param, UnitType) else
                name
            )
            for name, param in self.ctors.items()
        )
        return f'{self.name}{{{ctors}}}'


class OptionType(Type):
    def __init__(self, name, typ):
        super().__init__(name)
        self.typ = typ

    def __str__(self):
        return f'Option({self.typ})'


class ProductType(Type, Mapping):
    def __init__(self, name, items, names):
        super().__init__(name)
        self.items = list(items)
        self.names = names
        assert len(self.items) == len(self.names)

    def __getitem__(self, i):
        return dict(self.named_items)[i]

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.items)

    @property
    def named_items(self):
        return zip(self.names, self.items)

    def __str__(self):
        name = self.name or ''

        if len(self.items) > 1:
            params = ', '.join(f'{name}: {item}' for name, item in self.named_items)
        else:
            params = ', '.join(map(str, self.items))

        return f'{name}({params})'


class TupleProductType(ProductType):
    def __init__(self, name, items):
        super().__init__(
            name=name,
            items=items,
            names=list(range(len(items)))
        )


class StructProductType(ProductType):
    pass


def _listify(x):
    if not isinstance(x, Iterable) or isinstance(x, Mapping):
        return [x]
    else:
        return x


def _do_infer_adt(name, schema, deref):
    _name, schema = deref(schema)
    name = name or _name

    # "arrays" with min and max items are actually product types
    if schema.get('type') == 'array' and not ('minItems' in schema and 'maxItems' in schema):
        inner = _do_infer_adt(None, schema['items'], deref)
        return ArrayType(name, inner)
    elif 'items' in schema:
        items = [
            _do_infer_adt(None, _schema, deref)
            for _schema in _listify(schema['items'])
        ]
        return TupleProductType(name, items)
    elif 'properties' in schema:
        items = [
            (_name, _do_infer_adt(None, _schema, deref))
            for _name, _schema in schema['properties'].items()
        ]
        if items:
            names, items = zip(*items)
        else:
            names = []
            items = []

        return StructProductType(name, items, names)
    # Sum types with fields in variants
    elif 'oneOf' in schema or 'anyOf' in schema:
        try:
            variants = schema['oneOf']
        except KeyError:
            variants = schema['anyOf']

        ctors = {}
        for variant in variants:
            ctor, variant = deref(variant)

            if 'enum' in variant:
                _ctors = variant['enum']
                ctors.update(dict.fromkeys(_ctors, UnitType()))
            elif variant['type'] == 'object':
                ctor, = variant['required']
                ctors[ctor] = _do_infer_adt(None, variant['properties'][ctor], deref)
            else:
                ctors[ctor] = _do_infer_adt(None, variant, deref)

        if None in ctors:
            assert len(ctors) == 2
            ctor, = (ctor for ctor in ctors if ctor is not None)
            return OptionType(name, ctors[ctor])
        else:
            return SumType(name, ctors)
    # Sum type with no field in any variant
    elif 'enum' in schema:
        ctors = dict.fromkeys(schema['enum'], UnitType())
        return SumType(name, ctors)
    else:
        try:
            typ = schema['format']
        except KeyError:
            typ = schema['type']

        if typ == 'null':
            return UnitType()
        elif name is None:
            return BasicType(typ)
        else:
            return NewType(name, BasicType(typ))


def _infer_adt(whole_schema, sub_schema=None):
    if sub_schema is None:
        name = whole_schema['title']
        schema = whole_schema
    else:
        name = None
        schema = sub_schema

    def _deref(_schema):
        # Expand the $ref. We do not expect recursive types.
        if tuple(_schema) == ('$ref',):
            ref = _schema['$ref']
            path = [x for x in ref.split('/') if x != '#']
            return (path[-1], get_nested_key(whole_schema, path))
        else:
            return (None, _schema)

    def deref(_schema):
        new = (None, _schema)
        old = (None, None)
        while old[1] != new[1]:
            old = new
            new = _deref(new[1])

        # The fix point has name == None, so we backtrack one step to get
        # the actual name
        return (old[0], new[1])

    return _do_infer_adt(name, schema, deref)


def expand_adt(adt):
    if isinstance(adt, BasicType):
        def expand(l, i, x):
            l[i] = x
            return i + 1
        return ([(None, adt.typ, adt)], expand)

    elif isinstance(adt, NewType):
        [(col, typ, _)], expand = expand_adt(adt.typ)
        return ([(col, typ, adt)], expand)

    elif isinstance(adt, OptionType):
        return expand_adt(adt.typ)

    elif isinstance(adt, UnitType):
        def expand(l, i, x):
            return i
        return ([], expand)

    elif isinstance(adt, ProductType):
        if adt.items:
            cols, expands = zip(*map(expand_adt, adt.items))

            if isinstance(adt, TupleProductType):
                def expand(l, i, x):
                    for expand, _x in zip(expands, x):
                        i = expand(l, i, _x)
                    return i
            elif isinstance(adt, StructProductType):
                def expand(l, i, x):
                    for expand, field in zip(expands, adt.names):
                        i = expand(l, i, x[field])
                    return i
            else:
                raise TypeError(f'Unknown product type: {adt.__class__}')

            cols = [
                (
                    f'{i}.{col}' if col else str(i),
                    typ,
                    adt
                )
                for i, _cols in zip(adt.names, cols)
                for col, typ, adt in _cols
            ]
            return (cols, expand)
        else:
            raise ValueError(f'Product type with no constructor: {adt}')

    elif isinstance(adt, SumType):
        if adt.ctors:
            variants = {
                ctor: (cols, expand)
                for (ctor, (cols, expand)) in (
                    (ctor, expand_adt(variant_adt))
                    for ctor, variant_adt in adt.ctors.items()
                )
                if cols
            }
            variants = sorted(variants.items())

            cols = [
                (f'{name}.{col}' if col else name, typ, adt)
                for name, (cols, _) in variants
                for col, typ, adt in cols
            ]
            nr_cols = len(cols) + 1

            variant_names, _ = zip(*variants)
            variant_cols = [cols for _, (cols, _) in variants]
            variant_index = list(itertools.accumulate([0] + list(map(len, variant_cols))))
            variant_index = {
                name: (index + 1, expand)
                for name, index, (_, (_, expand)) in zip(variant_names, variant_index, variants)
            }

            def expand(l, i, x):
                # Fast check for dict as we typically expand loads of
                # them in dataframes
                if type(x) is dict:
                    x, v = x.popitem()
                    l[i] = x
                    index, expand = variant_index[x]
                    i = expand(l, i + index, v)
                else:
                    l[i] = x

                return i + nr_cols

            cols = [(None, 'category', adt)] + cols
            return (cols, expand)
        else:
            raise ValueError(f'Sum type with no constructor: {adt}')
    else:
        raise ValueError(f'Unknown type: {adt}')


def _post_process_data(data, normalize_time, out_path):
    schema = data['schema']
    value = data['value']

    adt = _infer_adt(schema)
    traverse, fmt = _make_traversal(adt, normalize_time)
    return (traverse(value, out_path), fmt)


def _traverse_identity(x, _):
    return x


def _make_traversal(adt, normalize_time):

    def make_traversal(*args, **kwargs):
        return _make_traversal(
            *args,
            **kwargs,
            normalize_time=normalize_time
        )[0]

    def _fixup_ts(x):
        if isinstance(x, pd.Series):
            x /= 1e9
            if normalize_time is not None:
                x -= normalize_time

            # Deduplicate timestamps, since pandas cannot deal properly with
            # duplicated indices
            return series_update_duplicates(x)
        else:
            return x / 1e9

    def make_newtype_fixer(adt):
        if adt.name == 'Timestamp':
            return _fixup_ts
        else:
            return _traverse_identity

    if isinstance(adt, StructProductType) and adt.name and adt.name.startswith('OutofbandTable_for_'):
        def traverse(value, out_path):
            path = Path(value['path'])
            path = out_path / path
            assert value['format'] == 'Feather'

            columns = value['columns']
            row_schema = value['schema']
            row_adt = _infer_adt(row_schema)

            assert len(row_adt.items) == len(columns)
            row_adt.names = columns

            typed_cols, _  = expand_adt(row_adt)
            columns, _, _ = zip(*typed_cols)

            # arrow2_convert represents Rust enums as UnionArray. For variants
            # that have no payload (unit payload), it pretends the type is
            # bool. When expanded, this would give rise to a bool column that
            # is of no use.
            df = load_feather(path, columns=set(columns))
            assert set(df.columns) == set(columns)

            for _col, _, _adt in typed_cols:
                if isinstance(_adt, NewType):
                    f = make_newtype_fixer(_adt)
                    if f is not _traverse_identity:
                        df[_col] = f(df[_col])

            df.set_index(columns[0], inplace=True)
            return df

        return (traverse, 'parquet')

    elif isinstance(adt, StructProductType) and adt.name and adt.name.startswith('InbandTable_for_'):
        # Each row is a tuple of values, so it will map to a
        # ProductType
        row_adt = adt['data'].typ

        def traverse(value, _):
            columns = value['columns']
            assert len(columns) > 0

            assert len(row_adt.items) == len(columns)
            _row_adt = copy.deepcopy(row_adt)
            _row_adt.names = columns

            typed_cols, _expand = expand_adt(_row_adt)
            columns, _, _ = zip(*typed_cols)

            nr_columns = len(columns)
            def expand(x):
                l = [None] * nr_columns
                _expand(l, 0, x)
                return l

            df = pd.DataFrame.from_records(
                tuple(map(expand, value['data'])),
                columns=columns,
            )

            # Use nullable types
            json_pd_dtypes = {
                'uint8': 'UInt8',
                'uint16': 'UInt16',
                'uint32': 'UInt32',
                'uint64': 'UInt64',
                'int8': 'Int8',
                'int16': 'Int16',
                'int32': 'Int32',
                'int64': 'Int64',
                'string': 'string',
                'bool': 'bool',
                'category': 'category',
            }
            dtypes = {
                col: json_pd_dtypes[typ]
                for col, typ, adt in typed_cols
                if typ in json_pd_dtypes
            }

            df = df.astype(dtypes, copy=False)

            for col, _, adt in typed_cols:
                if isinstance(adt, NewType):
                    f = make_newtype_fixer(adt)
                    if f is not _traverse_identity:
                        df[col] = f(df[col])

            df.set_index(columns[0], inplace=True)
            return df

        return (traverse, 'parquet')

    elif isinstance(adt, ArrayType) and adt.name and adt.name.startswith('Map_for'):
        k_adt, v_adt = adt.typ.items
        traverse_k = make_traversal(k_adt)
        traverse_v = make_traversal(v_adt)

        if traverse_k is _traverse_identity and traverse_v is _traverse_identity:
            traverse = lambda value, _: dict(x)
        else:
            def traverse(value, out_path):
                return {
                    traverse_k(k, out_path): traverse_v(v, out_path)
                    for k, v in value
                }

        return (traverse, 'json')

    elif isinstance(adt, StructProductType):
        traversals = {
            _name: make_traversal(_adt)
            for _name, _adt in adt.named_items
        }
        if all(f is _traverse_identity for f in traversals.values()):
            traverse = _traverse_identity
        else:
            def traverse(value, out_path):
                return {
                    _name: _traverse(value[_name], out_path)
                    for (_name, _traverse) in traversals.items()
                }
        return (traverse, 'json')

    elif isinstance(adt, TupleProductType):
        traversals = tuple(map(make_traversal, adt.items))

        if all(f is _traverse_identity for f in traversals):
            traverse = _traverse_identity
        else:
            def traverse(value, out_path):
                return tuple(
                    _traverse(_value, out_path)
                    for _traverse, _value in zip(traversals, value)
                )
        return (traverse, 'json')

    elif isinstance(adt, SumType):
        traversals = {
            _ctor: make_traversal(_adt)
            for _ctor, _adt in adt.ctors.items()
        }

        def traverse(value, out_path):
            if isinstance(value, Mapping):
                ctor, = value
                v = value[ctor]
                data = (ctor, traversals[ctor](v, out_path))
            else:
                data = (value, None)
            return data

        return (traverse, 'json')

    elif isinstance(adt, NewType):
        _traverse, _fmt = _make_traversal(adt.typ, normalize_time=normalize_time)
        _fixup = make_newtype_fixer(adt)

        if _traverse is _traverse_identity and _fixup is _traverse_identity:
            traverse = _traverse_identity
        else:
            def traverse(value, out_path):
                return _fixup(_traverse(value, out_path))

        return (traverse, _fmt)

    elif isinstance(adt, OptionType):
        _traverse = make_traversal(adt.typ)

        if _traverse is _traverse_identity:
            traverse = _traverse_identity
        else:
            def traverse(value, out_path):
                if value is None:
                    data = None
                else:
                    data = _traverse(value, out_path)
                return data

        # TODO: DataFrame is not a JSON encodable type, so Option<DataFrame>
        # will fail to serialize correctly.

        # Always return json data, otherwise we might populate the cache twice
        # with different formats and it will be ambiguous what to pick.
        return (traverse, 'json')

    elif isinstance(adt, ArrayType):
        _traverse = make_traversal(adt.typ)

        if _traverse is _traverse_identity:
            traverse = _traverse_identity
        else:
            def traverse(value, out_path):
                return [
                    _traverse(x, out_path)
                    for x in value
                ]

        # TODO: DataFrame is not a JSON encodable type, so Option<DataFrame>
        # will fail to serialize correctly.
        return (traverse, 'json')

    elif isinstance(adt, (BasicType, UnitType)):
        return (_traverse_identity, 'json')

    else:
        raise TypeError(f'Could not traverse unknown adt ({adt.__class__}): {adt}')


class RustAnalysis(TraceAnalysisBase, Loggable):
    """
    Support for Rust analysis.
    """

    name = '_rust'

    def __init__(self, trace, proxy=None):
        def get_base(trace):
            if isinstance(trace, TraceView):
                return get_base(trace.base_trace)
            else:
                return trace

        base_trace = get_base(trace)

        # We play a trick here: self.trace is the base trace and self._trace is
        # the actual trace that was passed. This makes @TraceAnalysisBase.cache
        # consider the base trace for the state, so we can manipulate any
        # window manually within this class
        self._trace = trace
        super().__init__(
            trace=base_trace,
            proxy=base_trace.ana,
        )

    @property
    def _metadata(self):
        return _RUST_ANALYSIS_METADATA

    @classmethod
    def _call_binary(cls, subcommand, *cli_args):
        logger = cls.get_logger()
        cmd = [_RUST_ANALYSIS_PATH, subcommand, *cli_args]
        pretty_cmd = ' '.join(map(lambda x: shlex.quote(str(x)), cmd))

        logger.debug(f'Running rust analysis: {pretty_cmd}')
        completed = subprocess.run(
            cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        logger.debug(f'Rust analysis stderr for {pretty_cmd}:\n{completed.stderr}')

        return completed.stdout

    def call_anas(self, analyses):
        trace = self._trace
        cache = trace._cache
        window = self._trace_window

        def resolve(spec):
            name = spec['name']
            try:
                metadata = self._metadata[name]
            except KeyError:
                raise ValueError(f'Unknown analysis "{name}", available analysis are: {sorted(self._metadata.keys())}')
            else:
                checker = metadata['event_checker']
                return (checker, spec)

        def make_cache_desc(spec, fmt):
            name = spec['name']
            args = spec['args']

            cache_spec = dict(
                func=RustAnalysis.call_anas.__qualname__,
                module=RustAnalysis.call_anas.__module__,

                ana_name=name,
                ana_args=copy.deepcopy(args),
                trace_state=trace.trace_state,
            )
            return _CacheDataDesc(spec=cache_spec, fmt=fmt)

        def try_cache(spec):
            for fmt in ('parquet', 'json'):
                cache_desc = make_cache_desc(spec, fmt)
                try:
                    data = cache.fetch(cache_desc)
                except KeyError:
                    continue
                else:
                    return data

            raise KeyError('Could not find data in cache')

        analyses = list(analyses)
        if analyses:
            analyses = list(map(FrozenDict, analyses))

            exceps = {}
            results = {}
            not_in_cache = list()
            for spec in analyses:
                try:
                    data = try_cache(spec)
                except KeyError:
                    not_in_cache.append(spec)
                else:
                    results[spec] = data

            if not_in_cache:
                checkers, cli_specs = zip(*map(resolve, not_in_cache))
                checker = AndTraceEventChecker(checkers)
                cli_spec = json.dumps(
                    list(map(dict, cli_specs)),
                    # Turn FrozenDict into a dict
                    default=dict,
                )
                if window is None:
                    window = '"none"'
                else:
                    window = f'{{"time": [{window[0]}, {window[1]}]}}'
                window = ['--window', window]

                with TemporaryDirectory() as out_path:
                    out_path = Path(out_path)
                    cli_args = [
                        cli_spec,
                        *window,
                        '--out-path', out_path,
                    ]

                    with measure_time() as measure:
                        stdout = self._do_call_ana(event_checker=checker, cli_args=cli_args)

                    # Consider that running multiple analysis takes the same time as
                    # running just one, as the time is dominated by iterating over the
                    # events.
                    compute_cost = measure.delta
                    computed = ujson.loads(stdout)

                    for spec, data in zip(not_in_cache, computed):
                        try:
                            data = data['ok']
                        except KeyError:
                            err = data['err']

                            if isinstance(err, str) and err.startswith('invalid type'):
                                excep = TypeError(err)
                            else:
                                excep = ValueError(err)

                            exceps[spec] = excep
                        else:

                            with measure_time() as measure:
                                data, fmt = _post_process_data(
                                    data=data,
                                    normalize_time=(self._trace.basetime if trace.normalize_time else None),
                                    out_path=out_path,
                                )
                            _compute_cost = compute_cost + measure.delta

                            cache_desc = make_cache_desc(spec=spec, fmt=fmt)
                            cache.insert(cache_desc, data, compute_cost=_compute_cost, write_swap=True)
                            results[spec] = data

            return [
                (results.get(spec), exceps.get(spec))
                for spec in analyses
            ]
        else:
            return []

    def _do_call_ana(self, event_checker, cli_args):
        trace = self._trace

        def run_ana(populated, json_path):
            if populated and json_path:
                # Checking the available events on this path should be cheap
                # since the available events have been updated on the other
                # path.
                event_checker.check_events(trace.available_events)
                stdout = self._call_binary('run', json_path, *cli_args)
            else:
                stdout, available_events = self._create_json_and_call_ana(
                    events=events,
                    cli_args=cli_args,
                    json_path=json_path,
                )
                if available_events is not None:
                    # Record the available events on the Trace object so that the
                    # other path can get them cheaply, as well as any other code in
                    # LISA
                    trace._update_parseable_events({
                        event: True
                        for event in available_events
                    })
                    # Check the events after the fact. The analysis is "pure", i.e.
                    # we only care about the JSON it returns so it's ok if the
                    # result is non-sensical since we will discard it.
                    event_checker.check_events(available_events)

            return stdout

        # The _dat_json_dumper does not need an actual event list and will just
        # dump all the events in the trace. This allows sharing the same JSON
        # for every single analysis which is a big performance boost.
        if self._dump_json_lines == self._dat_dump_json:
            events = None
        else:
            events = (event_checker.get_all_events())

        return self._with_json_trace(events=events, f=run_ana)

    @TraceAnalysisBase.cache(fmt='disk-only', ignored_params=['f'])
    def _with_json_trace(self, json_path, events, f):
        if json_path is None:
            return f(False, None)
        else:
            return f(os.path.exists(json_path), json_path)

    def _create_json_and_call_ana(self, events, json_path, cli_args):
        logger = self.logger
        cmd = [_RUST_ANALYSIS_PATH, 'run', '-', *cli_args]
        pretty_cmd = ' '.join(map(lambda x: shlex.quote(str(x)), cmd))
        bufsize = 128*1024*1024

        def reader_f(f, into):
            try:
                while True:
                    time.sleep(0.05)
                    x = f.readlines()
                    if x:
                        into.extend(x)
            # Exit when encountering:
            # ValueError: I/O operation on closed file.
            except ValueError:
                pass

        logger.debug(f'Running rust analysis: {pretty_cmd}')
        stdout = []
        stderr = []
        popen = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=bufsize,
        )
        popen = _PipeSafePopen(popen)

        if json_path is None:
            @contextlib.contextmanager
            def cm():
                with popen as p:
                    yield p.stdin
        else:
            @contextlib.contextmanager
            def cm():
                with popen as p, open(json_path, 'w', buffering=bufsize) as j:
                    with _TeeFile(p.stdin, j) as tf:
                        yield tf

        with popen as p, cm() as f:

            retcode = 0
            stdout_thread = threading.Thread(target=reader_f, args=(p.stdout, stdout), daemon=True)
            stderr_thread = threading.Thread(target=reader_f, args=(p.stderr, stderr), daemon=True)
            try:
                stdout_thread.start()
                stderr_thread.start()

                available_events = self._dump_json_lines(events=events, f=f)

            # We get BrokenPipeError for stdin in case the binary stops before
            # having consuming all the input we try to feed it.
            except BrokenPipeError:
                available_events = None
            finally:
                try:
                    f.flush()
                    f.close()
                except BrokenPipeError:
                    pass
                retcode = p.wait()

                p.stdout.close()
                p.stderr.close()

                def is_alive(thread):
                    return False if thread is None else thread.is_alive()

                while not all(map(is_alive, (stdout_thread, stderr_thread))):
                    time.sleep(0.01)
                stdout = ''.join(stdout)
                stderr = ''.join(stderr)

        logger.debug(f'Rust analysis stderr for {pretty_cmd}:\n{stderr}')
        if retcode:
            raise subprocess.CalledProcessError(retcode, pretty_cmd, output=stdout, stderr=stderr)
        else:
            return (stdout, available_events)

    @property
    def _trace_window(self):
        trace = self._trace
        if isinstance(trace, TraceView):
            window = trace.window
            if trace.normalize_time:
                window = map(lambda x: x + trace.basetime, window)
            window = map(lambda x: int(x * 1e9), window)
            return tuple(window)
        else:
            return None

    @property
    def _dump_json_lines(self):
        if self.trace.trace_path.endswith('.dat'):
            return self._dat_dump_json
        else:
            return self._generic_dump_json

    def _generic_dump_json(self, events, f):
        if events is None:
            raise ValueError('Generic trace.df_event()-based JSON generation needs a set of events to dump')

        def filternan(x):
            return {
                k: v
                for k, v in x.items()
                if not pd.isna(v)
            }

        def make_ts(seconds):
            return int(seconds * 1e9)

        def df_to_records(event, df):
            df = df.copy(deep=False)
            df['__type'] = event
            df['__ts'] = (df.index * 1e9).astype('uint64')
            # Remove any NA since it cannot be serialized to JSON,
            # and means that the data is missing anyway.
            return map(filternan, df.to_dict(orient='records'))

        def event_records(trace, events):
            # Merge the stream of events coming
            # from all dataframes based on the timestamp.
            return heapq.merge(
                *(
                    df_to_records(event, trace.df_event(event))
                    for event in events
                ),
                key=operator.itemgetter('__ts'),
            )

        trace = self.trace
        data = event_records(trace, events)
        start = {
            '__type': '__lisa_event_stream_start',
            '__ts': make_ts(trace.start),
        }
        end = {
            '__type': '__lisa_event_stream_end',
            '__ts': make_ts(trace.end),
        }
        data = itertools.chain([start], data, [end])

        for item in data:
            ujson.dump(item, f, reject_bytes=False, ensure_ascii=False, indent=0)
            f.write('\n')

        # If we made it this far, we have all the events that we need
        return set(events)

    def _dat_dump_json(self, events, f):
        """
        Specialized trace.dat handling that is much faster than
        :meth:`_generic_dump_json` as it does not require parsing the
        dataframes for each event prior to creating the JSON.

        .. note:: This might yield a different result since these events will
            not undergo sanitization as they do when parsed by
            :class:`lisa.trace.Trace` into dataframes. Problematic cases are
            usually unsupported, as it is the result of using things like
            bitmask in events which is too painful to support across the
            variety of data formats. If confronted with those, register your
            own event on the tracepoint and "unroll" the event (e.g. one event
            emitted per CPU in the cpumask).
        """
        trace = self.trace

        cmd = TxtTraceParser._tracecmd_report(
            path=trace.trace_path,
            events=events,
        )

        bufsize = 10 * 1024 * 1024
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=bufsize)

        dump_record = lambda record: f.write(ujson.dumps(record, reject_bytes=False, ensure_ascii=False, indent=0) + '\n')

        with popen as p, mp_spawn_pool() as pool:
            lines = p.stdout

            # Dump the event stream start event
            first_ts = None
            last_ts = None
            for first_line in lines:
                try:
                    record = _json_record(first_line)
                except ValueError:
                    continue
                else:
                    ts = record['__ts']
                    first_ts = ts / 1e9
                    dump_record({
                        '__type': '__lisa_event_stream_start',
                        '__ts': ts,
                    })
                    dump_record(record)
                    break


            # Dump the events by streaming the output of trace-cmd report into
            # a set of workers
            available_events = set()
            cpus = set()
            record = None
            for record, txt in pool.imap(_json_line, lines, chunksize=4096):
                available_events.add(record['__type'])

                try:
                    cpus.add(record['__cpu'])
                except KeyError:
                    pass
                f.write(txt + '\n')

            # Dump the event stream end event
            if record is not None:
                ts = record['__ts']
                dump_record({
                    '__type': '__lisa_event_stream_end',
                    '__ts': ts,
                })
                last_ts = ts / 1e9


            # The first and last timestamps are only reflecting the full range
            # if we asked for all events, otherwise we may have filtered the
            # actual first and last event in the trace.
            metadata = {}
            if events is None:
                time_range = (first_ts, last_ts)
                if None not in time_range:
                    metadata['time-range'] = time_range

            if cpus:
                metadata['cpus-count'] = len(cpus)

            # Update the metadata we gathered so far, to avoid costly
            # re-computation from the normal parser (especially time-range that
            # requires scanning the whole trace).
            self._trace._cache.update_metadata(metadata)

            return {
                event.decode()
                for event in available_events
            }

    def _run_coros(self, coros):
        """
        Run coroutines in lockstep, and execute the analyses request for each
        step.
        """
        coros_list = list(coros)
        coros = set(coros_list)

        final = {}
        to_send = dict.fromkeys(coros, None)

        def fixup_desc(desc):
            desc = dict(desc)
            desc['name'] = f"crate::analysis::{desc['name']}"
            # If the analysis uses unit for the args type, serde expects None
            args = desc.get('args')

            def traverse(x):
                if isinstance(x, Mapping):
                    return FrozenDict({
                        k: traverse(v)
                        for k, v in x.items()
                    })
                elif isinstance(x, str):
                    return x
                elif isinstance(x, Iterable):
                    return list(map(traverse, x))
                else:
                    return x

            desc['args'] = traverse(args)
            return FrozenDict(desc)


        while True:
            requests = {}
            for coro, x in list(to_send.items()):
                try:
                    request = coro.send(x)
                except StopIteration as e:
                    final[coro] = e.value
                    del to_send[coro]
                except Exception as e:
                    for _coro in to_send.keys() - {coro}:
                        try:
                            _coro.throw(CanceledAnalysis)
                        except CanceledAnalysis:
                            pass

                    raise
                else:
                    requests[coro] = request

            if to_send:
                descs = {
                    fixup_desc(desc): (coro, n)
                    for coro, request in requests.items()
                    # The user request might contain duplicates, so we remember
                    # how many of each request we got.
                    for (desc, n) in Counter(map(FrozenDict, request)).items()
                }
                results = dict(zip(
                    descs.keys(),
                    self.call_anas(descs.keys())
                ))

                to_send = {}
                for desc, res in results.items():
                    request, n = descs[desc]
                    to_send[request] = [res] * n
            else:
                return tuple(
                    final[coro]
                    for coro in coros_list
                )


class CanceledAnalysis(BaseException):
    pass


async def join(*coros):
    coros_list = list(coros)
    coros = set(coros_list)
    results = {}
    xs = [None] * len(coros)
    while True:
        _coros = list(coros)
        requests = []
        for coro, x in zip(_coros, xs):
            try:
                descs = coro.send(x)
            except StopIteration as e:
                results[coro] = e.value
                coros.remove(coro)
            else:
                requests.append(_RunMany(
                    Run(**desc)
                    for desc in descs
                ))

        if requests:
            xs = await _RunMany(requests)
        else:
            return tuple(
                results[coro]
                for coro in coros_list
            )


async def concurrently(*awaitables, raise_=True):
    coros = []
    requests = []

    for x in awaitables:
        (
            coros
            if inspect.iscoroutine(x) else
            requests
        ).append(x)

    async def proxy():
        cls = _RunManyRaise if raise_ else _RunMany
        return await cls(requests)

    if not raise_:
        def wrap_coro(coro):
            async def wrapper():
                try:
                    x = await coro
                except Exception as e:
                    excep = e
                    x = None
                else:
                    excep = None

                return (x, excep)
            return wrapper()

        coros = list(map(wrap_coro, coros))

    requests_res, *coros_res = await join(proxy(), *coros)
    coros_res = list(coros_res)
    requests_res = list(requests_res)

    return [
        (
            coros_res.pop()
            if inspect.iscoroutine(x) else
            requests_res.pop()
        )
        for x in awaitables
    ]



class _RunMany:
    def __init__(self, requests):
        self.requests = list(requests)

    def __await__(self):
        def expand_request(request):
            if isinstance(request, Run):
                return (request.desc,)
            elif isinstance(request, _RunMany):
                return tuple(
                    desc
                    for sub in request.requests
                    for desc in expand_request(sub)
                )
            else:
                raise TypeError('Unknown request type')

        descs = expand_request(self)

        xs = yield descs

        def rebuild_response(request, xs):
            if isinstance(request, Run):
                x, *xs = xs
                return (x, xs)
            elif isinstance(request, _RunMany):
                res = []
                for sub in request.requests:
                    x, xs = rebuild_response(sub, xs)
                    res.append(x)
                return (res, xs)
            else:
                raise TypeError('Unknown request')

        xs, remaining = rebuild_response(self, xs)
        assert not remaining
        return xs


class _RunManyRaise(_RunMany):
    def __await__(self):
        results = yield from super().__await__()

        if results:
            xs, exceps = zip(*results)
            exceps = [
                excep
                for excep in exceps
                if excep is not None
            ]
            if exceps:
                # Choose one arbitrarily, might benefit from PEP 654 exception
                # groups
                raise exceps[0]
            else:
                return tuple(xs)
        else:
            return []


class Run:
    def __init__(self, **desc):
        self.desc = desc

    def __await__(self):
        x, excep = (yield [self.desc])[0]
        if excep is None:
            return x
        else:
            raise excep


def _get_rust_analysis_metadata():
    ana_map = ujson.loads(RustAnalysis._call_binary('list'))
    return {
        name: dict(
            event_checker=_event_checker_from_json(spec['eventreq'])
        )
        for name, spec in ana_map.items()
    }


_RUST_ANALYSIS_METADATA = _get_rust_analysis_metadata()


@optional_kwargs
def rust_analysis(f, analyses=None):
    sig = inspect.signature(f)
    analyses = analyses or []

    event_checker = doc_events(*(
        _RUST_ANALYSIS_METADATA[f'crate::analysis::{name}']['event_checker']
        for name in analyses
    ))

    @event_checker
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        coro = f(self, *args, **kwargs)
        return self.trace.ana._rust._run_coros([coro])[0]

    wrapper.asyn = f
    return wrapper

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
