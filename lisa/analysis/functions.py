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

""" Functions Analysis Module """
import json
import os
from operator import itemgetter, attrgetter
from statistics import mean
from functools import reduce
from itertools import chain
from copy import copy
from collections.abc import Mapping
from enum import IntEnum

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from lisa.utils import groupby, memoized, FrozenDict, unzip_into
from lisa.datautils import df_merge
from lisa.analysis.base import TraceAnalysisBase, AnalysisHelpers
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.trace import requires_events, requires_one_event_of, MissingTraceEventError
from lisa.conf import ConfigKeyError
from lisa.stats import Stats
from lisa.pelt import PELT_SCALE


class FunctionsAnalysis(TraceAnalysisBase):
    """
    Support for ftrace events-based kernel functions profiling and analysis
    """

    name = 'functions'

    def df_resolve_ksym(self, df, addr_col, name_col='func_name', addr_map=None, exact=True):
        """
        Resolve the kernel function names.

        .. note:: If the ``addr_col`` is not of a numeric dtype, it will be
            assumed to be function names already and the content will be copied
            to ``name_col``.

        :param df: Dataframe to augment
        :type df: pandas.DataFrame

        :param addr_col: Name of the column containing a kernel address.
        :type addr_col: str

        :param name_col: Name of the column to create with symbol names
        :param name_col: str

        :param addr_map: If provided, the mapping of kernel addresses to symbol
            names. If missing, the symbols addresses from the
            :class:`lisa.platforms.platinfo.PlatformInfo` attached to the trace
            will be used.
        :type addr_map: dict(int, str)

        :param exact: If ``True``, an exact symbol address is expected. If
            ``False``, symbol addresses are sorted and paired to form
            intervals, which are then used to infer the name. This is suited to
            resolve an instruction pointer that could point anywhere inside of
            a function (but before the starting address of the next function).
        :type exact: bool
        """
        trace = self.trace
        df = df.copy(deep=False)

        # Names already resolved, we can just copy the address column to the
        # name one
        if not is_numeric_dtype(df[addr_col].dtype):
            df[name_col] = df[addr_col]
            return df

        if addr_map is None:
            addr_map = trace.plat_info['kernel']['symbols-address']

        if exact:
            df[name_col] = df[addr_col].map(addr_map)
        # Not exact means the function addresses will be used as ranges, so
        # we can find in which function any instruction point value is
        else:
            # Sort by address, so that each consecutive pair of address
            # constitue a range of address belonging to a given function.
            addr_list = sorted(
                addr_map.items(),
                key=itemgetter(0)
            )
            bins, labels = zip(*addr_list)
            # "close" the last bucket with the highest value possible of that column
            max_addr = np.iinfo(df[addr_col].dtype).max
            bins = list(bins) + [max_addr]
            name_i = pd.cut(
                df[addr_col],
                bins=bins,
                # Since our labels are not unique, we cannot pass it here
                # directly. Instead, use an index into the labels list
                labels=range(len(labels)),
                # Include the left boundary and exclude the right one
                include_lowest=True,
                right=False,
            )
            df[name_col] = name_i.apply(lambda x: labels[x])

        return df

    def _df_with_ksym(self, event, *args, **kwargs):
        df = self.trace.df_event(event)
        try:
            return self.df_resolve_ksym(df, *args, **kwargs)
        except ConfigKeyError:
            self.get_logger().warning(f'Missing symbol addresses, function names will not be resolved: {e}')
            return df

    @requires_one_event_of('funcgraph_entry', 'funcgraph_exit')
    @TraceAnalysisBase.cache
    def df_funcgraph(self, event):
        """
        Return augmented dataframe of the event with the following column:

            * ``func_name``: Name of the calling function if it could be
              resolved.

        :param event: One of:

            * ``entry`` (``funcgraph_entry`` event)
            * ``exit`` (``funcgraph_exit`` event)
        :type event: str
        """
        event = f'funcgraph_{event}'
        return self._df_with_ksym(event, 'func', 'func_name', exact=False)

    @df_funcgraph.used_events
    @LoadTrackingAnalysis.df_cpus_signal.used_events
    def _get_callgraph(self, tag_df=None, thread_root_functions=None):
        entry_df = self.df_funcgraph(event='entry').copy(deep=False)
        entry_df['event'] = _CallGraph._EVENT.ENTRY
        exit_df = self.df_funcgraph(event='exit').copy(deep=False)
        exit_df['event'] = _CallGraph._EVENT.EXIT

        # Attempt to get the CPU capacity signal to normalize the results
        capacity_cols = ['__cpu', 'event', 'capacity']
        try:
            capacity_df = self.trace.analysis.load_tracking.df_cpus_signal('capacity')
        except MissingTraceEventError:
            capacity_df = pd.DataFrame(columns=capacity_cols)
        else:
            capacity_df = capacity_df.copy(deep=False)
            capacity_df['__cpu'] = capacity_df['cpu']
            capacity_df['event'] = _CallGraph._EVENT.SET_CAPACITY
            capacity_df = capacity_df[capacity_cols]

        # Set a reasonable initial capacity
        try:
            orig_capacities = self.trace.plat_info['cpu-capacities']['orig']
        except KeyError:
            pass
        else:
            orig_capacities_df = pd.DataFrame.from_records(
                (
                    (-1 * cpu, cpu, _CallGraph._EVENT.SET_CAPACITY, cap)
                    for cpu, cap in orig_capacities.items()
                ),
                columns=['Time', '__cpu', 'event', 'capacity'],
                index='Time',
            )
            capacity_df = pd.concat((orig_capacities_df, capacity_df))

        to_merge = [entry_df, exit_df, capacity_df]

        if tag_df is not None:
            cpu = tag_df['__cpu']
            tag_df = tag_df.drop(columns=['__cpu'])
            tag_df = pd.DataFrame(dict(
                tags=tag_df.apply(pd.Series.to_dict, axis=1),
                __cpu=cpu,
            ))
            tag_df['event'] = _CallGraph._EVENT.SET_TAG
            to_merge.append(tag_df)

        df = df_merge(to_merge)
        return _CallGraph.from_df(
            df,
            thread_root_functions=thread_root_functions
        )

    @_get_callgraph.used_events
    def df_calls(self, tag_df=None, thread_root_functions=None, normalize=True):
        """
        Return a :class:`pandas.DataFrame` with a row for each function call,
        along some metrics:

            * ``cum_time``: cumulative time spent in that function. This
                includes the time spent in all children too.
            * ``self_time``: time spent in that function only. This
                excludes the time spent in all children.

        :param tag_df: Dataframe containing the tag event, which is used to tag
            paths in the callgraph. The ``__cpu`` column is mandatory in order
            to know which CPU is to be tagged at any index. Other colunms will
            be used as tag keys. Tags are inherited from from both parents and
            children. This allows a leaf function to emit an event and use it
            for the whole path that lead to there. Equally, if a function emits
            a tag, all the children of this call will inherit the tag too. This
            allows a top-level function to tag a whole subtree at once.
        :type tag_df: pandas.DataFrame

        :param thread_root_functions: Functions that are considered to be a
            root of threads. When they appear in the callgraph, the profiler
            will consider the current function to be preempted and will not
            register the call as a child of it and will avoid to count it in
            the cumulative time.
        :type thread_root_functions: list(str) or None

        :param normalize: Normalize metrics according to the current CPU
            capacity so that they appear to have run on the fastest CPU at
            maximum frequency. This allows merging calls regardless of their
            origin (CPU and frequency).

            .. note:: Normalization only currently takes into account the
                capacity of the CPU when the function is entered. If it changes
                during execution, the result will be somewhat wrong.
        :type normalize: bool

        .. note:: Calls during which the current function name changes are not
            accounted for. They are typically a sign of functions that did not
            properly return, for example functions triggering a context switch
            and returning to userspace.
        """
        graph = self._get_callgraph(
            tag_df=tag_df,
            thread_root_functions=thread_root_functions,
        )
        metrics = _CallGraphNode._METRICS

        def get_metric(node, metric):
            val = node[metric]
            if normalize:
                return (node.cpu_capacity / PELT_SCALE) * val
            else:
                return val

        return pd.DataFrame.from_records(
            (
                (
                    node.entry_time, node.cpu, node.func_name, FrozenDict(node.tags), node.tagged_name,
                    *(
                        get_metric(node, metric)
                        for metric in metrics
                    )
                )
                for node in graph.all_nodes
            ),
            columns=['Time', 'cpu', 'function', 'tags', 'tagged_name'] + metrics,
            index='Time',
        )

    def compare_with_traces(self, others, normalize=True, **kwargs):
        """
        Compare the :class:`~lisa.trace.Trace` it's called on with the other
        traces passed as ``others``. The reference is the trace it's called on.

        :returns: a :class:`lisa.stats.Stats` object just like
            :meth:`profile_stats`.

        :param others: List of traces to compare against.
        :type others: list(lisa.trace.Trace)

        :Variable keyword arguments: Forwarded to :meth:`profile_stats`.
        """
        ref = self.trace
        traces = [ref] + list(others)
        paths = [
            trace.trace_path
            for trace in traces
        ]
        common_prefix_len = len(os.path.commonprefix(paths))
        common_suffix_len = len(os.path.commonprefix(list(map(lambda x: str(reversed(x)), paths))))

        def get_name(trace):
            name = trace.trace_path[common_prefix_len:common_suffix_len]
            if not name:
                if trace is ref:
                    name = 'ref'
                else:
                    name = str(traces.index(trace))
            return name

        def get_df(trace):
            df = trace.analysis.functions.df_calls(normalize=normalize)
            df = df.copy(deep=False)
            df['trace'] = get_name(trace)
            return df

        df = df_merge(map(get_df, traces))
        ref_group = {
            'trace': get_name(ref)
        }
        return self._profile_stats_from_df(df, ref_group=ref_group, **kwargs)

    @df_calls.used_events
    def profile_stats(self, tag_df=None, normalize=True, ref_function=None, ref_tags=None, **kwargs):
        """
        Create a :class:`lisa.stats.Stats` out of profiling information of the
        trace.

        :param tag_df: Dataframe of tags, forwarded to :meth:`df_calls`
        :type tag_df: pandas.DataFrame or None

        :param normalize: Normalize execution time according to CPU capacity,
            forwarded to to :meth:`df_calls`
        :type normalize: bool

        :param metric: Name of the metric to use for statistics. Can be one of:

            * ``self_time``: Time spent in the function, not accounting for
              time spent in children
            * ``cum_time``: Total time spent in the function, including the
              time spent in children.

            Defaults to ``self_time``.
        :type metric: str

        :param functions: Restrict the statistics to the given list of
            function.
        :type functions: list(str) or None

        :param ref_function: Function to compare to.
        :type ref_function: str or None

        :param ref_tags: Function tags to compare to. Ignored if ``ref_function
            is None``.
        :type ref_tags: dict(str, set(object)) or None

        :param cpus: List of CPUs where the functions were called to take into
            account. If left to ``None``, all CPUs are considered.
        :type cpus: list(int) or None

        :param per_cpu: If ``True``, the per-function statistics are separated
            for each CPU they ran on. This is useful if the frequency was fixed and
            the only variation in speed was coming from the CPU it ran on.
        :type per_cpu: bool or None

        :param tags: Restrict the statistics to the function tagged with the
            given tag values. If a function has multiple values for a given tag
            and one of the value is in ``tags``, the function is selected.
        :type tags: dict(str, object)

        :Variable keyword arguments: Forwarded to :class:`lisa.stats.Stats`.

        .. note:: Recursive calls are treated as if they were inlined in their
            callers. This means that the count of calls will be counting the
            toplevel calls only, and that the ``self_time`` for a recursive
            function is directly linked to how much time each level consumes
            multiplied by the number of levels. ``cum_time`` will also be
            tracked on the top-level call only to provide a more accurate
            result.
        """
        df = self.df_calls(tag_df=tag_df, normalize=normalize)
        if ref_function:
            ref_tags = ref_tags or {}
            ref_group = {
                'f': _CallGraphNode.format_name(ref_function, ref_tags)
            }
        else:
            ref_group = None

        return self._profile_stats_from_df(df, ref_group=ref_group, **kwargs)

    @staticmethod
    def _profile_stats_from_df(df, metric='self_time', functions=None, per_cpu=True, cpus=None, tags=None, **kwargs):
        metrics = _CallGraphNode._METRICS
        # Get rid of the other value columns to avoid treating them as
        # tags
        other_metrics = set(metrics) - {metric}

        if functions:
            df = df[df['function'].isin(functions)]

        if cpus is not None:
            df = df[df['cpu'].isin(cpus)]

        if tags:
            # Select all rows that are a subset of the given tags
            def select_tag(row_tags):
                return all(
                    val in row_tags.get(tag, [])
                    for tag, val in tags.items()
                )
            df = df[df['tags'].apply(select_tag)]

        df = df.copy(deep=False)

        # Use tagged_name for display
        df['f'] = df['tagged_name']

        to_drop = list(other_metrics) + ['tags', 'function', 'tagged_name']
        # Calls are already uniquely identified by their timestamp, so grouping
        # per CPU is optional
        if not per_cpu:
            to_drop.append('cpu')
        df = df.drop(columns=to_drop)
        df['unit'] = 's'

        index_name = df.index.name
        df = df.reset_index()

        return Stats(
            df,
            agg_cols=[index_name],
            value_col=metric,
            **kwargs,
        )


class _CallGraph:
    class _EVENT(IntEnum):
        """
        To be used as events for the dataframe passed to
        :meth:`from_df`.
        """
        ENTRY = 1
        """Enter the given function"""
        EXIT = 2
        """Exit the given function"""
        SET_TAG = 3
        """
        Tag the current call graph path (parents and
        children) with the given value
        """
        SET_CAPACITY = 4
        """
        Set the capacity of the current CPU. Values are between 0 and
        :attr:`lisa.pelt.PELT_SCALE`.
        """

    def __init__(self, cpu_nodes):
        self.cpu_nodes = cpu_nodes

    @property
    def all_nodes(self):
        return chain.from_iterable(
            node.indirect_children
            for node in self.cpu_nodes.values()
        )

    @classmethod
    def from_df(cls, df, thread_root_functions=None, ts_cols=('calltime', 'rettime')):
        """
        Build a :class:`_CallGraph` from a :class:`pandas.DataFrame` with the
        following columns:

            * ``event``: One of :class:`_CallGraph._EVENT` enumeration.
            * ``func_name``: Name of the function for ``entry`` and ``exit``
              events.
            * ``tags``: ``dict(str, object)`` of tags for ``tag`` event.

        :param thread_root_functions: Functions that are considered to be a
            root of threads. When they appear in the callgraph, the profiler
            will consider the current function to be preempted and will not
            register the call as a child of it and will avoid to count it in
            the cumulative time.
        :type thread_root_functions: list(str) or None

        :param ts_cols: Name of the columns for the
            :attr:`_CallGraph._EVENT.EXIT` rows that contain timestamps for
            entry and exit. If they are provided, they will be used instead of
            the index.
        :type ts_cols: tuple(str) or None
        """
        thread_root_functions = set(thread_root_functions) if thread_root_functions else set()

        def make_visitor():
            _max_thread = -1
            def make_thread():
                nonlocal _max_thread
                _max_thread += 1
                return _max_thread

            root_node = _CallGraphNode(
                func_name=None,
                parent=None,
                cpu=None,
                cpu_capacity=None,
                logical_thread=make_thread(),
            )
            curr_node = root_node
            # This is expected to be overriden right away by a SET_CAPACITY
            # event
            curr_capacity = PELT_SCALE
            event_enum = cls._EVENT

            def visit(row):
                nonlocal curr_node, curr_capacity
                curr_event = row['event']

                if curr_event == event_enum.ENTRY:
                    func_name = row['func_name']
                    cpu = row['__cpu']

                    # If we got preempted by a function that is considered to
                    # be part of different logical thread (e.g. the toplevel
                    # function of an ISR), create a new ID
                    if func_name in thread_root_functions:
                        logical_thread = make_thread()
                    # Otherwise, just inherit it from the parent
                    else:
                        logical_thread = curr_node.logical_thread

                    child = _CallGraphNode(
                        func_name=func_name,
                        cpu=cpu,
                        parent=curr_node,
                        cpu_capacity=curr_capacity,
                        entry_time=row.name,
                        logical_thread=logical_thread,
                    )
                    curr_node._children.append(child)
                    curr_node = child

                elif curr_event == event_enum.EXIT:
                    # We are trying to exit the root, which is probably the sign of
                    # a missing entry event (could have been cropped out of the
                    # trace). We therefore just ignore it.
                    if curr_node is not root_node:
                        # That node is unusable for stats, since the function
                        # used to enter the call is not the same one as for the
                        # exit. This usually means that the kernel returned to
                        # userspace in between.
                        if row['func_name'] != curr_node.func_name:
                            curr_node.valid_metrics = False

                        if ts_cols is None:
                            curr_node.exit_time = row.name
                        else:
                            entry_ts, exit_ts = ts_cols
                            curr_node.entry_time = row[entry_ts] * 1e-9
                            curr_node.exit_time = row[exit_ts] * 1e-9

                        curr_node = curr_node.parent

                elif curr_event == event_enum.SET_TAG:
                    tags = row['tags']
                    curr_node.set_tags(tags)
                elif curr_event == event_enum.SET_CAPACITY:
                    curr_capacity = row['capacity']
                else:
                    raise ValueError(f'Unknown event "{curr_event}"')

            def finalize(df):
                # Fixup the exit time if there were missing exit events
                if curr_node is not root_node:
                    last_time = df.index[-1]
                    for node in chain([curr_node], curr_node.parents):
                        node.exit_time = last_time
                        node.valid_metrics = False

                root_children = root_node.children
                if root_children:
                    root_node.entry_time = min(map(attrgetter('entry_time'), root_children))
                    root_node.exit_time = max(map(attrgetter('exit_time'), root_children))
                else:
                    root_node.entry_time = 0
                    root_node.exit_time = 0

            return (root_node, visit, finalize)

        def build_graph(subdf):
            root_node, visitor, finalizer = make_visitor()
            subdf.apply(visitor, axis=1)
            finalizer(subdf)
            return root_node

        return cls(
            cpu_nodes = {
                cpu: build_graph(subdf)
                for cpu, subdf in df.groupby('__cpu', observed=True)
            }
        )



class _CallGraphNode(Mapping):
    """
    Represent a function call extracted from some profiling information.
    """
    __slots__ = [
        'func_name',
        'cpu',
        'cpu_capacity',
        '_tags',
        '_children',
        'parent',
        'logical_thread',
        'entry_time',
        'exit_time',
        'valid_metrics',
        '__weakref__',
    ]

    _METRICS = sorted((
        'cum_time',
        'self_time',
    ))

    def __init__(self, func_name, parent, logical_thread, cpu, cpu_capacity, entry_time=None, exit_time=None, valid_metrics=True):
        self.func_name = func_name
        self.cpu = cpu
        self.cpu_capacity = cpu_capacity
        self.parent = parent
        self.logical_thread = logical_thread
        self._children = []
        self._tags = {}
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.valid_metrics = valid_metrics

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    @memoized
    def _expanded_children(self):
        def visit(node):
            children = node._children

            children_visit = map(visit, children)
            is_recursive, children_expansion = unzip_into(2, children_visit)

            # Check if we are part of any recursive chain
            is_recursive = any(is_recursive) or node.func_name == self.func_name

            # If we are part of a recursion chain, expand all of our children
            # so that they are reparented into our caller
            if is_recursive:
                expansion = list(chain.from_iterable(children_expansion))
            else:
                expansion = [node]

            return (is_recursive, expansion)

        return visit(self)[1]

    @property
    @memoized
    def children(self):
        return [
            child
            for child in self._expanded_children
            if not self._is_preempted_by(child)
        ]

    @property
    def _preempting_children(self):
        return {
            child
            for child in self._expanded_children
            if self._is_preempted_by(child)
        }

    def _is_preempted_by(self, node):
        return self.logical_thread != node.logical_thread

    def _str(self, idt):
        idt_str = idt * '    '

        if self.children:
            children = ':\n' + '\n'.join(child._str(idt + 1) for child in self.children)
        else:
            children = ''

        return f'{idt_str}{self.func_name}, self={self["self_time"]}s cum={self["cum_time"]}s tags={self.tags}{children}'

    def __str__(self):
        return self._str(0)

    @property
    def tagged_name(self):
        return self.format_name(self.func_name, self.tags)

    @staticmethod
    def format_name(func_name, tags):
        tags = tags or {}
        tags = ', '.join(
            f'{tag}={"|".join(map(str, vals))}'
            for tag, vals in sorted(tags.items())
        )
        tags = f' ({tags})' if tags else ''
        return f'{func_name}{tags}'

    @memoized
    def __getitem__(self, key):
        if not self.valid_metrics:
            return np.NaN

        delta = self.exit_time - self.entry_time

        if key == 'self_time':
            return delta - sum(
                node.exit_time - node.entry_time
                # Substract the time spent in all the children, including the
                # ones that preempted us
                for node in self._expanded_children
            )
        elif key == 'cum_time':
            # Define cum_time in terms of self_time, so that preempting
            # children are properly accounted for recurisvely
            return self['self_time'] + sum(
                node['cum_time']
                for node in self.children
            )
        else:
            raise KeyError(f'Unknown metric "{key}"')

    def __iter__(self):
        return iter(self._METRICS)

    def __len__(self):
        return len(self._METRICS)

    @property
    def _inherited_tags(self):
        def merge_tags(tags1, tags2):
            common_keys = tags1.keys() & tags2.keys()
            new = {
                tag: tags1[tag] | tags2[tag]
                for tag in common_keys
            }
            for tags in (tags1, tags2):
                new.update({
                    tag: tags[tag]
                    for tag in tags.keys() - common_keys
                })
            return new

        # Since merge_tags() is commutative (merge_tags(a, b) == merge_tags(b,
        # a)), we don't need any specific ordering on the parents
        nodes = chain(self.parents, self.indirect_children)
        tags = reduce(merge_tags, map(attrgetter('_tags'), nodes), {})

        return tags

    @property
    @memoized
    def tags(self):
        return dict(
            (key, frozenset(vals))
            for key, vals in {
                **self._inherited_tags,
                **self._tags,
            }.items()
        )

    @property
    def parents(self):
        parent = self.parent
        if parent is not None:
            yield parent
            yield from parent.parents

    @property
    def indirect_children(self):
        for child in self.children:
            yield child
            yield from child.indirect_children

    def set_tags(self, tags):
        for tag, val in tags.items():
            self._tags.setdefault(tag, set()).add(val)


class JSONStatsFunctionsAnalysis(AnalysisHelpers):
    """
    Support for kernel functions profiling and analysis

    :param stats_path: Path to JSON function stats as returned by devlib
        :meth:`devlib.collector.ftrace.FtraceCollector.get_stats`
    :type stats_path: str
    """

    name = 'functions_json'

    def __init__(self, stats_path):
        self.stats_path = stats_path

        # Opening functions profiling JSON data file
        with open(self.stats_path) as f:
            stats = json.load(f)

        # Build DataFrame of function stats
        frames = {}
        for cpu, data in stats.items():
            frames[int(cpu)] = pd.DataFrame.from_dict(data, orient='index')

        # Build and keep track of the DataFrame
        self._df = pd.concat(list(frames.values()),
                             keys=list(frames.keys()))

    def get_default_plot_path(self, **kwargs):
        return super().get_default_plot_path(
            default_dir=os.path.dirname(self.stats_path),
            **kwargs,
        )

    def df_functions_stats(self, functions=None):
        """
        Get a DataFrame of specified kernel functions profile data

        For each profiled function a DataFrame is returned which reports stats
        on kernel functions execution time. The reported stats are per-CPU and
        includes: number of times the function has been executed (hits),
        average execution time (avg), overall execution time (time) and samples
        variance (s_2).
        By default returns a DataFrame of all the functions profiled.

        :param functions: the name of the function or a list of function names
                          to report
        :type functions: list(str)
        """
        df = self._df
        if functions:
            return df.loc[df.index.get_level_values(1).isin(functions)]
        else:
            return df

    @AnalysisHelpers.plot_method()
    def plot_profiling_stats(self, functions: str=None, axis=None, local_fig=None, metrics: str='avg'):
        """
        Plot functions profiling metrics for the specified kernel functions.

        For each speficied metric a barplot is generated which report the value
        of the metric when the kernel function has been executed on each CPU.
        By default all the kernel functions are plotted.

        :param functions: the name of list of name of kernel functions to plot
        :type functions: str or list(str)

        :param metrics: the metrics to plot
                        avg   - average execution time
                        time  - total execution time
        :type metrics: list(str)
        """
        df = self.df_functions_stats(functions)

        # Check that all the required metrics are acutally availabe
        available_metrics = df.columns.tolist()
        if not set(metrics).issubset(set(available_metrics)):
            msg = f'Metrics {(set(metrics) - set(available_metrics))} not supported, available metrics are {available_metrics}'
            raise ValueError(msg)

        for metric in metrics:
            if metric.upper() == 'AVG':
                title = 'Average Completion Time per CPUs'
                ylabel = 'Completion Time [us]'
            if metric.upper() == 'TIME':
                title = 'Total Execution Time per CPUs'
                ylabel = 'Execution Time [us]'
            data = df[metric.casefold()].unstack()
            data.plot(kind='bar',
                     ax=axis, figsize=(16, 8), legend=True,
                     title=title, table=True)
            axis.set_ylabel(ylabel)
            axis.get_xaxis().set_visible(False)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
