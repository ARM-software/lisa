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
from operator import itemgetter

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from lisa.analysis.base import TraceAnalysisBase, AnalysisHelpers
from lisa.trace import requires_events, requires_one_event_of
from lisa.conf import ConfigKeyError


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
