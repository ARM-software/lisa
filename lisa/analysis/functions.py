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

import pandas as pd
from lisa.analysis.base import AnalysisHelpers


class FunctionsAnalysis(AnalysisHelpers):
    """
    Support for kernel functions profiling and analysis

    :param stats_path: Path to JSON function stats as returned by devlib
        :meth:`devlib.FtraceCollector.get_stats`
    :type stats_path: str
    """

    name = 'functions'

    def __init__(self, stats_path):
        self.stats_path = stats_path

        # Opening functions profiling JSON data file
        with open(self.stats_path, 'r') as f:
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
            msg = 'Metrics {} not supported, available metrics are {}'\
                .format(set(metrics) - set(available_metrics),
                        available_metrics)
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
