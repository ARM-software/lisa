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

from trappy.utils import listify

from lisa.analysis.base import AnalysisBase


class FunctionsAnalysis(AnalysisBase):
    """
    Support for kernel functions profiling and analysis

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'functions'

    def __init__(self, trace):
        super(FunctionsAnalysis, self).__init__(trace)

    def plot_profiling_stats(self, functions=None, metrics='avg'):
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
        :type metrics: srt or list(str)
        """
        if not hasattr(self._trace, '_functions_stats_df'):
            self._log.warning('Functions stats data not available')
            return

        metrics = listify(metrics)
        df = self._trace.df_functions_stats(functions)

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
            data = df[metric.lower()].unstack()
            axes = data.plot(kind='bar',
                             figsize=(16, 8), legend=True,
                             title=title, table=True)
            axes.set_ylabel(ylabel)
            axes.get_xaxis().set_visible(False)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
