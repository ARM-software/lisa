# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

import os
import re
import glob

import pandas as pd

from lisa.utils import Loggable
from lisa.analysis.base import AnalysisHelpers

class PerfAnalysis(AnalysisHelpers):
    """
    Parse and analyse a set of RTApp log files

    :param task_log_map: Mapping of task names to log files
    :type task_log_map: dict

    .. note:: That is not a subclass of
        :class:`lisa.analysis.base.TraceAnalysisBase` since it does not uses traces.
    """

    RTA_LOG_PATTERN = 'rt-app-{task}.log'
    "Filename pattern matching RTApp log files"

    def __init__(self, task_log_map):
        """
        Load peformance data of an rt-app workload
        """
        logger = self.get_logger()

        if not task_log_map:
            raise ValueError('No tasks in the task log mapping')

        for task_name, logfile in task_log_map.items():
            logger.debug('rt-app task [{}] logfile: {}'.format(
                task_name, logfile
            ))

        self.perf_data = {
            task_name: {
                'logfile': logfile,
                'df': self._parse_df(logfile),
            }
            for task_name, logfile in task_log_map.items()
        }

    @classmethod
    def from_log_files(cls, rta_logs):
        """
        Build a :class:`PerfAnalysis` from a sequence of RTApp log files

        :param rta_logs: sequence of path to log files
        :type rta_logs: list(str)
        """

        def find_task_name(logfile):
            logfile = os.path.basename(logfile)
            regex = cls.RTA_LOG_PATTERN.format(task=r'(.+)-[0-9]+')
            match = re.search(regex, logfile)
            if not match:
                raise ValueError('The logfile [{}] is not from rt-app'.format(logfile))
            return match.group(1)

        task_log_map = {
            find_task_name(logfile): logfile
            for logfile in rta_logs
        }
        return cls(task_log_map)

    @classmethod
    def from_dir(cls, log_dir):
        """
        Build a :class:`PerfAnalysis` from a folder path

        :param log_dir: Folder containing RTApp log files
        :type log_dir: str
        """
        rta_logs = glob.glob(os.path.join(
            log_dir, cls.RTA_LOG_PATTERN.format(task='*'),
        ))
        return cls.from_log_files(rta_logs)

    @classmethod
    def from_task_names(cls, task_names, log_dir):
        """
        Build a :class:`PerfAnalysis` from a list of task names

        :param task_names: List of task names to look for
        :type task_names: list(str)

        :param log_dir: Folder containing RTApp log files
        :type log_dir: str
        """
        def find_log_file(task_name, log_dir):
            log_file = os.path.join(log_dir, cls.RTA_LOG_PATTERN.format(task_name))
            if not os.path.isfile(log_file):
                raise ValueError('No rt-app logfile found for task [{}]'.format(
                    task_name
                ))
            return log_file

        task_log_map = {
            task_name: find_log_file(task_name, log_dir)
            for task_name in tasks_names
        }
        return cls(task_log_map)

    @staticmethod
    def _parse_df(logfile):
        df = pd.read_csv(logfile,
                sep='\s+',
                skiprows=1,
                header=0,
                usecols=[1,2,3,4,7,8,9,10],
                names=[
                    'Cycles', 'Run' ,'Period', 'Timestamp',
                    'Slack', 'CRun', 'CPeriod', 'WKPLatency'
                ])
        # Normalize time to [s] with origin on the first event
        start_time = df['Timestamp'][0]/1e6
        df['Time'] = df['Timestamp']/1e6 - start_time
        df.set_index(['Time'], inplace=True)
        # Add performance metrics column, performance is defined as:
        #             slack
        #   perf = -------------
        #          period - run
        df['PerfIndex'] = df['Slack'] / (df['CPeriod'] - df['CRun'])

        return df

    @property
    def tasks(self):
        """
        List of tasks for which performance data have been loaded
        """
        return sorted(self.perf_data.keys())

    def get_log_file(self, task):
        """
        Return the logfile for the specified task

        :param task: Name of the task that we want the logfile of.
        :type task: str
        """
        return self.perf_data[task]['logfile']

    def get_df(self, task):
        """
        Return the pandas dataframe with the performance data for the
        specified task

        :param task: Name of the task that we want the performance dataframe of.
        :type task: str
        """
        return self.perf_data[task]['df']

    def get_default_plot_path(self, **kwargs):
        # If all logfiles are located in the same folder, use that folder
        # and the default_filename
        dirnames = {
            os.path.realpath(os.path.dirname(perf_data['logfile']))
            for perf_data in self.perf_data.values()
        }
        if len(dirnames) != 1:
            raise ValueError('A default folder cannot be inferred from logfiles location unambiguously: {}'.format(dirnames))

        default_dir = dirnames.pop()

        return super().get_default_plot_path(
            default_dir=default_dir,
            **kwargs,
        )

    @AnalysisHelpers.plot_method()
    def plot_perf(self, task, axis, local_fig):
        """
        Plot the performance Index
        """
        axis.set_title('Task [{0:s}] Performance Index'.format(task))
        data = self.get_df(task)[['PerfIndex',]]
        data.plot(ax=axis, drawstyle='steps-post')
        axis.set_ylim(0, 2)


    @AnalysisHelpers.plot_method()
    def plot_latency(self, task, axis, local_fig):
        """
        Plot the Latency/Slack and Performance data for the specified task.
        """
        axis.set_title('Task [{0:s}] (start) Latency and (completion) Slack'\
                .format(task))
        data = self.get_df(task)[['Slack', 'WKPLatency']]
        data.plot(ax=axis, drawstyle='steps-post')

    @AnalysisHelpers.plot_method()
    def plot_slack_histogram(self, task, axis, local_fig, bins=30):
        """
        Plot the slack histogram.

        :param task: rt-app task name to plot
        :type task: str

        :param bins: number of bins for the histogram.
        :type bins: int

        .. seealso:: :meth:`plot_perf_index_histogram`
        """
        ylabel = 'slack of "{}"'.format(task)
        series = self.get_df(task)['Slack']
        series.hist(bins=bins, ax=axis, alpha=0.4, label=ylabel)
        axis.axvline(series.mean(), linestyle='--', linewidth=2, label='mean')
        axis.legend()

        if local_fig:
            axis.set_title(ylabel)

    @AnalysisHelpers.plot_method()
    def plot_perf_index_histogram(self, task, axis, local_fig, bins=30):
        r"""
        Plot the perf index histogram.

        :param task: rt-app task name to plot
        :type task: str

        :param bins: number of bins for the histogram.
        :type bins: int

        The perf index is defined as:

        .. math::

            perfIndex = \frac{slack}{period - runtime}

        """
        ylabel = 'perf index of "{}"'.format(task)
        series = self.get_df(task)['PerfIndex']
        mean = series.mean()
        self.get_logger().info('perf index of task "{}": avg={:.2f} std={:.2f}'.format(
            task, mean, series.std()))

        series.hist(bins=bins, ax=axis, alpha=0.4, label=ylabel)
        axis.axvline(mean, linestyle='--', linewidth=2, label='mean')
        axis.legend()

        if local_fig:
            axis.set_title(ylabel)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
