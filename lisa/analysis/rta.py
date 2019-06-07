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

from lisa.utils import memoized

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
            task_name: find_log_file(task, log_dir)
            for task_name in tasks
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

    @memoized
    def get_df(self, task, start_time=None):
        """
        Return the pandas dataframe with the performance data for the
        specified task. A start time can be specified as a reference for the
        first event, for example to align events to a given (see :class:`TraceView`).

        :param task: Name of the task that we want the performance dataframe of.
        :type task: str

        :param start_time: The first event time in seconds
        :type start_time: float
        """
        task_df = self.perf_data[task]['df']
        if not start_time:
            return task_df

        # Let's keep a copy so that we can still access the original one
        task_df = task_df.reset_index()
        task_df['Time'] = task_df['Time'] + start_time

        return task_df.set_index('Time')

    def save_plot(self, figure, filepath=None, img_format=None):
        # If all logfiles are located in the same folder, use that folder
        # and the default_filename
        dirnames = {
            os.path.realpath(os.path.dirname(perf_data['logfile']))
            for perf_data in self.perf_data.values()
        }
        if not filepath and len(dirnames) != 1:
            raise ValueError('Please specify filepath or axis, since a default folder cannot be inferred from logfiles location')

        default_dir = dirnames.pop()
        return self._save_plot(figure, default_dir, filepath, img_format)

    @memoized
    def df_activations(self, task, start_time=None):
        """
        Get activation events

        :param task: Name of the task that we want the performance dataframe of.
        :type task: str

        :param start_time: The first activation time in seconds
        :type start_time: float

        :returns: A :class:`pandas.DataFrame` with index representing the start
        time of an activation and these columns:

        * ``Run``: the running time of the activation
        * ``End``: the end time of the activation
        * ``Slack``: the slack of the activation
        * ``WKPLatency``: the wakeup latency of the activation
        * ``PerfIndex``: the performance index of the activation
        """
        activations_df = self.get_df(task, start_time)[[
            'Run', 'Slack', 'WKPLatency', 'PerfIndex']].copy()
        activations_df['End'] = activations_df.index + activations_df['Run'] / 1e6

        # Reorder to keep "End" as the second column
        return activations_df[['Run', 'End', 'Slack', 'WKPLatency', 'PerfIndex']]

    def plot_activations(self, task, start_time=None, filepath=None, axis=None):
        """
        Draw the task's activations colored bands

        :param task: Name of the task that we want the performance dataframe of.
        :type task: str

        :param start_time: The first event time in seconds
        :type start_time: float

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        activations_df = self.df_activations(task, start_time)

        # Compute intervals in which the task was running
        bands = [(t, activations_df['End'][t]) for t in activations_df.index]

        def plotter(axis, local_fig):
            label = 'Activations'
            for (start, end) in bands:
                axis.axvspan(start, end, alpha=0.1, facecolor='r', label=label)
                if label:
                    label = None

            axis.legend()

            if local_fig:
                axis.set_title("Task [{}] activations".format(task))

        return self.do_plot(plotter, filepath, axis)

    @memoized
    def df_phases(self, task, start_time=None):
        """
        Get phases actual start times and durations

        :param task: Name of the task that we want the phases dataframe of
        :type task: def __str__(self):

        :param start_time: The first activation time in seconds
        :type start_time: float

        :returns: A :class:`pandas.DataFrame` with index representing the
        start time of a phase and these column:

        * ``Duration``: the measured phase duration.
        * ``CPeriod``: the configured activation periods for the phase
        * ``CRun``: the configured activation run time for the phase

        """
        activation_df = self.get_df(task, start_time)
        phase_df = activation_df[['CRun', 'CPeriod']].copy()

        # Shift Run and Periods for phase changes detection
        phase_df['PRun'] = phase_df['CRun'].shift(1)
        phase_df['PPeriod'] = phase_df['CPeriod'].shift(1)

        # Keep only new phase start events
        phase_df = phase_df[(phase_df.CRun != phase_df.PRun) |
                            (phase_df.CPeriod != phase_df.PPeriod)]

        # Compute duration of each phase, last one requires a "special" treatment
        durations = list(phase_df.index[1:] - phase_df.index[:-1])
        last_phase_duration = activation_df.index[-1] - phase_df.index[-1] + \
                              activation_df.iloc[-1].Period / 1e6

        # Compute phase durations
        phase_df.loc[:,'Duration'] = durations + [last_phase_duration]

        return phase_df[['Duration', 'CPeriod', 'CRun']]

    def plot_phases(self, task, start_time=None, filepath=None, axis=None):
        """
        Draw the task's phases colored bands

        :param task: Name of the task that we want the performance dataframe of.
        :type task: str

        :param start_time: The first event time in seconds
        :type start_time: float

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        phases_df = self.df_phases(task, start_time)

        # Compute phases intervals
        bands = [(t, t + phases_df['Duration'][t]) for t in phases_df.index]

        def plotter(axis, local_fig):
            for idx, (start, end) in enumerate(bands):
                color = self.get_next_color(axis)
                label = 'Phase_{}:{}'.format(int(phases_df.iloc[idx].CRun /1e3),
                                             int(phases_df.iloc[idx].CPeriod / 1e3))
                axis.axvspan(start, end, alpha=0.1, facecolor=color, label=label)

            axis.legend()

            if local_fig:
                axis.set_title("Task [{}] phases".format(task))

        return self.do_plot(plotter, filepath, axis)

    def plot_perf(self, task, **kwargs):
        """
        Plot the performance Index

        :param filepath: If no axis is specified, the figure will be saved to
            that path

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        def plotter(axis, local_fig):
            axis.set_title('Task [{0:s}] Performance Index'.format(task))
            data = self.get_df(task)[['PerfIndex',]]
            data.plot(ax=axis, drawstyle='steps-post')
            axis.set_ylim(0, 2)

        return self.do_plot(plotter, **kwargs)

    def plot_latency(self, task, **kwargs):
        """
        Plot the Latency/Slack and Performance data for the specified task.

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        def plotter(axis, local_fig):
            axis.set_title('Task [{0:s}] (start) Latency and (completion) Slack'\
                    .format(task))
            data = self.get_df(task)[['Slack', 'WKPLatency']]
            data.plot(ax=axis, drawstyle='steps-post')

        return self.do_plot(plotter, **kwargs)

    def plot_slack_histogram(self, task, **kwargs):
        """
        Plot the Slack Histogram

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`

        """
        def plotter(axis, local_fig):
            data = self.get_df(task)[['PerfIndex',]]
            data.hist(bins=30, ax=axis, alpha=0.4)
            pindex_avg = data.mean()[0]
            pindex_std = data.std()[0]
            self.get_logger().info('PerfIndex, Task [%s] avg: %.2f, std: %.2f',
                    task, pindex_avg, pindex_std)
            axis.axvline(pindex_avg, linestyle='--', linewidth=2)

        return self.do_plot(plotter, **kwargs)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
