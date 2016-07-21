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

import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pylab as pl
import re
import sys
import trappy
import json

from trappy.utils import listify

# Configure logging
import logging

class Trace(object):

    def __init__(self, platform, data_dir, events,
                 tasks=None, window=(0,None),
                 normalize_time=True,
                 trace_format='FTrace',
                 plots_dir=None,
                 plots_prefix=''):

        # The platform used to run the experiments
        self.platform = platform

        # TRAPpy Trace object
        self.ftrace = None

        # Trace format
        self.trace_format = trace_format

        # The time window used to limit trace parsing to
        self.window = window

        # Dynamically registered TRAPpy events
        self.trappy_cls = {}

        # Maximum timespan for all collected events
        self.time_range = 0

        # Time the system was overutilzied
        self.overutilized_time = 0
        self.overutilized_prc = 0

        # The dictionary of tasks descriptors available in the dataset
        self.tasks = {}

        # List of events required by user
        self.events = []

        # List of events available in the parsed trace
        self.available_events = []

        # Cluster frequency coherency flag
        self.freq_coherency = True

        # Folder containing all trace data
        self.data_dir = None

        # Folder containing trace
        if not os.path.isdir(data_dir):
            self.data_dir = os.path.dirname(data_dir)
        else:
            self.data_dir = data_dir

        # By deafult, use the trace dir to save plots
        self.plots_dir = plots_dir
        if self.plots_dir is None:
            self.plots_dir = self.data_dir
        self.plots_prefix = plots_prefix

        self.__registerTraceEvents(events)
        self.__parseTrace(data_dir, tasks, window, normalize_time, trace_format)
        self.__computeTimeSpan()

    def __registerTraceEvents(self, events):

        if isinstance(events, basestring):
            self.events = events.split(' ')
        elif isinstance(events, list):
            self.events = events
        else:
            raise ValueError('Events must be a string or a list of strings')


    def __parseTrace(self, path, tasks, window, normalize_time, trace_format):
        logging.debug('Loading [sched] events from trace in [%s]...', path)
        logging.debug("Parsing events: %s", self.events)
        if trace_format.upper() == 'SYSTRACE' or path.endswith('html'):
            logging.info('Parsing SysTrace format...')
            trace_class = trappy.SysTrace
            self.trace_format = 'SysTrace'
        elif trace_format.upper() == 'FTRACE':
            logging.info('Parsing FTrace format...')
            trace_class = trappy.FTrace
            self.trace_format = 'FTrace'
        else:
            raise ValueError("Unknown trace format {}".format(trace_format))

        self.ftrace = trace_class(path, scope="custom", events=self.events,
                                  window=window, normalize_time=normalize_time)

        # Load Functions profiling data
        has_function_stats = self._loadFunctionsStats(path)

        # Check for events available on the parsed trace
        self.__checkAvailableEvents()
        if len(self.available_events) == 0:
            if has_function_stats:
                logging.info('Trace contains only functions stats')
                return
            raise ValueError('The trace does not contain useful events '
                             'nor function stats')

        # Setup internal data reference to interesting events/dataframes

        self._sanitize_SchedLoadAvgCpu()
        self._sanitize_SchedLoadAvgTask()
        self._sanitize_SchedCpuCapacity()
        self._sanitize_SchedBoostCpu()
        self._sanitize_SchedBoostTask()
        self._sanitize_SchedEnergyDiff()
        self._sanitize_SchedOverutilized()
        self._sanitize_CpuFrequency()

        self.__loadTasksNames(tasks)


        # Compute plot window
        if not normalize_time:
            start = self.window[0]
            if self.window[1]:
                duration = min(self.ftrace.get_duration(), self.window[1])
            else:
                duration = self.ftrace.get_duration()
            self.window = (self.ftrace.basetime + start,
                           self.ftrace.basetime + duration)

    def __checkAvailableEvents(self, key=""):
        for val in self.ftrace.get_filters(key):
            obj = getattr(self.ftrace, val)
            if len(obj.data_frame):
                self.available_events.append(val)
        logging.debug('Events found on trace:')
        for evt in self.available_events:
            logging.debug(' - %s', evt)


    def __loadTasksNames(self, tasks):
        # Try to load tasks names using one of the supported events
        if 'sched_switch' in self.available_events:
            self.getTasks(self.df('sched_switch'), tasks,
                name_key='next_comm', pid_key='next_pid')
            self._scanTasks(self.df('sched_switch'),
                            name_key='next_comm', pid_key='next_pid')
            return
        if 'sched_load_avg_task' in self.available_events:
            self.getTasks(self.df('sched_load_avg_task'), tasks)
            self._scanTasks(self.df('sched_load_avg_task'))
            return
        logging.warning('Failed to load tasks names from trace events')

    def hasEvents(self, dataset):
        if dataset in self.available_events:
            return True
        return False

    def __computeTimeSpan(self):
        # Compute time axis range, considering all the parsed events
        ts = sys.maxint
        te = 0

        for events in self.available_events:
            df = self.df(events)
            if len(df) == 0:
                continue
            if (df.index[0]) < ts:
                ts = df.index[0]
            if (df.index[-1]) > te:
                te = df.index[-1]
            self.time_range = te - ts

        logging.info('Collected events spans a %.3f [s] time interval',
                self.time_range)

        # Build a stat on trace overutilization
        if self.hasEvents('sched_overutilized'):
            df = self.df('sched_overutilized')
            self.overutilized_time = df[df.overutilized == 1].len.sum()
            self.overutilized_prc = 100. * self.overutilized_time / self.time_range

            logging.info('Overutilized time: %.6f [s] (%.3f%% of trace time)',
                    self.overutilized_time, self.overutilized_prc)

    def _loadFunctionsStats(self, path='trace.stats'):
        if os.path.isdir(path):
            path = os.path.join(path, 'trace.stats')
        if path.endswith('dat') or path.endswith('html'):
            pre, ext = os.path.splitext(path)
            path = pre + '.stats'
        if not os.path.isfile(path):
            return False

        # Opening functions profiling JSON data file
        logging.debug('Loading functions profiling data from [%s]...', path)
        with open(os.path.join(path), 'r') as fh:
            trace_stats = json.load(fh)

        # Build DataFrame of function stats
        frames = {}
        for cpu, data in trace_stats.iteritems():
            frames[int(cpu)] = pd.DataFrame.from_dict(data, orient='index')

        # Build and keep track of the DataFrame
        self._functions_stats_df = pd.concat(frames.values(), keys=frames.keys())

        return len(self._functions_stats_df) > 0

    def functions_stats_df(self, functions=None):
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
        :type functions: str or list

        """
        if not hasattr(self, '_functions_stats_df'):
            return None
        df = self._functions_stats_df
        if not functions:
            return df
        return df.loc[df.index.get_level_values(1).isin(listify(functions))]

    def _scanTasks(self, df, name_key='comm', pid_key='pid'):
        df =  df[[name_key, pid_key]]
        self._tasks_by_name = df.set_index(name_key)
        self._tasks_by_pid  = df.set_index(pid_key)

    def getTaskByName(self, name):
        if name not in self._tasks_by_name.index:
            return []
        if len(self._tasks_by_name.ix[name].values) > 1:
            return list({task[0] for task in
                         self._tasks_by_name.ix[name].values})
        return [self._tasks_by_name.ix[name].values[0]]

    def getTaskByPid(self, pid):
        if pid not in self._tasks_by_pid.index:
            return []
        if len(self._tasks_by_pid.ix[pid].values) > 1:
            return list({task[0] for task in
                         self._tasks_by_pid.ix[pid].values})
        return [self._tasks_by_pid.ix[pid].values[0]]

    def getTasks(self, dataframe=None,
            task_names=None, name_key='comm', pid_key='pid'):
        # """ Helper function to get PIDs of specified tasks
        #
        #     This method requires a Pandas dataset in input to be used to
        #     fiter out the PIDs of all the specified tasks.
        #     In a dataset is not provided, previouslt filtered PIDs are
        #     returned.
        #     If a list of task names is not provided, the workload defined
        #     task names is used instead.
        #     The specified dataframe must provide at least two columns
        #     reporting the task name and the task PID. The default values of
        #     this colums could be specified using the provided parameters.
        #
        #     :param task_names: The list of tasks to get the PID of (by default
        #                        the workload defined tasks)
        #     :param dataframe: A Pandas datafram containing at least 'pid' and
        #                       'task name' columns. If None, the previously
        #                       filtered PIDs are returned
        #     :param name_key: The name of the dataframe columns containing
        #                      task names
        #     :param pid_key:  The name of the dataframe columns containing
        #                      task PIDs
        # """
        if dataframe is None:
            return self.tasks
        df = dataframe
        if task_names is None:
            task_names = self.tasks.keys()
        logging.debug("Lookup dataset for tasks...")
        for tname in task_names:
            logging.debug("Lookup for task [%s]...", tname)
            results = df[df[name_key] == tname][[name_key,pid_key]]
            if len(results)==0:
                logging.error('  task %16s NOT found', tname)
                continue
            (name, pid) = results.head(1).values[0]
            if name!=tname:
                logging.error('  task %16s NOT found', tname)
                continue
            if (tname not in self.tasks):
                self.tasks[tname] = {}
            pids = list(results[pid_key].unique())
            self.tasks[tname]['pid'] = pids
            logging.info('  task %16s found, pid: %s',
                    tname, self.tasks[tname]['pid'])
        return self.tasks

    def df(self, event):
        """
        Return the PANDAS dataframe with the performance data for the specified
        event
        """
        if self.data_dir is None:
            raise ValueError("trace data not (yet) loaded")
        if self.ftrace and hasattr(self.ftrace, event):
            return getattr(self.ftrace, event).data_frame
        raise ValueError('Event [{}] not supported. '\
                         'Supported events are: {}'\
                         .format(event, self.available_events))

    def _sanitize_SchedCpuCapacity(self):
        # Add more columns if the energy model is available
        if not self.hasEvents('cpu_capacity') \
           or 'nrg_model' not in self.platform:
            return

        df = self.df('cpu_capacity')

        # Add column with LITTLE and big CPUs max capacities
        nrg_model = self.platform['nrg_model']
        max_lcap = nrg_model['little']['cpu']['cap_max']
        max_bcap = nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [tip_lcap], tip_bcap)

    def _sanitize_SchedLoadAvgCpu(self):
        if not self.hasEvents('sched_load_avg_cpu'):
            return
        df = self.df('sched_load_avg_cpu')
        if 'utilization' in df:
            # Convert signals name from v5.0 to v5.1 format
            df.rename(columns={'utilization':'util_avg'}, inplace=True)
            df.rename(columns={'load':'load_avg'}, inplace=True)

    def _sanitize_SchedLoadAvgTask(self):
        if not self.hasEvents('sched_load_avg_task'):
            return
        df = self.df('sched_load_avg_task')
        if 'utilization' in df:
            # Convert signals name from v5.0 to v5.1 format
            df.rename(columns={'utilization':'util_avg'}, inplace=True)
            df.rename(columns={'load':'load_avg'}, inplace=True)
            df.rename(columns={'avg_period':'period_contrib'}, inplace=True)
            df.rename(columns={'runnable_avg_sum':'load_sum'}, inplace=True)
            df.rename(columns={'running_avg_sum':'util_sum'}, inplace=True)
        df['cluster'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                ['LITTLE'], 'big')

    def _sanitize_SchedBoostCpu(self):
        if not self.hasEvents('sched_boost_cpu'):
            return
        df = self.df('sched_boost_cpu')
        if 'usage' in df:
            # Convert signals name from to v5.1 format
            df.rename(columns={'usage':'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']


    def _sanitize_SchedBoostTask(self):
        if not self.hasEvents('sched_boost_task'):
            return
        df = self.df('sched_boost_task')
        if 'utilization' in df:
            # Convert signals name from to v5.1 format
            df.rename(columns={'utilization':'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']

    def _sanitize_SchedEnergyDiff(self):
        if not self.hasEvents('sched_energy_diff') \
           or 'nrg_model' not in self.platform:
            return
        nrg_model = self.platform['nrg_model']
        em_lcluster = nrg_model['little']['cluster']
        em_bcluster = nrg_model['big']['cluster']
        em_lcpu = nrg_model['little']['cpu']
        em_bcpu = nrg_model['big']['cpu']
        lcpus = len(self.platform['clusters']['little'])
        bcpus = len(self.platform['clusters']['big'])
        SCHED_LOAD_SCALE = 1024

        power_max = em_lcpu['nrg_max'] * lcpus + em_bcpu['nrg_max'] * bcpus + \
                    em_lcluster['nrg_max'] + em_bcluster['nrg_max']
        print "Maximum estimated system energy: {0:d}".format(power_max)

        df = self.df('sched_energy_diff')
        df['nrg_diff_pct'] = SCHED_LOAD_SCALE * df.nrg_diff / power_max

        # Tag columns by usage_delta
        ccol = df.usage_delta
        df['usage_delta_group'] = np.select(
            [ccol < 150, ccol < 400, ccol < 600],
            ['< 150', '< 400', '< 600'], '>= 600')

        # Tag columns by nrg_payoff
        ccol = df.nrg_payoff
        df['nrg_payoff_group'] = np.select(
            [ccol > 2e9, ccol > 0, ccol > -2e9],
            ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject'], 'Suboptimal Reject')

    def _sanitize_SchedOverutilized(self):
        if not self.hasEvents('sched_overutilized'):
            return
        # Add a column with overutilized status duration
        df = self.df('sched_overutilized')
        df['start'] = df.index
        df['len'] = (df.start - df.start.shift()).fillna(0).shift(-1)
        df.drop('start', axis=1, inplace=True)

    def _chunker(self, seq, size):
        return (seq.iloc[pos:pos + size] for pos in range(0, len(seq), size))

    def _sanitize_CpuFrequency(self):
        if not self.hasEvents('cpu_frequency'):
            return
        # Verify that all platform reported clusters are frequency choerent
        df = self.df('cpu_frequency')
        clusters = self.platform['clusters']
        for c, cpus in clusters.iteritems():
            cluster_df = df[df.cpu.isin(cpus)]
            for chunk in self._chunker(cluster_df, len(cpus)):
                f = chunk.iloc[0].frequency
                if any(chunk.frequency != f):
                    logging.warn('Cluster Frequency is not coherent! '\
                                'Failure in [cpu_frequency] events at:')
                    logging.warn(chunk)
                    self.freq_coherency = False
                    return
        logging.info("Platform clusters verified to be Frequency choerent")

