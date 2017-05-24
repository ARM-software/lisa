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

import numpy as np
import os
import pandas as pd
import sys
import trappy
import json
import warnings
import operator
import logging

from analysis_register import AnalysisRegister
from collections import namedtuple
from devlib.utils.misc import memoized
from trappy.utils import listify


NON_IDLE_STATE = -1
ResidencyTime = namedtuple('ResidencyTime', ['total', 'active'])
ResidencyData = namedtuple('ResidencyData', ['label', 'residency'])

class Trace(object):
    """
    The Trace object is the LISA trace events parser.

    :param platform: a dictionary containing information about the target
        platform
    :type platform: dict

    :param data_dir: folder containing all trace data
    :type data_dir: str

    :param events: events to be parsed (everything in the trace by default)
    :type events: list(str)

    :param tasks: filter data for the specified tasks only. If None (default),
        use data for all tasks found in the trace.
    :type tasks: list(str) or NoneType

    :param window: time window to consider when parsing the trace
    :type window: tuple(int, int)

    :param normalize_time: normalize trace time stamps
    :type normalize_time: bool

    :param trace_format: format of the trace. Possible values are:
        - FTrace
        - SysTrace
    :type trace_format: str

    :param plots_dir: directory where to save plots
    :type plots_dir: str

    :param plots_prefix: prefix for plots file names
    :type plots_prefix: str
    """

    def __init__(self, platform, data_dir, events,
                 tasks=None, window=(0, None),
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

        # Setup logging
        self._log = logging.getLogger('Trace')

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
        self.__parseTrace(data_dir, tasks, window, normalize_time,
                          trace_format)
        self.__computeTimeSpan()

        # Minimum and Maximum x_time to use for all plots
        self.x_min = 0
        self.x_max = self.time_range

        # Reset x axis time range to full scale
        t_min = self.window[0]
        t_max = self.window[1]
        self.setXTimeRange(t_min, t_max)

        self.data_frame = TraceData()
        self._registerDataFrameGetters(self)

        self.analysis = AnalysisRegister(self)

    def _registerDataFrameGetters(self, module):
        """
        Internal utility function that looks up getter functions with a "_dfg_"
        prefix in their name and bounds them to the specified module.

        :param module: module to which the function is added
        :type module: class
        """
        self._log.debug('Registering [%s] local data frames', module)
        for func in dir(module):
            if not func.startswith('_dfg_'):
                continue
            dfg_name = func.replace('_dfg_', '')
            dfg_func = getattr(module, func)
            self._log.debug('   %s', dfg_name)
            setattr(self.data_frame, dfg_name, dfg_func)

    def setXTimeRange(self, t_min=None, t_max=None):
        """
        Set x axis time range to the specified values.

        :param t_min: lower bound
        :type t_min: int or float

        :param t_max: upper bound
        :type t_max: int or float
        """
        if t_min is None:
            self.x_min = 0
        else:
            self.x_min = t_min
        if t_max is None:
            self.x_max = self.time_range
        else:
            self.x_max = t_max
        self._log.debug('Set plots time range to (%.6f, %.6f)[s]',
                       self.x_min, self.x_max)

    def __registerTraceEvents(self, events):
        """
        Save a copy of the parsed events.

        :param events: single event name or list of events names
        :type events: str or list(str)
        """
        if isinstance(events, basestring):
            self.events = events.split(' ')
        elif isinstance(events, list):
            self.events = events
        else:
            raise ValueError('Events must be a string or a list of strings')
        # Register devlib fake cpu_frequency events
        if 'cpu_frequency' in events:
            self.events.append('cpu_frequency_devlib')

    def __parseTrace(self, path, tasks, window, normalize_time, trace_format):
        """
        Internal method in charge of performing the actual parsing of the
        trace.

        :param path: path to the trace folder (or trace file)
        :type path: str

        :param tasks: filter data for the specified tasks only
        :type tasks: list(str)

        :param window: time window to consider when parsing the trace
        :type window: tuple(int, int)

        :param normalize_time: normalize trace time stamps
        :type normalize_time: bool

        :param trace_format: format of the trace. Possible values are:
            - FTrace
            - SysTrace
        :type trace_format: str
        """
        self._log.debug('Loading [sched] events from trace in [%s]...', path)
        self._log.debug('Parsing events: %s', self.events)
        if trace_format.upper() == 'SYSTRACE' or path.endswith('html'):
            self._log.debug('Parsing SysTrace format...')
            trace_class = trappy.SysTrace
            self.trace_format = 'SysTrace'
        elif trace_format.upper() == 'FTRACE':
            self._log.debug('Parsing FTrace format...')
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
                self._log.info('Trace contains only functions stats')
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
        """
        Internal method used to build a list of available events.

        :param key: key to be used for TRAPpy filtering
        :type key: str
        """
        for val in self.ftrace.get_filters(key):
            obj = getattr(self.ftrace, val)
            if len(obj.data_frame):
                self.available_events.append(val)
        self._log.debug('Events found on trace:')
        for evt in self.available_events:
            self._log.debug(' - %s', evt)

    def __loadTasksNames(self, tasks):
        """
        Try to load tasks names using one of the supported events.

        :param tasks: list of task names. If None, load all tasks found.
        :type tasks: list(str) or NoneType
        """
        def load(tasks, event, name_key, pid_key):
            df = self._dfg_trace_event(event)
            if tasks is None:
                tasks = df[name_key].unique()
            self.getTasks(df, tasks, name_key=name_key, pid_key=pid_key)
            self._scanTasks(df, name_key=name_key, pid_key=pid_key)

        if 'sched_switch' in self.available_events:
            load(tasks, 'sched_switch', 'next_comm', 'next_pid')
        elif 'sched_load_avg_task' in self.available_events:
            load(tasks, 'sched_load_avg_task', 'comm', 'pid')
        else:
            self._log.warning('Failed to load tasks names from trace events')

    def hasEvents(self, dataset):
        """
        Returns True if the specified event is present in the parsed trace,
        False otherwise.

        :param dataset: trace event name or list of trace events
        :type dataset: str or list(str)
        """
        if dataset in self.available_events:
            return True
        return False

    def __computeTimeSpan(self):
        """
        Compute time axis range, considering all the parsed events.
        """
        ts = sys.maxint
        te = 0

        for events in self.available_events:
            df = self._dfg_trace_event(events)
            if len(df) == 0:
                continue
            if (df.index[0]) < ts:
                ts = df.index[0]
            if (df.index[-1]) > te:
                te = df.index[-1]
            self.time_range = te - ts

        self._log.debug('Collected events spans a %.3f [s] time interval',
                       self.time_range)

        # Build a stat on trace overutilization
        if self.hasEvents('sched_overutilized'):
            df = self._dfg_trace_event('sched_overutilized')
            self.overutilized_time = df[df.overutilized == 1].len.sum()
            self.overutilized_prc = 100. * self.overutilized_time / self.time_range

            self._log.debug('Overutilized time: %.6f [s] (%.3f%% of trace time)',
                           self.overutilized_time, self.overutilized_prc)

    def _scanTasks(self, df, name_key='comm', pid_key='pid'):
        """
        Extract tasks names and PIDs from the input data frame. The data frame
        should contain a task name column and PID column.

        :param df: data frame containing trace events from which tasks names
            and PIDs will be extracted
        :type df: :mod:`pandas.DataFrame`

        :param name_key: The name of the dataframe columns containing task
            names
        :type name_key: str

        :param pid_key: The name of the dataframe columns containing task PIDs
        :type pid_key: str
        """
        df = df[[name_key, pid_key]]
        self._tasks_by_name = df.set_index(name_key)
        self._tasks_by_pid = df.set_index(pid_key)

    def getTaskByName(self, name):
        """
        Get the PIDs of all tasks with the specified name.

        :param name: task name
        :type name: str
        """
        if name not in self._tasks_by_name.index:
            return []
        if len(self._tasks_by_name.ix[name].values) > 1:
            return list({task[0] for task in
                         self._tasks_by_name.ix[name].values})
        return [self._tasks_by_name.ix[name].values[0]]

    def getTaskByPid(self, pid):
        """
        Get the names of all tasks with the specified PID.

        :param name: task PID
        :type name: int
        """
        if pid not in self._tasks_by_pid.index:
            return []
        if len(self._tasks_by_pid.ix[pid].values) > 1:
            return list({task[0] for task in
                         self._tasks_by_pid.ix[pid].values})
        return [self._tasks_by_pid.ix[pid].values[0]]

    def getTasks(self, dataframe=None,
                 task_names=None, name_key='comm', pid_key='pid'):
        """
        Helper function to get PIDs of specified tasks.

        This method can take a Pandas dataset in input to be used to fiter out
        the PIDs of all the specified tasks. If a dataset is not provided,
        previously filtered PIDs are returned.

        If a list of task names is not provided, all tasks detected in the trace
        will be used. The specified dataframe must provide at least two columns
        reporting the task name and the task PID. The default values of this
        colums could be specified using the provided parameters.

        :param dataframe: A Pandas dataframe containing at least 'name_key' and
            'pid_key' columns. If None, the all PIDs are returned.
        :type dataframe: :mod:`pandas.DataFrame`

        :param task_names: The list of tasks to get the PID of (default: all
            tasks)
        :type task_names: list(str)

        :param name_key: The name of the dataframe columns containing task
            names
        :type name_key: str

        :param pid_key: The name of the dataframe columns containing task PIDs
        :type pid_key: str
        """
        if task_names is None:
            task_names = self.tasks.keys()
        if dataframe is None:
            return {k: v for k, v in  self.tasks.iteritems() if k in task_names}
        df = dataframe
        self._log.debug('Lookup dataset for tasks...')
        for tname in task_names:
            self._log.debug('Lookup for task [%s]...', tname)
            results = df[df[name_key] == tname][[name_key, pid_key]]
            if len(results) == 0:
                self._log.error('  task %16s NOT found', tname)
                continue
            (name, pid) = results.head(1).values[0]
            if name != tname:
                self._log.error('  task %16s NOT found', tname)
                continue
            if tname not in self.tasks:
                self.tasks[tname] = {}
            pids = list(results[pid_key].unique())
            self.tasks[tname]['pid'] = pids
            self._log.debug('  task %16s found, pid: %s',
                            tname, self.tasks[tname]['pid'])
        return self.tasks


###############################################################################
# DataFrame Getter Methods
###############################################################################

    def df(self, event):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name
        :type event: str
        """
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("\n\tUse of Trace::df() is deprecated and will be soon removed."
                      "\n\tUse Trace::data_frame.trace_event(event_name) instead.",
                      category=DeprecationWarning)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return self._dfg_trace_event(event)

    def _dfg_trace_event(self, event):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name
        :type event: str
        """
        if self.data_dir is None:
            raise ValueError("trace data not (yet) loaded")
        if self.ftrace and hasattr(self.ftrace, event):
            return getattr(self.ftrace, event).data_frame
        raise ValueError('Event [{}] not supported. '
                         'Supported events are: {}'
                         .format(event, self.available_events))

    def _dfg_functions_stats(self, functions=None):
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
        :type functions: str or list(str)
        """
        if not hasattr(self, '_functions_stats_df'):
            return None
        df = self._functions_stats_df
        if not functions:
            return df
        return df.loc[df.index.get_level_values(1).isin(listify(functions))]


###############################################################################
# Trace Events Sanitize Methods
###############################################################################

    def _sanitize_SchedCpuCapacity(self):
        """
        Add more columns to cpu_capacity data frame if the energy model is
        available.
        """
        if not self.hasEvents('cpu_capacity') \
           or 'nrg_model' not in self.platform:
            return

        df = self._dfg_trace_event('cpu_capacity')

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
        """
        If necessary, rename certain signal names from v5.0 to v5.1 format.
        """
        if not self.hasEvents('sched_load_avg_cpu'):
            return
        df = self._dfg_trace_event('sched_load_avg_cpu')
        if 'utilization' in df:
            df.rename(columns={'utilization': 'util_avg'}, inplace=True)
            df.rename(columns={'load': 'load_avg'}, inplace=True)

    def _sanitize_SchedLoadAvgTask(self):
        """
        If necessary, rename certain signal names from v5.0 to v5.1 format.
        """
        if not self.hasEvents('sched_load_avg_task'):
            return
        df = self._dfg_trace_event('sched_load_avg_task')
        if 'utilization' in df:
            df.rename(columns={'utilization': 'util_avg'}, inplace=True)
            df.rename(columns={'load': 'load_avg'}, inplace=True)
            df.rename(columns={'avg_period': 'period_contrib'}, inplace=True)
            df.rename(columns={'runnable_avg_sum': 'load_sum'}, inplace=True)
            df.rename(columns={'running_avg_sum': 'util_sum'}, inplace=True)
        df['cluster'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                ['LITTLE'], 'big')
        # Add a column which represents the max capacity of the smallest
        # clustre which can accomodate the task utilization
        little_cap = self.platform['nrg_model']['little']['cpu']['cap_max']
        big_cap = self.platform['nrg_model']['big']['cpu']['cap_max']
        df['min_cluster_cap'] = df.util_avg.map(
            lambda util_avg: big_cap if util_avg > little_cap else little_cap
        )

    def _sanitize_SchedBoostCpu(self):
        """
        Add a boosted utilization signal as the sum of utilization and margin.

        Also, if necessary, rename certain signal names from v5.0 to v5.1
        format.
        """
        if not self.hasEvents('sched_boost_cpu'):
            return
        df = self._dfg_trace_event('sched_boost_cpu')
        if 'usage' in df:
            df.rename(columns={'usage': 'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']

    def _sanitize_SchedBoostTask(self):
        """
        Add a boosted utilization signal as the sum of utilization and margin.

        Also, if necessary, rename certain signal names from v5.0 to v5.1
        format.
        """
        if not self.hasEvents('sched_boost_task'):
            return
        df = self._dfg_trace_event('sched_boost_task')
        if 'utilization' in df:
            # Convert signals name from to v5.1 format
            df.rename(columns={'utilization': 'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']

    def _sanitize_SchedEnergyDiff(self):
        """
        If a energy model is provided, some signals are added to the
        sched_energy_diff trace event data frame.

        Also convert between existing field name formats for sched_energy_diff
        """
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
        self._log.debug(
            "Maximum estimated system energy: {0:d}".format(power_max))

        df = self._dfg_trace_event('sched_energy_diff')

        translations = {'nrg_d' : 'nrg_diff',
                        'utl_d' : 'usage_delta',
                        'payoff' : 'nrg_payoff'
        }
        df.rename(columns=translations, inplace=True)

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
            ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject'],
            'Suboptimal Reject')

    def _sanitize_SchedOverutilized(self):
        """ Add a column with overutilized status duration. """
        if not self.hasEvents('sched_overutilized'):
            return
        df = self._dfg_trace_event('sched_overutilized')
        df['start'] = df.index
        df['len'] = (df.start - df.start.shift()).fillna(0).shift(-1)
        df.drop('start', axis=1, inplace=True)

    def _chunker(self, seq, size):
        """
        Given a data frame or a series, generate a sequence of chunks of the
        given size.

        :param seq: data to be split into chunks
        :type seq: :mod:`pandas.Series` or :mod:`pandas.DataFrame`

        :param size: size of each chunk
        :type size: int
        """
        return (seq.iloc[pos:pos + size] for pos in range(0, len(seq), size))

    def _sanitize_CpuFrequency(self):
        """
        Verify that all platform reported clusters are frequency coherent (i.e.
        frequency scaling is performed at a cluster level).
        """
        if not self.hasEvents('cpu_frequency_devlib'):
            return

        devlib_freq = self._dfg_trace_event('cpu_frequency_devlib')
        devlib_freq.rename(columns={'cpu_id':'cpu'}, inplace=True)
        devlib_freq.rename(columns={'state':'frequency'}, inplace=True)

        df = self._dfg_trace_event('cpu_frequency')
        clusters = self.platform['clusters']

        # devlib always introduces fake cpu_frequency events, in case the
        # OS has not generated cpu_frequency envets there are the only
        # frequency events to report
        if len(df) == 0:
            # Register devlib injected events as 'cpu_frequency' events
            setattr(self.ftrace.cpu_frequency, 'data_frame', devlib_freq)
            df = devlib_freq
            self.available_events.append('cpu_frequency')

        # make sure fake cpu_frequency events are never interleaved with
        # OS generated events
        else:
            if len(devlib_freq) > 0:

                # Frequencies injection is done in a per-cluster based.
                # This is based on the assumption that clusters are
                # frequency choerent.
                # For each cluster we inject devlib events only if
                # these events does not overlaps with os-generated ones.

                # Inject "initial" devlib frequencies
                os_df = df
                dl_df = devlib_freq.iloc[:self.platform['cpus_count']]
                for _,c in self.platform['clusters'].iteritems():
                    dl_freqs = dl_df[dl_df.cpu.isin(c)]
                    os_freqs = os_df[os_df.cpu.isin(c)]
                    self._log.debug("First freqs for %s:\n%s", c, dl_freqs)
                    # All devlib events "before" os-generated events
                    self._log.debug("Min os freq @: %s", os_freqs.index.min())
                    if os_freqs.empty or \
                       os_freqs.index.min() > dl_freqs.index.max():
                        self._log.debug("Insert devlib freqs for %s", c)
                        df = pd.concat([dl_freqs, df])

                # Inject "final" devlib frequencies
                os_df = df
                dl_df = devlib_freq.iloc[self.platform['cpus_count']:]
                for _,c in self.platform['clusters'].iteritems():
                    dl_freqs = dl_df[dl_df.cpu.isin(c)]
                    os_freqs = os_df[os_df.cpu.isin(c)]
                    self._log.debug("Last freqs for %s:\n%s", c, dl_freqs)
                    # All devlib events "after" os-generated events
                    self._log.debug("Max os freq @: %s", os_freqs.index.max())
                    if os_freqs.empty or \
                       os_freqs.index.max() < dl_freqs.index.min():
                        self._log.debug("Append devlib freqs for %s", c)
                        df = pd.concat([df, dl_freqs])

                df.sort_index(inplace=True)

            setattr(self.ftrace.cpu_frequency, 'data_frame', df)

        # Frequency Coherency Check
        for _, cpus in clusters.iteritems():
            cluster_df = df[df.cpu.isin(cpus)]
            for chunk in self._chunker(cluster_df, len(cpus)):
                f = chunk.iloc[0].frequency
                if any(chunk.frequency != f):
                    self._log.warning('Cluster Frequency is not coherent! '
                                      'Failure in [cpu_frequency] events at:')
                    self._log.warning(chunk)
                    self.freq_coherency = False
                    return
        self._log.info('Platform clusters verified to be Frequency coherent')

###############################################################################
# Utility Methods
###############################################################################

    def integrate_square_wave(self, sq_wave):
        """
        Compute the integral of a square wave time series.

        :param sq_wave: square wave assuming only 1.0 and 0.0 values
        :type sq_wave: :mod:`pandas.Series`
        """
        sq_wave.iloc[-1] = 0.0
        # Compact signal to obtain only 1-0-1-0 sequences
        comp_sig = sq_wave.loc[sq_wave.shift() != sq_wave]
        # First value for computing the difference must be a 1
        if comp_sig.iloc[0] == 0.0:
            return sum(comp_sig.iloc[2::2].index - comp_sig.iloc[1:-1:2].index)
        else:
            return sum(comp_sig.iloc[1::2].index - comp_sig.iloc[:-1:2].index)

    def _loadFunctionsStats(self, path='trace.stats'):
        """
        Read functions profiling file and build a data frame containing all
        relevant data.

        :param path: path to the functions profiling trace file
        :type path: str
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'trace.stats')
        if path.endswith('dat') or path.endswith('html'):
            pre, ext = os.path.splitext(path)
            path = pre + '.stats'
        if not os.path.isfile(path):
            return False

        # Opening functions profiling JSON data file
        self._log.debug('Loading functions profiling data from [%s]...', path)
        with open(os.path.join(path), 'r') as fh:
            trace_stats = json.load(fh)

        # Build DataFrame of function stats
        frames = {}
        for cpu, data in trace_stats.iteritems():
            frames[int(cpu)] = pd.DataFrame.from_dict(data, orient='index')

        # Build and keep track of the DataFrame
        self._functions_stats_df = pd.concat(frames.values(),
                                             keys=frames.keys())

        return len(self._functions_stats_df) > 0

    @memoized
    def getCPUActiveSignal(self, cpu):
        """
        Build a square wave representing the active (i.e. non-idle) CPU time,
        i.e.:

          cpu_active[t] == 1 if the CPU is reported to be non-idle by cpuidle at
          time t
          cpu_active[t] == 0 otherwise

        :param cpu: CPU ID
        :type cpu: int

        :returns: A :mod:`pandas.Series` or ``None`` if the trace contains no
                  "cpu_idle" events
        """
        if not self.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, '
                              'cannot compute CPU active signal!')
            return None

        idle_df = self._dfg_trace_event('cpu_idle')
        cpu_df = idle_df[idle_df.cpu_id == cpu]

        cpu_active = cpu_df.state.apply(
            lambda s: 1 if s == NON_IDLE_STATE else 0
        )

        start_time = 0.0
        if not self.ftrace.normalized_time:
            start_time = self.ftrace.basetime

        if cpu_active.empty:
            cpu_active = pd.Series([0], index=[start_time])
        elif cpu_active.index[0] != start_time:
            entry_0 = pd.Series(cpu_active.iloc[0] ^ 1, index=[start_time])
            cpu_active = pd.concat([entry_0, cpu_active])

        return cpu_active

    @memoized
    def getClusterActiveSignal(self, cluster):
        """
        Build a square wave representing the active (i.e. non-idle) cluster
        time, i.e.:

          cluster_active[t] == 1 if at least one CPU is reported to be non-idle
          by CPUFreq at time t
          cluster_active[t] == 0 otherwise

        :param cluster: list of CPU IDs belonging to a cluster
        :type cluster: list(int)

        :returns: A :mod:`pandas.Series` or ``None`` if the trace contains no
                  "cpu_idle" events
        """
        if not self.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, '
                              'cannot compute cluster active signal!')
            return None

        active = self.getCPUActiveSignal(cluster[0]).to_frame(name=cluster[0])
        for cpu in cluster[1:]:
            active = active.join(
                self.getCPUActiveSignal(cpu).to_frame(name=cpu),
                how='outer'
            )

        active.fillna(method='ffill', inplace=True)

        # Cluster active is the OR between the actives on each CPU
        # belonging to that specific cluster
        cluster_active = reduce(
            operator.or_,
            [cpu_active.astype(int) for _, cpu_active in
             active.iteritems()]
        )

        return cluster_active


class TraceData:
    """ A DataFrame collector exposed to Trace's clients """
    pass

# vim :set tabstop=4 shiftwidth=4 expandtab
