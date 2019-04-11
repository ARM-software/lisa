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

import abc
import copy
import numpy as np
import os
import os.path
import pandas as pd
import sys
import trappy
import json
import warnings
import operator
import logging
import webbrowser
import inspect
from functools import reduce, wraps
from collections.abc import Sequence

from lisa.utils import Loggable, HideExekallID, memoized, deduplicate
from lisa.platforms.platinfo import PlatformInfo
from lisa.conf import SimpleMultiSrcConf, KeyDesc, TopLevelKeyDesc, StrList, Configurable
import devlib
from devlib.target import KernelVersion
from trappy.utils import listify, handle_duplicate_index

NON_IDLE_STATE = -1

class TraceBase(abc.ABC):
    """
    Base class for common functionalities between :class:`Trace` and :class:`TraceView`
    """

    @abc.abstractmethod
    def get_view(self, window):
        """
        Get a view on a trace cropped time-wise to fit in ``window``
        """
        pass

    def __getitem__(self, window):
        if not isinstance(window, slice):
            raise TypeError("Cropping window must be an instance of slice")

        if window.step is not None:
            raise ValueError("Slice step is not supported")

        return self.get_view((window.start, window.stop))

    def add_events_deltas(self, df, col_name='delta', inplace=True):
        """
        Store the time between each event in a new dataframe column

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
        if df.empty:
            return df

        if col_name in df.columns:
            raise RuntimeError("Column {} is already present in the dataframe".
                               format(col_name))

        if not inplace:
            df = df.copy()

        df[col_name] = df.index
        df[col_name] = df[col_name].diff().shift(-1)
        # Fix the last event, which will have a NaN duration
        # Set duration to trace_end - last_event
        df.loc[df.index[-1], col_name] = self.end - df.index[-1]

        return df

class TraceView(Loggable, TraceBase):
    """
    A view on a :class:`Trace`

    :param trace: The trace to trim
    :type trace: Trace

    :param window: The time window to base this view on
    :type window: tuple(float, float)

    :ivar base_trace: The original :class:`Trace` this view is based on.
    :ivar analysis: The analysis proxy on the trimmed down :class:`Trace`.

    :ivar start: The timestamp of the first trace event in the view (>= ``window[0]``)
    :ivar end: The timestamp of the last trace event in the view (<= ``window[1]``)

    You can substitute an instance of :class:`Trace` with an instance of
    :class:`TraceView`. This means you can create a view of a trimmed down trace
    and run analysis code/plots that will only use data within that window, e.g.::

      trace = Trace(...)
      view = trace.get_view((2, 4))

      # Alias for the above
      view = trace[2:4]

      # This will only use events in the (2, 4) time window
      df = view.analysis.tasks.df_tasks_runtime()

    **Design notes:**

      * :meth:`df_events` uses the underlying :meth:`lisa.trace.Trace.df_events`
        and trims the dataframe according to the given ``window`` before
        returning it.
      * ``self.start`` and ``self.end`` mimic the :class:`Trace` attributes but
        they are adjusted to match the given window. On top of this, this class
        mimics a regular :class:`Trace` using :func:`getattr`.
    """
    def __init__(self, trace, window):
        self.base_trace = trace

        t_min = window[0]
        t_max = window[1]

        trace_classes = [cls for cls in self.base_trace.ftrace.trace_classes
                         if not cls.data_frame.empty]

        if t_min is not None:
            start = self.base_trace.end
            for trace_class in trace_classes:
                df = trace_class.data_frame[t_min:]
                if not df.empty:
                    start = min(start, df.index[0])
            t_min = start
        else:
            t_min = self.base_trace.start

        if t_max is not None:
            end = self.base_trace.start
            for trace_class in trace_classes:
                df = trace_class.data_frame[:t_max]
                if not df.empty:
                    end = max(end, df.index[-1])
            t_max = end
        else:
            t_max = self.base_trace.end

        self.start = t_min
        self.end = t_max
        self.time_range = t_max - t_min

        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base
        from lisa.analysis.proxy import AnalysisProxy
        self.analysis = AnalysisProxy(self)

    def __getattr__(self, name):
        return getattr(self.base_trace, name)

    def df_events(self, event):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name
        :type event: str
        """
        df = self.base_trace.df_events(event)
        if not df.empty:
            df = df[self.start:self.end]

        return df

    def get_view(self, window):
        start = self.start
        end   = self.end

        if window[0]:
            start = max(start, window[0])

        if window[1]:
            end = min(end, window[1])

        return self.base_trace.get_view((start, end))

class Trace(Loggable, TraceBase):
    """
    The Trace object is the LISA trace events parser.

    :param trace_path: File containing the trace
    :type trace_path: str

    :param events: events to be parsed (all the events by default)
    :type events: str or list(str)

    :param platform: a dictionary containing information about the target
        platform
    :type platform: dict

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

    :ivar start: The timestamp of the first trace event in the trace
    :ivar end: The timestamp of the last trace event in the trace
    """

    def __init__(self,
                 trace_path,
                 plat_info=None,
                 events=None,
                 normalize_time=True,
                 trace_format='FTrace',
                 plots_dir=None,
                 plots_prefix=''):
        logger = self.get_logger()

        if plat_info is None:
            plat_info = PlatformInfo()

        # The platform information used to run the experiments
        self.plat_info = plat_info

        # Trace format
        self.trace_format = trace_format

        # Whether trace timestamps are normalized or not
        self.normalize_time = normalize_time

        # Dynamically registered TRAPpy events
        self.trappy_cls = {}

        # Maximum timespan for all collected events
        self.time_range = 0

        # Time the system was overutilzied
        self.overutilized_time = 0
        self.overutilized_prc = 0

        # List of events required by user
        self.events = self._process_events(events)

        # List of events available in the parsed trace
        self.available_events = []

        # Cluster frequency coherency flag
        self.freq_coherency = True

        # Folder containing trace
        self.trace_path = trace_path

        # By default, use the trace dir to save plots
        self.plots_dir = plots_dir if plots_dir else os.path.dirname(trace_path)

        self.plots_prefix = plots_prefix

        self._parse_trace(self.trace_path, trace_format)

        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base
        from lisa.analysis.proxy import AnalysisProxy
        self.analysis = AnalysisProxy(self)

    @property
    @memoized
    def cpus_count(self):
        try:
            return self.plat_info['cpus-count']
        # If we don't know the number of CPUs, check the trace for the
        # highest-numbered CPU that traced an event.
        except KeyError:
            max_cpu = max(int(self.df_events(e)['__cpu'].max())
                          for e in self.available_events)
            count = max_cpu + 1
            self.get_logger().info("Estimated CPU count from trace: %s", count)
            return count

    @staticmethod
    def _process_events(events):
        """
        Process the `events` parameter of :meth:`Trace.__init__`.

        :param events: single event name or list of events names
        :type events: str or list(str)
        """
        # Parse all events by default
        if events is None:
            events = []
        elif isinstance(events, str):
            events = [events]
        elif isinstance(events, Sequence):
            events = list(events)
        else:
            raise ValueError('Events must be a string or a sequence of strings')

        # Register devlib fake cpu_frequency events
        if 'cpu_frequency' in events:
            events.append('cpu_frequency_devlib')

        return events

    def _parse_trace(self, path, trace_format):
        """
        Internal method in charge of performing the actual parsing of the
        trace.

        :param path: path to the trace folder (or trace file)
        :type path: str

        :param trace_format: format of the trace. Possible values are:
            - FTrace
            - SysTrace
        :type trace_format: str
        """
        logger = self.get_logger()
        logger.debug('Loading [sched] events from trace in [%s]...', path)
        logger.debug('Parsing events: %s', self.events)
        if trace_format.upper() == 'SYSTRACE' or path.endswith('html'):
            logger.debug('Parsing SysTrace format...')
            trace_class = trappy.SysTrace
            self.trace_format = 'SysTrace'
        elif trace_format.upper() == 'FTRACE':
            logger.debug('Parsing FTrace format...')
            trace_class = trappy.FTrace
            self.trace_format = 'FTrace'
        else:
            raise ValueError("Unknown trace format {}".format(trace_format))

        # Make sure event names are not unicode strings
        self.ftrace = trace_class(path, scope="custom", events=self.events,
                                  normalize_time=self.normalize_time)

        # Load Functions profiling data
        has_function_stats = self._loadFunctionsStats(path)

        # Check for events available on the parsed trace
        self._check_available_events()
        if not self.available_events:
            if has_function_stats:
                logger.info('Trace contains only functions stats')
                return
            raise ValueError('The trace does not contain useful events '
                             'nor function stats')

        self._compute_timespan()
        # Index PIDs and Task names
        self._load_tasks_names()

        # Setup internal data reference to interesting events/dataframes
        self._sanitize_SchedLoadAvgCpu()
        self._sanitize_SchedLoadAvgTask()
        self._sanitize_SchedCpuCapacity()
        self._sanitize_SchedBoostCpu()
        self._sanitize_SchedBoostTask()
        self._sanitize_SchedEnergyDiff()
        self._sanitize_SchedOverutilized()
        self._sanitize_CpuFrequency()
        self._sanitize_ThermalPowerCpu()

    def _check_available_events(self, key=""):
        """
        Internal method used to build a list of available events.

        :param key: key to be used for TRAPpy filtering
        :type key: str
        """
        logger = self.get_logger()
        for val in self.ftrace.get_filters(key):
            obj = getattr(self.ftrace, val)
            if not obj.data_frame.empty:
                self.available_events.append(val)
        logger.debug('Events found on trace:')
        for evt in self.available_events:
            logger.debug(' - %s', evt)

    def _load_tasks_names(self):
        """
        Try to load tasks names using one of the supported events.
        """
        def load(event, name_key, pid_key):
            df = self.df_events(event)
            self._scan_tasks(df, name_key=name_key, pid_key=pid_key)

        if 'sched_switch' in self.available_events:
            load('sched_switch', 'prev_comm', 'prev_pid')
            return

        if 'sched_load_avg_task' in self.available_events:
            load('sched_load_avg_task', 'comm', 'pid')
            return

        self.get_logger().warning('Failed to load tasks names from trace events')

    def has_events(self, events):
        """
        Returns True if the specified event is present in the parsed trace,
        False otherwise.

        :param events: trace event name or list of trace events
        :type events: str or list(str) or TraceEventCheckerBase
        """
        if isinstance(events, str):
            return events in self.available_events
        elif isinstance(events, TraceEventCheckerBase):
            try:
                events.check_events(self.available_events)
            except MissingTraceEventError:
                return False
            else:
                return True
        else:
            return set(events).issubset(set(self.available_events))

    def get_view(self, window):
        return TraceView(self, window)

    def _compute_timespan(self):
        """
        Compute time axis range, considering all the parsed events.
        """
        self.start = 0 if self.ftrace.normalized_time else self.ftrace.basetime
        self.end = self.start + self.ftrace.get_duration()
        self.time_range = self.end - self.start

        self.get_logger().debug('Trace contains events from %s to %s',
                                self.start, self.end)

    def _scan_tasks(self, df, name_key='comm', pid_key='pid'):
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
        self._tasks_by_pid = (df.drop_duplicates(subset=pid_key, keep='last')
                .rename(columns={
                    pid_key : 'PID',
                    name_key : 'TaskName'})
                .set_index('PID').sort_index())

    def get_task_by_name(self, name):
        """
        Get the PIDs of all tasks with the specified name.

        The same PID can have different task names, mainly because once a task
        is generated it inherits the parent name and then its name is updated
        to represent what the task really is.

        This API works under the assumption that a task name is updated at
        most one time and it always considers the name a task had the last time
        it has been scheduled for execution in the current trace.

        :param name: task name
        :type name: str

        :return: a list of PID for tasks which name matches the required one,
                 the last time they ran in the current trace
        """
        return (self._tasks_by_pid[self._tasks_by_pid.TaskName == name]
                    .index.tolist())

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
        try:
            return self._tasks_by_pid.ix[pid].values[0]
        except KeyError:
            return None

    def get_task_pid(self, task):
        """
        Helper that takes either a name or a PID and always returns a PID

        :param task: Either the task name or the task PID
        :type task: int or str
        """
        if isinstance(task, str):
            pid_list = self.get_task_by_name(task)

            if not pid_list:
                raise ValueError('trace does not have any task named "{}".format(task)')

            if len(pid_list) > 1:
                self.get_logger().warning(
                    "More than one PID found for task {}, "
                    "using the first one ({})".format(task, pid_list[0]))

            pid = pid_list[0]
        else:
            pid = task

        return pid


    def get_tasks(self):
        """
        Get a dictionary of all the tasks in the Trace.

        :return: a dictionary which maps each PID to the corresponding task
                 name
        """
        return self._tasks_by_pid.TaskName.to_dict()

    def show(self):
        """
        Open the parsed trace using the most appropriate native viewer.

        The native viewer depends on the specified trace format:
        - ftrace: open using kernelshark
        - systrace: open using a browser

        In both cases the native viewer is assumed to be available in the host
        machine.
        """
        if isinstance(self.ftrace, trappy.FTrace):
            return os.popen("kernelshark '{}'".format(self.ftrace.trace_path))
        if isinstance(self.ftrace, trappy.SysTrace):
            return webbrowser.open(self.ftrace.trace_path)
        self.get_logger().warning('No trace data available')


###############################################################################
# DataFrame Getter Methods
###############################################################################

    def df_events(self, event):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name
        :type event: str
        """
        try:
            return getattr(self.ftrace, event).data_frame
        except AttributeError:
            raise ValueError('Event [{}] not supported. '
                             'Supported events are: {}'
                             .format(event, self.available_events))

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
        :type functions: str or list(str)
        """
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
        available and the platform is big.LITTLE.
        """
        if not self.has_events('cpu_capacity') \
           or 'nrg-model' not in self.plat_info \
           or not self.has_big_little:
            return

        df = self.df_events('cpu_capacity')

        # Add column with LITTLE and big CPUs max capacities
        nrg_model = self.plat_info['nrg-model']
        max_lcap = nrg_model['little']['cpu']['cap_max']
        max_bcap = nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.plat_info['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.plat_info['clusters']['little'])],
                [tip_lcap], tip_bcap)

    def _sanitize_SchedLoadAvgCpu(self):
        """
        If necessary, rename certain signal names from v5.0 to v5.1 format.
        """
        if not self.has_events('sched_load_avg_cpu'):
            return
        df = self.df_events('sched_load_avg_cpu')
        if 'utilization' in df:
            df.rename(columns={'utilization': 'util_avg'}, inplace=True)
            df.rename(columns={'load': 'load_avg'}, inplace=True)

    def _sanitize_SchedLoadAvgTask(self):
        """
        If necessary, rename certain signal names from v5.0 to v5.1 format.
        """
        if not self.has_events('sched_load_avg_task'):
            return
        df = self.df_events('sched_load_avg_task')
        if 'utilization' in df:
            df.rename(columns={'utilization': 'util_avg'}, inplace=True)
            df.rename(columns={'load': 'load_avg'}, inplace=True)
            df.rename(columns={'avg_period': 'period_contrib'}, inplace=True)
            df.rename(columns={'runnable_avg_sum': 'load_sum'}, inplace=True)
            df.rename(columns={'running_avg_sum': 'util_sum'}, inplace=True)

    def _sanitize_SchedBoostCpu(self):
        """
        Add a boosted utilization signal as the sum of utilization and margin.

        Also, if necessary, rename certain signal names from v5.0 to v5.1
        format.
        """
        if not self.has_events('sched_boost_cpu'):
            return
        df = self.df_events('sched_boost_cpu')
        if 'usage' in df:
            df.rename(columns={'usage': 'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']

    def _sanitize_SchedBoostTask(self):
        """
        Add a boosted utilization signal as the sum of utilization and margin.

        Also, if necessary, rename certain signal names from v5.0 to v5.1
        format.
        """
        if not self.has_events('sched_boost_task'):
            return
        df = self.df_events('sched_boost_task')
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
        logger = self.get_logger()
        if not self.has_events('sched_energy_diff') \
           or 'nrg-model' not in self.plat_info \
           or not self.has_big_little:
            return
        nrg_model = self.plat_info['nrg-model']
        em_lcluster = nrg_model['little']['cluster']
        em_bcluster = nrg_model['big']['cluster']
        em_lcpu = nrg_model['little']['cpu']
        em_bcpu = nrg_model['big']['cpu']
        lcpus = len(self.plat_info['clusters']['little'])
        bcpus = len(self.plat_info['clusters']['big'])
        SCHED_LOAD_SCALE = 1024

        power_max = em_lcpu['nrg_max'] * lcpus + em_bcpu['nrg_max'] * bcpus + \
            em_lcluster['nrg_max'] + em_bcluster['nrg_max']
        logger.debug(
            "Maximum estimated system energy: {0:d}".format(power_max))

        df = self.df_events('sched_energy_diff')

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
        if not self.has_events('sched_overutilized'):
            return

        # df = self.df_events('sched_overutilized')
        df = getattr(self.ftrace, "sched_overutilized").data_frame
        self.add_events_deltas(df, 'len')

        # Build a stat on trace overutilization
        self.overutilized_time = df[df.overutilized == 1].len.sum()
        self.overutilized_prc = 100. * self.overutilized_time / self.time_range

        self.get_logger().debug('Overutilized time: %.6f [s] (%.3f%% of trace time)',
                        self.overutilized_time, self.overutilized_prc)

    def _sanitize_ThermalPowerCpu(self):
        self._sanitize_ThermalPowerCpuGetPower()
        self._sanitize_ThermalPowerCpuLimit()

    def _sanitize_ThermalPowerCpuMask(self, mask):
        # Replace '00000000,0000000f' format in more usable int
        return int(mask.replace(',', ''), 16)

    def _sanitize_ThermalPowerCpuGetPower(self):
        if not self.has_events('thermal_power_cpu_get_power'):
            return

        df = self.df_events('thermal_power_cpu_get_power')

        df['cpus'] = df['cpus'].apply(
            self._sanitize_ThermalPowerCpuMask
        )

    def _sanitize_ThermalPowerCpuLimit(self):
        if not self.has_events('thermal_power_cpu_limit'):
            return

        df = self.df_events('thermal_power_cpu_limit')

        df['cpus'] = df['cpus'].apply(
            self._sanitize_ThermalPowerCpuMask
        )

    def _chunker(self, seq, size):
        """
        Given a data frame or a series, generate a sequence of chunks of the
        given size.

        :param seq: data to be split into chunks
        :type seq: :class:`pandas.Series` or :class:`pandas.DataFrame`

        :param size: size of each chunk
        :type size: int
        """
        return (seq.iloc[pos:pos + size] for pos in range(0, len(seq), size))

    def _sanitize_CpuFrequency(self):
        """
        Verify that all platform reported clusters are frequency coherent (i.e.
        frequency scaling is performed at a cluster level).
        """
        logger = self.get_logger()
        if not self.has_events('cpu_frequency_devlib') \
           or 'freq-domains' not in self.plat_info:
            return

        devlib_freq = self.df_events('cpu_frequency_devlib')
        devlib_freq.rename(columns={'cpu_id':'cpu'}, inplace=True)
        devlib_freq.rename(columns={'state':'frequency'}, inplace=True)

        df = self.df_events('cpu_frequency')
        domains = self.plat_info['freq-domains']

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
            if not devlib_freq.empty:

                # Frequencies injection is done in a per-cluster based.
                # This is based on the assumption that clusters are
                # frequency choerent.
                # For each cluster we inject devlib events only if
                # these events does not overlaps with os-generated ones.

                # Inject "initial" devlib frequencies
                os_df = df
                dl_df = devlib_freq.iloc[:self.cpus_count]
                for cpus in domains:
                    dl_freqs = dl_df[dl_df.cpu.isin(cpus)]
                    os_freqs = os_df[os_df.cpu.isin(cpus)]
                    logger.debug("First freqs for %s:\n%s", cpus, dl_freqs)
                    # All devlib events "before" os-generated events
                    logger.debug("Min os freq @: %s", os_freqs.index.min())
                    if os_freqs.empty or \
                       os_freqs.index.min() > dl_freqs.index.max():
                        logger.debug("Insert devlib freqs for %s", cpus)
                        df = pd.concat([dl_freqs, df])

                # Inject "final" devlib frequencies
                os_df = df
                dl_df = devlib_freq.iloc[self.cpus_count:]
                for cpus in domains:
                    dl_freqs = dl_df[dl_df.cpu.isin(cpus)]
                    os_freqs = os_df[os_df.cpu.isin(cpus)]
                    logger.debug("Last freqs for %s:\n%s", cpus, dl_freqs)
                    # All devlib events "after" os-generated events
                    logger.debug("Max os freq @: %s", os_freqs.index.max())
                    if os_freqs.empty or \
                       os_freqs.index.max() < dl_freqs.index.min():
                        logger.debug("Append devlib freqs for %s", cpus)
                        df = pd.concat([df, dl_freqs])

                df.sort_index(inplace=True)

            setattr(self.ftrace.cpu_frequency, 'data_frame', df)

        # Frequency Coherency Check
        for cpus in domains:
            cluster_df = df[df.cpu.isin(cpus)]
            for chunk in self._chunker(cluster_df, len(cpus)):
                f = chunk.iloc[0].frequency
                if any(chunk.frequency != f):
                    logger.warning('Cluster Frequency is not coherent! '
                                      'Failure in [cpu_frequency] events at:')
                    logger.warning(chunk)
                    self.freq_coherency = False
                    return
        logger.info('Platform clusters verified to be Frequency coherent')

###############################################################################
# Utility Methods
###############################################################################

    def integrate_square_wave(self, sq_wave):
        """
        Compute the integral of a square wave time series.

        :param sq_wave: square wave assuming only 1.0 and 0.0 values
        :type sq_wave: :class:`pandas.Series`
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
        if (path.endswith('dat') or
            path.endswith('txt') or
            path.endswith('html')):
            pre, ext = os.path.splitext(path)
            path = pre + '.stats'
        if not os.path.isfile(path):
            return False

        # Opening functions profiling JSON data file
        self.get_logger().debug('Loading functions profiling data from [%s]...', path)
        with open(os.path.join(path), 'r') as fh:
            trace_stats = json.load(fh)

        # Build DataFrame of function stats
        frames = {}
        for cpu, data in trace_stats.items():
            frames[int(cpu)] = pd.DataFrame.from_dict(data, orient='index')

        # Build and keep track of the DataFrame
        self._functions_stats_df = pd.concat(list(frames.values()),
                                             keys=list(frames.keys()))

        return len(self._functions_stats_df) > 0

    @memoized
    def get_peripheral_clock_effective_rate(self, clk_name):
        logger = self.get_logger()
        if clk_name is None:
            logger.warning('no specified clk_name in computing peripheral clock, returning None')
            return
        if not self.has_events('clock_set_rate'):
            logger.warning('Events [clock_set_rate] not found, returning None!')
            return
        rate_df = self.df_events('clock_set_rate')
        enable_df = self.df_events('clock_enable')
        disable_df = self.df_events('clock_disable')
        pd.set_option('display.expand_frame_repr', False)

        freq = rate_df[rate_df.clk_name == clk_name]
        if not enable_df.empty:
            enables = enable_df[enable_df.clk_name == clk_name]
        if not disable_df.empty:
            disables = disable_df[disable_df.clk_name == clk_name]

        freq = pd.concat([freq, enables, disables], sort=False).sort_index()
        if freq.empty:
            logger.warning('No events for clock ' + clk_name + ' found in trace')
            return

        freq['start'] = freq.index
        freq['len'] = (freq.start - freq.start.shift()).fillna(0).shift(-1)
        # The last value will be NaN, fix to be appropriate length
        freq.loc[freq.index[-1], 'len'] = self.end - freq.index[-1]

        freq = freq.fillna(method='ffill')
        freq['effective_rate'] = np.where(freq['state'] == 0, 0,
                                          np.where(freq['state'] == 1, freq['rate'], float('nan')))
        return freq

    @staticmethod
    def squash_df(df, start, end, column='delta'):
        """
        Slice a dataframe of deltas in [start:end] and ensure we have
        an event at exactly those boundaries.

        The input dataframe is expected to have a "column" which reports
        the time delta between consecutive rows, as for example dataframes
        generated by add_events_deltas().

        The returned dataframe is granted to have an initial and final
        event at the specified "start" ("end") index values, which values
        are the same of the last event before (first event after) the
        specified "start" ("end") time.

        Examples:

        Slice a dataframe to [start:end], and work on the time data so that it
        makes sense within the interval.

        Examples to make it clearer:

        df is:
        Time len state
        15    1   1
        16    1   0
        17    1   1
        18    1   0
        -------------

        slice_df(df, 16.5, 17.5) =>

        Time len state
        16.5  .5   0
        17    .5   1

        slice_df(df, 16.2, 16.8) =>

        Time len state
        16.2  .6   0

        :returns: a new df that fits the above description
        """
        if df.empty:
            return df

        end = min(end, df.index[-1] + df[column].values[-1])
        res_df = pd.DataFrame(data=[], columns=df.columns)

        if start > end:
            return res_df

        # There's a few things to keep in mind here, and it gets confusing
        # even for the people who wrote the code. Let's write it down.
        #
        # It's assumed that the data is continuous, i.e. for any row 'r' within
        # the trace interval, we will find a new row at (r.index + r.len)
        # For us this means we'll never end up with an empty dataframe
        # (if we started with a non empty one)
        #
        # What's we're manipulating looks like this:
        # (| = events; [ & ] = start,end slice)
        #
        # |   [   |   ]   |
        # e0  s0  e1  s1  e2
        #
        # We need to push e0 within the interval, and then tweak its duration
        # (len column). The mathemagical incantation for that is:
        # e0.len = min(e1.index - s0, s1 - s0)
        #
        # This takes care of the case where s1 isn't in the interval
        # If s1 is in the interval, we just need to cap its len to
        # s1 - e1.index

        prev_df = df[:start]
        middle_df = df[start:end]

        # Tweak the closest previous event to include it in the slice
        if not prev_df.empty and not (start in middle_df.index):
            res_df = res_df.append(prev_df.tail(1))
            res_df.index = [start]
            e1 = end

            if not middle_df.empty:
                e1 = middle_df.index[0]

            res_df[column] = min(e1 - start, end - start)

        if not middle_df.empty:
            res_df = res_df.append(middle_df)
            if end in res_df.index:
                # e_last and s1 collide, ditch e_last
                res_df = res_df.drop([end])
            else:
                # Fix the delta for the last row
                delta = min(end - res_df.index[-1], res_df[column].values[-1])
                res_df.at[res_df.index[-1], column] = delta

        return res_df

class TraceEventCheckerBase(abc.ABC, Loggable):
    """
    ABC for events checker classes.

    Event checking can be achieved using a boolean expression on expected
    events.
    """
    @abc.abstractmethod
    def check_events(self, event_set):
        """
        Check that certain trace events are available in the given set of
        events.

        :raises: MissingTraceEventError if some events are not available
        """
        pass

    @abc.abstractmethod
    def get_all_events(self):
        """
        Return a set of all events that are checked by this checker.

        That may be a superset of events that are strictly required, when the
        checker checks a logical OR combination of events for example.
        """
        pass

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
        try:
            used_events = f.used_events
        except AttributeError:
            checker = self
        else:
            checker = AndTraceEventChecker([self, used_events])

        sig = inspect.signature(f)
        if sig.parameters:
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
                    available_events = set(trace.available_events)
                    checker.check_events(available_events)

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
    def _str_internal(self, style=None, wrapped=True):
        """
        Format the boolean expression that this checker represents.

        :param style: When 'rst', a reStructuredText output is expected
        :type style: str

        :param wrapped: When True, the expression should be wrapped with
            parenthesis so it can be composed with other expressions.
        :type wrapped: bool
        """

        pass

    def doc_str(self):
        """
        Top-level function called by Sphinx's autodoc extension to augment
        docstrings of the functions.
        """
        return '\n    * {}'.format(self._str_internal(style='rst', wrapped=False))

    def __str__(self):
        return self._str_internal()

class TraceEventChecker(TraceEventCheckerBase):
    """
    Check for one single event.
    """
    def __init__(self, event):
        self.event = event

    def get_all_events(self):
        return {self.event}

    def check_events(self, event_set):
        if self.event not in event_set:
            raise MissingTraceEventError(self)

    def _str_internal(self, style=None, wrapped=True):
        template = '``{}``' if style == 'rst' else '{}'
        return template.format(self.event)

class AssociativeTraceEventChecker(TraceEventCheckerBase):
    """
    Base class for associative operators like `and` and `or`
    """
    def __init__(self, op_str, event_checkers):
        checker_list = []
        for checker in event_checkers:
            # "unwrap" checkers of the same type, to avoid useless levels of
            # nesting. This is valid since the operator is known to be
            # associative. We don't use isinstance to avoid merging checkers
            # that may have different semantics.
            if type(checker) is type(self):
                checker_list.extend(checker.checkers)
            else:
                checker_list.append(checker)

        # Avoid having the same event twice at the same level
        def key(checker):
            if isinstance(checker, TraceEventChecker):
                return checker.event
            else:
                return checker
        checker_list = deduplicate(checker_list, key=key)

        self.checkers = checker_list
        self.op_str = op_str

    def get_all_events(self):
        events = set()
        for checker in self.checkers:
            events.update(checker.get_all_events())
        return events

    @classmethod
    def from_events(cls, events):
        """
        Build an instance of the class, converting ``str`` to
        ``TraceEventChecker``.

        :param events: Sequence of events
        :type events: list(str or TraceEventCheckerBase)
        """
        return cls({
            e if isinstance(e, TraceEventCheckerBase) else TraceEventChecker(e)
            for e in events
        })

    def _str_internal(self, style=None, wrapped=True):
        op_str = ' {} '.format(self.op_str)
        # Sort for stable output
        checker_list = sorted(self.checkers, key=lambda c: str(c))
        unwrapped_str = op_str.join(
            c._str_internal(style=style, wrapped=True)
            for c in checker_list
        )

        template = '({})' if len(self.checkers) > 1 and wrapped else '{}'
        return template.format(unwrapped_str)

class OrTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that one of the given event checkers is satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """
    def __init__(self, event_checkers):
        super().__init__('or', event_checkers)

    def check_events(self, event_set):
        if not self.checkers:
            return

        failed_checker_set = set()
        for checker in self.checkers:
            try:
                checker.check_events(event_set)
            except MissingTraceEventError as e:
                failed_checker_set.add(e.missing_events)
            else:
                break
        else:
            cls = type(self)
            raise MissingTraceEventError(
                cls(failed_checker_set)
            )

class AndTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that all the given event checkers are satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """
    def __init__(self, event_checkers):
        super().__init__('and', event_checkers)

    def check_events(self, event_set):
        if not self.checkers:
            return

        failed_checker_set = set()
        for checker in self.checkers:
            try:
                checker.check_events(event_set)
            except MissingTraceEventError as e:
                failed_checker_set.add(e.missing_events)

        if failed_checker_set:
            cls = type(self)
            raise MissingTraceEventError(
                cls(failed_checker_set)
            )

    def doc_str(self):
        joiner = '\n' + '    '
        rst = joiner + joiner.join(
            '* {}'.format(c._str_internal(style='rst', wrapped=False))
            # Sort for stable output
            for c in sorted(self.checkers, key=lambda c: str(c))
        )
        return rst

def requires_events(*events):
    """
    Decorator for methods that require some given trace events.

    :param events: The list of required events
    :type events: list(str or TraceEventCheckerBase)

    The decorated method must operate on instances that have a
    ``self.trace`` attribute.
    """
    return AndTraceEventChecker.from_events(events)

def requires_one_event_of(*events):
    """
    Same as :func:``used_events`` with logical `OR` semantic.
    """
    return OrTraceEventChecker.from_events(events)

class MissingTraceEventError(RuntimeError):
    """
    :param missing_events: The missing trace events
    :type missing_events: TraceEventCheckerBase
    """
    def __init__(self, missing_events):
        super().__init__(
            "Trace is missing the following required events: {}".format(missing_events))

        self.missing_events = missing_events

class FtraceConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class of :class:`FtraceCollector`

    Available keys:
    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('ftrace-conf', 'FTrace configuration', (
        KeyDesc('events', 'FTrace events to trace', [StrList]),
        KeyDesc('functions', 'FTrace functions to trace', [StrList]),
        KeyDesc('buffer-size', 'FTrace buffer size', [int]),
    ))

    def add_merged_src(self, src, conf, **kwargs):
        """
        Merge-in a configuration source.

        :param src: Name of the merged source
        :type src: str

        :param conf: Conf to merge in
        :type conf: FtraceConf
        """
        def merge_conf(key, val):
            if key in ('events', 'functions'):
                return sorted(set(val) | set(self.get(key, [])))
            elif key == 'buffer-size':
                return max(val, self.get(key, 0))
            else:
                raise KeyError('Cannot merge key "{}"'.format(key))

        merged = {
            key: merge_conf(key, val)
            for key, val in conf.items()
        }

        def is_modified(key, val):
            try:
                existing_val = self[key]
            except KeyError:
                return True
            else:
                return val != existing_val

        # We merge some keys with their current value in the conf
        return self.add_src(src,
            conf={
                key: val
                for key, val in merged.items()
                # Only add to the source if the result is different than what is
                # already set
                if is_modified(key, val)
            },
            **kwargs,
        )

class FtraceCollector(devlib.FtraceCollector, Loggable, Configurable):
    """
    Thin wrapper around :class:`devlib.FtraceCollector`.

    {configurable_params}
    """

    CONF_CLASS = FtraceConf

    def __init__(self, target, events=[], functions=[], buffer_size=10240, autoreport=False, **kwargs):
        kwargs.update(dict(
            events=events,
            functions=functions,
            buffer_size=buffer_size,
            autoreport=autoreport,
        ))
        self.check_init_param(**kwargs)

        super().__init__(target, **kwargs)
        # Ensure we have trace-cmd on the target
        self.target.install_tools(['trace-cmd'])

    @classmethod
    def from_conf(cls, target, conf):
        """
        Build an :class:`FtraceCollector` from a :class:`FtraceConf`

        :param target: Target to use when collecting the trace
        :type target: lisa.target.Target

        :param conf: Configuration object
        :type conf: FtraceConf
        """
        cls.get_logger().info('Ftrace configuration:\n{}'.format(conf))
        kwargs = cls.conf_to_init_kwargs(conf)
        return cls(target, **kwargs)

    @classmethod
    def from_user_conf(cls, target, base_conf=None, user_conf=None, merged_src='merged'):
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
        """
        user_conf = user_conf or FtraceConf()
        base_conf = base_conf or FtraceConf()

        # Make a copy of the conf, since it may be shared by multiple classes
        conf = copy.copy(base_conf)

        # Merge user configuration with the test's configuration
        conf.add_merged_src(
            src=merged_src,
            conf=user_conf,
        )
        return cls.from_conf(target, conf)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
