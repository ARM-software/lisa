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
import numbers
import os
import os.path
import sys
import json
import warnings
import operator
import logging
import webbrowser
import inspect
import shlex
import contextlib
import tempfile
from functools import reduce, wraps
from collections.abc import Sequence
from collections import namedtuple

import numpy as np
import pandas as pd

import trappy
import devlib
from devlib.target import KernelVersion

from lisa.utils import Loggable, HideExekallID, memoized, deduplicate, deprecate, nullcontext
from lisa.platforms.platinfo import PlatformInfo
from lisa.conf import SimpleMultiSrcConf, KeyDesc, TopLevelKeyDesc, StrList, Configurable


class TaskID(namedtuple('TaskID', ('pid', 'comm'))):
    """
    Unique identifier of a logical task in a :class:`Trace`.

    :param pid: PID of the task. ``None`` indicates the PID is not important.
    :type pid: int

    :param comm: Name of the task. ``None`` indicates the name is not important.
        This is useful to describe tasks like PID0, which can have multiple
        names associated.
    :type comm: str
    """

    # Prevent creation of a __dict__. This allows a more compact representation
    __slots__ = []

    def __init__(self, *args, **kwargs):
        # TODO: remove that once this trace-cmd issue is solved in one way or another:
        # https://bugzilla.kernel.org/show_bug.cgi?id=204979
        if self.comm == '<...>':
            raise ValueError('Invalid comm name "<...>"')

    def __str__(self):
        if self.pid is not None and self.comm is not None:
            out = '{}:{}'.format(self.pid, self.comm)
        else:
            out = str(self.comm if self.comm is not None else self.pid)

        return '[{}]'.format(out)


class TraceBase(abc.ABC):
    """
    Base class for common functionalities between :class:`Trace` and :class:`TraceView`
    """

    def __init__(self):
        # Import here to avoid a circular dependency issue at import time
        # with lisa.analysis.base
        from lisa.analysis.proxy import AnalysisProxy
        self.analysis = AnalysisProxy(self)

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

        This function assumes that at time [n] the event starts and at [n+1]
        the event stops, so the formula for the returned value is::

                  |        |            |
                  |        |            |
                  |        |            |
            ------+--------+------------+------
                [n-1]     [n]         [n+1]

            delta[n] = index[n+1] - index[n]

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

        if col_name in df.columns:
            raise RuntimeError("Column {} is already present in the dataframe".
                               format(col_name))

        if not inplace:
            df = df.copy()

        df[col_name] = df.index

        if not df.empty:
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
        super().__init__()

        self.base_trace = trace

        t_min = window[0]
        t_max = window[1]

        df_list = [
            trace.df_events(event)
            for event in trace.available_events
        ]

        if t_min is not None:
            start = self.base_trace.end
            for df in df_list:
                df = df[t_min:]
                if not df.empty:
                    start = min(start, df.index[0])
            t_min = start
        else:
            t_min = self.base_trace.start

        if t_max is not None:
            end = self.base_trace.start
            for df in df_list:
                df = df[:t_max]
                if not df.empty:
                    end = max(end, df.index[-1])
            t_max = end
        else:
            t_max = self.base_trace.end

        self.start = t_min
        self.end = t_max
        self.time_range = t_max - t_min

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

    :param events: events to be parsed (all the events used by analysis by
        default)
    :type events: str or list(str)

    :param platform: a dictionary containing information about the target
        platform
    :type platform: dict

    :param window: time window to consider when parsing the trace
    :type window: tuple(int, int)

    :param normalize_time: Make the first timestamp in the trace 0 instead
        of the system timestamp that was captured when tracing.
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
    :ivar time_range: Maximum timespan for all collected events
    :ivar available_events: List of events available in the parsed trace
    """

    def __init__(self,
                 trace_path,
                 plat_info=None,
                 events=None,
                 normalize_time=True,
                 trace_format='FTrace',
                 plots_dir=None,
                 plots_prefix=''):

        super().__init__()

        logger = self.get_logger()

        if plat_info is None:
            plat_info = PlatformInfo()

        # The platform information used to run the experiments
        self.plat_info = plat_info

        self.normalize_time = normalize_time

        proxy_cls = type(self.analysis)
        self.events = self._process_events(events, proxy_cls)

        # Path to the trace file
        self.trace_path = trace_path

        # By default, use the trace dir to save plots
        self.plots_dir = plots_dir if plots_dir else os.path.dirname(trace_path)

        self.plots_prefix = plots_prefix

        self._parse_trace(self.trace_path, trace_format, normalize_time)

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

    @deprecate('Direct access to underlying ftrace object is discouraged as this is now an implementation detail of that class which could change in the future',
        deprecated_in='2.0',
        removed_in='2.1'
    )
    @property
    def ftrace(self):
        """
        Underlying :class:`trappy.ftrace.FTrace`.
        """
        return self._ftrace


    @classmethod
    @contextlib.contextmanager
    def from_target(cls, target, events=None, buffer_size=10240, filepath=None, **kwargs):
        """
        Context manager that can be used to collect a :class:`Trace` directly
        from a :class:`lisa.target.Target` without needing to setup an
        :class:`FtraceCollector`.

        **Example**::

            from lisa.trace import Trace
            from lisa.target import Target

            target = Target.from_default_conf()

            with Trace.from_target(target, events=['sched_switch', 'sched_wakeup']) as trace:
                target.execute('echo hello world')
                # DO NOT USE trace object inside the `with` statement

            trace.analysis.tasks.plot_tasks_total_residency(filepath='plot.png')


        :param target: Target to connect to.
        :type target: Target

        :param events: ftrace events to collect and parse in the trace.
        :type events: list(str)

        :param buffer_size: Size of the ftrace ring buffer.
        :type buffer_size: int

        :param filepath: If set, the trace file will be saved at that location.
            Otherwise, a temporary file is created and removed as soon as the
            parsing is finished.
        :type filepath: str or None

        :Variable keyword arguments: Forwarded to :class:`Trace`.
        """
        ftrace_coll = FtraceCollector(target, events=events, buffer_size=buffer_size)
        plat_info = target.plat_info

        class TraceProxy(TraceBase):
            def get_view(self, *args, **kwargs):
                return self.base_trace.get_view(*args, **kwargs)

            def __getattr__(self, attr):
                try:
                    base_trace = self.__dict__['base_trace']
                except KeyError:
                    raise RuntimeError('The trace instance can only be used outside its "with" statement.')
                else:
                    return getattr(base_trace, attr)

        proxy = TraceProxy()

        with ftrace_coll:
            yield proxy

        if filepath:
            cm = nullcontext(filepath)
        else:
            @contextlib.contextmanager
            def cm_func():
                with tempfile.NamedTemporaryFile(suffix='.dat', delete=True) as temp:
                    yield temp.name

            cm = cm_func()

        with cm as path:
            ftrace_coll.get_trace(path)
            trace = cls(
                path,
                events=events,
                plat_info=plat_info,
                **kwargs
            )

        proxy.base_trace = trace

    @staticmethod
    def _process_events(events, proxy_cls):
        """
        Process the `events` parameter of :meth:`Trace.__init__`.

        :param events: single event name or list of events names
        :type events: str or list(str)
        """
        # Merge all the events that we expect to see, plus all the ones that
        # can be useful to any analysis method
        if events is None:
            events = sorted(proxy_cls.get_all_events())
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

    def _parse_trace(self, path, trace_format, normalize_time):
        """
        Internal method in charge of performing the actual parsing of the
        trace.

        :param path: path to the trace file
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
        elif trace_format.upper() == 'FTRACE':
            logger.debug('Parsing FTrace format...')
            trace_class = trappy.FTrace
        else:
            raise ValueError("Unknown trace format {}".format(trace_format))

        # Make sure event names are not unicode strings
        self._ftrace = trace_class(path, scope="custom", events=self.events,
                                  normalize_time=normalize_time)

        # trappy sometimes decides to be "clever" and overrules the path to be
        # used, even though it was specifically asked for a given file path
        assert path == self._ftrace.trace_path

        # Check for events available on the parsed trace
        self.available_events = self._check_available_events()
        if not self.available_events:
            raise ValueError('The trace does not contain useful events')

        self.basetime = self._ftrace.basetime

        self._compute_timespan()

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
        available_events = []
        for val in self._ftrace.get_filters(key):
            obj = getattr(self._ftrace, val)
            if not obj.data_frame.empty:
                available_events.append(val)
        logger.debug('Events found on trace: {}'.format(', '.join(available_events)))
        for evt in available_events:
            logger.debug(' - %s', evt)
        return available_events

    @memoized
    def _get_task_maps(self):
        """
        Give the mapping from PID to task names, and the opposite.

        The names or PIDs are listed in appearance order.
        """

        name_to_pid = {}
        pid_to_name = {}

        # Merge the list values if the key already exists rather than
        # overriding them
        def update_mapping(existing, new):
            for key, new_val in new.items():
                existing.setdefault(key, []).extend(new_val)

        # Keep only the values, in appearance order according to the timestamp
        # index
        def finalize_mapping(mapping):
            keep_values = lambda items: list(zip(*items))[1]
            sort_by_index = lambda values: sorted(values, key=lambda index_v: index_v[0])
            return {
                # Remove duplicates and only keep the first occurence of each
                k: deduplicate(
                    # Sort by index values, i.e. appearance order
                    keep_values(sort_by_index(values)),
                    keep_last=False
                )
                for k, values in mapping.items()
            }

        def create_mapping(df, key_col, value_col):
            return {
                k: [
                    # save the index at which that value appeared so we
                    # conserve appearance order across events
                    (df[df[value_col] == value].index[0], value)
                    for value in df[df[key_col] == k][value_col].unique()
                ]
                for k in df[key_col].unique()
            }

        def load(event, name_col, pid_col):
            df = self.df_events(event)
            update_mapping(name_to_pid, create_mapping(df, name_col, pid_col))
            update_mapping(pid_to_name, create_mapping(df, pid_col, name_col))

        if 'sched_load_avg_task' in self.available_events:
            load('sched_load_avg_task', 'comm', 'pid')

        if 'sched_wakeup' in self.available_events:
            load('sched_wakeup', '__comm', '__pid')

        if 'sched_switch' in self.available_events:
            load('sched_switch', 'prev_comm', 'prev_pid')
            load('sched_switch', 'next_comm', 'next_pid')

        if not (name_to_pid and pid_to_name):
            raise RuntimeError('Failed to load tasks names, sched_switch, sched_wakeup, or sched_load_avg_task events are needed')

        name_to_pid = finalize_mapping(name_to_pid)
        pid_to_name = finalize_mapping(pid_to_name)

        return (name_to_pid, pid_to_name)

    @property
    def _task_name_map(self):
        return self._get_task_maps()[0]

    @property
    def _task_pid_map(self):
        return self._get_task_maps()[1]

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
        start = []
        end = []
        for event in self.available_events:
            df = self.df_events(event)
            start.append(df.index[0])
            end.append(df.index[-1])

        duration = max(end) - min(start)

        self.start = 0 if self.normalize_time else self.basetime
        self.end = self.start + duration
        self.time_range = self.end - self.start

        self.get_logger().debug('Trace contains events from %s to %s',
                                self.start, self.end)

    def get_task_name_pids(self, name, ignore_fork=True):
        """
        Get the PIDs of all tasks with the specified name.

        The same PID can have different task names, mainly because once a task
        is generated it inherits the parent name and then its name is updated
        to represent what the task really is.

        :param name: task name
        :type name: str

        :param ignore_fork: Hide the PIDs of tasks that initially had ``name``
            but were later renamed. This is common for shell processes for
            example, which fork a new task, inheriting the shell name, and then
            being renamed with the final "real" task name
        :type ignore_fork: bool

        :return: a list of PID for tasks which name matches the required one.
        """
        pids = self._task_name_map[name]

        if ignore_fork:
            pids = [
                pid
                for pid in pids
                # Only keep the PID if its last name was the name we are
                # looking for.
                if self._task_pid_map[pid][-1] == name
            ]

        return pids

    @deprecate('This method has been deprecated and is an alias',
        deprecated_in='2.0',
        removed_in='2.1',
        replaced_by=get_task_name_pids,
    )
    def get_task_by_name(self, name):
        return self.get_task_name_pids(name, ignore_fork=True)

    def get_task_pid_names(self, pid):
        """
        Get the all the names of the task(s) with the specified PID, in
        appearance order.

        The same PID can have different task names, mainly because once a task
        is generated it inherits the parent name and then its name is
        updated to represent what the task really is.

        :param name: task PID
        :type name: int

        :return: the name of the task which PID matches the required one,
                 the last time they ran in the current trace
        """
        return self._task_pid_map[pid]

    @deprecate('This function raises exceptions when faced with ambiguity instead of giving the choice to the user',
        deprecated_in='2.0',
        removed_in='2.1',
        replaced_by=get_task_pid_names,
    )
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
        name_list = self.get_task_pid_names(pid)

        if len(name_list) > 2:
            raise RuntimeError('The PID {} had more than two names in its life: {}'.format(
                pid, name_list,
            ))

        return name_list[-1]

    def get_task_ids(self, task, update=True):
        """
        Similar to :meth:`get_task_id` but returns a list with all the
        combinations, instead of raising an exception.

        :param task: Either the task name, the task PID, or a tuple ``(pid, comm)``
        :type task: int or str or tuple(int, str)

        :param update: If a partially-filled :class:`TaskID` is passed (one of
            the fields set to ``None``), returns a complete :class:`TaskID`
            instead of leaving the ``None`` fields.
        :type update: bool
        """

        def comm_to_pid(comm):
            try:
                pid_list = self._task_name_map[comm]
            except IndexError:
                raise ValueError('trace does not have any task named "{}"'.format(comm))

            return pid_list

        def pid_to_comm(pid):
            try:
                comm_list = self._task_pid_map[pid]
            except IndexError:
                raise ValueError('trace does not have any task PID {}'.format(pid))

            return comm_list


        if isinstance(task, str):
            task_ids = [
                TaskID(pid=pid, comm=task)
                for pid in comm_to_pid(task)
            ]
        elif isinstance(task, numbers.Number):
            task_ids = [
                TaskID(pid=task, comm=comm)
                for comm in pid_to_comm(task)
            ]
        else:
            pid, comm = task
            if pid is None and comm is None:
                raise ValueError('TaskID needs to have at least one of PID or comm specified')

            if update:
                non_none = pid if comm is None else comm
                task_ids = self.get_task_ids(non_none)
            else:
                task_ids = [TaskID(pid=pid, comm=comm)]

        return task_ids

    def get_task_id(self, task, update=True):
        """
        Helper that resolves a task PID or name to a :class:`TaskID`.

        :param task: Either the task name, the task PID, or a tuple ``(pid, comm)``
        :type task: int or str or tuple(int, str)

        :param update: If a partially-filled :class:`TaskID` is passed (one of
            the fields set to ``None``), returns a complete :class:`TaskID`
            instead of leaving the ``None`` fields.
        :type update: bool

        :raises ValueError: If there the input matches multiple tasks in the trace.
            See :meth:`get_task_ids` to get all the ambiguous alternatives
            instead of an exception.
        """
        task_ids = self.get_task_ids(task, update=update)
        if len(task_ids) > 1:
            raise ValueError('More than one TaskID matching: {}'.format(task_ids))

        return task_ids[0]


    @deprecate(deprecated_in='2.0', removed_in='2.1', replaced_by=get_task_id)
    def get_task_pid(self, task):
        """
        Helper that takes either a name or a PID and always returns a PID

        :param task: Either the task name or the task PID
        :type task: int or str or tuple(int, str)
        """
        return self.get_task_id(task).pid

    def get_tasks(self):
        """
        Get a dictionary of all the tasks in the Trace.

        :return: a dictionary which maps each PID to the corresponding list of
                 task name
        """
        return self._task_pid_map

    @property
    @memoized
    def task_ids(self):
        """
        List of all the :class:`TaskID` in the trace, sorted by PID.
        """
        key = lambda k_v: k_v[0]

        return [
            TaskID(pid=pid, comm=comm)
            for pid, comms in sorted(self._task_pid_map.items(), key=key)
            for comm in comms
        ]

    def show(self):
        """
        Open the parsed trace using the most appropriate native viewer.

        The native viewer depends on the specified trace format:
        - ftrace: open using kernelshark
        - systrace: open using a browser

        In both cases the native viewer is assumed to be available in the host
        machine.
        """
        if isinstance(self._ftrace, trappy.FTrace):
            return os.popen("kernelshark {}".format(shlex.quote(self.trace_path)))
        if isinstance(self._ftrace, trappy.SysTrace):
            return webbrowser.open(self.trace_path)

    def df_events(self, event):
        """
        Get a dataframe containing all occurrences of the specified trace event
        in the parsed trace.

        :param event: Trace event name
        :type event: str
        """

        if event not in self.available_events:
            raise MissingTraceEventError(
                TraceEventChecker(event),
                available_events=self.available_events,
            )

        return getattr(self._ftrace, event).data_frame

###############################################################################
# Trace Events Sanitize Methods
###############################################################################
    def _sanitize_SchedCpuCapacity(self):
        """
        Add more columns to cpu_capacity data frame if the energy model is
        available and the platform is big.LITTLE.
        """
        if not self.has_events('cpu_capacity') \
           or 'nrg-model' not in self.plat_info:
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
           or 'nrg-model' not in self.plat_info:
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

        df = self.df_events('sched_overutilized')
        self.add_events_deltas(df, 'len')

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

    def _sanitize_CpuFrequency(self):
        """
        Rename some columns and add fake devlib frequency events
        """
        logger = self.get_logger()
        if not self.has_events('cpu_frequency_devlib') \
           or 'freq-domains' not in self.plat_info:
            return

        devlib_freq = self.df_events('cpu_frequency_devlib')
        devlib_freq.rename(columns={'cpu_id':'cpu'}, inplace=True)
        devlib_freq.rename(columns={'state':'frequency'}, inplace=True)

        domains = self.plat_info['freq-domains']

        # devlib always introduces fake cpu_frequency events, in case the
        # OS has not generated cpu_frequency envets there are the only
        # frequency events to report
        if not self.has_events('cpu_frequency'):
            # Register devlib injected events as 'cpu_frequency' events
            self._ftrace.cpu_frequency.data_frame = devlib_freq
            df = devlib_freq
            self.available_events.append('cpu_frequency')

        # make sure fake cpu_frequency events are never interleaved with
        # OS generated events
        else:
            df = self.df_events('cpu_frequency')
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

            self._ftrace.cpu_frequency.data_frame = df


class TraceEventCheckerBase(abc.ABC, Loggable):
    """
    ABC for events checker classes.

    Event checking can be achieved using a boolean expression on expected
    events.

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated
        method is applied.  If no such attribute is found, no check will be
        done.
    :type check: bool
    """
    def __init__(self, check=True):
        self.check = check

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
        def unwrap_down_to(obj):
            return hasattr(obj, 'used_events')

        try:
            # we want to see through all other kinds of wrappers, down to the
            # one that matters to us
            unwrapped_f = inspect.unwrap(f, stop=unwrap_down_to)
            used_events = unwrapped_f.used_events
        except AttributeError:
            checker = self
        else:
            # Update the existing checker inplace to avoid adding an extra
            # level of wrappers.
            checker = AndTraceEventChecker([self, used_events])
            unwrapped_f.used_events = checker
            return f

        sig = inspect.signature(f)
        if self.check and sig.parameters:
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

    :param event: Name of the event to check for.
    :type event: str

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated
        method is applied.  If no such attribute is found, no check will be
        done.
    :type check: bool
    """
    def __init__(self, event, check=True):
        super().__init__(check=check)
        self.event = event

    def get_all_events(self):
        return {self.event}

    def check_events(self, event_set):
        if self.event not in event_set:
            raise MissingTraceEventError(self, available_events=event_set)

    def _str_internal(self, style=None, wrapped=True):
        template = '``{}``' if style == 'rst' else '{}'
        return template.format(self.event)

class AssociativeTraceEventChecker(TraceEventCheckerBase):
    """
    Base class for associative operators like `and` and `or`
    """
    def __init__(self, op_str, event_checkers, check=True, prefix_str=''):
        super().__init__(check=check)
        checker_list = []
        optional_checker_list = []
        for checker in event_checkers:
            # "unwrap" checkers of the same type, to avoid useless levels of
            # nesting. This is valid since the operator is known to be
            # associative. We don't use isinstance to avoid merging checkers
            # that may have different semantics.
            if type(checker) is type(self):
                checker_list.extend(checker.checkers)
            # Aggregate them separately to avoid having multiple of them
            elif isinstance(checker, OptionalTraceEventChecker):
                optional_checker_list.append(checker)
            else:
                checker_list.append(checker)

        if optional_checker_list:
            checker_list.append(OptionalTraceEventChecker(optional_checker_list))

        # Avoid having the same event twice at the same level
        def key(checker):
            if isinstance(checker, TraceEventChecker):
                return checker.event
            else:
                return checker
        checker_list = deduplicate(checker_list, key=key)

        self.checkers = checker_list
        self.op_str = op_str
        self.prefix_str = prefix_str

    def get_all_events(self):
        events = set()
        for checker in self.checkers:
            events.update(checker.get_all_events())
        return events

    @classmethod
    def from_events(cls, events, **kwargs):
        """
        Build an instance of the class, converting ``str`` to
        ``TraceEventChecker``.

        :param events: Sequence of events
        :type events: list(str or TraceEventCheckerBase)
        """
        def make_event(e):
            if isinstance(e, TraceEventCheckerBase):
                return e
            else:
                return TraceEventChecker(e)

        return cls({
            make_event(e)
            for e in events
        }, **kwargs)

    def _str_internal(self, style=None, wrapped=True):
        op_str = ' {} '.format(self.op_str)
        # Sort for stable output
        checker_list = sorted(self.checkers, key=lambda c: str(c))
        unwrapped_str = self.prefix_str + op_str.join(
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
    def __init__(self, event_checkers, **kwargs):
        super().__init__('or', event_checkers, **kwargs)

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
                cls(failed_checker_set),
                available_events=event_set,
            )

class OptionalTraceEventChecker(AssociativeTraceEventChecker):
    """
    Do not check anything, but exposes the information that the events may be
    used if present.

    :param event_checkers: Event checkers that may be used
    :type event_checkers: list(TraceEventCheckerBase)
    """
    def __init__(self, event_checkers, **kwargs):
        super().__init__(',', event_checkers, prefix_str='optional: ', **kwargs)

    def check_events(self, event_set):
        return

class AndTraceEventChecker(AssociativeTraceEventChecker):
    """
    Check that all the given event checkers are satisfied.

    :param event_checkers: Event checkers to check for
    :type event_checkers: list(TraceEventCheckerBase)
    """
    def __init__(self, event_checkers, **kwargs):
        super().__init__('and', event_checkers, **kwargs)

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
                cls(failed_checker_set),
                available_events=event_set,
            )

    def doc_str(self):
        joiner = '\n' + '    '
        rst = joiner + joiner.join(
            '* {}'.format(c._str_internal(style='rst', wrapped=False))
            # Sort for stable output
            for c in sorted(self.checkers, key=lambda c: str(c))
        )
        return rst

def requires_events(*events, **kwargs):
    """
    Decorator for methods that require some given trace events.

    :param events: The list of required events
    :type events: list(str or TraceEventCheckerBase)

    :param check: Check that the listed events are present in the
        ``self.trace`` attribute of the instance on which the decorated method
        is applied.  If no such attribute is found, no check will be done.
    :type check: bool

    """
    return AndTraceEventChecker.from_events(events, **kwargs)

def requires_one_event_of(*events, **kwargs):
    """
    Same as :func:`requires_events` with logical `OR` semantic.
    """
    return OrTraceEventChecker.from_events(events, **kwargs)

def may_use_events(*events, **kwargs):
    """
    Same as :func:`requires_events` but just exposes some events that may be used
    if presents.
    """
    return OptionalTraceEventChecker.from_events(events, **kwargs)

class MissingTraceEventError(RuntimeError, ValueError):
    """
    :param missing_events: The missing trace events
    :type missing_events: TraceEventCheckerBase
    """
    def __init__(self, missing_events, available_events=None):
        msg = "Trace is missing the following required events: {}".format(missing_events)
        if available_events:
            msg += '. Available events are: {}'.format(
                ', '.join(available_events))

        super().__init__(msg)
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

class FtraceCollector(Loggable, Configurable):
    """
    Thin wrapper around :class:`devlib.FtraceCollector`.

    {configurable_params}
    """

    CONF_CLASS = FtraceConf

    def __init__(self, target, events=None, functions=None, buffer_size=10240, autoreport=False, **kwargs):
        events = events or []
        functions = functions or []
        kwargs.update(dict(
            target=target,
            events=events,
            functions=functions,
            buffer_size=buffer_size,
            autoreport=autoreport,
        ))
        self.check_init_param(**kwargs)

        self.events = events
        kernel_events = [
            event for event in events
            if self._is_kernel_event(event)
        ]
        # Only pass true kernel events to devlib, as it will reject any other.
        kwargs['events'] = kernel_events

        self._collector = devlib.FtraceCollector(**kwargs)

        # Ensure we have trace-cmd on the target
        self.target.install_tools(['trace-cmd'])

    def __getattr__(self, attr):
        return getattr(self._collector, attr)

    def __enter__(self):
        return self._collector.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._collector.__exit__(*args, **kwargs)

    def _is_kernel_event(self, event):
        """
        Return ``True`` if the event is a kernel event.

        This allows events to be passed around in the collector to be inspected
        and used by other entities and then ignored when actually setting up
        ``trace-cmd``. An example are the userspace events generated by
        ``rt-app``.
        """
        return not event.startswith('rtapp_')

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
        kwargs['target'] = target
        cls.check_init_param(**kwargs)
        return cls(**kwargs)

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


class DmesgCollector(devlib.DmesgCollector):
    """
    Wrapper around :class:`devlib.trace.dmesg.DmesgCollector`.

    It installs the ``dmesg`` tool automatically on the target upon creation,
    so we know what version is being is used.
    """
    def __init__(self, target, *args, **kwargs):
        # Make sure we use the binary that is known to work
        target.install_tools(['dmesg'])
        super().__init__(target, *args, **kwargs)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
