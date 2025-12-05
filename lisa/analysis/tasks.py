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
"""
Linux kernel scheduler tasks analysis.
"""

from enum import Enum
import itertools
import warnings
import typing
from numbers import Number
from operator import itemgetter
import re
import functools
from typing import NamedTuple, Optional
import operator

import numpy as np
import pandas as pd
import holoviews as hv
import bokeh.models
import polars as pl
from colorcet import glasbey_hv

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import memoized, kwargs_forwarded_to, deprecate, order_as, is_running_ipython
from lisa.datautils import df_filter_task_ids, series_rolling_apply, series_refit_index, df_refit_index, df_deduplicate, df_split_signals, df_add_delta, df_window, df_update_duplicates, df_combine_duplicates, SignalDesc, _df_to, NO_INDEX, _polars_fast_collect, _polars_fast_collect_all
from lisa.trace import requires_events, will_use_events_from, may_use_events, CPU, MissingTraceEventError, OrTraceEventChecker
from lisa.notebook import _hv_neutral, plot_signal
from lisa._typeclass import FromString


# We cannot override __init__ in a NamedTuple so we inherit instead:
# https://github.com/python/typing/issues/526
class _TaskID(NamedTuple):
    pid: Optional[int]
    comm: Optional[str]

class TaskID(_TaskID):
    """
    Unique identifier of a logical task in a :class:`lisa.trace.Trace`.

    :param pid: PID of the task. ``None`` indicates the PID is not important.
    :type pid: int

    :param comm: Name of the task. ``None`` indicates the name is not important.
        This is useful to describe tasks like PID0, which can have multiple
        names associated.
    :type comm: str
    """
    def __init__(self, *args, **kwargs):
        # pylint: disable=unused-argument
        super().__init__()
        # This happens when the number of saved PID/comms entries in the trace
        # is too low
        if self.comm == '<...>':
            raise ValueError('Invalid comm name "<...>", please increase saved_cmdlines_nr value on FtraceCollector')

    def __str__(self):
        if self.pid is not None and self.comm is not None:
            out = f'{self.pid}:{self.comm}'
        else:
            out = str(self.comm if self.comm is not None else self.pid)

        return f'[{out}]'

    _STR_PARSE_REGEX = re.compile(r'\[?([0-9]+):([a-zA-Z0-9_-]+)\]?')


class _TaskIDFromStringInstance(FromString, types=TaskID):
    """
    Instance of :class:`lisa._typeclass.FromString` for :class:`TaskID` type.
    """
    @classmethod
    def from_str(cls, string):
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        try:
            pid = int(string)
            comm = None
        except ValueError:
            match = cls._STR_PARSE_REGEX.match(string)
            if match:
                pid = int(match.group(1))
                comm = match.group(2)
            else:
                pid = None
                comm = string

        return cls(pid=pid, comm=comm)

    @classmethod
    def get_format_description(cls, short):
        if short:
            return 'task ID'
        else:
            return textwrap.dedent("""
            Can be any of:
               * a PID
               * a task name
               * a PID (first) and a name (second): pid:name
            """).strip()


class _TaskIDSeqFromStringInstance(FromString, types=(typing.List[TaskID], typing.Sequence[TaskID])):
    """
    Instance of :class:`lisa._typeclass.FromString` for lists :class:`TaskID` type.
    """
    @classmethod
    def from_str(cls, string):
        """
        The format is a comma-separated list of :class:`TaskID`.
        """
        from_str = FromString(TaskID).from_str
        return [
            from_str(string.strip())
            for string in string.split(',')
        ]

    @classmethod
    def get_format_description(cls, short):
        return 'comma-separated TaskIDs'


class StateInt(int):
    """
    An tweaked int for :class:`lisa.analysis.tasks.TaskState`
    """
    def __new__(cls, value, char="", doc=""):
        new = super().__new__(cls, value)
        new.char = char
        new.__doc__ = doc
        return new

    def __or__(self, other):
        char = self.char

        if other.char:
            char = "|".join(char + other.char)

        return type(self)(
            int(self) | int(other),
            char=(self.char + other.char))

    # This is needed for some obscure reason (maybe a bug in std library ?)
    # In any case, if we don't provide that, Enum's metaclass will "sabbotage"
    # pickling, and by doing so, will set cls.__module__ = '<unknown>'
    __reduce__ = int.__reduce__


class TaskState(StateInt, Enum):
    """
    Represents the task state as visible in sched_switch

    * Values are extracted from include/linux/sched.h
    * Chars are extracted from fs/proc/array.c:get_task_state()
    """
    # pylint-suppress: bad-whitespace
    TASK_RUNNING = 0x0000, "R", "Running"
    TASK_INTERRUPTIBLE = 0x0001, "S", "Sleeping"
    TASK_UNINTERRUPTIBLE = 0x0002, "D", "Disk sleep"
    # __ has a special meaning in Python so let's not do that
    TASK_STOPPED = 0x0004, "T", "Stopped"
    TASK_TRACED = 0x0008, "t", "Tracing stop"

    EXIT_DEAD = 0x0010, "X", "Dead"
    EXIT_ZOMBIE = 0x0020, "Z", "Zombie"

    # Apparently not visible in traces
    # EXIT_TRACE           = (EXIT_ZOMBIE[0] | EXIT_DEAD[0])

    TASK_PARKED = 0x0040, "P", "Parked"
    TASK_DEAD = 0x0080, "I", "Idle"
    TASK_WAKEKILL = 0x0100
    TASK_WAKING = 0x0200, "W", "Waking"  # LISA-only char definition
    TASK_NOLOAD = 0x0400
    TASK_NEW = 0x0800
    TASK_STATE_MAX = 0x1000

    # LISA synthetic states

    # Used to differenciate runnable (R) vs running (A)
    TASK_ACTIVE = 0x2000, "A", "Active"
    TASK_RENAMED = 0x2001, "N", "Renamed"
    # Used when the task state is unknown
    TASK_UNKNOWN = -1, "U", "Unknown"

    @classmethod
    def list_reported_states(cls):
        """
        List the states that can be reported in a ``sched_switch`` trace

        See include/linux/sched.h:TASK_REPORT
        """
        return [state for state in cls if 0 <= state <= cls.TASK_DEAD]

    # Could use IntFlag instead once we move to Python 3.6
    @classmethod
    @memoized
    def sched_switch_str(cls, value):
        """
        Get the task state string that would be used in a ``sched_switch`` event

        :param value: The task state value
        :type value: int

        Tries to emulate what is done in include/trace/events:TRACE_EVENT(sched_switch)
        """
        def find_states(value, states):
            return [
                state.char
                for state in states
                if value & state.value
            ]

        reported_states = cls.list_reported_states()
        res = '|'.join(find_states(value, reported_states))
        res = res if res else cls.TASK_RUNNING.char

        # Flag the presence of unreportable states with a "+"
        unreportable_states = [
            state for state in cls
            if state.value >= 0 and state not in reported_states
        ]
        if find_states(value, unreportable_states):
            res += '+'

        return res

    @classmethod
    def from_sched_switch_str(cls, string):
        """
        Build a :class:`StateInt` from a string as it would be used in
        ``sched_switch`` event's ``prev_state`` field.

        :param string: String to parse.
        :type string: str
        """
        state = 0
        for _state in cls:
            if _state.char in string:
                state |= _state

        return state


class TasksAnalysis(TraceAnalysisBase):
    """
    Support for Tasks signals analysis.

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'tasks'


    @memoized
    def _get_task_maps(self):
        """
        Give the mapping from PID to task names, and the opposite.

        The names or PIDs are listed in appearance order.
        """
        trace = self.trace.get_view(df_fmt='polars-lazyframe')

        mapping_df_list = []
        def _load(event, name_col, pid_col):
            df = trace.df_event(event)
            grouped = df.group_by(name_col, pid_col)

            # Get timestamp of first occurrences of each key/value combinations
            mapping_df = grouped.first().select(
                'Time',
                pid=pl.col(pid_col),
                # Ensure we have a Categorical dtype, otherwise we might not be
                # able to successfully concatenate a String and Categorical
                # column
                name=pl.col(name_col).cast(pl.Categorical),
            )
            mapping_df_list.append(mapping_df)

        missing = []
        def load(event, *args, **kwargs):
            try:
                _load(event, *args, **kwargs)
            except MissingTraceEventError as e:
                missing.append(e.missing_events)

        # The "pid" field of task_rename has disappeared since:
        # https://lkml.org/lkml/2024/11/8/665
        load('task_rename', 'oldcomm', '__pid')
        load('task_rename', 'newcomm', '__pid')

        load('sched_switch', 'prev_comm', 'prev_pid')
        load('sched_switch', 'next_comm', 'next_pid')

        if not mapping_df_list:
            missing = OrTraceEventChecker.from_events(events=missing)
            raise MissingTraceEventError(missing, available_events=trace.available_events)

        df = pl.concat(mapping_df_list).sort('Time')
        df = df.unique(
            subset=['name', 'pid'],
            keep='first',
            maintain_order=True,
        )
        df = df.select('name', 'pid')

        with pl.StringCache():
            df = _polars_fast_collect(df)

        def finalize(df, key_col):
            assert len(df.columns) == 2
            # Aggregate the values for each key and convert to python types
            return {
                key: [x[0] for x in values]
                for key, values in df.rows_by_key(key_col).items()
            }

        name_to_pid = finalize(df, 'name')
        pid_to_name = finalize(df, 'pid')

        return (name_to_pid, pid_to_name)

    @property
    def _task_name_map(self):
        return self._get_task_maps()[0]

    @property
    def _task_pid_map(self):
        return self._get_task_maps()[1]

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
        removed_in='4.0',
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
            raise RuntimeError(f'The PID {pid} had more than two names in its life: {name_list}')

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
            except KeyError:
                # pylint: disable=raise-missing-from
                raise ValueError(f'trace does not have any task named "{comm}"')

            return pid_list

        def pid_to_comm(pid):
            try:
                comm_list = self._task_pid_map[pid]
            except KeyError:
                # pylint: disable=raise-missing-from
                raise ValueError(f'trace does not have any task PID {pid}')

            return comm_list

        if isinstance(task, str):
            task_ids = [
                TaskID(pid=pid, comm=task)
                for pid in comm_to_pid(task)
            ]
        elif isinstance(task, Number):
            task_ids = [
                TaskID(pid=task, comm=comm)
                for comm in pid_to_comm(task)
            ]
        else:
            pid, comm = task
            if pid is None and comm is None:
                raise ValueError('TaskID needs to have at least one of PID or comm specified')

            if update and (pid is None or comm is None):
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
            raise ValueError(f'More than one TaskID matching: {task_ids}')

        return task_ids[0]

    @deprecate(deprecated_in='2.0', removed_in='4.0', replaced_by=get_task_id)
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
        return [
            TaskID(pid=pid, comm=comm)
            for pid, comms in sorted(self._task_pid_map.items(), key=itemgetter(0))
            for comm in comms
        ]

    @requires_events('sched_switch')
    def cpus_of_tasks(self, tasks):
        """
        Return the list of CPUs where the ``tasks`` executed.

        :param tasks: Task names or PIDs or ``(pid, comm)`` to look for.
        :type tasks: list(int or str or tuple(int, str))
        """
        trace = self.trace
        df = trace.df_event('sched_switch')[['next_pid', 'next_comm', '__cpu']]

        task_ids = [self.get_task_id(task, update=False) for task in tasks]
        df = df_filter_task_ids(df, task_ids, pid_col='next_pid', comm_col='next_comm')
        cpus = df['__cpu'].unique()

        return sorted(cpus)

    def _get_task_pid_name(self, pid):
        """
        Get the last name the given PID had.
        """
        return self.get_task_pid_names(pid)[-1]

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.df_method
    @requires_events('sched_wakeup')
    def df_tasks_wakeups(self):
        """
        The number of wakeups per task

        :returns: a :class:`pandas.DataFrame` with:

          * Task PIDs as index
          * A ``wakeups`` column (The number of wakeups)
        """
        df = self.trace.df_event('sched_wakeup')

        wakeups = df.groupby('pid', observed=True, sort=False, group_keys=False).count()["comm"]
        df = pd.DataFrame(wakeups).rename(columns={"comm": "wakeups"})
        df["comm"] = df.index.map(self._get_task_pid_name)

        return df

    @TraceAnalysisBase.df_method
    @df_tasks_wakeups.used_events
    def df_top_wakeup(self, min_wakeups=100):
        """
        Tasks which wakeup more frequently than a specified threshold.

        :param min_wakeups: minimum number of wakeups
        :type min_wakeups: int
        """
        df = self.df_tasks_wakeups()

        # Compute number of samples above threshold
        df = df[df.wakeups > min_wakeups]
        df = df.sort_values(by="wakeups", ascending=False)

        return df

    @TraceAnalysisBase.df_method
    @requires_events('sched_switch')
    def df_rt_tasks(self, min_prio=99):
        """
        Tasks with RT priority

        .. note:: priorities uses scheduler values, thus: the lower the value the
          higher is the task priority.
          RT   Priorities: [  0..99]
          FAIR Priorities: [100..120]

        .. note:: RT and DL (deadline) tasks share the priority 0 on modern
            kernels. Since this method only filters by priority, it will also
            select DL tasks as a result.

        :param min_prio: minimum priority
        :type min_prio: int

        :returns: a :class:`pandas.DataFrame` with:

          * Task PIDs as index
          * A ``prio`` column (The priority of the task)
          * A ``comm`` column (The name of the task)
        """
        trace = self.trace.get_view(df_fmt='polars-lazyframe')
        df = trace.df_event('sched_switch')

        df = df.filter(pl.col('next_prio') <= pl.lit(min_prio))
        df = df.select(('next_pid', 'next_prio'))
        df = df.unique(['next_pid', 'next_prio'])
        # Order by priority
        df = df.sort(['next_prio', 'next_pid'], descending=False)
        df = df.rename({'next_pid': 'pid', 'next_prio': 'prio'})

        df = df.with_columns(
            comm=pl.col('pid').replace_strict(
                {
                    pid: comms[-1]
                    for pid, comms in self._task_pid_map.items()
                    if comms
                },
                default=None,
                return_dtype=pl.Categorical,
            )
        )
        return df

    @requires_events('sched_switch', 'sched_wakeup')
    @will_use_events_from('task_rename')
    @may_use_events('sched_wakeup_new')
    @TraceAnalysisBase.df_method
    def _df_tasks_states(self, tasks=None):
        """
        Compute tasks states for all tasks.

        :param tasks: If specified, states of these tasks only will be yielded.
            The :class:`lisa.analysis.tasks.TaskID` must have a ``pid`` field specified,
            since the task state is per-PID.
        :type tasks: list(lisa.analysis.tasks.TaskID) or list(int)
        """
        ######################################################
        # A) Assemble the sched_switch and sched_wakeup events
        ######################################################
        def filters_comm(task):
            try:
                return task.comm is not None
            except AttributeError:
                return isinstance(task, str)

        # Add the rename events if we are interested in the comm of tasks
        add_rename = any(map(filters_comm, tasks or []))

        trace = self.trace.get_view(
            df_fmt='polars-lazyframe',
            signals=[
                SignalDesc('sched_switch', ['prev_pid', 'prev_comm']),
                SignalDesc('sched_switch', ['next_pid', 'next_comm']),
                SignalDesc('sched_wakeup', ['pid', 'comm']),
                SignalDesc('sched_wakeup_new', ['pid', 'comm']),
                SignalDesc('task_rename', ['__pid']),
            ],
            compress_signals_init=True,
        )

        wk_df = trace.df_event('sched_wakeup')
        sw_df = trace.df_event('sched_switch')

        try:
            wkn_df = trace.df_event('sched_wakeup_new')
        except MissingTraceEventError:
            pass
        else:
            wk_df = pl.concat([wk_df, wkn_df], how='diagonal_relaxed')

        def state(value):
            return pl.lit(
                value,
                dtype=sw_df.collect_schema()['prev_state'],
            )

        wk_df = wk_df.select(["Time", "pid", "comm", "target_cpu", "__cpu"])
        wk_df = wk_df.with_columns(
            curr_state=state(TaskState.TASK_WAKING)
        )

        prev_sw_df = sw_df.select(["Time", "__cpu", "prev_pid", "prev_state", "prev_comm"])
        next_sw_df = sw_df.select(["Time", "__cpu", "next_pid", "next_comm"])

        prev_sw_df = (
            prev_sw_df
            .drop('pid', 'curr_state', 'comm', strict=False)
            .rename({
                "prev_pid": "pid",
                "prev_state": "curr_state",
                "prev_comm": "comm",
            })
        )

        next_sw_df = next_sw_df.with_columns(
            curr_state=state(TaskState.TASK_ACTIVE)
        )
        next_sw_df = (
            next_sw_df
            .drop('pid', 'comm', strict=False)
            .rename({
                'next_pid': 'pid',
                'next_comm': 'comm'
            })
        )
        all_sw_df = pl.concat([prev_sw_df, next_sw_df], how='diagonal_relaxed')

        if add_rename:
            rename_df = (
                trace.df_event('task_rename')
                .drop('pid', 'comm', strict=False)
                .rename({
                    'oldcomm': 'comm',
                    '__pid': 'pid',
                })
            )
            rename_df = rename_df.select(['Time', 'pid', 'comm'])
            rename_df = rename_df.with_columns(
                curr_state=state(TaskState.TASK_RENAMED),
            )
            all_sw_df = pl.concat([all_sw_df, rename_df], how='diagonal_relaxed')

        # Integer values are prefered here, otherwise the whole column
        # is converted to float64
        # FIXME: should we just use null here ?
        all_sw_df = all_sw_df.with_columns(
            target_cpu=pl.lit(
                -1,
                # Ensure we use the exact same dtype so that there is no
                # mismatch when concatenating
                dtype=wk_df.collect_schema()['target_cpu']
            )
        )

        df = pl.concat([all_sw_df, wk_df], how='diagonal_relaxed')
        df = df.sort('Time')
        df = (
            df
            .drop('cpu', strict=False)
            .rename({'__cpu': 'cpu'})
        )

        # Restrict the set of data we will process to a given set of tasks
        if tasks is not None:
            def resolve_task(task):
                """
                Get a TaskID for each task, and only update existing TaskID if
                they lack a PID field, since that's what we care about in that
                function.
                """
                try:
                    do_update = task.pid is None
                except AttributeError:
                    do_update = False

                return self.get_task_id(task, update=do_update)

            tasks = list(map(resolve_task, tasks))
            df = df_filter_task_ids(df, tasks)

        df = df.with_columns(
            next_state=pl.col('curr_state').shift(
                -1,
                fill_value=state(TaskState.TASK_UNKNOWN)
            ).over(pl.col('pid')),
            duration_delta=pl.col('Time').diff().shift(-1).over(pl.col('pid')),
        )
        df = df.with_columns(
            delta=pl.col('duration_delta').dt.total_nanoseconds() / 1e9,
        )

        return df

    @staticmethod
    def _reorder_tasks_states_columns(df):
        order = ['Time', 'pid', 'comm', 'target_cpu', 'cpu', 'curr_state', 'next_state', 'delta']
        return df.select(order_as(list(df.collect_schema().names()), order))

    @_df_tasks_states.used_events
    @TraceAnalysisBase.df_method
    def df_tasks_states(self):
        """
        DataFrame of all tasks state updates events

        :returns: a :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU where the event took place)
          * A ``pid`` column (the PID of the task)
          * A ``comm`` column (the name of the task)
          * A ``target_cpu`` column (the CPU where the task has been scheduled).
            Will be ``NaN`` for non-wakeup events
          * A ``curr_state`` column (the current task state, see :class:`~TaskState`)
          * A ``delta`` column (the duration for which the task will remain in
            this state)
          * A ``next_state`` column (the next task state)

        .. warning:: Since ``sched_switch`` event multiplexes the update to two
            PIDs at the same time, the resulting dataframe would contain
            duplicated indices, breaking some Pandas functions. In order to
            avoid that, the duplicated timestamps are updated with the minimum
            increment possible to remove duplication.
        """
        df = self._df_tasks_states(df_fmt='polars-lazyframe')
        return self._reorder_tasks_states_columns(df)

    @TraceAnalysisBase.df_method
    @_df_tasks_states.used_events
    def df_task_states(self, task, stringify=False):
        """
        DataFrame of task's state updates events

        :param task: The task's name or PID or tuple ``(pid, comm)``
        :type task: int or str or tuple(int, str)

        :param stringify: Include stringifed :class:`TaskState` columns
        :type stringify: bool

        :returns: a :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU where the event took place)
          * A ``target_cpu`` column (the CPU where the task has been scheduled).
            Will be ``-1`` for non-wakeup events
          * A ``curr_state`` column (the current task state, see :class:`~TaskState`)
          * A ``next_state`` column (the next task state)
          * A ``delta`` column (the duration for which the task will remain in
            this state)
        """
        df = self._df_tasks_states(tasks=[task], df_fmt='polars-lazyframe')
        df = df.drop(["pid", "comm"])

        if stringify:
            df = self.stringify_df_task_states(
                df,
                ["curr_state", "next_state"],
                inplace=True
            )

        return self._reorder_tasks_states_columns(df)

    @classmethod
    def stringify_task_state_series(cls, series):
        """
        Stringify a series containing :class:`TaskState` values

        :param series: The series
        :type series: pandas.Series

        The common use case for this will be to pass a dataframe column::

            df["state_str"] = stringify_task_state_series(df["state"])
        """
        def stringify_state(state):
            # Same logic as in sched_switch format string
            if state & 0xff:
                try:
                    return TaskState(state).char
                except ValueError:
                    return TaskState.sched_switch_str(state)
            else:
                return TaskState.sched_switch_str(state)

        return series.apply(stringify_state)

    @classmethod
    def stringify_df_task_states(cls, df, columns, inplace=False):
        """
        Adds stringified :class:`TaskState` columns to a Dataframe

        :param df: The DataFrame to operate on
        :type df: pandas.DataFrame

        :param columns: The columns to stringify
        :type columns: list

        :param inplace: Do the modification on the original DataFrame
        :type inplace: bool
        """

        if isinstance(df, pd.DataFrame):
            df = df if inplace else df.copy()

            for col in columns:
                df[f"{col}_str"] = cls.stringify_task_state_series(df[col])

            return df
        elif isinstance(df, pl.LazyFrame):
            mapping = {
                int(state): state.char
                for state in TaskState.list_reported_states()
            }

            def fixup(df, col):
                str_col = (pl.col(col) & 0xff).replace_strict(
                    mapping,
                    default=None,
                    return_dtype=pl.Categorical,
                )
                str_col = (
                    pl.when(str_col.is_null() & (pl.col(col) > 0))
                    .then(
                        pl.col(col).map_elements(
                            TaskState.sched_switch_str,
                            return_dtype=pl.String,
                        )
                    )
                    .otherwise(str_col)
                )

                return df.with_columns(
                    str_col.alias(f'{col}_str')
                )

            return functools.reduce(fixup, columns, df)
        else:
            raise TypeError(f'Cannot handle type dataframe of type {df.__class__}')

    @TraceAnalysisBase.df_method
    @_df_tasks_states.used_events
    def df_tasks_runtime(self):
        """
        DataFrame of the time each task spent in TASK_ACTIVE (:class:`TaskState`)

        :returns: a :class:`pandas.DataFrame` with:

          * PIDs as index
          * A ``comm`` column (the name of the task)
          * A ``runtime`` column (the time that task spent running)

        .. note:: This function only tracks time spent by each PID. The
            reported name is the last name associated with the PID in chronological
            order.
        """

        df = self._df_tasks_states(df_fmt='polars-lazyframe')
        df = df.group_by('pid').agg(
            comm=pl.col('comm').last(),
            runtime=pl.col('delta').filter(pl.col('curr_state') == TaskState.TASK_ACTIVE).sum()
        )
        return df

    @TraceAnalysisBase.df_method
    @df_task_states.used_events
    def df_task_total_residency(self, task):
        """
        DataFrame of a task's execution time on each CPU

        :param task: the task to report runtimes for
        :type task: int or str or tuple(int, str)

        :returns: a :class:`pandas.DataFrame` with:

          * CPU IDs as index
          * A ``runtime`` column (the time the task spent being active)
        """
        df = self.df_task_states(task)
        # Get the correct delta for the window we want.
        df = df_add_delta(df, window=self.trace.window, col='runtime')
        df = df[df['curr_state'] == TaskState.TASK_ACTIVE]

        # For each CPU, sum the time spent on each by each task
        by_cpu = df.groupby('cpu', observed=True, sort=False, group_keys=False)
        residency_df = by_cpu['runtime'].sum().to_frame()

        # Add runtime for CPUs that did not appear in the window
        residency_df = residency_df.reindex(
            residency_df.index.union(range(self.trace.cpus_count))
        )
        return residency_df.fillna(0).sort_index()

    @TraceAnalysisBase.df_method(index='task')
    @df_task_total_residency.used_events
    def df_tasks_total_residency(self, tasks=None, ascending=False, count=None):
        """
        DataFrame of tasks execution time on each CPU

        :param tasks: List of tasks to report, all trace tasks by default
        :type tasks: list(int or str or tuple(int, str))

        :param ascending: Set True to order plot by ascending task runtime
                          False by default
        :type ascending: bool

        :param count: Maximum number of tasks to report
        :type count: int
        """
        if tasks is None:
            task_ids = self.task_ids
        else:
            task_ids = itertools.chain.from_iterable(
                self.get_task_ids(task)
                for task in tasks
            )

        def get_task_df(task):
            try:
                df = self.ana.tasks.df_task_total_residency(task)
            except MissingTraceEventError:
                raise
            # Not all tasks may be available, e.g. tasks outside the _TraceView
            # window
            except Exception:
                return None
            else:
                return df.T.rename(index={'runtime': str(task)})

        res_df = pd.concat(
            df
            for df in map(get_task_df, task_ids)
            if df is not None
        )

        res_df['Total'] = res_df.sum(axis=1)
        res_df.sort_values(by='Total', ascending=ascending, inplace=True)
        if count is not None:
            res_df = res_df.head(count)

        res_df.index.name = 'task'

        # Ensure column names are all strings, so it can be serialized to
        # parquet
        res_df.columns = [str(col) for col in res_df.columns]
        return res_df

    @TraceAnalysisBase.df_method
    @df_task_states.used_events
    def df_task_activation(self, task, cpu=None, active_value=1, sleep_value=0, preempted_value=np.nan):
        """
        DataFrame of a task's active time on a given CPU

        :param task: the task to report activations of
        :type task: int or str or tuple(int, str)

        :param cpu: the CPUs to look at. If ``None``, all CPUs will be used.
        :type cpu: int or None

        :param active_value: the value to use in the series when task is
            active.
        :type active_value: float

        :param sleep_value: the value to use in the series when task is
            sleeping.
        :type sleep_value: float

        :param preempted_value: the value to use in the series when task is
            preempted (runnable but not actually executing).
        :type sleep_value: float

        :returns: a :class:`pandas.DataFrame` with:

          * A timestamp as index
          * A ``active`` column, containing ``active_value`` when the task is
            running, ``sleep_value`` when sleeping, and ``preempted_value``
            otherwise.
          * A ``cpu`` column with the CPU the task was running on.
          * A ``duration`` column containing the duration of the current sleep or activation.
          * A ``duty_cycle`` column containing the duty cycle in ``[0...1]`` of
            the task, updated at each pair of activation and sleep.
        """

        trace = self.trace.get_view(df_fmt='polars-lazyframe')
        ana = trace.ana.tasks
        df = ana.df_task_states(task)

        if cpu is not None:
            df = df.filter(pl.col('cpu') == cpu)

        # TASK_WAKING can just be removed. The delta will then be computed
        # without it, which means the time spent in WAKING state will be
        # accounted into the previous state.
        df = df.filter(pl.col('curr_state') != TaskState.TASK_WAKING)

        df = df.with_columns(
            active=pl.col('curr_state').replace_strict(
                {
                    TaskState.TASK_ACTIVE: active_value,
                    # TASK_RUNNING happens when a task is preempted (so it's not
                    # TASK_ACTIVE anymore but still runnable).
                    # Use null regardless of preempted_value, since some below code
                    # relies on that
                    TaskState.TASK_RUNNING: None,
                },
                default=sleep_value
            )
        )
        df = df.select(('Time', 'active', 'cpu'))

        # Only keep first occurence of each adjacent duplicates, since we get
        # events when the signal changes
        on = pl.struct(('active', 'cpu'))
        df = df.filter((on != on.shift()).fill_null(True))

        # Once we removed the duplicates, we can compute the time spent while sleeping or activating
        df = df_add_delta(df, col='duration')

        if not np.isnan(preempted_value):
            df = df.with_columns(
                active=pl.col('active').fill_null(preempted_value)
            )

        # Merge consecutive activations' duration. They could have been split
        # in two by a bit of preemption, and we don't want that to affect the
        # duty cycle.
        active = pl.col('active')
        df = df.group_by_dynamic(
            # Assign an id to each consecutive run of "active", which is used
            # as a "group id". We then create a window that only contains a
            # given group ID.
            index_column=active.rle_id().cast(pl.Int32).alias('__active_rle_id'),
            every='1i',
        ).agg(
            # In each group, we take the first row for all columns except the
            # duration that is summed
            pl.all().exclude('duration').first(),
            pl.when(
                pl.col('duration').count() > 0,
            ).then(
                pl.col('duration').sum(),
            )
        ).drop('__active_rle_id')

        # Make a dataframe where the rows corresponding to preempted time are
        # removed, unless preempted_value is set to non-NA
        df = df.drop_nulls('active')

        # Pair an activation time with it's following sleep time. Since we
        # combined all active runs, we are guaranteed an active row is
        # immediately succeeded by a non-active row.
        df = df.join_asof(
            df.filter(
                active == sleep_value
            ).select('Time', sleep_duration='duration'),
            on='Time',
            strategy='forward',
        )

        df = df.with_columns(
            duty_cycle=pl.when(
                active == active_value,
            ).then(
                pl.col('duration') / (pl.col('duration') + pl.col('sleep_duration'))
            )
        )
        df = df.drop('sleep_duration')
        return df

###############################################################################
# Plotting Methods
###############################################################################

    def _plot_markers(self, df, label):
        return hv.Scatter(df, label=label).options(marker='+').options(
            backend='bokeh',
            size=5,
        )

    def _plot_overutilized(self):
        try:
            return self.ana.status.plot_overutilized()
        except MissingTraceEventError:
            return _hv_neutral()

    @TraceAnalysisBase.plot_method
    @requires_events('sched_switch')
    def plot_task_residency(self, task: TaskID):
        """
        Plot on which CPUs the task ran on over time

        :param task: Task to track
        :type task: int or str or tuple(int, str)
        """
        task_id = self.get_task_id(task, update=False)

        sw_df = self.trace.df_event("sched_switch")
        sw_df = df_filter_task_ids(sw_df, [task_id], pid_col='next_pid', comm_col='next_comm')

        def plot_residency():
            if "freq-domains" in self.trace.plat_info:
                # If we are aware of frequency domains, use one color per domain
                for domain in self.trace.plat_info["freq-domains"]:
                    series = sw_df[sw_df["__cpu"].isin(domain)]["__cpu"]
                    series = series_refit_index(series, window=self.trace.window)

                    if series.empty:
                        return _hv_neutral()
                    else:
                        return self._plot_markers(
                            series,
                            label=f"Task running in domain {domain}"
                        )
            else:
                return self._plot_markers(
                    series_refit_index(sw_df['__cpu'], window=self.trace.window),
                    label=str(task),
                )

        return (
            plot_residency().options(ylabel='cpu') *
            self._plot_overutilized()
        ).options(
            title=f'CPU residency of task {task}'
        )

    @TraceAnalysisBase.plot_method
    @df_task_total_residency.used_events
    def plot_task_total_residency(self, task: TaskID):
        """
        Plot a task's total time spent on each CPU

        :param task: The task's name or PID or tuple ``(pid, comm)``
        :type task: str or int or tuple(int, str)
        """
        df = self.df_task_total_residency(task)

        return hv.Bars(df['runtime']).options(
            title=f"CPU residency of task {task}",
            xlabel='CPU',
            ylabel='Runtime (s)',
            invert_axes=True,
        )

    @TraceAnalysisBase.plot_method
    @df_tasks_total_residency.used_events
    def plot_tasks_total_residency(self, tasks: typing.Sequence[TaskID]=None, ascending: bool=False,
                                   count: bool=None):
        """
        Plot the stacked total time spent by each task on each CPU

        :param tasks: List of tasks to plot, all trace tasks by default
        :type tasks: list(int or str or tuple(int, str))

        :param ascending: Set True to order plot by ascending task runtime,
                          False by default
        :type ascending: bool

        :param count: Maximum number of tasks to report
        :type count: int
        """
        df = self.df_tasks_total_residency(tasks, ascending, count)
        df = df.copy(deep=False)
        df['task'] = df.index
        df.columns = list(map(str, df.columns))
        df = df.melt(id_vars=['task'], var_name='cpu', value_name='Runtime (s)')
        return hv.Bars(
            df,
            kdims=['cpu', 'task']
        ).options(
            stacked=True,
            invert_axes=True,
            title=f"Stacked CPU residency of [{len(df.index)}] selected tasks",
        ).sort('cpu')

    def _plot_cpu_heatmap(self, event, bins, xbins, cmap):
        """
        Plot some data in a heatmap-style 2d histogram
        """
        df = self.trace.df_event(event)
        df = df_window(df, window=self.trace.window, method='exclusive')
        x = df.index
        y = df['target_cpu']

        if xbins:
            warnings.warn('"xbins" parameter is deprecated and will be removed, use "bins" instead', DeprecationWarning)
            bins = xbins

        nr_cpus = self.trace.cpus_count
        hist = np.histogram2d(y, x, bins=[nr_cpus, bins])
        z, _, x = hist
        y = list(range(nr_cpus))
        return hv.HeatMap(
            (x, y, z),
            kdims=[
                # Manually set dimension name/label so that shared_axes works
                # properly.
                # Also makes hover tooltip better.
                hv.Dimension('Time'),
                hv.Dimension('CPU'),
            ],
            vdims=[
                hv.Dimension(event),
            ]
        ).options(
            colorbar=True,
            xlabel='Time (s)',
            ylabel='CPU',
            # Viridis works both on bokeh and matplotlib
            cmap=cmap or 'Viridis',
            yticks=[
                (cpu, f'CPU{cpu}')
                for cpu in y
            ]
        )

    @TraceAnalysisBase.plot_method
    @requires_events("sched_wakeup")
    def _plot_tasks_X(self, event, name, target_cpus, window, per_sec):
        df = self.trace.df_event(event)

        if target_cpus:
            df = df[df['target_cpu'].isin(target_cpus)]

        series = series_rolling_apply(
            df["target_cpu"],
            lambda x: x.count() / (window if per_sec else 1),
            window,
            window_float_index=False,
            center=True
        )

        if per_sec:
            label = f"Number of task {name} per second ({window}s windows)"
        else:
            label = f"Number of task {name} within {window}s windows"
        series = series_refit_index(series, window=self.trace.window)
        series.name = name
        return plot_signal(series, name=label)

    @TraceAnalysisBase.plot_method
    def plot_tasks_wakeups(self, target_cpus: typing.Sequence[CPU]=None, window: float=1e-2, per_sec: bool=False):
        """
        Plot task wakeups over time

        :param target_cpus:
        :type target_cpus:

        :param window: The rolling window size for wakeup counts.
        :type window: float

        :param per_sec: Display wakeups per second if True, else wakeup counts
          within the window
        :type per_sec: bool
        """
        return self._plot_tasks_X(
            event='sched_wakeup',
            name='wakeups',
            target_cpus=target_cpus,
            window=window,
            per_sec=per_sec
        )

    @TraceAnalysisBase.plot_method
    @requires_events("sched_wakeup")
    def plot_tasks_wakeups_heatmap(self, bins: int=100, xbins=None, colormap=None):
        """
        Plot tasks wakeups heatmap

        :param bins: Number of x-axis bins, i.e. in how many slices should
          time be arranged
        :type bins: int

        :param colormap: The name of a colormap:

            * matplotlib backend: https://matplotlib.org/stable/tutorials/colors/colormaps.html
            * bokeh backend: https://docs.bokeh.org/en/latest/docs/reference/palettes.html
        :type colormap: str
        """
        return self._plot_cpu_heatmap(
            event='sched_wakeup',
            bins=bins,
            xbins=xbins,
            cmap=colormap,
        ).options(
            title="Tasks wakeups over time",
        )

    @TraceAnalysisBase.plot_method
    @requires_events("sched_wakeup_new")
    def plot_tasks_forks(self, target_cpus: typing.Sequence[CPU]=None, window: float=1e-2, per_sec: bool=False):
        """
        Plot task forks over time

        :param target_cpus:
        :type target_cpus:

        :param window: The rolling window size for fork counts.
        :type window: float

        :param per_sec: Display wakeups per second if True, else wakeup counts
          within the window
        :type per_sec: bool
        """
        return self._plot_tasks_X(
            event='sched_wakeup_new',
            name='forks',
            target_cpus=target_cpus,
            window=window,
            per_sec=per_sec
        )

    @TraceAnalysisBase.plot_method
    @requires_events("sched_wakeup_new")
    def plot_tasks_forks_heatmap(self, bins: int=100, xbins=None, colormap=None):
        """
        Plot number of task forks over time as a heatmap.

        :param bins: Number of x-axis bins, i.e. in how many slices should
          time be arranged
        :type bins: int

        :param colormap: The name of a colormap:

            * matplotlib backend: https://matplotlib.org/stable/tutorials/colors/colormaps.html
            * bokeh backend: https://docs.bokeh.org/en/latest/docs/reference/palettes.html
        :type colormap: str
        """

        return self._plot_cpu_heatmap(
            event='sched_wakeup_new',
            bins=bins,
            xbins=xbins,
            cmap=colormap,
        ).options(
            title="Tasks forks over time",
        )

    # Use a class attribute so that there will be only one extra hover tool in
    # the toolbar rather than one per task when stacking them
    _BOKEH_TASK_HOVERTOOL = bokeh.models.HoverTool(
        description='Task activations tooltip',
        tooltips=[
            ('Task', '[@pid:@comm]'),
            ('CPU', '@cpu'),
            ('#', '$index'),
            ('Start', '@start'),
            ('Duration', '@duration'),
            ('Duty cycle', '@duty_cycle'),
        ]
    )

    @df_task_activation.used_events
    def _plot_tasks_activation(self, tasks, show_legend=None, cpu: CPU=None, alpha:
            float=None, overlay: bool=None, duration: bool=False, duty_cycle:
            bool=False, which_cpu: bool=False, height_duty_cycle: bool=False,
            best_effort: bool =False,
        ):
        logger = self.logger
        trace = self.trace.get_view(df_fmt='polars-lazyframe')
        ana = trace.ana

        if overlay is not None:
            warnings.warn('"overlay" parameter is deprecated and will be removed, use holoviews twin axis support and plot customization to change e.g. Rectangles alpha instead', DeprecationWarning)

        def ensure_last_rectangle(df):
            """
            Make sure we will draw the last rectangle, which could be
            critical for tasks that are never sleeping
            """
            window = trace.window

            # Regenerate the duration so they match the boundaries of the
            # window
            df = df_add_delta(df, window=window, col='duration')
            return df

        if which_cpu:
            def make_rect_df(df):
                half_height = pl.col('active') / 2
                return df.select((
                    pl.col('Time'),
                    pl.col('cpu').alias('CPU'),

                    (pl.col('cpu') - half_height).alias('y0'),
                    (pl.col('Time') + pl.col('duration')).alias('x1'),
                    (pl.col('cpu') + half_height).alias('y1'),
                ))
        else:
            def make_rect_df(df):
                return df.select((
                    pl.col('Time'),
                    pl.col('cpu').alias('CPU'),

                    pl.lit(0).alias('y0').cast(pl.Float64),
                    (pl.col('Time') + pl.col('duration')).alias('x1'),
                    pl.col('active').alias('y1').cast(pl.Float64),
                ))

        def plot_extra(task, df):
            figs = []
            if duty_cycle:
                figs.append(
                    plot_signal(df.select(('Time', 'duty_cycle')), name=f'Duty cycle of {task}').redim(
                        duty_cycle='Duty cycle'
                    )
                )

            if duration:
                def plot_duration(active, label):
                    duration_df = df.filter(pl.col('active') == pl.lit(active)).select(('Time', 'duration'))
                    # Add blanks in the plot when the state is not the one we care about
                    _, duration_df = pl.align_frames(df, duration_df, on='Time', how='left')
                    return plot_signal(duration_df, name=f'{label} duration of {task}')

                figs.extend(
                    plot_duration(active, label).redim(
                        duration='Duration'
                    )
                    for active, label in (
                        (True, 'Activations'),
                        (False, 'Sleep')
                    )
                )

            return figs

        def check_df(task, df, empty_is_none):
            try:
                ana.tasks.get_task_pid_names(task.pid)
            except KeyError:
                msg = f'Could not find events associated to task {task}'
                if empty_is_none:
                    logger.debug(msg)
                    return None
                else:
                    raise ValueError(msg)
            else:
                df = df.with_columns(
                    pl.col('duration').dt.total_nanoseconds() / 1e9
                )
                return ensure_last_rectangle(df)

        def get_task_data(task, df):
            df = df.with_columns(
                # Preempted == sleep for plots
                pl.col('active').fill_null(0)
            )
            if height_duty_cycle:
                df = df.with_columns(
                    active=(
                        pl.col('active') *
                        # Ensure we don't end up with null here, otherwise the
                        # rectangle will disappear.
                        pl.col('duty_cycle').fill_null(1)
                    )
                )

            data_df = make_rect_df(df.filter(pl.col('active') != 0))

            comm_df = trace.df_event('sched_switch')
            comm_df = (
                comm_df
                .filter(pl.col('next_pid') == task.pid)
                .select(('Time', pl.col('next_comm').alias('comm')))
            )

            data_df = data_df.join_asof(
                comm_df,
                on='Time',
            )

            last_comm = ana.tasks.get_task_pid_names(task.pid)[-1]
            data_df = data_df.with_columns(
                # If there was no sched_switch with next_pid matching
                # task.pid, we simply take the last known name of the
                # task, which could originate from another field or
                # another event.
                #
                # Note: This prevent an null value, which makes bokeh
                # choke.
                pl.col('comm').fill_null(last_comm)
            )

            data_df = data_df.join_asof(
                df.select((
                    'Time',
                    'cpu',
                    'active',
                    'duration',
                    'duty_cycle',
                )),
                on='Time',
            )

            data_df = data_df.with_columns(
                pid=pl.lit(
                    task.pid
                    if use_rasterize else
                    # Use a string so it is recognized as a categorical column
                    str(task.pid)
                ),
                start=pl.col('Time'),
                cpu=pl.col('CPU'),
            )
            return data_df

        def plot_rect(data):
            data = data.lazy()

            if show_legend or use_rasterize:
                opts = {}
            else:
                # If there is no legend, we are gonna plot all the rectangles
                # at once so we use colormapping to distinguish the tasks
                opts = dict(
                    color='pid',
                    # Colormap from colorcet with a large number of color, so it is
                    # suitable for plotting many tasks
                    cmap='glasbey_hv',
                )

            if use_rasterize:
                channels = ('red', 'green', 'blue')
                data = data.with_columns(
                    pl.col('pid').replace_strict(
                        color_key,
                        return_dtype=pl.Array(
                            pl.UInt8, 3
                        )
                    ).arr.to_struct(
                        fields=channels
                    ).struct.unnest()
                )

            fig = hv.Rectangles(
                _df_to(data, fmt='pandas', index=NO_INDEX),
                kdims=[
                    # Ensure we have Time as kdim, otherwise some generic code
                    # in AnalysisBase will not recognize this plot uses Time as
                    # X axis and will not link the events dataframe by default
                    # in output="ui"
                    hv.Dimension('Time'),
                    hv.Dimension('y0'),
                    hv.Dimension('x1'),
                    hv.Dimension('y1'),
                ]
            ).options(
                show_legend=show_legend,
                alpha=alpha,
                xlabel='Time',
                **opts,
            ).options(
                backend='bokeh',
                line_width=0,
                line_color=None,
                tools=[self._BOKEH_TASK_HOVERTOOL],
            )

            if use_rasterize:

                interactive = is_running_ipython()
                # Merge on each RGB channels. This way we can pre-assign the color
                # of each rectangle in the data set and we just end up with blended
                # color when zooming out.
                from holoviews.operation.datashader import datashade, stack
                import datashader as ds

                fig = functools.reduce(
                    operator.mul,
                    (
                        datashade(
                            fig,
                            # Select the rectangle with the longest duration, so
                            # that shorter ones are displayed "as part of it"
                            aggregator=ds.where(ds.max('duration'), chan),
                            # We use a single color name, so the aggregated value
                            # for each pixel is used as alpha for that color.
                            cmap=chan,
                            # Ensure the value that shade() picks is the value
                            # we asked for. We do not want any re-normalization
                            # or offset of any kind.
                            cnorm='linear',
                            clims=(0, 255),
                            min_alpha=0,
                            precompute=True,
                            dynamic=interactive,
                        )
                        for chan in channels
                    )
                )
                fig = stack(fig, compositor='add', link_inputs=True)
            return fig

        if alpha is None:
            if duty_cycle or duration:
                alpha = 0.2
            else:
                alpha = 1


        cpus_count = trace.cpus_count

        if show_legend:
            use_rasterize = False
        else:
            event_per_sec = 1000
            approx_rectangles = (trace.end - trace.start) * event_per_sec * cpus_count
            use_rasterize = approx_rectangles > 5_000

        # TODO: undo that when this gets fixed:
        # https://github.com/holoviz/holoviews/issues/6736
        use_rasterize=False

        # For performance reasons, plot all the tasks as one hv.Rectangles
        # invocation when we get too many tasks
        if show_legend is None and not use_rasterize:
            show_legend = len(tasks) < 5

        task_dfs = {
            task: check_df(
                task,
                ana.tasks.df_task_activation(task, cpu=cpu),
                empty_is_none=best_effort,
            )
            for task in tasks
        }
        task_dfs = {
            task: df
            for task, df in task_dfs.items()
            if df is not None
        }
        tasks = sorted(task_dfs.keys())

        # Assign a color to each PID
        color_key = {
            pid: tuple(int(c * 255) for c in color)
            for pid, color in zip(
                sorted({
                    task.pid
                    for task in tasks
                }),
                itertools.cycle(list(glasbey_hv))
            )
        }

        if show_legend:
            task_dfs = dict(zip(
                task_dfs.keys(),
                _polars_fast_collect_all(
                    itertools.starmap(
                        get_task_data,
                        task_dfs.items()
                    )
                )
            ))
            fig = hv.Overlay(
                [
                    plot_rect(df).relabel(
                        f'Activations of {task.pid} (' +
                        ', '.join(
                            task_id.comm
                            for task_id in ana.tasks.get_task_ids(task)
                        ) +
                        ')',
                    )
                    for task, df in task_dfs.items()
                ]
            ).options(
                legend_limit=len(tasks) * 100,
            )
        else:
            data = pl.concat(
                get_task_data(task, df)
                for task, df in task_dfs.items()
            )
            fig = plot_rect(data)

        if which_cpu:
            fig = fig.options(
                'Rectangles',
                ylabel='CPU',
                yticks=[
                    (cpu, f'CPU{cpu}')
                    for cpu in range(cpus_count)
                ],
            )
        else:
            fig = fig.options(
                'Rectangles',
                ylabel='Task Activations',
            )

        if duty_cycle or duration:
            fig = hv.Overlay(
                [fig] +
                [
                    fig
                    for task, df in task_dfs.items()
                    for fig in plot_extra(task, df)
                ]
            ).options(
                # Enable twin axes, so that the duty cycle has a separate scale.
                multi_y=True,
            )

        return fig.options(
            title='Activations of {}'.format(
                ', '.join(map(str, tasks))
            ),
        )

    @TraceAnalysisBase.plot_method
    @_plot_tasks_activation.used_events
    @kwargs_forwarded_to(_plot_tasks_activation, ignore=['tasks', 'best_effort'])
    def plot_tasks_activation(self, tasks: typing.Sequence[TaskID]=None, hide_tasks: typing.Sequence[TaskID]=None, which_cpu: bool=True, overlay: bool=None, **kwargs):
        """
        Plot all tasks activations, in a style similar to kernelshark.

        :param tasks: Tasks to plot. If ``None``, all tasks in the trace will
            be used.
        :type tasks: list(TaskID) or None

        :param hide_tasks: Tasks to hide. Note that PID 0 (idle task) will
            always be hidden.
        :type hide_tasks: list(TaskID) or None

        :param alpha: transparency level of the plot.
        :type task: float

        :param overlay: If ``True``, adjust the transparency and plot
            activations on a separate hidden scale so existing scales are not
            modified.
        :type task: bool

        :param duration: Plot the duration of each sleep/activation.
        :type duration: bool

        :param duty_cycle: Plot the duty cycle of each pair of sleep/activation.
        :type duty_cycle: bool

        :param which_cpu: If ``True``, plot the activations on each CPU in a
            separate row like kernelshark does.
        :type which_cpu: bool

        :param height_duty_cycle: Height of each activation's rectangle is
            proportional to the duty cycle during that activation.
        :type height_duty_cycle: bool

        .. seealso:: :meth:`df_task_activation`
        """
        hidden = set(itertools.chain.from_iterable(
            self.get_task_ids(task)
            for task in (hide_tasks or [])
        ))
        if tasks:
            best_effort = False
            task_ids = list(itertools.chain.from_iterable(
                map(self.get_task_ids, tasks)
            ))
        else:
            best_effort = True
            task_ids = self.task_ids

        full_task_ids = sorted(
            task
            for task in task_ids
            if (
                task not in hidden and
                task.pid != 0
            )
        )

        # Only consider the PIDs in order to:
        # * get the same color for the same PID during its whole life
        # * avoid potential issues around task renaming
        # Note: The task comm will still be displayed in the hover tool
        task_ids = [
            TaskID(pid=pid, comm=None)
            for pid in sorted(set(x.pid for x in full_task_ids))
        ]

        title = 'Activations of ' + ', '.join(
            map(str, full_task_ids)
        )
        if len(title) > 50:
            title = 'Task activations'

        return self._plot_tasks_activation(
            tasks=task_ids,
            which_cpu=which_cpu,
            overlay=overlay,
            best_effort=best_effort,
            **kwargs
        ).options(
            title=title
        )

    @TraceAnalysisBase.plot_method
    @plot_tasks_activation.used_events
    @kwargs_forwarded_to(plot_tasks_activation, ignore=['tasks'])
    @deprecate('Deprecated since it does not provide anything more than plot_tasks_activation', deprecated_in='2.0', removed_in='4.0', replaced_by=plot_tasks_activation)
    def plot_task_activation(self, task: TaskID, **kwargs):
        """
        Plot task activations, in a style similar to kernelshark.

        :param task: the task to report activations of
        :type task: int or str or tuple(int, str)

        .. seealso:: :meth:`plot_tasks_activation`
        """
        return self.plot_tasks_activation(tasks=[task], **kwargs)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
