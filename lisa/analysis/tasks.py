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

from enum import Enum
from collections import defaultdict
import itertools
import functools

import numpy as np
import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import memoized
from lisa.datautils import df_filter_task_ids, series_rolling_apply, series_refit_index, df_refit_index, df_deduplicate, df_split_signals, df_add_delta, df_window, df_update_duplicates, df_combine_duplicates
from lisa.trace import requires_events, TaskID, CPU
from lisa.pelt import PELT_SCALE
from lisa.generic import TypedList


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
    # Used when the task state is unknown
    TASK_UNKNOWN = -1, "U", "Unknown"

    @classmethod
    def list_reported_states(cls):
        """
        List the states that can be reported in a ``sched_switch`` trace

        See include/linux/sched.h:TASK_REPORT
        """
        return [state for state in list(cls) if state <= cls.TASK_DEAD]

    # Could use IntFlag instead once we move to Python 3.6
    @classmethod
    def sched_switch_str(cls, value):
        """
        Get the task state string that would be used in a ``sched_switch`` event

        :param value: The task state value
        :type value: int

        Tries to emulate what is done in include/trace/events:TRACE_EVENT(sched_switch)
        """
        if any([value & state.value for state in cls.list_reported_states()]):
            res = "|".join([state.char for state in cls.list_reported_states()
                            if state.value & value])
        else:
            res = cls.TASK_RUNNING.char

        # Flag the presence of unreportable states with a "+"
        if any([value & state.value for state in list(cls)
                if state not in cls.list_reported_states()]):
            res += "+"

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
    :type trace: :class:`trace.Trace`
    """

    name = 'tasks'

    @requires_events('sched_switch')
    def cpus_of_tasks(self, tasks):
        """
        Return the list of CPUs where the ``tasks`` executed.

        :param tasks: Task names or PIDs or ``(pid, comm)`` to look for.
        :type tasks: list(int or str or tuple(int, str))
        """
        trace = self.trace
        df = trace.df_event('sched_switch')[['next_pid', 'next_comm', '__cpu']]

        task_ids = [trace.get_task_id(task, update=False) for task in tasks]
        df = df_filter_task_ids(df, task_ids, pid_col='next_pid', comm_col='next_comm')
        cpus = df['__cpu'].unique()

        return sorted(cpus)

    def _get_task_pid_name(self, pid):
        """
        Get the last name the given PID had.
        """
        return self.trace.get_task_pid_names(pid)[-1]

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.cache
    @requires_events('sched_wakeup')
    def df_tasks_wakeups(self):
        """
        The number of wakeups per task

        :returns: a :class:`pandas.DataFrame` with:

          * Task PIDs as index
          * A ``wakeups`` column (The number of wakeups)
        """
        df = self.trace.df_event('sched_wakeup')

        wakeups = df.groupby('pid', observed=True, sort=False).count()["comm"]
        df = pd.DataFrame(wakeups).rename(columns={"comm": "wakeups"})
        df["comm"] = df.index.map(self._get_task_pid_name)

        return df

    @TraceAnalysisBase.cache
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

    @TraceAnalysisBase.cache
    @requires_events('sched_switch')
    def df_rt_tasks(self, min_prio=100):
        """
        Tasks with RT priority

        .. note:: priorities uses scheduler values, thus: the lower the value the
          higher is the task priority.
          RT   Priorities: [  0..100]
          FAIR Priorities: [101..120]

        :param min_prio: minimum priority
        :type min_prio: int

        :returns: a :class:`pandas.DataFrame` with:

          * Task PIDs as index
          * A ``prio`` column (The priority of the task)
          * A ``comm`` column (The name of the task)
        """
        df = self.trace.df_event('sched_switch')

        # Filters tasks which have a priority bigger than threshold
        df = df[df.next_prio <= min_prio]

        # Filter columns of interest
        rt_tasks = df[['next_pid', 'next_prio']]
        rt_tasks = rt_tasks.drop_duplicates()

        # Order by priority
        rt_tasks.sort_values(
            by=['next_prio', 'next_pid'], ascending=True, inplace=True)
        rt_tasks.rename(
            columns={'next_pid': 'pid', 'next_prio': 'prio'}, inplace=True)

        rt_tasks.set_index('pid', inplace=True)
        rt_tasks['comm'] = rt_tasks.index.map(self._get_task_pid_name)

        return rt_tasks

    @requires_events('sched_switch', 'sched_wakeup')
    def _df_tasks_states(self, tasks=None, return_one_df=False):
        """
        Compute tasks states for all tasks.

        :param tasks: If specified, states of these tasks only will be yielded.
            The :class:`lisa.trace.TaskID` must have a ``pid`` field specified,
            since the task state is per-PID.
        :type tasks: list(lisa.trace.TaskID) or list(int)

        :param return_one_df: If ``True``, a single dataframe is returned with
            new extra columns. If ``False``, a generator is returned that
            yields tuples of ``(TaskID, task_df)``. Each ``task_df`` contains
            the new columns.
        :type return_one_df: bool
        """
        ######################################################
        # A) Assemble the sched_switch and sched_wakeup events
        ######################################################

        wk_df = self.trace.df_event('sched_wakeup')
        sw_df = self.trace.df_event('sched_switch')

        if self.trace.has_events('sched_wakeup_new'):
            wkn_df = self.trace.df_event('sched_wakeup_new')
            wk_df = pd.concat([wk_df, wkn_df])

        wk_df = wk_df[["pid", "comm", "target_cpu", "__cpu"]].copy(deep=False)
        wk_df["curr_state"] = TaskState.TASK_WAKING

        prev_sw_df = sw_df[["__cpu", "prev_pid", "prev_state", "prev_comm"]].copy()
        next_sw_df = sw_df[["__cpu", "next_pid", "next_comm"]].copy()

        prev_sw_df.rename(
            columns={
                "prev_pid": "pid",
                "prev_state": "curr_state",
                "prev_comm": "comm",
            },
            inplace=True
        )

        next_sw_df["curr_state"] = TaskState.TASK_ACTIVE
        next_sw_df.rename(columns={'next_pid': 'pid', 'next_comm': 'comm'}, inplace=True)

        all_sw_df = prev_sw_df.append(next_sw_df, sort=False)

        # Integer values are prefered here, otherwise the whole column
        # is converted to float64
        all_sw_df['target_cpu'] = -1

        df = all_sw_df.append(wk_df, sort=False)
        df.sort_index(inplace=True)
        df.rename(columns={'__cpu': 'cpu'}, inplace=True)

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

                return self.trace.get_task_id(task, update=do_update)

            tasks = list(map(resolve_task, tasks))
            df = df_filter_task_ids(df, tasks)

        # Return a unique dataframe with new columns added
        if return_one_df:
            df.sort_index(inplace=True)
            df.index.name = 'Time'
            df.reset_index(inplace=True)

            # Since sched_switch is split in two df (next and prev), we end up with
            # duplicated indices. Avoid that by incrementing them by the minimum
            # amount possible.
            df = df_update_duplicates(df, col='Time', inplace=True)

            grouped = df.groupby('pid', observed=True, sort=False)
            new_columns = dict(
                next_state=grouped['curr_state'].shift(-1, fill_value=TaskState.TASK_UNKNOWN),
                # GroupBy.transform() will run the function on each group, and
                # concatenate the resulting series to create a new column.
                # Note: We actually need transform() to chain 2 operations on
                # the group, otherwise the first operation returns a final
                # Series, and the 2nd is not applied on groups
                delta=grouped['Time'].transform(lambda time: time.diff().shift(-1)),
            )
            df = df.assign(**new_columns)
            df.set_index('Time', inplace=True)

            return df

        # Return a generator yielding (TaskID, task_df) tuples
        else:
            def make_pid_df(pid_df):
                # Even though the initial dataframe contains duplicated indices due to
                # using both prev_pid and next_pid in sched_switch event, we should
                # never end up with prev_pid == next_pid, so task-specific dataframes
                # are expected to be free from duplicated timestamps.
                # assert not df.index.duplicated().any()

                # Copy the df to add new columns
                pid_df = pid_df.copy(deep=False)

                # For each PID, add the time it spent in each state
                pid_df['delta'] = pid_df.index.to_series().diff().shift(-1)
                pid_df['next_state'] = pid_df['curr_state'].shift(-1, fill_value=TaskState.TASK_UNKNOWN)
                return pid_df

            signals = df_split_signals(df, ['pid'])
            return (
                (TaskID(pid=col['pid'], comm=None), make_pid_df(pid_df))
                for col, pid_df in signals
            )

    @staticmethod
    def _reorder_tasks_states_columns(df):
        """
        Reorder once at the end of computation, since doing it for each tasks'
        dataframe turned out to be very costly
        """
        order = ['pid', 'comm', 'target_cpu', 'cpu', 'curr_state', 'next_state', 'delta']
        cols = set(order)
        available_cols = set(df.columns)
        displayed_cols = [
            col
            for col in order
            if col in (available_cols & cols)
        ]
        extra_cols = sorted(available_cols - cols)

        col_list = displayed_cols + extra_cols
        return df[col_list]

    @TraceAnalysisBase.cache
    @_df_tasks_states.used_events
    def df_tasks_states(self):
        """
        DataFrame of all tasks state updates events

        :returns: a :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU where the task was on)
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
        df = self._df_tasks_states(return_one_df=True)
        return self._reorder_tasks_states_columns(df)

    @TraceAnalysisBase.cache
    @_df_tasks_states.used_events
    def df_task_states(self, task, stringify=False):
        """
        DataFrame of task's state updates events

        :param task: The task's name or PID or tuple ``(pid, comm)``
        :type task: int or str or tuple(int, str)

        :param stringify: Include stringifed :class:`TaskState` columns
        :type stringify: bool

        :returns: a :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU where the task was on)
          * A ``target_cpu`` column (the CPU where the task has been scheduled).
            Will be ``-1`` for non-wakeup events
          * A ``curr_state`` column (the current task state, see :class:`~TaskState`)
          * A ``next_state`` column (the next task state)
          * A ``delta`` column (the duration for which the task will remain in
            this state)
        """
        tasks_df = list(self._df_tasks_states(tasks=[task]))
        task_id, task_df = tasks_df[0]
        task_df = task_df.drop(columns=["pid", "comm"])

        if stringify:
            self.stringify_df_task_states(task_df, ["curr_state", "next_state"], inplace=True)

        return self._reorder_tasks_states_columns(task_df)

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
            try:
                return TaskState(state).char
            except ValueError:
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
        df = df if inplace else df.copy()

        for col in columns:
            df["{}_str".format(col)] = cls.stringify_task_state_series(df[col])

        return df

    @TraceAnalysisBase.cache
    @_df_tasks_states.used_events
    def df_tasks_runtime(self):
        """
        DataFrame of the time each task spent in TASK_ACTIVE (:class:`TaskState`)

        :returns: a :class:`pandas.DataFrame` with:

          * PIDs as index
          * A ``comm`` column (the name of the task)
          * A ``runtime`` column (the time that task spent running)
        """

        runtimes = {}
        for task, pid_df in self._df_tasks_states():
            pid = task.pid
            # Make sure to only look at the relevant portion of the dataframe
            # with the window, since we are going to make a time-based sum
            pid_df = df_refit_index(pid_df, window=self.trace.window)
            pid_df = df_add_delta(pid_df)
            # Resolve the comm to the last name of the PID in that window
            comms = pid_df['comm'].unique()
            comm = comms[-1]
            pid_df = pid_df[pid_df['curr_state'] == TaskState.TASK_ACTIVE]
            runtimes[pid] = (pid_df['delta'].sum(skipna=True), comm)

        df = pd.DataFrame.from_dict(runtimes, orient="index", columns=["runtime", 'comm'])

        df.index.name = "pid"
        df.sort_values(by="runtime", ascending=False, inplace=True)

        return df

    @TraceAnalysisBase.cache
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
        cpus = set(range(self.trace.cpus_count))

        df = self.df_task_states(task)
        # Get the correct delta for the window we want.
        df = df_add_delta(df, window=self.trace.window)
        df = df[df.curr_state == TaskState.TASK_ACTIVE]

        residency_df = pd.DataFrame(df.groupby("cpu", observed=True, sort=False)["delta"].sum())
        residency_df.rename(columns={"delta": "runtime"}, inplace=True)

        cpus_present = set(residency_df.index.unique())

        for cpu in cpus.difference(cpus_present):
            residency_df.loc[cpu] = 0.

        residency_df.sort_index(inplace=True)

        return residency_df

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
            task_ids = self.trace.task_ids
        else:
            task_ids = itertools.chain.from_iterable(
                self.trace.get_task_ids(task)
                for task in tasks
            )

        res_df = pd.DataFrame()
        for task_id in task_ids:
            mapping = {'runtime': str(task_id)}
            try:
                _df = self.trace.analysis.tasks.df_task_total_residency(task_id).T.rename(index=mapping)
            # Not all tasks may be available, e.g. tasks outside the TraceView
            # window
            except Exception:
                continue
            res_df = res_df.append(_df)

        res_df['Total'] = res_df.iloc[:, :].sum(axis=1)
        res_df.sort_values(by='Total', ascending=ascending, inplace=True)
        if count is None:
            count = len(res_df)

        return res_df[:count]

    @TraceAnalysisBase.cache
    @df_task_states.used_events
    def df_task_activation(self, task, cpu=None, active_value=1, sleep_value=0, preempted_value=np.NaN):
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

        df = self.df_task_states(task)

        def f(state):
            if state == TaskState.TASK_ACTIVE:
                return active_value
            # TASK_RUNNING happens when a task is preempted (so it's not
            # TASK_ACTIVE anymore but still runnable)
            elif state == TaskState.TASK_RUNNING:
                # Return NaN regardless of preempted_value, since some below
                # code relies on that
                return np.NaN
            else:
                return sleep_value

        if cpu is not None:
            df = df[df['cpu'] == cpu]

        df = df.copy()

        # TASK_WAKING can just be removed. The delta will then be computed
        # without it, which means the time spent in WAKING state will be
        # accounted into the previous state.
        df = df[df['curr_state'] != TaskState.TASK_WAKING]

        df['active'] = df['curr_state'].map(f)
        df = df[['active', 'cpu']]

        # Only keep first occurence of each adjacent duplicates, since we get
        # events when the signal changes
        df = df_deduplicate(df, consecutives=True, keep='first')

        # Once we removed the duplicates, we can compute the time spent while sleeping or activating
        df_add_delta(df, col='duration', inplace=True)

        # Make a dataframe where the rows corresponding to preempted time are removed
        preempt_free_df = df.dropna().copy()

        # Merge consecutive activations' duration. They could have been
        # split in two by a bit of preemption, and we don't want that to
        # affect the duty cycle.
        df_combine_duplicates(preempt_free_df, cols=['active'], func=lambda df: df['duration'].sum(), output_col='duration', inplace=True)

        sleep = preempt_free_df[preempt_free_df['active'] == sleep_value]['duration']
        active = preempt_free_df[preempt_free_df['active'] == active_value]['duration']
        # Pair an activation time with it's following sleep time
        sleep = sleep.reindex(active.index, method='bfill')
        duty_cycle = active / (active + sleep)

        df['duty_cycle'] = duty_cycle
        df['duty_cycle'].fillna(inplace=True, method='ffill')

        if not np.isnan(preempted_value):
            df['active'].fillna(preempted_value, inplace=True)

        return df

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method()
    @requires_events('sched_switch')
    def plot_task_residency(self, task: TaskID, axis, local_fig):
        """
        Plot on which CPUs the task ran on over time

        :param task: Task to track
        :type task: int or str or tuple(int, str)
        """

        task_id = self.trace.get_task_id(task, update=False)

        sw_df = self.trace.df_event("sched_switch")
        sw_df = df_filter_task_ids(sw_df, [task_id], pid_col='next_pid', comm_col='next_comm')

        if "freq-domains" in self.trace.plat_info:
            # If we are aware of frequency domains, use one color per domain
            for domain in self.trace.plat_info["freq-domains"]:
                series = sw_df[sw_df["__cpu"].isin(domain)]["__cpu"]

                if series.empty:
                    # Cycle the colours to stay consistent
                    self.cycle_colors(axis)
                else:
                    series = series_refit_index(series, window=self.trace.window)
                    series.plot(ax=axis, style='+',
                            label="Task running in domain {}".format(domain))
        else:
            series = series_refit_index(sw_df['__cpu'], window=self.trace.window)
            series.plot(ax=axis, style='+')

        plot_overutilized = self.trace.analysis.status.plot_overutilized
        if self.trace.has_events(plot_overutilized.used_events):
            plot_overutilized(axis=axis)

        # Add an extra CPU lane to make room for the legend
        axis.set_ylim(-0.95, self.trace.cpus_count - 0.05)

        axis.set_title("CPU residency of task \"{}\"".format(task))
        axis.set_ylabel('CPUs')
        axis.grid(True)
        axis.legend()

    @TraceAnalysisBase.plot_method(return_axis=True)
    @df_task_total_residency.used_events
    def plot_task_total_residency(self, task: TaskID, axis=None, **kwargs):
        """
        Plot a task's total time spent on each CPU

        :param task: The task's name or PID or tuple ``(pid, comm)``
        :type task: str or int or tuple(int, str)
        """
        df = self.df_task_total_residency(task)

        def plotter(axis, local_fig):
            df["runtime"].plot.bar(ax=axis)
            axis.set_title("CPU residency of task \"{}\"".format(task))
            axis.set_xlabel("CPU")
            axis.set_ylabel("Runtime (s)")
            axis.grid(True)

        return self.do_plot(plotter, height=8, axis=axis, **kwargs)

    @TraceAnalysisBase.plot_method()
    @df_tasks_total_residency.used_events
    def plot_tasks_total_residency(self, tasks: TypedList[TaskID]=None, ascending: bool=False,
                                   count: bool=None, axis=None, local_fig=None):
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
        df.T.plot.barh(ax=axis, stacked=True)
        axis.set_title("Stacked CPU residency of [{}] selected tasks"
                       .format(len(df.index)))
        axis.set_ylabel("CPU")
        axis.set_xlabel("Runtime (s)")
        axis.grid(True)
        axis.legend(loc='upper left', ncol=5, bbox_to_anchor=(0, -.15))

    def _plot_cpu_heatmap(self, x, y, xbins, colorbar_label, cmap, **kwargs):
        """
        Plot some data in a heatmap-style 2d histogram
        """
        nr_cpus = self.trace.cpus_count
        fig, axis = self.setup_plot(
            height=min(4, nr_cpus // 2),
            width=20,
            **kwargs
        )

        # According to this thread, passing Pandas object upsets matplotlib for
        # some reasons, so we pass numpy arrays instead:
        # https://stackoverflow.com/questions/34128232/keyerror-when-trying-to-plot-or-histogram-pandas-data-in-matplotlib
        x = x.values
        y = y.values

        _, _, _, img = axis.hist2d(x, y, bins=[xbins, nr_cpus])
        fig.colorbar(img, label=colorbar_label)
        return fig, axis

    @TraceAnalysisBase.plot_method()
    @requires_events("sched_wakeup")
    def plot_tasks_wakeups(self, target_cpus: TypedList[CPU]=None, window: float=0.01, per_sec: bool=False,
                           axis=None, local_fig=None):
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

        df = self.trace.df_event("sched_wakeup")

        if target_cpus:
            df = df[df.target_cpu.isin(target_cpus)]

        series = series_rolling_apply(df["target_cpu"],
                                      lambda x: x.count() / (window if per_sec else 1),
                                      window, window_float_index=False, center=True)

        series = series_refit_index(series, window=self.trace.window)
        series.plot(ax=axis, legend=False)

        if per_sec:
            axis.set_title("Number of task wakeups per second ({}s windows)".format(window))
        else:
            axis.set_title("Number of task wakeups within {}s windows".format(window))

    @TraceAnalysisBase.plot_method(return_axis=True)
    @requires_events("sched_wakeup")
    def plot_tasks_wakeups_heatmap(self, xbins: int=100, colormap=None, axis=None, **kwargs):
        """
        Plot tasks wakeups heatmap

        :param xbins: Number of x-axis bins, i.e. in how many slices should
          time be arranged
        :type xbins: int

        :param colormap: The name of a colormap (see
          https://matplotlib.org/users/colormaps.html), or a Colormap object
        :type colormap: str or matplotlib.colors.Colormap
        """

        df = self.trace.df_event("sched_wakeup")
        df = df_window(df, window=self.trace.window, method='exclusive', clip_window=False)

        fig, axis = self._plot_cpu_heatmap(
            df.index, df.target_cpu, xbins, "Number of wakeups",
            cmap=colormap,
            **kwargs,
        )
        axis.set_title("Tasks wakeups over time")

        return axis

    @TraceAnalysisBase.plot_method()
    @requires_events("sched_wakeup_new")
    def plot_tasks_forks(self, target_cpus: TypedList[CPU]=None, window: float=0.01, per_sec: bool=False,
                         axis=None, local_fig=None):
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

        df = self.trace.df_event("sched_wakeup_new")

        if target_cpus:
            df = df[df.target_cpu.isin(target_cpus)]

        series = series_rolling_apply(df["target_cpu"],
                                      lambda x: x.count() / (window if per_sec else 1),
                                      window, window_float_index=False, center=True)

        series = series_refit_index(series, window=self.trace.window)
        series.plot(ax=axis, legend=False)

        if per_sec:
            axis.set_title("Number of task forks per second ({}s windows)".format(window))
        else:
            axis.set_title("Number of task forks within {}s windows".format(window))

    @TraceAnalysisBase.plot_method(return_axis=True)
    @requires_events("sched_wakeup_new")
    def plot_tasks_forks_heatmap(self, xbins: int=100, colormap=None, axis=None, **kwargs):
        """
        :param xbins: Number of x-axis bins, i.e. in how many slices should
          time be arranged
        :type xbins: int

        :param colormap: The name of a colormap (see
          https://matplotlib.org/users/colormaps.html), or a Colormap object
        :type colormap: str or matplotlib.colors.Colormap
        """

        df = self.trace.df_event("sched_wakeup_new")
        df = df_window(df, window=self.trace.window, method='exclusive', clip_window=False)

        fig, axis = self._plot_cpu_heatmap(
            df.index, df.target_cpu, xbins, "Number of forks",
            cmap=colormap,
            **kwargs
        )

        axis.set_title("Tasks forks over time")
        return axis

    @TraceAnalysisBase.plot_method()
    @df_task_activation.used_events
    def plot_task_activation(self, task: TaskID, cpu: CPU=None, active_value: float=None,
            sleep_value: float=None, alpha: float=None, overlay: bool=False,
            duration: bool=False, duty_cycle: bool=False, which_cpu:
            bool=False, height_duty_cycle: bool=False,
            axis=None, local_fig=None):
        """
        Plot task activations, in a style similar to kernelshark.

        :param task: the task to report activations of
        :type task: int or str or tuple(int, str)

        :param alpha: transparency level of the plot.
        :type task: float

        :param overlay: If ``True``, assumes that ``axis`` already contains a
            plot, so it will adjust automatically the height and transparency of
            the plot to blend with existing data.
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
        # Adapt the steps height to the existing limits. This allows
        # re-using an existing axis that already contains some data.
        min_lim, max_lim = axis.get_ylim()

        df = self.df_task_activation(task, cpu=cpu)

        if overlay:
            active_default = max_lim / 4
            _alpha = alpha if alpha is not None else 0.5
        else:
            active_default = max_lim
            _alpha = alpha

        if duration or duty_cycle:
            active_default = max((
                1 if duty_cycle else 0,
                df['duration'].max() * 1.2 if duration else 0
            ))

        active_value = active_value if active_value is not None else active_default
        sleep_value = sleep_value if sleep_value is not None else 0
        preempted_value = sleep_value

        if not df.empty:
            color = self.get_next_color(axis)

            if which_cpu:
                cpus_count = self.trace.cpus_count
                if overlay:
                    level_height = max_lim / cpus_count
                else:
                    level_height = 1
                    axis.set_yticks(range(cpus_count))
                    # Reversed limits so 0 is at the top, kernelshark-style
                    axis.set_ylim((cpus_count, 0))

                for cpu in range(cpus_count):
                    # The y tick is not reversed, so we need to change the
                    # level manually to match kernelshark behavior
                    if overlay:
                        y_level = level_height * (cpus_count - cpu)
                    else:
                        y_level = level_height * cpu

                    cpu_df = df[df['cpu'] == cpu]
                    active = cpu_df['active'].fillna(preempted_value)
                    if height_duty_cycle:
                        active *= cpu_df['duty_cycle']

                    axis.fill_between(
                        x=cpu_df.index,
                        y1=y_level,
                        y2=y_level + active * level_height,
                        step='post',
                        alpha=_alpha,
                        color=color,
                        # Avoid ugly lines striking through sleep times
                        linewidth=0,
                    )

                if not overlay:
                    axis.set_ylabel('CPU')
            else:
                axis.fill_between(
                    x=df.index,
                    y1=sleep_value,
                    y2=df['active'].map({1: active_value, 0: sleep_value}).fillna(preempted_value),
                    step='post',
                    alpha=_alpha,
                    color=color,
                    linewidth=0,
                )

            if duty_cycle or duration:
                if which_cpu and not overlay:
                    duration_axis = axis.twinx()
                else:
                    duration_axis = axis

                # For some reason fill_between does not advance in the color cycler so let's do that manually.
                self.cycle_colors(duration_axis)

                if duty_cycle:
                    df['duty_cycle'].plot(ax=duration_axis, drawstyle='steps-post', label='Duty cycle of {}'.format(task))
                    duration_axis.set_ylabel('Duty cycle')

                if duration:
                    duration_axis.set_ylabel('Duration in seconds')
                    for active, label in (
                            (True, 'Activations'),
                            (False, 'Sleep')
                        ):
                        duration_series = df[df['active'] == active]['duration']
                        # Add blanks in the plot when the state is not the one we care about
                        duration_series = duration_series.reindex_like(df)
                        duration_series.plot(ax=duration_axis, drawstyle='steps-post', label='{} duration of {}'.format(label, task))

                duration_axis.legend()

        if local_fig:
            axis.set_title('Activations of {}'.format(task))
            axis.grid(True)

    @TraceAnalysisBase.plot_method()
    @plot_task_activation.used_events
    def plot_tasks_activation(self, hide_tasks: TypedList[TaskID]=None, which_cpu: bool=True, **kwargs):
        """
        Plot all tasks activations, in a style similar to kernelshark.

        :param hide_tasks: PIDs to hide. Note that PID 0 (idle task) will
            always be hidden.
        :type hide_tasks: list(TaskID) or None

        :Variable keyword arguments: Forwarded to :meth:`plot_task_activation`.
        """
        trace = self.trace
        hide_tasks = hide_tasks or []
        hidden_pids = {
            trace.get_task_id(task, update=True).pid
            for task in hide_tasks
        }
        # Hide idle task
        hidden_pids.add(0)

        # Plot per-PID, to avoid quirks around task renaming
        for pid, comms in self.trace.get_tasks().items():
            if pid in hidden_pids:
                continue

            self.plot_task_activation(TaskID(pid=pid, comm=None), which_cpu=which_cpu, **kwargs)

    plot_tasks_activation.__annotations__ = plot_task_activation.__annotations__

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
