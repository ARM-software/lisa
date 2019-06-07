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

import numpy as np
import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import memoized
from lisa.trace import requires_events

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
    #pylint-suppress: bad-whitespace
    TASK_RUNNING         = 0x0000, "R", "Running"
    TASK_INTERRUPTIBLE   = 0x0001, "S", "Sleeping"
    TASK_UNINTERRUPTIBLE = 0x0002, "D", "Disk sleep"
    # __ has a special meaning in Python so let's not do that
    TASK_STOPPED         = 0x0004, "T", "Stopped"
    TASK_TRACED          = 0x0008, "t", "Tracing stop"

    EXIT_DEAD            = 0x0010, "X", "Dead"
    EXIT_ZOMBIE          = 0x0020, "Z", "Zombie"

    # Apparently not visible in traces
    # EXIT_TRACE           = (EXIT_ZOMBIE[0] | EXIT_DEAD[0])

    TASK_PARKED          = 0x0040, "P", "Parked"
    TASK_DEAD            = 0x0080, "I", "Idle"
    TASK_WAKEKILL        = 0x0100
    TASK_WAKING          = 0x0200, "W", "Waking" # LISA-only char definition
    TASK_NOLOAD          = 0x0400
    TASK_NEW             = 0x0800
    TASK_STATE_MAX       = 0x1000

    # LISA-only, used to differenciate runnable (R) vs running (A)
    TASK_ACTIVE          = 0x2000, "A", "Active"

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

class TasksAnalysis(TraceAnalysisBase):
    """
    Support for Tasks signals analysis.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'tasks'

    def __init__(self, trace):
        super(TasksAnalysis, self).__init__(trace)

    @requires_events('sched_switch')
    def cpus_of_tasks(self, tasks):
        """
        Return the list of CPUs where the ``tasks`` executed.

        :param tasks: Task names or PIDs to look for.
        :type tasks: list(int or str)
        """
        trace = self.trace
        pids = [trace.get_task_pid(task) for task in tasks]
        df = trace.df_events('sched_switch')[['next_pid', '__cpu']]
        df = df[df['next_pid'].isin(pids)]
        cpus = df['__cpu'].unique()

        return sorted(cpus)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @requires_events('sched_wakeup')
    def df_tasks_wakeups(self):
        """
        The number of wakeups per task

        :returns: a :class:`pandas.DataFrame` with:

          * Task PIDs as index
          * A ``wakeups`` column (The number of wakeups)
        """
        df = self.trace.df_events('sched_wakeup')

        wakeups = df.groupby('pid').count()["comm"]
        df = pd.DataFrame(wakeups).rename(columns={"comm" : "wakeups"})
        df["comm"] = df.index.map(self.trace.get_task_by_pid)

        return df

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
        df = self.trace.df_events('sched_switch')

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
        rt_tasks['comm'] = rt_tasks.index.map(self.trace.get_task_by_pid)

        return rt_tasks

    @memoized
    @requires_events('sched_switch', 'sched_wakeup')
    def df_tasks_states(self):
        """
        DataFrame of all tasks state updates events

        :returns: a :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU where the task was on)
          * A ``pid`` column (the PID of the task)
          * A ``target_cpu`` column (the CPU where the task has been scheduled).
            Will be ``NaN`` for non-wakeup events
          * A ``curr_state`` column (the current task state, see :class:`~TaskState`)
          * A ``delta`` column (the duration for which the task will remain in
            this state)
          * A ``next_state`` column (the next task state)
        """
        ######################################################
        # A) Assemble the sched_switch and sched_wakeup events
        ######################################################

        wk_df = self.trace.df_events('sched_wakeup')
        sw_df = self.trace.df_events('sched_switch')

        if "sched_wakeup_new" in self.trace.events:
            wkn_df = self.trace.df_events('sched_wakeup_new')
            wk_df = pd.concat([wk_df, wkn_df])

        wk_df = wk_df[wk_df.success == 1][["pid", "target_cpu", "__cpu"]]
        wk_df["curr_state"] = TaskState.TASK_WAKING

        prev_sw_df = sw_df[["__cpu", "prev_pid", "prev_state"]].copy()
        next_sw_df = sw_df[["__cpu", "next_pid"]].copy()

        prev_sw_df.rename(columns={"prev_pid" : "pid", "prev_state" : "curr_state"},
                          inplace=True)

        next_sw_df["curr_state"] = TaskState.TASK_ACTIVE
        next_sw_df.rename(columns={'next_pid' : 'pid'}, inplace=True)

        all_sw_df = prev_sw_df.append(next_sw_df, sort=False)

        # Integer values are prefered here, otherwise the whole column
        # is converted to float64
        all_sw_df['target_cpu'] = -1

        df = all_sw_df.append(wk_df, sort=False)
        df.sort_index(inplace=True)
        df.rename(columns={'__cpu' : 'cpu'}, inplace=True)

        # Move the target_cpu column to the 2nd position
        columns = df.columns.to_list()
        columns = columns[:1] + ["target_cpu"] + \
                  [col for col in columns[1:] if col != "target_cpu"]

        df = df[columns]

        ######################################################
        # B) Compute the deltas for each PID
        ######################################################

        # We have duplicate index values (timestamps) in there, so to make
        # merging easier use an integer indexing instead.
        df.reset_index(inplace=True)

        # To speed up the sorting, we'll append all of the values sequentially
        # and just sort them once at the very end
        index = []
        deltas = []
        states = []

        pids = df.pid.unique()

        for pid in pids:
            df_slice = df[df.pid == pid]
            time = df_slice.Time
            state = df_slice.curr_state

            index += time.index.to_list()
            deltas += list(time.values[1:] - time.values[:-1]) + \
                      [self.trace.end - time.values[-1]]
            states += list(state.values[1:]) + [state.values[-1]]

        merged_df = pd.DataFrame(index=index,
                                 data={"delta" : deltas, "next_state" : states})
        merged_df.sort_index(inplace=True)

        df["delta"] = merged_df.delta
        df["next_state"] = merged_df.next_state
        df.set_index("Time", inplace=True)

        return df

    @df_tasks_states.used_events
    def df_task_states(self, task, stringify=False):
        """
        DataFrame of task's state updates events

        :param task: The task's name or PID
        :type task: int or str

        :param stringify: Include stringifed :class:`TaskState` columns
        :type stringify: bool

        :returns: a :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU where the task was on)
          * A ``target_cpu`` column (the CPU where the task has been scheduled).
            Will be ``NaN`` for non-wakeup events
          * A ``curr_state`` column (the current task state, see :class:`~TaskState`)
          * A ``next_state`` column (the next task state)
          * A ``delta`` column (the duration for which the task will remain in
            this state)
        """
        pid = self.trace.get_task_pid(task)
        df = self.df_tasks_states()

        df = df[df.pid == pid].copy()
        df.drop("pid", axis=1, inplace=True)

        if stringify:
            self.stringify_df_task_states(df, ["curr_state", "next_state"], inplace=True)

        return df

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

    @df_tasks_states.used_events
    def df_tasks_runtime(self):
        """
        DataFrame of the time each task spent in TASK_ACTIVE (:class:`TaskState`)

        :returns: a :class:`pandas.DataFrame` with:

          * PIDs as index
          * A ``comm`` column (the name of the task)
          * A ``runtime`` column (the time that task spent running)
        """
        df = self.df_tasks_states()

        runtimes = {}
        for pid in df.pid.unique():
            runtimes[pid] = df[
                (df.pid == pid) &
                (df.curr_state == TaskState.TASK_ACTIVE)
            ].delta.sum()

        df = pd.DataFrame.from_dict(runtimes, orient="index", columns=["runtime"])

        df.index.name = "pid"
        df.sort_values(by="runtime", ascending=False, inplace=True)
        df.insert(0, "comm", df.index.map(self.trace.get_task_by_pid))

        return df

    @df_task_states.used_events
    def df_task_total_residency(self, task):
        """
        DataFrame of a task's execution time on each CPU

        :param task: the task to report runtimes for
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * CPU IDs as index
          * A ``runtime`` column (the time the task spent being active)
        """
        cpus = set(range(self.trace.cpus_count))

        df = self.df_task_states(task)
        df = df[df.curr_state == TaskState.TASK_ACTIVE]

        residency_df = pd.DataFrame(df.groupby("cpu")["delta"].sum())
        residency_df.rename(columns={"delta" : "runtime"}, inplace=True)

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
        :type tasks: list(int or str)

        :param ascending: Set True to order plot by ascending task runtime
                          False by default
        :type ascending: bool

        :param count: Maximum number of tasks to report
        :type count: int
        """
        if tasks is None:
            tasks = list(self.trace.get_tasks().keys())
        res_df = pd.DataFrame()

        for pid in [self.trace.get_task_pid(task) for task in tasks]:
            task = self.trace.get_task_by_pid(pid)
            mapping = {'runtime': '{}:[{}]'.format(pid, task)}
            _df = self.trace.analysis.tasks.df_task_total_residency(pid).T.rename(index=mapping)
            res_df = res_df.append(_df)

        res_df['Total'] = res_df.iloc[:, :].sum(axis=1)
        res_df.sort_values(by='Total', ascending=ascending, inplace=True)
        if count is None:
            count = len(res_df)

        return res_df[:count]

    @df_task_states.used_events
    def df_task_activation(self, task, cpu=None, active_value=1, sleep_value=0):
        """
        DataFrame of a task's active time on a given CPU

        :param task: the task to report activations of
        :type task: int or str

        :param cpu: the CPUs to look at. If ``None``, all CPUs will be used.
        :type task: int or None

        :param active_value: the value to use in the series when task is
            active.
        :type task: float

        :param active_value: the value to use in the series when task is
            sleeping.
        :type task: float

        :returns: a :class:`pandas.DataFrame` with:

          * A timestamp as index
          * A ``active`` column, containing ``active_value`` when the task is
            not sleeping, ``sleep_value`` otherwise.
        """

        df = self.df_task_states(task)

        def f(state):
            if state == TaskState.TASK_ACTIVE:
                return active_value
            elif state == TaskState.TASK_INTERRUPTIBLE:
                return sleep_value

        if cpu is not None:
            df = df[df['cpu'] == cpu]

        curr_state = df.curr_state
        active_series = curr_state[
            (curr_state == TaskState.TASK_ACTIVE) |
            (curr_state == TaskState.TASK_INTERRUPTIBLE)
        ].map(f)

        return active_series.to_frame('active')

###############################################################################
# Plotting Methods
###############################################################################

    @requires_events('sched_switch')
    def plot_task_residency(self, task, **kwargs):
        """
        Plot on which CPUs the task ran on over time

        :param task: Task to track
        :type task: int or str

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """

        pid = self.trace.get_task_pid(task)

        sw_df = self.trace.df_events("sched_switch")
        sw_df = sw_df[sw_df.next_pid == pid]

        def plotter(axis, local_fig):
            if "freq-domains" in self.trace.plat_info:
                # If we are aware of frequency domains, use one color per domain
                for domain in self.trace.plat_info["freq-domains"]:
                    df = sw_df[sw_df["__cpu"].isin(domain)]["__cpu"]

                    print(domain)

                    if df.empty:
                        print(df.empty)
                        # Cycle the colours to stay consistent
                        self.cycle_colors(axis, 1)
                    else:
                        print(df.unique())
                        df.plot(ax=axis, style='+',
                                label="Task running in domain {}".format(domain))
            else:
                sw_df["__cpu"].plot(ax=axis, style='+')

            plot_overutilized = self.trace.analysis.status.plot_overutilized
            if self.trace.has_events(plot_overutilized.used_events):
                plot_overutilized(axis=axis)

            # Add an extra CPU lane to make room for the legend
            axis.set_ylim(-0.95, self.trace.cpus_count - 0.05)

            axis.set_title("CPU residency of task \"{}\"".format(task))
            axis.set_ylabel('CPUs')
            axis.grid(True)
            axis.legend()
            axis.set_xlim(self.trace.start, self.trace.end)

        return self.do_plot(plotter, **kwargs)

    @df_task_total_residency.used_events
    def plot_task_total_residency(self, task, **kwargs):
        """
        Plot a task's total time spent on each CPU

        :param task: The task's name or PID
        :type task: str or int

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        df = self.df_task_total_residency(task)

        def plotter(axis, local_fig):
            df["runtime"].plot.bar(ax=axis)
            axis.set_title("CPU residency of task \"{}\"".format(task))
            axis.set_xlabel("CPU")
            axis.set_ylabel("Runtime (s)")
            axis.grid(True)

        return self.do_plot(plotter, height=8, **kwargs)

    @df_tasks_total_residency.used_events
    def plot_tasks_total_residency(self, tasks=None, ascending=False,
                                   count=None, **kwargs):
        """
        Plot the stacked total time spent by each task on each CPU

        :param tasks: List of tasks to plot, all trace tasks by default
        :type tasks: list(int or str)

        :param ascending: Set True to order plot by ascending task runtime,
                          False by default
        :type ascending: bool

        :param count: Maximum number of tasks to report
        :type count: int

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        df = self.df_tasks_total_residency(tasks, ascending, count)

        def plotter(axis, local_fig):
            df.T.plot.barh(ax=axis, stacked=True)
            axis.set_title("Stacked CPU residency of [{}] selected tasks"\
                           .format(len(df.index)))
            axis.set_ylabel("CPU")
            axis.set_xlabel("Runtime (s)")
            axis.grid(True)
            axis.legend(loc='upper left',ncol=5, bbox_to_anchor=(0, -.15))

        return self.do_plot(plotter, **kwargs)

    def _df_discretize_series(self, series, time_delta, name):
        """
        Discrete the contents of ``series`` in ``time_delta`` buckets
        """
        left = self.trace.start
        data = []
        index = []
        for right in np.arange(left + time_delta, self.trace.end, time_delta):
            index.append(left)
            data.append(series[left:right].count())
            left = right

        return pd.DataFrame(data=data, index=index, columns=[name])

    def _plot_cpu_heatmap(self, x, y,  xbins, colorbar_label, **kwargs):
        """
        Plot some data in a heatmap-style 2d histogram
        """
        nr_cpus = self.trace.cpus_count
        fig, axis = self.setup_plot(height=min(4, nr_cpus // 2), width=20)

        _, _, _, img = axis.hist2d(x, y, bins=[xbins, nr_cpus], **kwargs)
        fig.colorbar(img, label=colorbar_label)

        return fig, axis

    @requires_events("sched_wakeup")
    def plot_tasks_wakeups(self, target_cpus=None, time_delta=0.01, **kwargs):
        """
        Plot task wakeups over time

        :param target_cpus:
        :type target_cpus:

        :param time_delta: The discretization delta for summing up wakeups in a
          given time delta.
        :type time_delta: float

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """

        df = self.trace.df_events("sched_wakeup")

        if target_cpus:
            df = df[df.target_cpu.isin(target_cpus)]

        df = self._df_discretize_series(df["target_cpu"], time_delta, "Wakeup count")

        def plotter(axis, local_fig):
            df.plot(ax=axis, legend=False)

            axis.set_title("Number of task wakeups within {}s windows".format(time_delta))
            axis.set_xlim(self.trace.start, self.trace.end)

        return self.do_plot(plotter, **kwargs)

    @requires_events("sched_wakeup")
    def plot_tasks_wakeups_heatmap(self, xbins=100, colormap=None, filepath=None):
        """
        :param xbins: Number of x-axis bins, i.e. in how many slices should
          time be arranged
        :type xbins: int

        :param colormap: The name of a colormap (see
          https://matplotlib.org/users/colormaps.html), or a Colormap object
        :type colormap: str or matplotlib.colors.Colormap
        """

        df = self.trace.df_events("sched_wakeup")

        fig, axis = self._plot_cpu_heatmap(
            df.index, df.target_cpu, xbins, "Number of wakeups", cmap=colormap)

        axis.set_title("Tasks wakeups over time")
        axis.set_xlim(self.trace.start, self.trace.end)

        self.save_plot(fig, filepath)

        return axis

    @requires_events("sched_wakeup_new")
    def plot_tasks_forks(self, target_cpus=None, time_delta=0.01, **kwargs):
        """
        Plot task forks over time

        :param target_cpus:
        :type target_cpus:

        :param time_delta: The discretization delta for summing up forks in a
          given time delta.
        :type time_delta: float

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """

        df = self.trace.df_events("sched_wakeup_new")

        if target_cpus:
            df = df[df.target_cpu.isin(target_cpus)]

        df = self._df_discretize_series(df["target_cpu"], time_delta, "Forks count")

        def plotter(axis, local_fig):
            df.plot(ax=axis, legend=False)

            axis.set_title("Number of task forks within {}s windows".format(time_delta))
            axis.set_xlim(self.trace.start, self.trace.end)

        return self.do_plot(plotter, **kwargs)

    @requires_events("sched_wakeup_new")
    def plot_tasks_forks_heatmap(self, xbins=100, colormap=None, filepath=None):
        """
        :param xbins: Number of x-axis bins, i.e. in how many slices should
          time be arranged
        :type xbins: int

        :param colormap: The name of a colormap (see
          https://matplotlib.org/users/colormaps.html), or a Colormap object
        :type colormap: str or matplotlib.colors.Colormap
        """

        df = self.trace.df_events("sched_wakeup_new")

        fig, axis = self._plot_cpu_heatmap(
            df.index, df.target_cpu, xbins, "Number of forks", cmap=colormap)

        axis.set_title("Tasks forks over time")
        axis.set_xlim(self.trace.start, self.trace.end)

        self.save_plot(fig, filepath)

        return axis

    @df_task_activation.used_events
    def plot_task_activation(self, task, cpu=None, active_value=None,
            sleep_value=None, alpha=None, overlay=False,
            **kwargs):
        """
        Plot task activations, in a style similar to kernelshark.

        :param task: the task to report activations of
        :type task: int or str

        :param alpha: transparency level of the plot.
        :type task: float

        :param overlay: If ``True``, assumes that ``axis`` already contains a
            plot, so it will adjust automatically the height and transparency of
            the plot to blend with existing data.
        :type task: bool

        .. seealso:: :meth:`df_task_activation` and
            :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """

        def plotter(axis, local_fig):
            nonlocal active_value, sleep_value

            # Adapt the steps height to the existing limits. This allows
            # re-using an existing axis that already contains some data.
            min_lim, max_lim = axis.get_ylim()
            height = abs(max_lim - min_lim)

            if overlay:
                height /= 4
                _alpha = alpha if alpha is not None else 0.5
            else:
                _alpha = alpha

            active_value = active_value if active_value is not None else height
            sleep_value = sleep_value if sleep_value is not None else min_lim

            df = self.df_task_activation(task,
                cpu=cpu, active_value=active_value, sleep_value=sleep_value,
            )
            if not df.empty:
                axis.fill_between(df.index, df['active'], step='post',
                    alpha=_alpha
                )

        return self.do_plot(plotter, **kwargs)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
