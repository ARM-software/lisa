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

from lisa.analysis.base import AnalysisBase, requires_events
from lisa.utils import memoized

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

class TasksAnalysis(AnalysisBase):
    """
    Support for Tasks signals analysis.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'tasks'

    def __init__(self, trace):
        super(TasksAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @requires_events(['sched_wakeup'])
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

    @requires_events(df_tasks_wakeups.required_events)
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

    @requires_events(['sched_switch'])
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

    @requires_events(['sched_switch', 'sched_wakeup'])
    def df_task_states(self, task):
        """
        DataFrame of task's state updates events

        :param task: The task's name or PID
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * A ``target_cpu`` column (the CPU where the task has been scheduled).
            Will be ``NaN`` for non-wakeup events
          * A ``curr_state`` column (the current task state, see :class:`~TaskState`)
          * A ``next_state`` column (the next task state, see :class:`~TaskState`)
          * A ``delta`` column (the duration for which the task will remain in
            this state)
        """
        pid = self.trace.get_task_pid(task)

        wk_df = self.trace.df_events('sched_wakeup')
        sw_df = self.trace.df_events('sched_switch')

        if "sched_wakeup_new" in self.trace.events:
            wkn_df = self.trace.df_events('sched_wakeup_new')
            wk_df = pd.concat([wk_df, wkn_df]).sort_index()

        task_wakeup = wk_df[wk_df.pid == pid][['target_cpu', '__cpu']]
        task_wakeup['curr_state'] = TaskState.TASK_WAKING.char

        task_switches_df = sw_df[
            (sw_df.prev_pid == pid) |
            (sw_df.next_pid == pid)
        ][['__cpu', 'prev_pid', 'prev_state']]

        def stringify_row_state(row):
            if row.prev_pid != pid:
                # This is a switch-in event
                # (we don't care about the status of a task we are replacing)
                return TaskState.TASK_ACTIVE.char

            return TaskState.sched_switch_str(row.prev_state)

        task_switches_df.prev_state = task_switches_df.apply(
            stringify_row_state, axis=1)

        task_switches_df = task_switches_df.drop(columns=["prev_pid"])

        task_switches_df.rename(columns={'prev_state' : 'curr_state'}, inplace=True)

        # Integer values are prefered here, otherwise the whole column
        # is converted to float64
        task_switches_df['target_cpu'] = -1

        task_state_df = task_wakeup.append(task_switches_df, sort=True).sort_index()

        task_state_df.rename(columns={'__cpu' : 'cpu'}, inplace=True)
        task_state_df = task_state_df[['target_cpu', 'cpu', 'curr_state']]
        task_state_df['next_state'] = task_state_df.curr_state.shift(-1)
        self.trace.add_events_deltas(task_state_df, inplace=True)

        return task_state_df

    @requires_events(df_task_states.required_events)
    def df_task_total_residency(self, task):
        """
        DataFrame of a task's execution time on each CPU

        :param task: the task to report runtimes for
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * CPU IDs as index
          * A ``runtime`` column (the time the task spent being active)
        """
        cpus = set(range(self.trace.plat_info['cpus-count']))

        df = self.df_task_states(task)
        df = df[df.curr_state == TaskState.TASK_ACTIVE.char]

        residency_df = pd.DataFrame(df.groupby("cpu")["delta"].sum())
        residency_df.rename(columns={"delta" : "runtime"}, inplace=True)

        cpus_present = set(residency_df.index.unique())

        for cpu in cpus.difference(cpus_present):
            residency_df.loc[cpu] = 0.

        residency_df.sort_index(inplace=True)

        return residency_df

###############################################################################
# Plotting Methods
###############################################################################

    @requires_events(['sched_switch'])
    def plot_task_residency(self, task, filepath=None):
        """
        Plot on which CPUs the task ran on over time

        :param task:
        """
        fig, axis = self.setup_plot()

        pid = self.trace.get_task_pid(task)

        sw_df = self.trace.df_events("sched_switch")
        sw_df = sw_df[sw_df.next_pid == pid]

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
        if self.trace.has_events(plot_overutilized.required_events):
            plot_overutilized(axis=axis)

        # Add an extra CPU lane to make room for the legend
        axis.set_ylim(-0.95, self.trace.cpus_count - 0.05)

        axis.set_title("CPU residency of task \"{}\"".format(task))
        axis.set_ylabel('CPUs')
        axis.grid(True)
        axis.legend()
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)

        return axis

    @requires_events(df_task_total_residency.required_events)
    def plot_task_total_residency(self, task, filepath=None):
        """
        Plot a task's total time spent on each CPU

        :param task: The task's name or PID
        :type task: str or int
        """
        fig, axis = self.setup_plot(height=8)

        df = self.df_task_total_residency(task)

        df["runtime"].plot.bar(ax=axis)
        axis.set_title("CPU residency of task \"{}\"".format(task))
        axis.set_xlabel("CPU")
        axis.set_ylabel("Runtime (s)")
        axis.grid(True)

        self.save_plot(fig, filepath)

        return axis

    def _df_discretize_series(self, series, time_delta, name):
        """
        Discrete the contents of ``series`` in ``time_delta`` buckets
        """
        left = self.trace.x_min
        data = []
        index = []
        for right in np.arange(left + time_delta, self.trace.x_max, time_delta):
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

    @requires_events(["sched_wakeup"])
    def plot_tasks_wakeups(self, target_cpus=None, time_delta=0.01, filepath=None):
        """
        Plot task wakeups over time

        :param target_cpus:
        :type target_cpus:

        :param time_delta: The discretization delta for summing up wakeups in a
          given time delta.
        :type time_delta: float
        """
        fig, axis = self.setup_plot()

        df = self.trace.df_events("sched_wakeup")

        if target_cpus:
            df = df[df.target_cpu.isin(target_cpus)]

        df = self._df_discretize_series(df["target_cpu"], time_delta, "Wakeup count")
        df.plot(ax=axis, legend=False)

        axis.set_title("Number of task wakeups within {}s windows".format(time_delta))
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)

        return axis

    @requires_events(["sched_wakeup"])
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
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)

        return axis

    @requires_events(["sched_wakeup_new"])
    def plot_tasks_forks(self, target_cpus=None, time_delta=0.01, filepath=None):
        """
        Plot task forks over time

        :param target_cpus:
        :type target_cpus:

        :param time_delta: The discretization delta for summing up forks in a
          given time delta.
        :type time_delta: float
        """
        fig, axis = self.setup_plot()

        df = self.trace.df_events("sched_wakeup_new")

        if target_cpus:
            df = df[df.target_cpu.isin(target_cpus)]

        df = self._df_discretize_series(df["target_cpu"], time_delta, "Forks count")
        df.plot(ax=axis, legend=False)

        axis.set_title("Number of task forks within {}s windows".format(time_delta))
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)

        return axis

    @requires_events(["sched_wakeup_new"])
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
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)

        return axis

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
