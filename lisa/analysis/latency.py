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

import pandas as pd
import numpy as np

from lisa.analysis.base import AnalysisBase, COLOR_CYCLES
from lisa.analysis.tasks import TaskState, TasksAnalysis
from lisa.utils import memoized

class LatencyAnalysis(AnalysisBase):
    """
    Support for plotting Latency Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'latency'

    LATENCY_THRESHOLD_ZONE_COLOR=COLOR_CYCLES[2]
    LATENCY_THRESHOLD_COLOR=COLOR_CYCLES[3]

    def __init__(self, trace):
        super(LatencyAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TasksAnalysis.df_task_states.used_events
    def df_latency_wakeup(self, task):
        """
        DataFrame of a task's wakeup latencies

        :param task: The task's name or PID
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * A ``wakeup_latency`` column (the wakeup latency at that timestamp).
        """

        df = self.trace.analysis.tasks.df_task_states(task)

        df = df[(df.curr_state == TaskState.TASK_WAKING) &
                (df.next_state == TaskState.TASK_ACTIVE)][["delta"]]

        df.rename(columns={'delta' : 'wakeup_latency'}, inplace=True)
        return df

    @TasksAnalysis.df_task_states.used_events
    def df_latency_preemption(self, task):
        """
        DataFrame of a task's preemption latencies

        :param task: The task's name or PID
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * A ``preempt_latency`` column (the preemption latency at that timestamp).
        """
        df = self.trace.analysis.tasks.df_task_states(task)

        df = df[(df.curr_state.str.contains(TaskState.TASK_RUNNING)) &
                (df.next_state == TaskState.TASK_ACTIVE)][["delta"]]

        df.rename(columns={'delta' : 'preempt_latency'}, inplace=True)
        return df

    @TasksAnalysis.df_task_states.used_events
    def df_activations(self, task):
        """
        DataFrame of a task's activations

        :param task: The task's name or PID
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * An ``activation_interval`` column (the time since the last activation).
        """
        wkp_df = self.trace.analysis.tasks.df_task_states(task)
        wkp_df = wkp_df[wkp_df.curr_state == TaskState.TASK_WAKING]

        index = wkp_df.index.to_frame()
        wkp_df['activation_interval'] = (index.shift(-1) - index).shift(1)

        return wkp_df[["activation_interval"]]

    @TasksAnalysis.df_task_states.used_events
    def df_runtimes(self, task):
        """
        DataFrame of task's runtime each time the task blocks

        :param task: The task's name or PID
        :type task: int or str

        :returns: a :class:`pandas.DataFrame` with:

          * The times where the task stopped running as an index
          * A ``curr_state`` column (the current task state, see
            :class:`lisa.analysis.tasks.TaskState`)
          * A ``running_time`` column (the cumulated running time since the
            last activation).
        """
        df = self.trace.analysis.tasks.df_task_states(task)

        runtimes = []
        spurious_wkp = False

        # Using df.apply() is risky for counting (can be called more than once
        # on the same row), so use a loop instead
        for index, row in df.iterrows():
            runtime = runtimes[-1] if len(runtimes) else 0

            if row.curr_state == TaskState.TASK_WAKING:
                # This is required to capture strange trace sequences where
                # a switch_in event is followed by a wakeup_event.
                # This sequence is not expected, but we found it in some traces.
                # Possible reasons could be:
                # - misplaced sched_wakeup events
                # - trace buffer artifacts
                # TO BE BETTER investigated in kernel space.
                # For the time being, we account this interval as RUNNING time,
                # which is what kernelshark does.
                if spurious_wkp:
                    runtime += row.delta
                    spurious_wkp = False
                else:
                    # This is a new activation, reset the runtime counter
                    runtime = 0

            elif row.curr_state == TaskState.TASK_ACTIVE:
                # This is the spurious wakeup thing mentionned above
                if row.next_state == TaskState.TASK_WAKING:
                    spurious_wkp = True

                runtime += row.delta

            runtimes.append(runtime)

        df["running_time"] = runtimes

        # The runtime column is not entirely correct - at a task's first
        # TASK_ACTIVE occurence, the running_time will be non-zero, even
        # though the task has not run yet. However, it's much simpler to
        # accumulate the running_time the way we do and shift it later.
        df.running_time = df.running_time.shift(1)
        df.running_time = df.running_time.fillna(0)

        return df[~df.curr_state.isin([
            TaskState.TASK_ACTIVE,
            TaskState.TASK_WAKING
        ])][["curr_state", "running_time"]]

###############################################################################
# Plotting Methods
###############################################################################

    @df_latency_wakeup.used_events
    def plot_latencies(self, task, wakeup=True, preempt=True, threshold_ms=1,
                       filepath=None):
        """
        Plot the latencies of a task over time

        :param task: The task's name or PID
        :type task: int or str

        :param wakeup: Whether to plot wakeup latencies
        :type wakeup: bool

        :param preempt: Whether to plot preemption latencies
        :type preempt: bool

        :param threshold_ms: The latency threshold to plot
        :type threshold_ms: int or float
        """
        fig, axis = self.setup_plot()

        axis.axhline(threshold_ms / 1e3, linestyle='--', color=self.LATENCY_THRESHOLD_COLOR,
                     label="{}ms threshold".format(threshold_ms))

        if wakeup:
            df = self.df_latency_wakeup(task)
            if df.empty:
                self.get_logger().warning("No data to plot for wakeups")
            else:
                df.plot(ax=axis, style='+', label="Wakeup")

        if preempt:
            df = self.df_latency_preemption(task)
            if df.empty:
                self.get_logger().warning("No data to plot for preemption")
            else:
                df.plot(ax=axis, style='+', label="Preemption")


        axis.set_title("Latencies of task \"{}\"".format(task))
        axis.set_ylabel("Latency (s)")
        axis.legend()
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)
        return axis

    def _get_cdf(self, data, threshold):
        """
        Build the "Cumulative Distribution Function" (CDF) for the given data
        """

        # Build the series of sorted values
        ser = data.sort_values()
        df = pd.Series(np.linspace(0., 1., len(ser)), index=ser)

        # Compute percentage of samples above/below the specified threshold
        below = float(max(df[:threshold]))
        above = 1 - below
        return df, above, below

    @df_latency_wakeup.used_events
    def _get_latencies_df(self, task, wakeup, preempt):
        wkp_df = None
        prt_df = None

        if wakeup:
            wkp_df = self.df_latency_wakeup(task)
            wkp_df.rename(columns={'wakeup_latency' : 'latency'}, inplace=True)

        if preempt:
            prt_df = self.df_latency_preemption(task)
            prt_df.rename(columns={'preempt_latency' : 'latency'}, inplace=True)

        if wakeup and preempt:
            df = wkp_df.append(prt_df)
        else:
            df = wkp_df or prt_df

        return df

    @_get_latencies_df.used_events
    def plot_latencies_cdf(self, task, wakeup=True, preempt=True,
                           threshold_ms=1, filepath=None):
        """
        Plot the latencies Cumulative Distribution Function of a task

        :param task: The task's name or PID
        :type task: int or str

        :param wakeup: Whether to plot wakeup latencies
        :type wakeup: bool

        :param preempt: Whether to plot preemption latencies
        :type preempt: bool

        :param threshold_ms: The latency threshold to plot
        :type threshold_ms: int or float
        """

        fig, axis = self.setup_plot()

        df = self._get_latencies_df(task, wakeup, preempt)
        threshold_s = threshold_ms / 1e3
        cdf_df, above, below = self._get_cdf(df.latency, threshold_s)

        cdf_df.plot(ax=axis, xlim=(0, None), label="CDF")
        axis.axhline(below, linestyle='--', color=self.LATENCY_THRESHOLD_COLOR,
                     label="Latencies below {}ms".format(threshold_ms))
        axis.axvspan(0, threshold_s, facecolor=self.LATENCY_THRESHOLD_ZONE_COLOR,
                     alpha=0.5, label="{}ms threshold zone".format(threshold_ms));

        axis.set_title("Latencies CDF of task \"{}\"".format(task))
        axis.set_xlabel("Latency (s)")
        axis.set_ylabel("Latencies below the x value (%)")
        axis.legend()

        self.save_plot(fig, filepath)
        return axis

    @_get_latencies_df.used_events
    def plot_latencies_histogram(self, task, wakeup=True, preempt=True,
                                 threshold_ms=1, bins=64, filepath=None):
        """
        Plot the latencies histogram of a task

        :param task: The task's name or PID
        :type task: int or str

        :param wakeup: Whether to plot wakeup latencies
        :type wakeup: bool

        :param preempt: Whether to plot preemption latencies
        :type preempt: bool

        :param threshold_ms: The latency threshold to plot
        :type threshold_ms: int or float
        """
        fig, axis= self.setup_plot()

        df = self._get_latencies_df(task, wakeup, preempt)
        threshold_s = threshold_ms / 1e3

        df.latency.plot.hist(bins=bins, ax=axis, xlim=(0, 1.1 * df.latency.max()))
        axis.axvspan(0, threshold_s, facecolor=self.LATENCY_THRESHOLD_ZONE_COLOR, alpha=0.5,
                     label="{}ms threshold zone".format(threshold_ms));

        axis.set_title("Latencies histogram of task \"{}\"".format(task))
        axis.set_xlabel("Latency (s)")
        axis.legend()

        self.save_plot(fig, filepath)
        return axis

    @df_latency_wakeup.used_events
    def plot_latency_bands(self, task, filepath=None, axis=None):
        """
        Draw the task wakeup/preemption latencies as colored bands

        :param task: The task's name or PID
        :type task: int or str

        :param axis: If provided, overlay the bands on this axis
        :type axis: matplotlib.axes.Axes
        """
        local_fig = axis is None

        if local_fig:
            fig, axis = self.setup_plot()

        wkl_df = self.df_latency_wakeup(task)
        prt_df = self.df_latency_preemption(task)

        def plot_bands(df, column, label):
            bands = [(t, df[column][t]) for t in df.index]
            color = self.get_next_color(axis)
            for idx, (start, duration) in enumerate(bands):
                if idx > 0:
                    label = None

                end = start + duration
                axis.axvspan(start, end, facecolor=color, alpha=0.5,
                             label=label)

        plot_bands(wkl_df, "wakeup_latency", "Wakeup latencies")
        plot_bands(prt_df, "preempt_latency", "Preemption latencies")
        axis.legend()
        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        if local_fig:
            self.save_plot(fig, filepath)

        return axis

    @df_activations.used_events
    def plot_activations(self, task, filepath=None):
        """
        Plot the :meth:`lisa.analysis.latency.LatencyAnalysis.df_activations` of a task

        :param task: The task's name or PID
        :type task: int or str
        """
        fig, axis = self.setup_plot()

        wkp_df = self.df_activations(task)

        wkp_df.plot(style='+', logy=False, ax=axis)

        plot_overutilized = self.trace.analysis.status.plot_overutilized
        if self.trace.has_events(plot_overutilized.used_events):
            plot_overutilized(axis=axis)

        axis.set_title("Activation intervals of task \"{}\"".format(task))

        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)
        return axis

    @df_runtimes.used_events
    def plot_runtimes(self, task, filepath=None):
        """
        Plot the :meth:`lisa.analysis.latency.LatencyAnalysis.df_runtimes` of a task

        :param task: The task's name or PID
        :type task: int or str
        """
        fig, axis = self.setup_plot()

        df = self.df_runtimes(task)

        df.plot(style='+', ax=axis)

        plot_overutilized = self.trace.analysis.status.plot_overutilized
        if self.trace.has_events(plot_overutilized.used_events):
            plot_overutilized(axis=axis)

        axis.set_title("Per-activation runtimes of task \"{}\"".format(task))

        axis.set_xlim(self.trace.x_min, self.trace.x_max)

        self.save_plot(fig, filepath)
        return axis

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
