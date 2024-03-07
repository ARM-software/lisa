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
import holoviews as hv

from lisa.analysis.base import TraceAnalysisBase
from lisa.notebook import COLOR_CYCLE, _hv_neutral
from lisa.analysis.tasks import TaskState, TasksAnalysis, TaskID
from lisa.datautils import df_refit_index
from lisa.trace import MissingTraceEventError


class LatencyAnalysis(TraceAnalysisBase):
    """
    Support for plotting Latency Analysis data

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'latency'

    LATENCY_THRESHOLD_ZONE_COLOR = COLOR_CYCLE[2]
    LATENCY_THRESHOLD_COLOR = COLOR_CYCLE[3]

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.df_method
    @TasksAnalysis.df_task_states.used_events
    def _df_latency(self, task, name, curr_state, next_state):
        df = self.ana.tasks.df_task_states(task)
        df = df[
            (df.curr_state == curr_state) &
            (df.next_state == next_state)
        ][["delta", "cpu", "target_cpu"]]

        df = df.rename(columns={'delta': name}, copy=False)
        return df

    @_df_latency.used_events
    def df_latency_wakeup(self, task):
        """
        DataFrame of a task's wakeup latencies

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :returns: a :class:`pandas.DataFrame` with:

          * A ``wakeup_latency`` column (the wakeup latency at that timestamp)
          * A ``cpu`` column (the CPU where the event took place)
          * A ``target_cpu`` column (the CPU where the task has been scheduled)
        """
        return self._df_latency(
            task,
            'wakeup_latency',
            TaskState.TASK_WAKING,
            TaskState.TASK_ACTIVE
        )

    @_df_latency.used_events
    def df_latency_preemption(self, task):
        """
        DataFrame of a task's preemption latencies

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :returns: a :class:`pandas.DataFrame` with:

          * A ``preempt_latency`` column (the preemption latency at that timestamp)
          * A ``cpu`` column (the CPU where the event took place)
        """
        return self._df_latency(
            task,
            'preempt_latency',
            TaskState.TASK_RUNNING,
            TaskState.TASK_ACTIVE
        )[['preempt_latency', 'cpu']]

    @TraceAnalysisBase.df_method
    @TasksAnalysis.df_task_states.used_events
    def df_activations(self, task):
        """
        DataFrame of a task's activations

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :returns: a :class:`pandas.DataFrame` with:

          * An ``activation_interval`` column (the time since the last activation).
        """
        wkp_df = self.ana.tasks.df_task_states(task)
        wkp_df = wkp_df[wkp_df.curr_state == TaskState.TASK_WAKING]

        index = wkp_df.index.to_series()
        activation_interval = (index.shift(-1) - index).shift(1)
        return pd.DataFrame({'activation_interval': activation_interval})

    @TraceAnalysisBase.df_method
    @TasksAnalysis.df_task_states.used_events
    def df_runtimes(self, task):
        """
        DataFrame of task's runtime each time the task blocks

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :returns: a :class:`pandas.DataFrame` with:

          * The times where the task stopped running as an index
          * A ``curr_state`` column (the current task state, see
            :class:`lisa.analysis.tasks.TaskState`)
          * A ``running_time`` column (the cumulated running time since the
            last activation).
        """
        df = self.ana.tasks.df_task_states(task)

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
    def _plot_threshold(self, y, **kwargs):
        return hv.HLine(
            y,
            group='threshold',
            **kwargs,
        ).options(
            color=self.LATENCY_THRESHOLD_COLOR
        ).options(
            backend='matplotlib',
            linestyle='--',
        ).options(
            backend='bokeh',
            line_dash='dashed',
        )

    def _plot_markers(self, df, label):
        return hv.Scatter(df, label=label).options(marker='+').options(
            backend='bokeh',
            size=5,
        ).options(
            backend='matplotlib',
            s=30,
        )

    def _plot_overutilized(self):
        try:
            return self.ana.status.plot_overutilized()
        except MissingTraceEventError:
            return _hv_neutral()

    @TraceAnalysisBase.plot_method
    @df_latency_wakeup.used_events
    @df_latency_preemption.used_events
    def plot_latencies(self, task: TaskID, wakeup: bool=True, preempt: bool=True,
            threshold_ms: float=1):
        """
        Plot the latencies of a task over time

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :param wakeup: Whether to plot wakeup latencies
        :type wakeup: bool

        :param preempt: Whether to plot preemption latencies
        :type preempt: bool

        :param threshold_ms: The latency threshold to plot
        :type threshold_ms: int or float
        """
        def make_fig(name, df_getter, label):
            df = df_getter(task)
            if df.empty:
                self.logger.warning(f"No data to plot for {name}")
            else:
                df = df_refit_index(df, window=self.trace.window)
                return self._plot_markers(df, label)

        return hv.Overlay(
            [
                make_fig(name, df_getter, label)
                for do_plot, name, label, df_getter in (
                    (wakeup, 'wakeup', 'Wakeup', self.df_latency_wakeup),
                    (preempt, 'preempt', 'Preemption', self.df_latency_preemption),
                )
                if do_plot
            ] + [
                self._plot_threshold(
                    threshold_ms / 1e3,
                    label=f"{threshold_ms}ms threshold",
                )
            ]
        ).options(
            title=f'Latencies of task "{task}"',
            ylabel='Latency (s)',
        )

    def _get_cdf(self, data, threshold):
        """
        Build the "Cumulative Distribution Function" (CDF) for the given data
        """
        index = data.sort_values()
        index.name = None
        series = pd.Series(np.linspace(0, 1, len(index)), index=index)
        series.name = data.name

        # Compute percentage of samples above/below the specified threshold
        below = float(max(series[:threshold]))
        above = 1 - below
        return series, above, below


    @df_latency_wakeup.used_events
    @df_latency_preemption.used_events
    def _get_latencies_df(self, task, wakeup, preempt):
        wkp_df = None
        prt_df = None

        if wakeup:
            wkp_df = self.df_latency_wakeup(task)
            wkp_df = wkp_df.rename(columns={'wakeup_latency': 'latency'}, copy=False)

        if preempt:
            prt_df = self.df_latency_preemption(task)
            prt_df = prt_df.rename(columns={'preempt_latency': 'latency'}, copy=False)

        if wakeup and preempt:
            df = pd.concat([wkp_df, prt_df])
        else:
            df = wkp_df or prt_df

        return df

    @TraceAnalysisBase.plot_method
    @_get_latencies_df.used_events
    def plot_latencies_cdf(self, task: TaskID, wakeup: bool=True, preempt: bool=True,
            threshold_ms: float=1):
        """
        Plot the latencies Cumulative Distribution Function of a task

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :param wakeup: Whether to plot wakeup latencies
        :type wakeup: bool

        :param preempt: Whether to plot preemption latencies
        :type preempt: bool

        :param threshold_ms: The latency threshold to plot
        :type threshold_ms: int or float
        """

        df = self._get_latencies_df(task, wakeup, preempt)
        threshold_s = threshold_ms / 1e3
        cdf, above, below = self._get_cdf(df['latency'], threshold_s)

        return (
            hv.Curve(cdf, label='CDF') *
            self._plot_threshold(
                below,
                label=f"Latencies below {threshold_ms}ms",
            ) *
            hv.VSpan(
                0, threshold_s,
                label=f"{threshold_ms}ms threshold zone",
            ).options(
                alpha=0.5,
                color=self.LATENCY_THRESHOLD_ZONE_COLOR,
            )
        ).options(
            title=f'Latencies CDF of task "{task}"',
            xlabel="Latency (s)",
            ylabel="Latencies below the x value (%)",
        )

    @TraceAnalysisBase.plot_method
    @_get_latencies_df.used_events
    def plot_latencies_histogram(self, task: TaskID, wakeup: bool=True,
            preempt: bool=True, threshold_ms: float=1, bins: int=64):
        """
        Plot the latencies histogram of a task

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)

        :param wakeup: Whether to plot wakeup latencies
        :type wakeup: bool

        :param preempt: Whether to plot preemption latencies
        :type preempt: bool

        :param threshold_ms: The latency threshold to plot
        :type threshold_ms: int or float
        """

        df = self._get_latencies_df(task, wakeup, preempt)
        threshold_s = threshold_ms / 1e3
        name = f'Latencies histogram of task {task}'
        return (
            hv.Histogram(
                np.histogram(df['latency'], bins=bins),
                label=name,
            ) *
            hv.VSpan(
                0, threshold_s,
                label=f"{threshold_ms}ms threshold zone",
            ).options(
                color=self.LATENCY_THRESHOLD_ZONE_COLOR,
                alpha=0.5,
            )
        ).options(
            xlabel='Latency (s)',
            title=name,
        )

    @TraceAnalysisBase.plot_method
    @df_latency_wakeup.used_events
    @df_latency_preemption.used_events
    def plot_latency_bands(self, task: TaskID):
        """
        Draw the task wakeup/preemption latencies as colored bands

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)
        """

        wkl_df = self.df_latency_wakeup(task)
        prt_df = self.df_latency_preemption(task)

        def plot_bands(df, column, label):
            df = df_refit_index(df, window=self.trace.window)
            if df.empty:
                return _hv_neutral()

            return hv.Overlay(
                [
                    hv.VSpan(
                        start,
                        start + duration,
                        label=label,
                    ).options(
                        alpha=0.5,
                    )
                    for start, duration in df[[column]].itertuples()
                ]
            )

        return (
            plot_bands(wkl_df, "wakeup_latency", "Wakeup latencies") *
            plot_bands(prt_df, "preempt_latency", "Preemption latencies")
        )

    @TraceAnalysisBase.plot_method
    @df_activations.used_events
    def plot_activations(self, task: TaskID):
        """
        Plot the :meth:`lisa.analysis.latency.LatencyAnalysis.df_activations` of a task

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)
        """

        wkp_df = self.df_activations(task)
        wkp_df = df_refit_index(wkp_df, window=self.trace.window)
        name = f'Activation intervals of task {task}'

        return (
            self._plot_markers(wkp_df, name) *
            self._plot_overutilized()
        ).options(
            title=name,
            ylabel='Activation interval (s)',
        )

    @TraceAnalysisBase.plot_method
    @df_runtimes.used_events
    def plot_runtimes(self, task: TaskID):
        """
        Plot the :meth:`lisa.analysis.latency.LatencyAnalysis.df_runtimes` of a task

        :param task: The task's name or PID
        :type task: int or str or tuple(int, str)
        """
        df = self.df_runtimes(task)
        df = df_refit_index(df, window=self.trace.window)

        name = f'Per-activation runtimes of task {task}'
        return (
            self._plot_markers(df, name) *
            self._plot_overutilized()
        ).options(
            title=name,
        )

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
