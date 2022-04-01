# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

""" Scheduler load tracking analysis module """

import operator
import itertools

import holoviews as hv
import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.analysis.status import StatusAnalysis
from lisa.trace import requires_one_event_of, may_use_events, will_use_events_from, TaskID, CPU, MissingTraceEventError, OrTraceEventChecker
from lisa.utils import deprecate
from lisa.datautils import df_refit_index, series_refit_index, df_filter_task_ids, df_split_signals
from lisa._generic import TypedList
from lisa.notebook import plot_signal, _hv_neutral


class LoadTrackingAnalysis(TraceAnalysisBase):
    """
    Support for scheduler load tracking analysis

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'load_tracking'

    _SCHED_PELT_SE_NAMES = [
        'sched_pelt_se',
        'sched_load_se',
        'sched_load_avg_task'
    ]
    """
    All the names that the per-task load tracking event ever had in various
    kernel versions (Android, mainline etc)
    """

    _SCHED_PELT_CFS_NAMES = [
        'sched_pelt_cfs',
        'sched_load_cfs_rq',
        'sched_load_avg_cpu',
    ]
    """
    All the names that the per-CPU load tracking event ever had in various
    kernel versions (Android, mainline etc)
    """

    @classmethod
    def _columns_renaming(cls, event):
        """
        Columns to rename to unify dataframes between trace event versions
        """
        if event in ['sched_load_avg_cpu', 'sched_load_avg_task']:
            return {
                'load_avg': 'load',
                'util_avg': 'util',
                "utilization": 'util',
                'avg_period': 'period_contrib',
                'runnable_avg_sum': 'load_sum',
                'running_avg_sum': 'util_sum',
            }
        elif event == 'cpu_capacity':
            return {
                'cpu_id': 'cpu',
                'state': 'capacity',
            }
        else:
            return {}

    @classmethod
    def _columns_to_drop(cls, event):
        """
        The extra columns not shared between trace event versions
        """
        if event in [*cls._SCHED_PELT_CFS_NAMES, 'sched_load_se', 'sched_pelt_se']:
            return ['path', 'rbl_load', 'runnable']

        if event in ['sched_load_avg_task']:
            return ['load_sum', 'period_contrib', 'util_sum']

        return []

    def _df_uniformized_signal(self, event):
        df = self.trace.df_event(event)

        df = df.rename(columns=self._columns_renaming(event), copy=True)

        # Legacy sched_load_avg_* events don't have a `path` field.
        if not event.startswith('sched_load_avg_'):
            if event in self._SCHED_PELT_SE_NAMES:
                df = df[df.path == "(null)"]

            if event in self._SCHED_PELT_CFS_NAMES:
                df = df[df.path == "/"]

        to_drop = self._columns_to_drop(event)
        df.drop(columns=to_drop, inplace=True, errors='ignore')

        return df

    def _df_either_event(self, events):
        missing = []
        for event in events:
            try:
                return self._df_uniformized_signal(event)
            except MissingTraceEventError as e:
                missing.append(e.missing_events)

        raise MissingTraceEventError(
            OrTraceEventChecker(missing),
            self.trace.available_events
        )

    @will_use_events_from(
        requires_one_event_of(*_SCHED_PELT_CFS_NAMES),
        'sched_util_est_cfs',
        'sched_cpu_capacity',
    )
    def df_cpus_signal(self, signal, cpus: TypedList[CPU]=None):
        """
        Get the load-tracking signals for the CPUs

        :returns: a :class:`pandas.DataFrame` with a column of the same name as
            the specified ``signal``, and additional context columns such as
            ``cpu``.

        :param signal: Signal name to get. Can be any of:

            * ``util``
            * ``load``
            * ``enqueued`` (util est enqueued)
            * ``capacity``

        :type signal: str

        :param cpus: If specified, list of CPUs to select.
        :type cpus: list(lisa.trace.CPU) or None
        """

        if signal in ('util', 'load'):
            df = self._df_either_event(self._SCHED_PELT_CFS_NAMES)
        elif signal == 'enqueued':
            df = self._df_uniformized_signal('sched_util_est_cfs')
        elif signal == 'capacity':
            df = self._df_uniformized_signal('sched_cpu_capacity')
        else:
            raise ValueError(f'Signal "{signal}" not supported')

        if signal in ('util', 'load'):
            columns = {'cpu', 'update_time', signal}
        elif signal == 'capacity':
            columns = {'cpu', 'capacity_curr', 'capacity', 'capacity_orig'}
        else:
            columns = {'cpu', signal}

        # Select the available columns among
        columns = sorted(set(df.columns) & columns)
        df = df[columns]

        if cpus is not None:
            df = df[df['cpu'].isin(cpus)]
        return df

    @deprecate(replaced_by=df_cpus_signal, deprecated_in='2.0', removed_in='4.0')
    @requires_one_event_of(*_SCHED_PELT_CFS_NAMES)
    def df_cpus_signals(self):
        """
        Get the load-tracking signals for the CPUs

        :returns: a :class:`pandas.DataFrame` with:

          * A ``util`` column (the average utilization of a CPU at time t)
          * A ``load`` column (the average load of a CPU at time t)
        """
        return self._df_either_event(self._SCHED_PELT_CFS_NAMES)

    @TraceAnalysisBase.cache
    @will_use_events_from(
        requires_one_event_of(*_SCHED_PELT_SE_NAMES),
        'sched_util_est_se'
    )
    def df_tasks_signal(self, signal):
        """
        Get the load-tracking signals for the tasks

        :returns: a :class:`pandas.DataFrame` with a column of the same name as
            the specified ``signal``, and additional context columns such as
            ``cpu``.

        :param signal: Signal name to get. Can be any of:

            * ``util``
            * ``load``
            * ``enqueued`` (util est enqueued)
            * ``ewma`` (util est ewma)
            * ``required_capacity``

        :type signal: str
        """
        if signal in ('util', 'load'):
            df = self._df_either_event(self._SCHED_PELT_SE_NAMES)

        elif signal in ('enqueued', 'ewma'):
            df = self._df_uniformized_signal('sched_util_est_se')

        elif signal == 'required_capacity':
            # Add a column which represents the max capacity of the smallest
            # CPU which can accomodate the task utilization
            capacities = sorted(self.trace.plat_info["cpu-capacities"]['orig'].values())

            def fits_capacity(util):
                for capacity in capacities:
                    if util <= capacity:
                        return capacity

                return capacities[-1]
            df = self._df_either_event(self._SCHED_PELT_SE_NAMES)
            df['required_capacity'] = df['util'].map(fits_capacity)

        else:
            raise ValueError(f'Signal "{signal}" not supported')

        # Select the available columns among
        columns = {'cpu', 'comm', 'pid', 'update_time', signal}
        columns = sorted(set(df.columns) & columns)
        return df[columns]

    @TraceAnalysisBase.cache
    @df_tasks_signal.used_events
    def df_task_signal(self, task, signal):
        """
        Same as :meth:`df_tasks_signal` but for one task only.

        :param task: The name or PID of the task, or a tuple ``(pid, comm)``
        :type task: str or int or tuple

        :param signal: See :meth:`df_tasks_signal`.
        """
        task_id = self.trace.get_task_id(task, update=False)
        df = self.df_tasks_signal(signal=signal)
        return df_filter_task_ids(df, [task_id])

    @deprecate(replaced_by=df_tasks_signal, deprecated_in='2.0', removed_in='4.0')
    @requires_one_event_of(*_SCHED_PELT_SE_NAMES)
    def df_tasks_signals(self):
        """
        Get the load-tracking signals for the tasks

        :returns: a :class:`pandas.DataFrame` with:

          * A ``util`` column (the average utilization of a task at time t)
          * A ``load`` column (the average load of a task at time t)

          If CPU capacity information is available:

          * A ``required_capacity`` column (the minimum available CPU capacity
            required to run this task without being CPU-bound)
        """
        df = self._df_either_event(self._SCHED_PELT_SE_NAMES)

        if "orig" in self.trace.plat_info['cpu-capacities']:
            df['required_capacity'] = self.df_tasks_signal('required_capacity')['required_capacity']
        return df

    @TraceAnalysisBase.cache
    @df_tasks_signal.used_events
    def df_top_big_tasks(self, util_threshold, min_samples=100):
        """
        Tasks which had 'utilization' samples bigger than the specified
        threshold

        :param min_samples: minumum number of samples over the min_utilization
        :type min_samples: int

        :param min_utilization: minimum utilization used to filter samples
            default: capacity of a little cluster
        :type min_utilization: int

        :returns: a :class:`pandas.DataFrame` with:

          * Task PIDs as index
          * A ``samples`` column (The number of util samples above the threshold)
        """
        df = self.df_tasks_signal('util')

        # Compute number of samples above threshold
        samples = df[df.util > util_threshold].groupby('pid', observed=True, sort=False).count()["util"]
        samples = samples[samples > min_samples]
        samples = samples.sort_values(ascending=False)

        top_df = pd.DataFrame(samples).rename(columns={"util": "samples"})

        def get_name(pid):
            return self.trace.get_task_pid_names(pid)[-1]
        top_df["comm"] = top_df.index.map(get_name)

        return top_df

    def _plot_overutilized(self):
        try:
            return self.ana.status.plot_overutilized()
        except MissingTraceEventError:
            return _hv_neutral()

    @TraceAnalysisBase.plot_method
    @may_use_events(StatusAnalysis.plot_overutilized.used_events)
    @df_cpus_signal.used_events
    def plot_cpus_signals(self, cpus: TypedList[CPU]=None, signals: TypedList[str]=['util', 'load']):
        """
        Plot the CPU-related load-tracking signals

        :param cpus: list of CPUs to be plotted
        :type cpus: list(int)

        :param signals: List of signals to plot.
        :type signals: list(str)
        """
        cpus = cpus or list(range(self.trace.cpus_count))
        window = self.trace.window

        def _plot_signal(cpu, signal):
            df = self.df_cpus_signal(signal, cpus=[cpu])
            df = df_refit_index(df, window=window)
            return plot_signal(df[signal], name=signal).options(
                dict(Curve=dict(alpha=0.5)),
            )

        def plot_capacity(cpu):
            try:
                df = self.df_cpus_signal('capacity', cpus=[cpu])
            except MissingTraceEventError:
                return _hv_neutral()
            else:
                if df.empty:
                    return _hv_neutral()
                else:
                    return plot_signal(
                        series_refit_index(df['capacity'], window=window),
                        name='capacity'
                    )

        def plot_cpu(cpu):
            return hv.Overlay(
                [
                    _plot_signal(cpu=cpu, signal=signal)
                    for signal in signals
                ] + [
                    self.ana.cpus.plot_orig_capacity(cpu),
                    plot_capacity(cpu),
                    self._plot_overutilized()
                ]
            ).options(
                title=f'CPU{cpu} signals',
            )

        return hv.Layout(list(map(plot_cpu, cpus))).cols(1)

    @TraceAnalysisBase.plot_method
    @df_task_signal.used_events
    def plot_task_signals(self, task: TaskID,  signals: TypedList[str]=['util', 'load']):
        """
        Plot the task-related load-tracking signals

        :param task: The name or PID of the task, or a tuple ``(pid, comm)``
        :type task: str or int or tuple

        :param signals: List of signals to plot.
        :type signals: list(str)
        """
        window = self.trace.window
        task = self.trace.get_task_id(task, update=False)

        def _plot_signal(signal):
            df = self.df_task_signal(task, signal)
            df = df_refit_index(df, window=window)
            return plot_signal(
                df[signal],
                name=signal,
            ).options(
                dict(Curve=dict(alpha=0.5)),
            )

        return hv.Overlay(
            [
                _plot_signal(signal)
                for signal in signals
            ] + [
                self._plot_overutilized()
            ]
        ).options(
            title=f'Load-tracking signals of task {task}',
            ylabel='/'.join(sorted(signals)),
        )

    @TraceAnalysisBase.plot_method
    @df_tasks_signal.used_events
    def plot_task_required_capacity(self, task: TaskID):
        """
        Plot the minimum required capacity of a task

        :param task: The name or PID of the task, or a tuple ``(pid, comm)``
        :type task: str or int or tuple
        """
        window = self.trace.window

        task_ids = self.trace.get_task_ids(task)
        df = self.df_tasks_signal('required_capacity')
        df = df_filter_task_ids(df, task_ids)
        df = df_refit_index(df, window=window)

        # Build task names (there could be multiple, during the task lifetime)
        task_name = f"Task ({', '.join(map(str, task_ids))})"

        return plot_signal(
            df['required_capacity'],
            name='required_capacity',
        ).options(
            title=f'Required CPU capacity for task {task}',
            ylabel='Utilization',
        )

    @TraceAnalysisBase.plot_method
    @df_task_signal.used_events
    def plot_task_placement(self, task: TaskID):
        """
        Plot the CPU placement of the task

        :param task: The name or PID of the task, or a tuple ``(pid, comm)``
        :type task: str or int or tuple
        """
        task_id = self.trace.get_task_id(task, update=False)

        # Get all utilization update events
        df = self.df_task_signal(task_id, 'required_capacity').copy()
        cpu_capacities = self.trace.plat_info["cpu-capacities"]['orig']

        df['capacity'] = df['cpu'].map(cpu_capacities)
        nr_cpus = df['cpu'].max() + 1

        def add_placement(df, comp, comp_str):
            placement = f"CPU capacity {comp_str} required capacity"
            condition = comp(df['capacity'], df['required_capacity'])
            df.loc[condition, 'placement'] = placement

        add_placement(df, operator.lt, '<')
        add_placement(df, operator.gt, '>')
        add_placement(df, operator.eq, '==')

        def plot_placement(cols, placement_df):
            placement = cols['placement']
            series = df["cpu"]
            series = series_refit_index(series, window=self.trace.window)
            series.name = 'cpu'
            return hv.Scatter(
                series,
                label=placement,
            ).options(
                marker='+',
                yticks=list(range(nr_cpus)),
            )

        return hv.Overlay(
            list(itertools.starmap(
                plot_placement,
                df_split_signals(df, ['placement'])
            )) + [
                self._plot_overutilized()
            ]
        ).options(
            title=f'Utilization vs placement of task {task}',
            ylabel='CPU',
        )


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
