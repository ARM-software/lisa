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

import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.analysis.status import StatusAnalysis
from lisa.trace import requires_one_event_of, may_use_events
from lisa.utils import deprecate


class LoadTrackingAnalysis(TraceAnalysisBase):
    """
    Support for scheduler load tracking analysis

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'load_tracking'

    def __init__(self, trace):
        super().__init__(trace)

    @classmethod
    def _columns_renaming(cls, event):
        """
        Columns to rename to unify dataframes between trace event versions
        """
        if event in ['sched_load_avg_cpu', 'sched_load_avg_task']:
            return {
                "util_avg" : "util",
                "load_avg" : "load"
            }

        if event == 'sched_util_est_task':
            return {
                'est_enqueued': 'util_est_enqueued',
                'est_ewma': 'util_est_ewma',
            }

        return {}

    @classmethod
    def _columns_to_drop(cls, event):
        """
        The extra columns not shared between trace event versions
        """
        if event in ['sched_load_cfs_rq', 'sched_load_se']:
            return ['path', 'rbl_load']

        if event in ['sched_load_avg_task']:
            return ['load_sum', 'period_contrib', 'util_sum']

        return []

    def _df_uniformized_signal(self, event):
        df = self.trace.df_events(event)

        df = df.rename(columns=self._columns_renaming(event), copy=True)

        if event == 'sched_load_se':
            df = df[df.path == "(null)"]

        if event == 'sched_load_cfs_rq':
            df = df[df.path == "/"]

        to_drop = self._columns_to_drop(event)
        df.drop(columns=to_drop, inplace=True)

        return df


    def _df_either_event(self, events):
        for event in events:
            if event not in self.trace.available_events:
                continue

            return self._df_uniformized_signal(event)

        raise RuntimeError("Trace is missing one of either events: {}".format(events))

    @may_use_events(
        requires_one_event_of('sched_load_cfs_rq', 'sched_load_avg_cpu'),
        'sched_util_est_cpu'
    )
    def df_cpus_signal(self, signal):
        """
        Get the load-tracking signals for the CPUs

        :returns: a :class:`pandas.DataFrame` with a column of the same name as
            the specified ``signal``, and additional context columns such as
            ``cpu``.

        :param signal: Signal name to get. Can be any of:
            * ``util``
            * ``load``
            * ``util_est_enqueued``

        :type signal: str
        """
        common_fields = ['cpu']

        if signal in ('util', 'load'):
            df = self._df_either_event(['sched_load_cfs_rq', 'sched_load_avg_cpu'])
        elif signal == 'util_est_enqueued':
            df = self._df_uniformized_signal('sched_util_est_cpu')
        else:
            raise ValueError('Signal "{}" not supported'.format(signal))

        return df[[*common_fields, signal]]

    @deprecate(replaced_by=df_cpus_signal, deprecated_in='2.0', removed_in='2.1')
    @requires_one_event_of('sched_load_cfs_rq', 'sched_load_avg_cpu')
    def df_cpus_signals(self):
        """
        Get the load-tracking signals for the CPUs

        :returns: a :class:`pandas.DataFrame` with:

          * A ``util`` column (the average utilization of a CPU at time t)
          * A ``load`` column (the average load of a CPU at time t)
        """
        return self._df_either_event(['sched_load_cfs_rq', 'sched_load_avg_cpu'])

    @may_use_events(
        requires_one_event_of('sched_load_se', 'sched_load_avg_task'),
        'sched_util_est_task'
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
            * ``util_est_enqueued``
            * ``util_est_ewma``
            * ``required_capacity``

        :type signal: str
        """
        if signal in ('util', 'load', 'required_capacity'):
            df = self._df_either_event(['sched_load_se', 'sched_load_avg_task'])
            common_fields = ['cpu', 'comm', 'pid']

        elif signal in ('util_est_enqueued', 'util_est_ewma'):
            df = self._df_uniformized_signal('sched_util_est_task')
            common_fields = ['cpu', 'pid']
        else:
            raise ValueError('Signal "{}" not supported'.format(signal))

        if signal == 'required_capacity':
            # Add a column which represents the max capacity of the smallest
            # CPU which can accomodate the task utilization
            capacities = sorted(self.trace.plat_info["cpu-capacities"].values())

            def fits_capacity(util):
                for capacity in capacities:
                    if util <= capacity:
                        return capacity

                return capacities[-1]
            df['required_capacity'] = df.util.map(fits_capacity)

        return df[[*common_fields, signal]]

    @deprecate(replaced_by=df_tasks_signal, deprecated_in='2.0', removed_in='2.1')
    @requires_one_event_of('sched_load_se', 'sched_load_avg_task')
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
        df =  self._df_either_event(['sched_load_se', 'sched_load_avg_task'])

        if "cpu-capacities" in self.trace.plat_info:
            df['required_capacity'] = self.df_tasks_signal('required_capacity')['required_capacity']
        return df

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
        samples = df[df.util > util_threshold].groupby('pid').count()["util"]
        samples = samples[samples > min_samples]
        samples = samples.sort_values(ascending=False)

        top_df = pd.DataFrame(samples).rename(columns={"util" : "samples"})
        top_df["comm"] = top_df.index.map(self.trace.get_task_by_pid)

        return top_df

    @may_use_events(
        StatusAnalysis.plot_overutilized.used_events,
        'cpu_capacity',
    )
    @df_cpus_signal.used_events
    def plot_cpus_signals(self, cpus=None, signals=['util', 'load'], **kwargs):
        """
        Plot the CPU-related load-tracking signals

        :param cpus: list of CPUs to be plotted
        :type cpus: list(int)

        :param signals: List of signals to plot.
        :type signals: list(str)

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        cpus = cpus or list(range(self.trace.cpus_count))

        def plotter(axes, local_fig):
            for cpu in cpus:
                axis = axes[cpu] if len(cpus) > 1 else axes

                # Add CPU utilization
                axis.set_title('CPU{}'.format(cpu))

                for signal in signals:
                    df = self.df_cpus_signal(signal)
                    df = df[df['cpu'] == cpu]
                    df[signal].plot(ax=axis, drawstyle='steps-post', alpha=0.4)

                self.trace.analysis.cpus.plot_orig_capacity(cpu, axis=axis)

                # Add capacities data if available
                if self.trace.has_events('cpu_capacity'):
                    df = self.trace.df_events('cpu_capacity')
                    df = df[df["__cpu"] == cpu]
                    if len(df):
                        data = df[['capacity', 'tip_capacity']]
                        data.plot(ax=axis, style=['m', '--y'],
                                  drawstyle='steps-post')

                # Add overutilized signal to the plot
                plot_overutilized = self.trace.analysis.status.plot_overutilized
                if self.trace.has_events(plot_overutilized.used_events):
                    plot_overutilized(axis=axis)

                axis.set_ylim(0, 1100)
                axis.set_xlim(self.trace.start, self.trace.end)
                axis.legend()

        return self.do_plot(plotter, nrows=len(cpus), sharex=True, **kwargs)

    @df_tasks_signal.used_events
    def plot_task_signals(self, task, signals=['util', 'load'], **kwargs):
        """
        Plot the task-related load-tracking signals

        :param task: The name or PID of the task
        :type task: str or int

        :param signals: List of signals to plot.
        :type signals: list(str)

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """
        pid = self.trace.get_task_pid(task)

        def plotter(axis, local_fig):
            for signal in signals:
                df = self.df_tasks_signal(signal)
                df = df[df.pid == pid]
                df[signal].plot(ax=axis, drawstyle='steps-post', alpha=0.4)

            plot_overutilized = self.trace.analysis.status.plot_overutilized
            if self.trace.has_events(plot_overutilized.used_events):
                plot_overutilized(axis=axis)

            axis.set_title('Load-tracking signals of task "{}"'.format(task))
            axis.legend()
            axis.grid(True)
            axis.set_xlim(self.trace.start, self.trace.end)

        return self.do_plot(plotter, **kwargs)

    @df_tasks_signal.used_events
    def plot_task_required_capacity(self, task, **kwargs):
        """
        Plot the minimum required capacity of a task

        :param task: The name or PID of the task
        :type task: str or int

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """

        pid = self.trace.get_task_pid(task)

        df = self.df_tasks_signal('required_capacity')
        df = df[df.pid == pid]

        # Build task names (there could be multiple, during the task lifetime)
        task_name = 'Task ({}:{})'.format(pid, self.trace.get_task_by_pid(pid))

        def plotter(axis, local_fig):
            df["required_capacity"].plot(
                drawstyle='steps-post',
                ax=axis)

            axis.legend()
            axis.grid(True)

            if local_fig:
                axis.set_title(task_name)
                axis.set_ylim(0, 1100)
                axis.set_xlim(self.trace.start, self.trace.end)
                axis.set_ylabel('Utilization')
                axis.set_xlabel('Time (s)')

        return self.do_plot(plotter, height=8, **kwargs)

    @df_tasks_signal.used_events
    def plot_task_placement(self, task, **kwargs):
        """
        Plot the CPU placement of the task

        :param task: The name or PID of the task
        :type task: str or int

        .. seealso:: :meth:`lisa.analysis.base.AnalysisHelpers.do_plot`
        """

        # Get all utilization update events
        df = self.df_tasks_signal('required_capacity')

        pid = self.trace.get_task_pid(task)
        df = df[df.pid == pid]

        cpu_capacities = self.trace.plat_info["cpu-capacities"]

        def evaluate_placement(cpu, required_capacity):
            capacity = cpu_capacities[cpu]

            if capacity < required_capacity:
                return "CPU capacity < required capacity"
            elif capacity == required_capacity:
                return "CPU capacity == required capacity"
            else:
                return "CPU capacity > required capacity"

        def plotter(axis, local_fig):
            df["placement"] = df.apply(
                lambda row: evaluate_placement(
                    row["__cpu"],
                    row["required_capacity"]), axis=1)

            for stat in df["placement"].unique():
                df[df.placement == stat]["__cpu"].plot(ax=axis, style="+", label=stat)

            plot_overutilized = self.trace.analysis.status.plot_overutilized
            if self.trace.has_events(plot_overutilized.used_events):
                plot_overutilized(axis=axis)

            axis.set_title("Utilization vs placement of task \"{}\"".format(task))

            axis.set_xlim(self.trace.start, self.trace.end)
            axis.grid(True)
            axis.legend()

        return self.do_plot(plotter, **kwargs)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
