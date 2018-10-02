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

""" Tasks Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import re

from lisa.analysis.base import AnalysisBase
from lisa.utils import memoized
from trappy.utils import listify


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

    def df_top_big_tasks(self, min_samples=100, min_utilization=None):
        """
        Tasks which had 'utilization' samples bigger than the specified
        threshold

        :param min_samples: minumum number of samples over the min_utilization
        :type min_samples: int

        :param min_utilization: minimum utilization used to filter samples
            default: capacity of a little cluster
        :type min_utilization: int
        """
        if self.df_load() is None:
            self._log.warning('No trace events for task signals, plot DISABLED')
            return None

        if min_utilization is None:
            min_utilization = self._little_cap

        # Get utilization samples >= min_utilization
        df = self.df_load()
        big_tasks_events = df[df.util_avg > min_utilization]
        if not len(big_tasks_events):
            self._log.warning('No tasks with with utilization samples > %d',
                              min_utilization)
            return None

        # Report the number of tasks which match the min_utilization condition
        big_tasks = big_tasks_events.pid.unique()
        self._log.info('%5d tasks with samples of utilization > %d',
                       len(big_tasks), min_utilization)

        # Compute number of samples above threshold
        desc = big_tasks_events.groupby('pid').describe(include=['object'])
        if isinstance(desc.index, pd.MultiIndex):
            # We must be running on a pre-0.20.0 version of pandas.
            # unstack will convert the old output format to the new.
            # http://pandas.pydata.org/pandas-docs/version/0.20/whatsnew.html#groupby-describe-formatting
            desc = desc.unstack()
        big_tasks_stats = desc['comm'].sort_values(by=['count'], ascending=False)

        # Filter for number of occurrences
        big_tasks_stats = big_tasks_stats[big_tasks_stats['count'] > min_samples]
        if not len(big_tasks_stats):
            self._log.warning('      but none with more than %d samples',
                              min_samples)
            return None

        self._log.info('      %d with more than %d samples',
                       len(big_tasks_stats), min_samples)

        # Add task name column
        big_tasks_stats['comm'] = big_tasks_stats.index.map(
            lambda pid: self._trace.getTaskByPid(pid))

        # Filter columns of interest
        big_tasks_stats = big_tasks_stats[['count', 'comm']]
        big_tasks_stats.rename(columns={'count': 'samples'}, inplace=True)

        return big_tasks_stats

    def df_top_wakeup(self, min_wakeups=100):
        """
        Tasks which wakeup more frequently than a specified threshold.

        :param min_wakeups: minimum number of wakeups
        :type min_wakeups: int
        """
        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Events [sched_wakeup] not found')
            return None

        df = self._trace.df_events('sched_wakeup')

        # Compute number of wakeups above threshold
        wkp_tasks_stats = df.groupby('pid').describe(include=['object'])
        wkp_tasks_stats = wkp_tasks_stats.unstack()['comm']\
                          .sort_values(by=['count'], ascending=False)

        # Filter for number of occurrences
        wkp_tasks_stats = wkp_tasks_stats[
            wkp_tasks_stats['count'] > min_wakeups]
        if not len(df):
            self._log.warning('No tasks with more than %d wakeups',
                              len(wkp_tasks_stats))
            return None
        self._log.info('%5d tasks with more than %d wakeups',
                       len(df), len(wkp_tasks_stats))

        # Add task name column
        wkp_tasks_stats['comm'] = wkp_tasks_stats.index.map(
            lambda pid: self._trace.getTaskByPid(pid))

        # Filter columns of interest
        wkp_tasks_stats = wkp_tasks_stats[['count', 'comm']]
        wkp_tasks_stats.rename(columns={'count': 'samples'}, inplace=True)

        return wkp_tasks_stats

    def df_rt_tasks(self, min_prio=100):
        """
        Tasks with RT priority

        NOTE: priorities uses scheduler values, thus: the lower the value the
              higher is the task priority.
              RT   Priorities: [  0..100]
              FAIR Priorities: [101..120]

        :param min_prio: minumum priority
        :type min_prio: int
        """
        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Events [sched_switch] not found')
            return None

        df = self._trace.df_events('sched_switch')

        # Filters tasks which have a priority bigger than threshold
        df = df[df.next_prio <= min_prio]

        # Filter columns of interest
        rt_tasks = df[['next_pid', 'next_prio']]

        # Remove all duplicateds
        rt_tasks = rt_tasks.drop_duplicates()

        # Order by priority
        rt_tasks.sort_values(by=['next_prio', 'next_pid'], ascending=True,
                             inplace=True)
        rt_tasks.rename(columns={'next_pid': 'pid', 'next_prio': 'prio'},
                        inplace=True)

        # Set PID as index
        rt_tasks.set_index('pid', inplace=True)

        # Add task name column
        rt_tasks['comm'] = rt_tasks.index.map(
            lambda pid: self._trace.getTaskByPid(pid))

        return rt_tasks

    def df_load(self):
        """
        Get a DataFrame with the scheduler's per-task load-tracking signals

        Parse the relevant trace event and return a DataFrame with the
        scheduler's load tracking update events for each task.

        :returns: DataFrame with at least the following columns:
                  'comm', 'pid', 'load_avg', 'util_avg'.
        """
        df = None

        if 'sched_load_avg_task' in self._trace.available_events:
            df = self._trace.df_events('sched_load_avg_task')

        elif 'sched_load_se' in self._trace.available_events:
            df = self._trace.df_events('sched_load_se')
            df = df.rename(columns={'util': 'util_avg', 'load': 'load_avg'})
            # In sched_load_se, PID shows -1 for task groups.
            df = df[df.pid != -1]

        if not self._trace.has_big_little:
            return df

        df['cluster'] = np.select(
                [df.cpu.isin(self._trace.plat_info['clusters']['little'])],
                ['LITTLE'], 'big')

        if 'nrg-model' in self._trace.plat_info:
            # Add a column which represents the max capacity of the smallest
            # clustre which can accomodate the task utilization
            little_cap = self._trace.plat_info['nrg-model']['little']['cpu']['cap_max']
            big_cap = self._trace.plat_info['nrg-model']['big']['cpu']['cap_max']
            df['min_cluster_cap'] = df.util_avg.map(
                lambda util_avg: big_cap if util_avg > little_cap else little_cap
            )

        return df

###############################################################################
# Plotting Methods
###############################################################################

    def plot_tasks(self, tasks, signals=None):
        """
        Generate a common set of useful plots for each of the specified tasks

        This method allows to filter which signals should be plot, if data are
        available in the input trace. The list of signals supported are:
        Tasks signals plot:
                load_avg, util_avg, boosted_util, sched_overutilized
        Tasks residencies on CPUs:
                residencies, sched_overutilized
        Tasks PELT signals:
                load_sum, util_sum, period_contrib, sched_overutilized

        At least one of the previous signals must be specified to get a valid
        plot.

        Addidional custom signals can be specified and they will be represented
        in the "Task signals plots" if they represent valid keys of the task
        load/utilization trace event (e.g. sched_load_avg_task).

        Note:
            sched_overutilized: enable the plotting of overutilization bands on
                                top of each subplot
            residencies: enable the generation of the CPUs residencies plot

        :param tasks: the list of task names and/or PIDs to plot.
                      Numerical PIDs and string task names can be mixed
                      in the same list.
        :type tasks: list(str) or list(int)

        :param signals: list of signals (and thus plots) to generate
                        default: all the plots and signals available in the
                        current trace
        :type signals: list(str)
        """
        if not signals:
            signals = ['load_avg', 'util_avg', 'boosted_util',
                       'sched_overutilized',
                       'load_sum', 'util_sum', 'period_contrib',
                       'residencies']

        # Check for the minimum required signals to be available
        if self.df_load() is None:
            self._log.warning('No trace events for task signals, plot DISABLED')
            return

        # Defined list of tasks to plot
        if tasks and \
            not isinstance(tasks, str) and \
            not isinstance(tasks, list):
            raise ValueError('Wrong format for tasks parameter')

        if tasks:
            tasks_to_plot = listify(tasks)
        else:
            raise ValueError('No tasks to plot specified')

        # Compute number of plots to produce
        plots_count = 0
        plots_signals = [
                # Fist plot: task's utilization
                {'load_avg', 'util_avg', 'boosted_util'},
                # Second plot: task residency
                {'residencies'},
                # Third plot: tasks's load
                {'load_sum', 'util_sum', 'period_contrib'}
        ]
        hr = []
        ysize = 0
        for plot_id, signals_to_plot in enumerate(plots_signals):
            signals_to_plot = signals_to_plot.intersection(signals)
            if len(signals_to_plot):
                plots_count = plots_count + 1
                # Use bigger size only for the first plot
                hr.append(3 if plot_id == 0 else 1)
                ysize = ysize + (8 if plot_id else 4)

        # Grid
        gs = gridspec.GridSpec(plots_count, 1, height_ratios=hr)
        gs.update(wspace=0.1, hspace=0.1)

        # Build list of all PIDs for each task_name to plot
        pids_to_plot = []
        for task in tasks_to_plot:
            # Add specified PIDs to the list
            if isinstance(task, int):
                pids_to_plot.append(task)
                continue
            # Otherwise: add all the PIDs for task with the specified name
            pids_to_plot.extend(self._trace.getTaskByName(task))

        for tid in pids_to_plot:
            savefig = False

            task_name = self._trace.getTaskByPid(tid)
            self._log.info('Plotting [%d:%s]...', tid, task_name)
            plot_id = 0

            # For each task create a figure with plots_count plots
            plt.figure(figsize=(16, ysize))
            plt.suptitle('Task Signals',
                         y=.94, fontsize=16, horizontalalignment='center')

            # Plot load and utilization
            signals_to_plot = {'load_avg', 'util_avg', 'boosted_util'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 0:
                axes = plt.subplot(gs[plot_id, 0])
                axes.set_title('Task [{0:d}:{1:s}] Signals'
                               .format(tid, task_name))
                plot_id = plot_id + 1
                is_last = (plot_id == plots_count)
                self._plot_task_signals(axes, tid, signals, is_last)
                savefig = True

            # Plot CPUs residency
            signals_to_plot = {'residencies'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 0:
                if not self._trace.has_big_little:
                    self._log.warning(
                        'No big.LITTLE platform data, residencies plot disabled')
                else:
                    axes = plt.subplot(gs[plot_id, 0])
                    axes.set_title(
                        'Task [{0:d}:{1:s}] Residency (green: LITTLE, red: big)'
                        .format(tid, task_name)
                    )
                    plot_id = plot_id + 1
                    is_last = (plot_id == plots_count)
                    if 'sched_overutilized' in signals:
                        signals_to_plot.append('sched_overutilized')
                    self._plot_task_residencies(axes, tid, signals_to_plot, is_last)
                    savefig = True

            # Plot PELT signals
            signals_to_plot = {'load_sum', 'util_sum', 'period_contrib'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 0:
                axes = plt.subplot(gs[plot_id, 0])
                axes.set_title('Task [{0:d}:{1:s}] PELT Signals'
                               .format(tid, task_name))
                plot_id = plot_id + 1
                if 'sched_overutilized' in signals:
                    signals_to_plot.append('sched_overutilized')
                self._plot_task_pelt(axes, tid, signals_to_plot)
                savefig = True

            if not savefig:
                self._log.warning('Nothing to plot for %s', task_name)
                continue

            # Save generated plots into datadir
            if isinstance(task_name, list):
                task_name = re.sub('[:/]', '_', task_name[0])
            else:
                task_name = re.sub('[:/]', '_', task_name)
            figname = '{}/{}task_util_{}_{}.png'\
                      .format(self._trace.plots_dir, self._trace.plots_prefix,
                              tid, task_name)
            pl.savefig(figname, bbox_inches='tight')

    def plot_big_tasks(self, max_tasks=10, min_samples=100,
                     min_utilization=None):
        """
        For each big task plot utilization and show the smallest cluster
        capacity suitable for accommodating task utilization.

        :param max_tasks: maximum number of tasks to consider
        :type max_tasks: int

        :param min_samples: minumum number of samples over the min_utilization
        :type min_samples: int

        :param min_utilization: minimum utilization used to filter samples
            default: capacity of a little cluster
        :type min_utilization: int
        """

        # Get PID of big tasks
        big_frequent_task_df = self.df_top_big_tasks(
            min_samples, min_utilization)
        if big_frequent_task_df is None:
            # (Logged already)
            return

        if max_tasks > 0:
            big_frequent_task_df = big_frequent_task_df.head(max_tasks)
        big_frequent_task_pids = big_frequent_task_df.index.values

        big_frequent_tasks_count = len(big_frequent_task_pids)
        if big_frequent_tasks_count == 0:
            self._log.warning('No big/frequent tasks to plot')
            return

        # Get the list of events for all big frequent tasks
        df = self.df_load()
        big_frequent_tasks_events = df[df.pid.isin(big_frequent_task_pids)]

        # Define axes for side-by-side plottings
        fig, axes = plt.subplots(big_frequent_tasks_count, 1,
                                 figsize=(16, big_frequent_tasks_count*4))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        plot_idx = 0
        for pid, group in big_frequent_tasks_events.groupby('pid'):

            # # Build task names (there could be multiple, during the task lifetime)
            task_name = 'Task [%d:%s]'.format(pid, self._trace.getTaskByPid(pid))

            # Plot title
            if big_frequent_tasks_count == 1:
                ax = axes
            else:
                ax = axes[plot_idx]
            ax.set_title(task_name)

            # Left axis: utilization
            ax = group.plot(y=['util_avg', 'min_cluster_cap'],
                            style=['r.', '-b'],
                            drawstyle='steps-post',
                            linewidth=1,
                            ax=ax)
            ax.set_xlim(self._trace.x_min, self._trace.x_max)
            ax.set_ylim(0, 1100)
            ax.set_ylabel('util_avg')
            ax.set_xlabel('')
            ax.grid(True)
            self._trace.analysis.status.plot_overutilized(ax)

            plot_idx += 1

        ax.set_xlabel('Time [s]')

        self._log.info('Tasks which have been a "utilization" of %d for at least %d samples',
                       self._little_cap, min_samples)

    def plot_wakeup(self, max_tasks=10, min_wakeups=0, per_cluster=False):
        """
        Show waking up tasks over time and newly forked tasks in two separate
        plots.

        :param max_tasks: maximum number of tasks to consider
        :param max_tasks: int

        :param min_wakeups: minimum number of wakeups of each task
        :type min_wakeups: int

        :param per_cluster: if True get per-cluster wakeup events
        :type per_cluster: bool
        """
        if per_cluster is True and \
           not self._trace.hasEvents('sched_wakeup_new'):
            self._log.warning('Events [sched_wakeup_new] not found, '
                              'plots DISABLED!')
            return
        elif  not self._trace.hasEvents('sched_wakeup') and \
              not self._trace.hasEvents('sched_wakeup_new'):
            self._log.warning('Events [sched_wakeup, sched_wakeup_new] not found, '
                              'plots DISABLED!')
            return

        # Define axes for side-by-side plottings
        fig, axes = plt.subplots(2, 1, figsize=(14, 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        if per_cluster:

            # Get per cluster wakeup events
            df = self._trace.df_events('sched_wakeup_new')
            big_frequent = df.target_cpu.isin(self._big_cpus)
            ntbc = df[big_frequent]
            ntbc_count = len(ntbc)
            little_frequent = df.target_cpu.isin(self._little_cpus)
            ntlc = df[little_frequent];
            ntlc_count = len(ntlc)

            self._log.info('%5d tasks forked on big cluster    (%3.1f %%)',
                           ntbc_count,
                           100. * ntbc_count / (ntbc_count + ntlc_count))
            self._log.info('%5d tasks forked on LITTLE cluster (%3.1f %%)',
                           ntlc_count,
                           100. * ntlc_count / (ntbc_count + ntlc_count))

            ax = axes[0]
            ax.set_title('Tasks Forks on big CPUs');
            ntbc.pid.plot(style=['g.'], ax=ax);
            ax.set_xlim(self._trace.x_min, self._trace.x_max);
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.grid(True)
            self._trace.analysis.status.plot_overutilized(ax)

            ax = axes[1]
            ax.set_title('Tasks Forks on LITTLE CPUs');
            ntlc.pid.plot(style=['g.'], ax=ax);
            ax.set_xlim(self._trace.x_min, self._trace.x_max);
            ax.grid(True)
            self._trace.analysis.status.plot_overutilized(ax)

            return

        # Keep events of defined big tasks
        wkp_task_pids = self.df_top_wakeup(min_wakeups)
        if len(wkp_task_pids):
            wkp_task_pids = wkp_task_pids.index.values[:max_tasks]
            self._log.info('Plotting %d frequent wakeup tasks',
                           len(wkp_task_pids))

        ax = axes[0]
        ax.set_title('Tasks WakeUps Events')
        df = self._trace.df_events('sched_wakeup')
        if len(df):
            df = df[df.pid.isin(wkp_task_pids)]
            df.pid.astype(int).plot(style=['b.'], ax=ax)
            ax.set_xlim(self._trace.x_min, self._trace.x_max)
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.grid(True)
            self._trace.analysis.status.plot_overutilized(ax)

        ax = axes[1]
        ax.set_title('Tasks Forks Events')
        df = self._trace.df_events('sched_wakeup_new')
        if len(df):
            df = df[df.pid.isin(wkp_task_pids)]
            df.pid.astype(int).plot(style=['r.'], ax=ax)
            ax.set_xlim(self._trace.x_min, self._trace.x_max)
            ax.grid(True)
            self._trace.analysis.status.plot_overutilized(ax)

    def plot_big_tasks_vs_capacity(self, min_samples=1,
                               min_utilization=None, big_cluster=True):
        """
        Draw a plot that shows whether tasks are placed on the correct cluster
        based on their utilization and cluster capacity. Green dots mean the
        task was placed on the correct cluster, Red means placement was wrong

        :param min_samples: minumum number of samples over the min_utilization
        :type min_samples: int

        :param min_utilization: minimum utilization used to filter samples
            default: capacity of a little cluster
        :type min_utilization: int

        :param big_cluster:
        :type big_cluster: bool
        """

        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warning('Events [cpu_frequency] not found')
            return

        # Get all utilization update events
        df = self.df_load()
        if df is None:
            self._log.warning('No trace events for task signals, plot DISABLED')
            return

        if big_cluster:
            cluster_correct = 'big'
            cpus = self._big_cpus
        else:
            cluster_correct = 'LITTLE'
            cpus = self._little_cpus

        # Keep events of defined big tasks
        big_task_pids = self.df_top_big_tasks(
            min_samples, min_utilization)
        if big_task_pids is not None:
            big_task_pids = big_task_pids.index.values
            df = df[df.pid.isin(big_task_pids)]
        if not df.size:
            self._log.warning('No events for tasks with more then %d utilization '
                              'samples bigger than %d, plots DISABLED!')
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        # Add column of expected cluster depending on:
        # a) task utilization value
        # b) capacity of the selected cluster
        bu_bc = ((df['util_avg'] > self._little_cap) &
                 (df['cpu'].isin(self._big_cpus)))
        su_lc = ((df['util_avg'] <= self._little_cap) &
                 (df['cpu'].isin(self._little_cpus)))

        # The Cluster CAPacity Matches the UTILization (ccap_mutil) iff:
        # - tasks with util_avg  > little_cap are running on a BIG cpu
        # - tasks with util_avg <= little_cap are running on a LITTLe cpu
        df.loc[:,'ccap_mutil'] = np.select([(bu_bc | su_lc)], [True], False)

        df_freq = self._trace.df_events('cpu_frequency')
        df_freq = df_freq[df_freq.cpu == cpus[0]]

        ax = axes[0]
        ax.set_title('Tasks Utilization vs Allocation')
        for ucolor, umatch in zip('gr', [True, False]):
            cdata = df[df['ccap_mutil'] == umatch]
            if len(cdata) > 0:
                cdata['util_avg'].plot(ax=ax,
                        style=[ucolor+'.'], legend=False)
        ax.set_xlim(self._trace.x_min, self._trace.x_max)
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.grid(True)
        self._trace.analysis.status.plot_overutilized(ax)

        ax = axes[1]
        ax.set_title('Frequencies on "{}" cluster'.format(cluster_correct))
        df_freq['frequency'].plot(style=['-b'], ax=ax, drawstyle='steps-post')
        ax.set_xlim(self._trace.x_min, self._trace.x_max);
        ax.grid(True)
        self._trace.analysis.status.plot_overutilized(ax)

        legend_y = axes[0].get_ylim()[1]
        axes[0].annotate('Utilization-Capacity Matches',
                         xy=(0, legend_y),
                         xytext=(-50, 45), textcoords='offset points',
                         fontsize=18)
        axes[0].annotate('Task schduled (green) or not (red) on min cluster',
                         xy=(0, legend_y),
                         xytext=(-50, 25), textcoords='offset points',
                         fontsize=14)


###############################################################################
# Utility Methods
###############################################################################

    def _plot_task_signals(self, axes, tid, signals, is_last=False):
        """
        For task with ID `tid` plot the specified signals.

        :param axes: axes over which to generate the plot
        :type axes: :mod:`matplotlib.axes.Axes`

        :param tid: task ID
        :type tid: int

        :param signals: signals to be plot
        :param signals: list(str)

        :param is_last: if True this is the last plot
        :type is_last: bool
        """
        # Get dataframe for the required task
        util_df = self.df_load()
        if util_df is None:
            self._log.warning('No trace events for task signals, plot DISABLED')
            return

        # Plot load and util
        signals_to_plot = set(signals).difference({'boosted_util'})
        for signal in signals_to_plot:
            if signal not in util_df.columns:
                continue
            data = util_df[util_df.pid == tid][signal]
            data.plot(ax=axes, drawstyle='steps-post', legend=True)

        # Plot boost utilization if available
        if 'boosted_util' in signals and \
           self._trace.hasEvents('sched_boost_task'):
            boost_df = self._trace.df_events('sched_boost_task')
            data = boost_df[boost_df.pid == tid][['boosted_util']]
            if len(data):
                data.plot(ax=axes, style=['y-'], drawstyle='steps-post')
            else:
                task_name = self._trace.getTaskByPid(tid)
                self._log.warning('No "boosted_util" data for task [%d:%s]',
                                  tid, task_name)

        # Add Capacities data if avilable
        if 'nrg-model' in self._trace.plat_info:
            nrg_model = self._trace.plat_info['nrg-model']
            max_lcap = nrg_model['little']['cpu']['cap_max']
            max_bcap = nrg_model['big']['cpu']['cap_max']
            tip_lcap = 0.8 * max_lcap
            tip_bcap = 0.8 * max_bcap
            self._log.debug(
                'LITTLE capacity tip/max: %d/%d, big capacity tip/max: %d/%d',
                tip_lcap, max_lcap, tip_bcap, max_bcap
            )
            axes.axhline(tip_lcap, color='y', linestyle=':', linewidth=2)
            axes.axhline(max_lcap, color='y', linestyle='--', linewidth=2)
            axes.axhline(tip_bcap, color='r', linestyle=':', linewidth=2)
            axes.axhline(max_bcap, color='r', linestyle='--', linewidth=2)

        axes.set_ylim(0, 1100)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.grid(True)
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self._trace.analysis.status.plot_overutilized(axes)

    def _plot_task_residencies(self, axes, tid, signals, is_last=False):
        """
        For task with ID `tid` plot residency information.

        :param axes: axes over which to generate the plot
        :type axes: :mod:`matplotlib.axes.Axes`

        :param tid: task ID
        :type tid: int

        :param signals: signals to be plot
        :param signals: list(str)

        :param is_last: if True this is the last plot
        :type is_last: bool
        """
        util_df = self.df_load()
        if util_df is None:
            self._log.warning('No trace events for task signals, plot DISABLED')
            return
        data = util_df[util_df.pid == tid][['cluster', 'cpu']]
        for ccolor, clabel in zip('gr', ['LITTLE', 'big']):
            cdata = data[data.cluster == clabel]
            if len(cdata) > 0:
                cdata.plot(ax=axes, style=[ccolor+'+'], legend=False)
        # Y Axis - placeholders for legend, acutal CPUs. topmost empty lane
        cpus = [str(n) for n in range(self._trace.plat_info['cpus-count'])]
        ylabels = [''] + cpus
        axes.set_yticklabels(ylabels)
        axes.set_ylim(-1, len(cpus))
        axes.set_ylabel('CPUs')
        # X Axis
        axes.set_xlim(self._trace.x_min, self._trace.x_max)

        axes.grid(True)
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self._trace.analysis.status.plot_overutilized(axes)

    def _plot_task_pelt(self, axes, tid, signals):
        """
        For task with ID `tid` plot PELT-related signals.

        :param axes: axes over which to generate the plot
        :type axes: :mod:`matplotlib.axes.Axes`

        :param tid: task ID
        :type tid: int

        :param signals: signals to be plot
        :param signals: list(str)
        """
        if not self._trace.hasEvents('sched_load_avg_task'):
            self._log.warning(
                'No sched_load_avg_task events, skipping PELT plot')
            return

        util_df = self._trace.df_events('sched_load_avg_task')
        data = util_df[util_df.pid == tid][['load_sum',
                                            'util_sum',
                                            'period_contrib']]
        data.plot(ax=axes, drawstyle='steps-post')
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.ticklabel_format(style='scientific', scilimits=(0, 0),
                              axis='y', useOffset=False)
        axes.grid(True)
        if 'sched_overutilized' in signals:
            self._trace.analysis.status.plot_overutilized(axes)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
