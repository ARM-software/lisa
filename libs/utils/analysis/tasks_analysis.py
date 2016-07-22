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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pylab as pl
import re

from analysis_module import AnalysisModule

# Configure logging
import logging

class TasksAnalysis(AnalysisModule):

    def __init__(self, trace):
        """
        Support for Tasks signals analysis
        """
        super(TasksAnalysis, self).__init__(trace)


################################################################################
# DataFrame Getter Methods
################################################################################


################################################################################
# Plotting Methods
################################################################################

    def plotTasks(self, tasks=None, signals=None):
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

        Note:
            sched_overutilized: enable the plotting of overutilization bands on
                                top of each subplot
            residencies: enable the generation of the CPUs residencies plot

        :param tasks: the list of task names and/or PIDs to plot.
                      Numerical PIDs and string task names can be mixed
                      in the same list.
                      default: all tasks defined in Trace
                      creation time are plotted
        :type tasks: list

        :param signals: list of signals (and thus plots) to generate
                        default: all the plots and signals available in the
                        current trace
        :type signals: list
        """
        if not signals:
            signals = ['load_avg', 'util_avg', 'boosted_util',
                       'sched_overutilized',
                       'load_sum', 'util_sum', 'period_contrib',
                       'residencies']

        # Check for the minimum required signals to be available
        if not self._trace.hasEvents('sched_load_avg_task'):
            logging.warn('Events [sched_load_avg_task] not found, '\
                    'plot DISABLED!')
            return

        # Defined list of tasks to plot
        if tasks:
            tasks_to_plot = tasks
        elif self._tasks:
            tasks_to_plot = sorted(self._tasks)
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
        for signals_to_plot in plots_signals:
            signals_to_plot = signals_to_plot.intersection(signals)
            if len(signals_to_plot):
                plots_count = plots_count + 1

        # Grid
        gs = gridspec.GridSpec(plots_count, 1, height_ratios=[2,1,1]);
        gs.update(wspace=0.1, hspace=0.1);

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
            task_name = self._trace.getTaskByPid(tid)
            if len(task_name) == 1:
                task_name = task_name[0]
                logging.info('Plotting %5d: %s...', tid, task_name)
            else:
                logging.info('Plotting %5d: %s...', tid, ', '.join(task_name))
            plot_id = 0

            # Figure
            plt.figure(figsize=(16, 2*6+3));
            plt.suptitle("Task Signals",
                         y=.94, fontsize=16, horizontalalignment='center');

            # Plot load and utilization
            signals_to_plot = {'load_avg', 'util_avg',
                               'boosted_util', 'sched_overutilized'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 0:
                axes = plt.subplot(gs[plot_id,0]);
                axes.set_title('Task [{0:d}:{1:s}] Signals'\
                               .format(tid, task_name));
                plot_id = plot_id + 1
                is_last = (plot_id == plots_count)
                self._plotTaskSignals(axes, tid, signals_to_plot, is_last)

            # Plot CPUs residency
            signals_to_plot = {'residencies', 'sched_overutilized'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 0:
                axes = plt.subplot(gs[plot_id,0]);
                axes.set_title('Task [{0:d}:{1:s}] Residency (green: LITTLE, red: big)'\
                               .format(tid, task_name));
                plot_id = plot_id + 1
                is_last = (plot_id == plots_count)
                self._plotTaskResidencies(axes, tid, signals_to_plot, is_last)

            # Plot PELT signals
            signals_to_plot = {
                'load_sum', 'util_sum',
                'period_contrib', 'sched_overutilized'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 0:
                axes = plt.subplot(gs[plot_id,0]);
                axes.set_title('Task [{0:d}:{1:s}] PELT Signals'\
                               .format(tid, task_name));
                plot_id = plot_id + 1
                self._plotTaskPelt(axes, tid, signals_to_plot)

            # Save generated plots into datadir
            if isinstance(task_name, list):
                task_name = re.sub('[:/]', '_', task_name[0])
            else:
                task_name = re.sub('[:/]', '_', task_name)
            figname = '{}/{}task_util_{}_{}.png'.format(
                self._trace.plots_dir, self._trace.plots_prefix, tid, task_name)
            pl.savefig(figname, bbox_inches='tight')


################################################################################
# Utility Methods
################################################################################

    def _plotTaskSignals(self, axes, tid, signals, is_last=False):
        # Get dataframe for the required task
        util_df = self._dfg_trace_event('sched_load_avg_task')

        # Plot load and util
        signals_to_plot = list({'load_avg', 'util_avg'}.intersection(signals))
        if len(signals_to_plot):
            data = util_df[util_df.pid == tid][signals_to_plot]
            data.plot(ax=axes, drawstyle='steps-post');

        # Plot boost utilization if available
        if 'boosted_util' in signals and \
            self._trace.hasEvents('sched_boost_task'):
            boost_df = self._dfg_trace_event('sched_boost_task')
            data = boost_df[boost_df.pid == tid][['boosted_util']]
            if len(data):
                data.plot(ax=axes, style=['y-'], drawstyle='steps-post');
            else:
                task_name = self._trace.getTaskByPid(tid)
                logging.warning("No 'boosted_util' data for task [%d:%s]",
                                tid, task_name)

        # Add Capacities data if avilable
        if 'nrg_model' in self._platform:
            nrg_model = self._platform['nrg_model']
            max_lcap = nrg_model['little']['cpu']['cap_max']
            max_bcap = nrg_model['big']['cpu']['cap_max']
            tip_lcap = 0.8 * max_lcap
            tip_bcap = 0.8 * max_bcap
            logging.debug('LITTLE capacity tip/max: %d/%d, big capacity tip/max: %d/%d',
                          tip_lcap, max_lcap, tip_bcap, max_bcap)
            axes.axhline(tip_lcap, color='g', linestyle='--', linewidth=1);
            axes.axhline(max_lcap, color='g', linestyle='-', linewidth=2);
            axes.axhline(tip_bcap, color='r', linestyle='--', linewidth=1);
            axes.axhline(max_bcap, color='r', linestyle='-', linewidth=2);

        axes.set_ylim(0, 1100);
        axes.set_xlim(self._trace.x_min, self._trace.x_max);
        axes.grid(True);
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self._trace.analysis.status.plotOverutilized(axes)

    def _plotTaskResidencies(self, axes, tid, signals, is_last=False):
        util_df = self._dfg_trace_event('sched_load_avg_task')
        data = util_df[util_df.pid == tid][['cluster', 'cpu']]
        for ccolor, clabel in zip('gr', ['LITTLE', 'big']):
            cdata = data[data.cluster == clabel]
            if (len(cdata) > 0):
                cdata.plot(ax=axes, style=[ccolor+'+'], legend=False);
        # Y Axis - placeholders for legend, acutal CPUs. topmost empty lane
        cpus = [str(n) for n in range(self._platform['cpus_count'])]
        ylabels = [''] + cpus
        axes.set_yticklabels(ylabels)
        axes.set_ylim(-1, self._platform['cpus_count'])
        axes.set_ylabel('CPUs')
        # X Axis
        axes.set_xlim(self._trace.x_min, self._trace.x_max);

        axes.grid(True);
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self._trace.analysis.status.plotOverutilized(axes)

    def _plotTaskPelt(self, axes, tid, signals):
        util_df = self._dfg_trace_event('sched_load_avg_task')
        data = util_df[util_df.pid == tid][['load_sum', 'util_sum', 'period_contrib']]
        data.plot(ax=axes, drawstyle='steps-post');
        axes.set_xlim(self._trace.x_min, self._trace.x_max);
        axes.ticklabel_format(style='scientific', scilimits=(0,0),
                              axis='y', useOffset=False)
        axes.grid(True);
        if 'sched_overutilized' in signals:
            self._trace.analysis.status.plotOverutilized(axes)

