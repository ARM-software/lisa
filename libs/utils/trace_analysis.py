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

import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pylab as pl
import re
import sys
import trappy
import operator
from trappy.utils import listify
from devlib.utils.misc import memoized
from collections import namedtuple

# Configure logging
import logging

NON_IDLE_STATE = 4294967295

ResidencyTime = namedtuple('ResidencyTime', ['total', 'active'])
ResidencyData = namedtuple('ResidencyData', ['label', 'residency'])

class TraceAnalysis(object):

    def __init__(self, trace, tasks=None, plotsdir=None, prefix=''):
        """
        Support for plotting a standard set of trace singals and events
        """

        self.trace = trace
        self.tasks = tasks
        self.plotsdir = plotsdir
        self.prefix = prefix

        # Keep track of the Trace::platform
        self.platform = trace.platform

        # Plotsdir is byb default the trace dir
        if self.plotsdir is None:
            self.plotsdir = self.trace.data_dir

        # Minimum and Maximum x_time to use for all plots
        self.x_min = 0
        self.x_max = self.trace.time_range

        # Reset x axis time range to full scale
        t_min = self.trace.window[0]
        t_max = self.trace.window[1]
        self.setXTimeRange(t_min, t_max)

    def setXTimeRange(self, t_min=None, t_max=None):
        if t_min is None:
            self.x_min = 0
        else:
            self.x_min = t_min
        if t_max is None:
            self.x_max = self.trace.time_range
        else:
            self.x_max = t_max
        logging.info('Set plots time range to (%.6f, %.6f)[s]',
                self.x_min, self.x_max)

    def plotFunctionStats(self, functions=None, metrics='avg'):
        """
        Plot functions profiling metrics for the specified kernel functions.

        For each speficied metric a barplot is generated which report the value
        of the metric when the kernel function has been executed on each CPU.
        By default all the kernel functions are plotted.

        :param functions: the name of list of name of kernel functions to plot
        :type functions: str or list

        :param metrics: the metrics to plot
                        avg   - average execution time
                        time  - total execution time
        :type metrics: srt or list
        """
        if not hasattr(self.trace, '_functions_stats_df'):
            logging.warning('Functions stats data not available')
            return

        metrics = listify(metrics)
        df = self.trace.functions_stats_df(functions)

        # Check that all the required metrics are acutally availabe
        available_metrics = df.columns.tolist()
        if not set(metrics).issubset(set(available_metrics)):
            msg = 'Metrics {} not supported, available metrics are {}'\
                    .format(set(metrics) - set(available_metrics),
                            available_metrics)
            raise ValueError(msg)

        for _m in metrics:
            if _m.upper() == 'AVG':
                title = 'Average Completion Time per CPUs'
                ylabel = 'Completion Time [us]'
            if _m.upper() == 'TIME':
                title = 'Total Execution Time per CPUs'
                ylabel = 'Execution Time [us]'
            data = df[_m.lower()].unstack()
            axes = data.plot(kind='bar',
                             figsize=(16,8), legend=True,
                             title=title, table=True)
            axes.set_ylabel(ylabel)
            axes.get_xaxis().set_visible(False)


    def __addCapacityColum(self):
        df = self.trace.df('cpu_capacity')
        # Rename CPU and Capacity columns
        df.rename(columns={'cpu_id':'cpu'}, inplace=True)
        # Add column with LITTLE and big CPUs max capacities
        nrg_model = self.platform['nrg_model']
        max_lcap = nrg_model['little']['cpu']['cap_max']
        max_bcap = nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [tip_lcap], tip_bcap)

    def __plotCPU(self, cpus=None, label=''):
        if cpus is None or len(cpus) == 0:
            return
        if label != '':
            label1 = '{} '.format(label)
            label2 = '_{}s'.format(label.lower())

        # Plot required CPUs
        fig, pltaxes = plt.subplots(len(cpus), 1, figsize=(16, 3*(len(cpus))));
        plt.suptitle("{}CPUs Signals".format(label1),
                     y=.99, fontsize=16, horizontalalignment='center');

        idx = 0
        for cpu in cpus:

            # Reference axes to be used
            axes = pltaxes
            if (len(cpus) > 1):
                axes = pltaxes[idx]

            # Add CPU utilization
            axes.set_title('{0:s}CPU [{1:d}]'.format(label1, cpu));
            df = self.trace.df('sched_load_avg_cpu')
            df = df[df.cpu == cpu]
            if len(df):
                df[['util_avg']].plot(ax=axes, drawstyle='steps-post', alpha=0.4);

            # if self.trace.hasEvents('sched_boost_cpu'):
            #     df = self.trace.df('sched_boost_cpu')
            #     df = df[df.cpu == cpu]
            #     if len(df):
            #         df[['usage', 'boosted_usage']].plot(
            #             ax=axes,
            #             style=['m-', 'r-'],
            #             drawstyle='steps-post');

            # Add Capacities data if avilable
            if self.trace.hasEvents('cpu_capacity'):
                df = self.trace.df('cpu_capacity')
                df = df[df.cpu == cpu]
                if len(df):
                    # data = df[['capacity', 'tip_capacity', 'max_capacity']]
                    # data.plot(ax=axes, style=['m', 'y', 'r'],
                    data = df[['capacity', 'tip_capacity' ]]
                    data.plot(ax=axes, style=['m', '--y' ],
                              drawstyle='steps-post')

            axes.set_ylim(0, 1100);
            axes.set_xlim(self.x_min, self.x_max);

            # Disable x-axis timestamp for top-most cpus
            if (len(cpus) > 1 and idx < len(cpus)-1):
                axes.set_xticklabels([])
                axes.set_xlabel('')
            axes.grid(True);

            idx+=1

        # Save generated plots into datadir
        figname = '{}/{}cpus{}.png'.format(self.plotsdir, self.prefix, label2)
        pl.savefig(figname, bbox_inches='tight')

    def plotCPU(self, cpus=None):
        if not self.trace.hasEvents('sched_load_avg_cpu'):
            logging.warn('Events [sched_load_avg_cpu] not found, '\
                    'plot DISABLED!')
            return

        # Filter on specified cpus
        if cpus is None:
            cpus = sorted(self.platform['clusters']['little'] + self.platform['clusters']['big'])

        # Plot: big CPUs
        bcpus = set(cpus) & set(self.platform['clusters']['big'])
        self.__plotCPU(bcpus, "big")

        # Plot: LITTLE CPUs
        lcpus = set(cpus) & set(self.platform['clusters']['little'])
        self.__plotCPU(lcpus, "LITTLE")

    def _plotTaskSignals(self, axes, tid, signals, is_last=False):
        # Get dataframe for the required task
        util_df = self.trace.df('sched_load_avg_task')

        # Plot load and util
        signals_to_plot = list({'load_avg', 'util_avg'}.intersection(signals))
        if len(signals_to_plot):
            data = util_df[util_df.pid == tid][signals_to_plot]
            data.plot(ax=axes, drawstyle='steps-post');

        # Plot boost utilization if available
        if 'boosted_util' in signals and \
            self.trace.hasEvents('sched_boost_task'):
            boost_df = self.trace.df('sched_boost_task')
            data = boost_df[boost_df.pid == tid][['boosted_util']]
            if len(data):
                data.plot(ax=axes, style=['y-'], drawstyle='steps-post');
            else:
                task_name = self.trace.getTaskByPid(tid)
                logging.warning("No 'boosted_util' data for task [%d:%s]",
                                tid, task_name)

        # Add Capacities data if avilable
        if 'nrg_model' in self.trace.platform:
            nrg_model = self.trace.platform['nrg_model']
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
        axes.set_xlim(self.x_min, self.x_max);
        axes.grid(True);
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self.plotOverutilized(axes)

    def _plotTaskResidencies(self, axes, tid, signals, is_last=False):
        util_df = self.trace.df('sched_load_avg_task')
        data = util_df[util_df.pid == tid][['cluster', 'cpu']]
        for ccolor, clabel in zip('gr', ['LITTLE', 'big']):
            cdata = data[data.cluster == clabel]
            if (len(cdata) > 0):
                cdata.plot(ax=axes, style=[ccolor+'+'], legend=False);
        # Y Axis - placeholders for legend, acutal CPUs. topmost empty lane
        cpus = [str(n) for n in range(self.platform['cpus_count'])]
        ylabels = [''] + cpus
        axes.set_yticklabels(ylabels)
        axes.set_ylim(-1, self.platform['cpus_count'])
        axes.set_ylabel('CPUs')
        # X Axis
        axes.set_xlim(self.x_min, self.x_max);

        axes.grid(True);
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self.plotOverutilized(axes)

    def _plotTaskPelt(self, axes, tid, signals):
        util_df = self.trace.df('sched_load_avg_task')
        data = util_df[util_df.pid == tid][['load_sum', 'util_sum', 'period_contrib']]
        data.plot(ax=axes, drawstyle='steps-post');
        axes.set_xlim(self.x_min, self.x_max);
        axes.ticklabel_format(style='scientific', scilimits=(0,0),
                              axis='y', useOffset=False)
        axes.grid(True);
        if 'sched_overutilized' in signals:
            self.plotOverutilized(axes)

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
                      default: all tasks defined at TraceAnalysis
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
        if not self.trace.hasEvents('sched_load_avg_task'):
            logging.warn('Events [sched_load_avg_task] not found, '\
                    'plot DISABLED!')
            return

        # Defined list of tasks to plot
        if tasks:
            tasks_to_plot = tasks
        elif self.tasks:
            tasks_to_plot = sorted(self.tasks)
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
            pids_to_plot.extend(self.trace.getTaskByName(task))

        for tid in pids_to_plot:
            task_name = self.trace.getTaskByPid(tid)
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
                self.plotsdir, self.prefix, tid, task_name)
            pl.savefig(figname, bbox_inches='tight')

