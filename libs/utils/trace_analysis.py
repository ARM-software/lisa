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
            self.plotsdir = self.trace.datadir

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

    def plotClusterFrequencies(self, title='Clusters Frequencies'):
        if not self.trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, '\
                    'plot DISABLED!')
            return
        df = self.trace.df('cpu_frequency')

        pd.options.mode.chained_assignment = None

        # Extract LITTLE and big clusters frequencies
        # and scale them to [MHz]
        if len(self.platform['clusters']['little']):
            lfreq = df[df.cpu == self.platform['clusters']['little'][-1]]
            lfreq['frequency'] = lfreq['frequency']/1e3
        else:
            lfreq = []
        if len(self.platform['clusters']['big']):
            bfreq = df[df.cpu == self.platform['clusters']['big'][-1]]
            bfreq['frequency'] = bfreq['frequency']/1e3
        else:
            bfreq = []

        # Compute AVG frequency for LITTLE cluster
        avg_lfreq = 0
        if len(lfreq) > 0:
            lfreq['timestamp'] = lfreq.index;
            lfreq['delta'] = (lfreq['timestamp'] - lfreq['timestamp'].shift()).fillna(0).shift(-1);
            lfreq['cfreq'] = (lfreq['frequency'] * lfreq['delta']).fillna(0);
            timespan = lfreq.iloc[-1].timestamp - lfreq.iloc[0].timestamp;
            avg_lfreq = lfreq['cfreq'].sum()/timespan;

        # Compute AVG frequency for big cluster
        avg_bfreq = 0
        if len(bfreq) > 0:
            bfreq['timestamp'] = bfreq.index;
            bfreq['delta'] = (bfreq['timestamp'] - bfreq['timestamp'].shift()).fillna(0).shift(-1);
            bfreq['cfreq'] = (bfreq['frequency'] * bfreq['delta']).fillna(0);
            timespan = bfreq.iloc[-1].timestamp - bfreq.iloc[0].timestamp;
            avg_bfreq = bfreq['cfreq'].sum()/timespan;

        pd.options.mode.chained_assignment = 'warn'

        # Setup a dual cluster plot
        fig, pltaxes = plt.subplots(2, 1, figsize=(16, 8));
        plt.suptitle(title, y=.97, fontsize=16,
                horizontalalignment='center');

        # Plot Cluster frequencies
        axes = pltaxes[0]
        axes.set_title('big Cluster');
        if avg_bfreq > 0:
            axes.axhline(avg_bfreq, color='r', linestyle='--', linewidth=2);
        axes.set_ylim(
                (self.platform['freqs']['big'][0] - 100000)/1e3,
                (self.platform['freqs']['big'][-1] + 100000)/1e3
        );
        if len(bfreq) > 0:
            bfreq['frequency'].plot(style=['r-'], ax=axes,
                    drawstyle='steps-post', alpha=0.4);
        else:
            logging.warn('NO big CPUs frequency events to plot')
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_ylabel('MHz')
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self.plotOverutilized(axes)

        axes = pltaxes[1]
        axes.set_title('LITTLE Cluster');
        if avg_lfreq > 0:
            axes.axhline(avg_lfreq, color='b', linestyle='--', linewidth=2);
        axes.set_ylim(
                (self.platform['freqs']['little'][0] - 100000)/1e3,
                (self.platform['freqs']['little'][-1] + 100000)/1e3
        );
        if len(lfreq) > 0:
            lfreq['frequency'].plot(style=['b-'], ax=axes,
                    drawstyle='steps-post', alpha=0.4);
        else:
            logging.warn('NO LITTLE CPUs frequency events to plot')
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_ylabel('MHz')
        axes.grid(True);
        self.plotOverutilized(axes)

        # Save generated plots into datadir
        figname = '{}/{}cluster_freqs.png'.format(self.plotsdir, self.prefix)
        pl.savefig(figname, bbox_inches='tight')

        logging.info('LITTLE cluster average frequency: %.3f GHz',
                avg_lfreq/1e3)
        logging.info('big    cluster average frequency: %.3f GHz',
                avg_bfreq/1e3)

        return (avg_lfreq/1e3, avg_bfreq/1e3)

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

    def plotOverutilized(self, axes=None):
        if not self.trace.hasEvents('sched_overutilized'):
            logging.warn('Events [sched_overutilized] not found, '\
                    'plot DISABLED!')
            return

        # Build sequence of overutilization "bands"
        df = self.trace.df('sched_overutilized')

        # Remove duplicated index events, keep only last event which is the
        # only one with a non null length
        df = df[df.len != 0]
        # This filtering can also be achieved by removing events happening at
        # the same time, but perhaps this filtering is more complex
        # df = df.reset_index()\
        #         .drop_duplicates(subset='Time', keep='last')\
        #         .set_index('Time')

        # Compute intervals in which the system is reported to be overutilized
        bands = [(t, df['len'][t], df['overutilized'][t]) for t in df.index]

        # If not axis provided: generate a standalone plot
        if not axes:
            gs = gridspec.GridSpec(1, 1)
            plt.figure(figsize=(16, 1))
            axes = plt.subplot(gs[0,0])
            axes.set_title('System Status {white: EAS mode, red: Non EAS mode}');
            axes.set_xlim(self.x_min, self.x_max);
            axes.grid(True);

        # Otherwise: draw overutilized bands on top of the specified plot
        for (t1,td,overutilized) in bands:
            if not overutilized:
                continue
            t2 = t1+td
            axes.axvspan(t1, t2, facecolor='r', alpha=0.1)

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

    def plotEDiffTime(self, tasks=None,
            min_usage_delta=None, max_usage_delta=None,
            min_cap_delta=None, max_cap_delta=None,
            min_nrg_delta=None, max_nrg_delta=None,
            min_nrg_diff=None, max_nrg_diff=None):
        if not self.trace.hasEvents('sched_energy_diff'):
            logging.warn('Events [sched_energy_diff] not found, plot DISABLED!')
            return
        df = self.trace.df('sched_energy_diff')

        # Filter on 'tasks'
        if tasks is not None:
            logging.info('Plotting EDiff data just for task(s) [%s]', tasks)
            df = df[df['comm'].isin(tasks)]

        # Filter on 'usage_delta'
        if min_usage_delta is not None:
            logging.info('Plotting EDiff data just with minimum usage_delta of [%d]', min_usage_delta)
            df = df[abs(df['usage_delta']) >= min_usage_delta]
        if max_usage_delta is not None:
            logging.info('Plotting EDiff data just with maximum usage_delta of [%d]', max_usage_delta)
            df = df[abs(df['usage_delta']) <= max_usage_delta]

        # Filter on 'cap_delta'
        if min_cap_delta is not None:
            logging.info('Plotting EDiff data just with minimum cap_delta of [%d]', min_cap_delta)
            df = df[abs(df['cap_delta']) >= min_cap_delta]
        if max_cap_delta is not None:
            logging.info('Plotting EDiff data just with maximum cap_delta of [%d]', max_cap_delta)
            df = df[abs(df['cap_delta']) <= max_cap_delta]

        # Filter on 'nrg_delta'
        if min_nrg_delta is not None:
            logging.info('Plotting EDiff data just with minimum nrg_delta of [%d]', min_nrg_delta)
            df = df[abs(df['nrg_delta']) >= min_nrg_delta]
        if max_nrg_delta is not None:
            logging.info('Plotting EDiff data just with maximum nrg_delta of [%d]', max_nrg_delta)
            df = df[abs(df['nrg_delta']) <= max_nrg_delta]

        # Filter on 'nrg_diff'
        if min_nrg_diff is not None:
            logging.info('Plotting EDiff data just with minimum nrg_diff of [%d]', min_nrg_diff)
            df = df[abs(df['nrg_diff']) >= min_nrg_diff]
        if max_nrg_diff is not None:
            logging.info('Plotting EDiff data just with maximum nrg_diff of [%d]', max_nrg_diff)
            df = df[abs(df['nrg_diff']) <= max_nrg_diff]

        # Grid: setup stats for gris
        gs = gridspec.GridSpec(4, 3, height_ratios=[2,4,2,4]);
        gs.update(wspace=0.1, hspace=0.1);

        # Configure plot
        fig = plt.figure(figsize=(16, 8*2+4*2+2));
        plt.suptitle("EnergyDiff Data",
                     y=.92, fontsize=16, horizontalalignment='center');

        # Plot1: src and dst CPUs
        axes = plt.subplot(gs[0,:]);
        axes.set_title('Source and Destination CPUs');
        df[['src_cpu', 'dst_cpu']].plot(ax=axes, style=['bo', 'r+']);
        axes.set_ylim(-1, self.platform['cpus_count']+1)
        axes.set_xlim(self.x_min, self.x_max);
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self.plotOverutilized(axes)

        # Plot2: energy and capacity variations
        axes = plt.subplot(gs[1,:]);
        axes.set_title('Energy vs Capacity Variations');

        for color, label in zip('gbyr', ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject', 'Suboptimal Reject']):
            subset = df[df.nrg_payoff_group == label]
            if (len(subset) == 0):
                continue
            subset[['nrg_diff_pct']].plot(ax=axes, style=[color+'o']);
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_yscale('symlog')
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self.plotOverutilized(axes)

        # Plot3: energy payoff
        axes = plt.subplot(gs[2,:]);
        axes.set_title('Energy Payoff Values');
        for color, label in zip('gbyr', ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject', 'Suboptimal Reject']):
            subset = df[df.nrg_payoff_group == label]
            if (len(subset) == 0):
                continue
            subset[['nrg_payoff']].plot(ax=axes, style=[color+'o']);
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_yscale('symlog')
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self.plotOverutilized(axes)

        # Plot4: energy deltas (kernel and host computed values)
        axes = plt.subplot(gs[3,:]);
        axes.set_title('Energy Deltas Values');
        df[['nrg_delta', 'nrg_diff_pct']].plot(ax=axes, style=['ro', 'b+']);
        axes.set_xlim(self.x_min, self.x_max);
        axes.grid(True);
        self.plotOverutilized(axes)

        # Save generated plots into datadir
        figname = '{}/{}ediff_time.png'.format(self.plotsdir, self.prefix)
        pl.savefig(figname, bbox_inches='tight')


        # Grid: setup stats for gris
        gs = gridspec.GridSpec(1, 3, height_ratios=[2]);
        gs.update(wspace=0.1, hspace=0.1);

        fig = plt.figure(figsize=(16, 4));

        # Plot: usage, capacity and energy distributuions
        axes = plt.subplot(gs[0,0]);
        df[['usage_delta']].hist(ax=axes, bins=60)
        axes = plt.subplot(gs[0,1]);
        df[['cap_delta']].hist(ax=axes, bins=60)
        axes = plt.subplot(gs[0,2]);
        df[['nrg_delta']].hist(ax=axes, bins=60)

        # Save generated plots into datadir
        figname = '{}/{}ediff_stats.png'.format(self.plotsdir, self.prefix)
        pl.savefig(figname, bbox_inches='tight')


    def plotEDiffSpace(self, tasks=None,
            min_usage_delta=None, max_usage_delta=None,
            min_cap_delta=None, max_cap_delta=None,
            min_nrg_delta=None, max_nrg_delta=None,
            min_nrg_diff=None, max_nrg_diff=None):
        if not self.trace.hasEvents('sched_energy_diff'):
            logging.warn('Events [sched_energy_diff] not found, plot DISABLED!')
            return
        df = self.trace.df('sched_energy_diff')

        # Filter on 'tasks'
        if tasks is not None:
            logging.info('Plotting EDiff data just for task(s) [%s]', tasks)
            df = df[df['comm'].isin(tasks)]

        # Filter on 'usage_delta'
        if min_usage_delta is not None:
            logging.info('Plotting EDiff data just with minimum usage_delta of [%d]', min_usage_delta)
            df = df[abs(df['usage_delta']) >= min_usage_delta]
        if max_usage_delta is not None:
            logging.info('Plotting EDiff data just with maximum usage_delta of [%d]', max_usage_delta)
            df = df[abs(df['usage_delta']) <= max_usage_delta]

        # Filter on 'cap_delta'
        if min_cap_delta is not None:
            logging.info('Plotting EDiff data just with minimum cap_delta of [%d]', min_cap_delta)
            df = df[abs(df['cap_delta']) >= min_cap_delta]
        if max_cap_delta is not None:
            logging.info('Plotting EDiff data just with maximum cap_delta of [%d]', max_cap_delta)
            df = df[abs(df['cap_delta']) <= max_cap_delta]

        # Filter on 'nrg_delta'
        if min_nrg_delta is not None:
            logging.info('Plotting EDiff data just with minimum nrg_delta of [%d]', min_nrg_delta)
            df = df[abs(df['nrg_delta']) >= min_nrg_delta]
        if max_nrg_delta is not None:
            logging.info('Plotting EDiff data just with maximum nrg_delta of [%d]', max_nrg_delta)
            df = df[abs(df['nrg_delta']) <= max_nrg_delta]

        # Filter on 'nrg_diff'
        if min_nrg_diff is not None:
            logging.info('Plotting EDiff data just with minimum nrg_diff of [%d]', min_nrg_diff)
            df = df[abs(df['nrg_diff']) >= min_nrg_diff]
        if max_nrg_diff is not None:
            logging.info('Plotting EDiff data just with maximum nrg_diff of [%d]', max_nrg_diff)
            df = df[abs(df['nrg_diff']) <= max_nrg_diff]

        # Grid: setup grid for P-E space
        gs = gridspec.GridSpec(1, 2, height_ratios=[2]);
        gs.update(wspace=0.1, hspace=0.1);

        fig = plt.figure(figsize=(16, 8));

        # Get min-max of each axes
        x_min = df.nrg_diff_pct.min()
        x_max = df.nrg_diff_pct.max()
        y_min = df.cap_delta.min()
        y_max = df.cap_delta.max()
        axes_min = min(x_min, y_min)
        axes_max = max(x_max, y_max)


        # # Tag columns by usage_delta
        # ccol = df.usage_delta
        # df['usage_delta_group'] = np.select(
        #     [ccol < 150, ccol < 400, ccol < 600],
        #     ['< 150', '< 400', '< 600'], '>= 600')
        #
        # # Tag columns by nrg_payoff
        # ccol = df.nrg_payoff
        # df['nrg_payoff_group'] = np.select(
        #     [ccol > 2e9, ccol > 0, ccol > -2e9],
        #     ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject'], 'Suboptimal Reject')

        # Plot: per usage_delta values
        axes = plt.subplot(gs[0,0]);

        for color, label in zip('bgyr', ['< 150', '< 400', '< 600', '>= 600']):
            subset = df[df.usage_delta_group == label]
            if (len(subset) == 0):
                continue
            plt.scatter(subset.nrg_diff_pct, subset.cap_delta,
                        s=subset.usage_delta,
                        c=color, label='task_usage ' + str(label),
                        axes=axes)

        # Plot space axes
        plt.plot((0, 0), (-1025, 1025), 'y--', axes=axes)
        plt.plot((-1025, 1025), (0,0), 'y--', axes=axes)

        # # Perf cuts
        # plt.plot((0, 100), (0,100*delta_pb), 'b--', label='PB (Perf Boost)')
        # plt.plot((0, -100), (0,-100*delta_pc), 'r--', label='PC (Perf Constraint)')
        #
        # # Perf boost setups
        # for y in range(0,6):
        #     plt.plot((0, 500), (0,y*100), 'g:')
        # for x in range(0,5):
        #     plt.plot((0, x*100), (0,500), 'g:')

        axes.legend(loc=4, borderpad=1);

        plt.xlim(1.1*axes_min, 1.1*axes_max);
        plt.ylim(1.1*axes_min, 1.1*axes_max);

        # axes.title('Performance-Energy Space')
        axes.set_xlabel('Energy diff [%]');
        axes.set_ylabel('Capacity diff [%]');


        # Plot: per usage_delta values
        axes = plt.subplot(gs[0,1]);

        for color, label in zip('gbyr', ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject', 'Suboptimal Reject']):
            subset = df[df.nrg_payoff_group == label]
            if (len(subset) == 0):
                continue
            plt.scatter(subset.nrg_diff_pct, subset.cap_delta,
                        s=60,
                        c=color,
                        marker='+',
                        label='{} Region'.format(label),
                        axes=axes)
                        # s=subset.usage_delta,

        # Plot space axes
        plt.plot((0, 0), (-1025, 1025), 'y--', axes=axes)
        plt.plot((-1025, 1025), (0,0), 'y--', axes=axes)

        # # Perf cuts
        # plt.plot((0, 100), (0,100*delta_pb), 'b--', label='PB (Perf Boost)')
        # plt.plot((0, -100), (0,-100*delta_pc), 'r--', label='PC (Perf Constraint)')
        #
        # # Perf boost setups
        # for y in range(0,6):
        #     plt.plot((0, 500), (0,y*100), 'g:')
        # for x in range(0,5):
        #     plt.plot((0, x*100), (0,500), 'g:')

        axes.legend(loc=4, borderpad=1);

        plt.xlim(1.1*axes_min, 1.1*axes_max);
        plt.ylim(1.1*axes_min, 1.1*axes_max);

        # axes.title('Performance-Energy Space')
        axes.set_xlabel('Energy diff [%]');
        axes.set_ylabel('Capacity diff [%]');

        plt.title('Performance-Energy Space')

        # Save generated plots into datadir
        figname = '{}/{}ediff_space.png'.format(self.plotsdir, self.prefix)
        pl.savefig(figname, bbox_inches='tight')


    def plotSchedTuneConf(self):
        """
        Plot the configuration of the SchedTune
        """
        if not self.trace.hasEvents('sched_tune_config'):
            logging.warn('Events [sched_tune_config] not found, plot DISABLED!')
            return
        # Grid
        gs = gridspec.GridSpec(2, 1, height_ratios=[4,1]);
        gs.update(wspace=0.1, hspace=0.1);

        # Figure
        plt.figure(figsize=(16, 2*6));
        plt.suptitle("SchedTune Configuration",
                     y=.97, fontsize=16, horizontalalignment='center');

        # Plot: Margin
        axes = plt.subplot(gs[0,0]);
        axes.set_title('Margin');
        data = self.trace.df('sched_tune_config')[['margin']]
        data.plot(ax=axes, drawstyle='steps-post', style=['b']);
        axes.set_ylim(0, 110);
        axes.set_xlim(self.x_min, self.x_max);
        axes.xaxis.set_visible(False);

        # Plot: Boost mode
        axes = plt.subplot(gs[1,0]);
        axes.set_title('Boost mode');
        data = self.trace.df('sched_tune_config')[['boostmode']]
        data.plot(ax=axes, drawstyle='steps-post');
        axes.set_ylim(0, 4);
        axes.set_xlim(self.x_min, self.x_max);
        axes.xaxis.set_visible(True);

        # Save generated plots into datadir
        figname = '{}/{}schedtune_conf.png'.format(self.plotsdir, self.prefix)
        pl.savefig(figname, bbox_inches='tight')

    @memoized
    def getCPUActiveSignal(self, cpu):
        """
        Build a square wave representing the active (i.e. non-idle) CPU time,
        i.e.:
            cpu_active[t] == 1 if at least one CPU is reported to be
                               non-idle by CPUFreq at time t
            cpu_active[t] == 0 otherwise

        :param cpu: CPU ID
        :type cpu: int
        """
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'cannot compute CPU active signal!')
            return None

        idle_df = self.trace.df('cpu_idle')
        cpu_df = idle_df[idle_df.cpu_id == cpu]

        cpu_active = cpu_df.state.apply(
            lambda s: 1 if s == NON_IDLE_STATE else 0
        )

        start_time = 0.0
        if not self.trace.ftrace.normalized_time:
            start_time = self.trace.ftrace.basetime
        if cpu_active.index[0] != start_time:
            entry_0 = pd.Series(cpu_active.iloc[0] ^ 1, index=[start_time])
            cpu_active = pd.concat([entry_0, cpu_active])

        return cpu_active

    @memoized
    def getClusterActiveSignal(self, cluster):
        """
        Build a square wave representing the active (i.e. non-idle) cluster
        time, i.e.:
            cluster_active[t] == 1 if at least one CPU is reported to be
                                   non-idle by CPUFreq at time t
            cluster_active[t] == 0 otherwise

        :param cluster: list of CPU IDs belonging to a cluster
        :type cluster: list(int)
        """
        cpu_active = {}
        for cpu in cluster:
            cpu_active[cpu] = self.getCPUActiveSignal(cpu)

        active = pd.DataFrame(cpu_active)
        active.fillna(method='ffill', inplace=True)

        # Cluster active is the OR between the actives on each CPU
        # belonging to that specific cluster
        cluster_active = reduce(
            operator.or_,
            [cpu_active.astype(int) for _, cpu_active in
             active.iteritems()]
        )

        return cluster_active

    def _integrate_square_wave(self, sq_wave):
        """
        Compute the integral of a square wave time series.

        :param sq_wave: square wave assuming only 1.0 and 0.0 values
        :type sq_wave: :mod:`pandas.Series`
        """
        sq_wave.iloc[-1] = 0.0
        # Compact signal to obtain only 1-0-1-0 sequences
        comp_sig = sq_wave.loc[sq_wave.shift() != sq_wave]
        # First value for computing the difference must be a 1
        if comp_sig.iloc[0] == 0.0:
            return sum(comp_sig.iloc[2::2].index - comp_sig.iloc[1:-1:2].index)
        else:
            return sum(comp_sig.iloc[1::2].index - comp_sig.iloc[:-1:2].index)

    @memoized
    def getClusterFrequencyResidency(self, cluster):
        """
        Get a DataFrame with per cluster frequency residency, i.e. amount of
        time spent at a given frequency in each cluster.

        :param cluster: this can be either a single CPU ID or a list of CPU IDs
            belonging to a cluster or the cluster name as specified in the
            platform description
        :type cluster: str or int or list(int)

        :returns: namedtuple(ResidencyTime) - tuple of total and active time
            dataframes

        :raises: KeyError
        """
        if not self.trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, '\
                         'frequency residency computation not possible!')
            return None
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'frequency residency computation not possible!')
            return None

        if isinstance(cluster, str):
            try:
                _cluster = self.platform['clusters'][cluster.lower()]
            except KeyError:
                logging.warn('%s cluster not found!', cluster)
                return None
        else:
            _cluster = listify(cluster)

        freq_df = self.trace.df('cpu_frequency')
        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU. This assumption is verified
        # by the Trace module when parsing the trace.
        if len(_cluster) > 1 and not self.trace.freq_coherency:
            logging.warn('Cluster frequency is NOT coherent,'\
                         'cannot compute residency!')
            return None
        cluster_freqs = freq_df[freq_df.cpu == _cluster[0]]

        ### Compute TOTAL Time ###
        time_intervals = cluster_freqs.index[1:] - cluster_freqs.index[:-1]
        total_time = pd.DataFrame({
            'time' : time_intervals,
            'frequency' : [f/1000 for f in cluster_freqs.iloc[:-1].frequency]
        })
        total_time = total_time.groupby(['frequency']).sum()

        ### Compute ACTIVE Time ###
        cluster_active = self.getClusterActiveSignal(_cluster)

        # In order to compute the active time spent at each frequency we
        # multiply 2 square waves:
        # - cluster_active, a square wave of the form:
        #     cluster_active[t] == 1 if at least one CPU is reported to be
        #                            non-idle by CPUFreq at time t
        #     cluster_active[t] == 0 otherwise
        # - freq_active, square wave of the form:
        #     freq_active[t] == 1 if at time t the frequency is f
        #     freq_active[t] == 0 otherwise
        available_freqs = sorted(cluster_freqs.frequency.unique())
        new_idx = sorted(cluster_freqs.index.tolist() + \
                         cluster_active.index.tolist())
        cluster_freqs = cluster_freqs.reindex(new_idx, method='ffill')
        cluster_active = cluster_active.reindex(new_idx, method='ffill')
        nonidle_time = []
        for f in available_freqs:
            freq_active = cluster_freqs.frequency.apply(
                lambda x: 1 if x == f else 0
            )
            active_t = cluster_active * freq_active
            # Compute total time by integrating the square wave
            nonidle_time.append(self._integrate_square_wave(active_t))

        active_time = pd.DataFrame({'time' : nonidle_time},
                                   index=[f/1000 for f in available_freqs])
        active_time.index.name = 'frequency'
        return ResidencyTime(total_time, active_time)

    def getCPUFrequencyResidency(self, cpu):
        """
        Get a DataFrame with per-CPU frequency residency, i.e. amount of
        time CPU `cpu` spent at each frequency. Both total and active times
        will be computed.

        :param cpu: CPU ID
        :type cpu: int

        :returns: namedtuple(ResidencyTime) - tuple of total and active time
            dataframes
        """
        return self.getClusterFrequencyResidency(cpu)

    def _plotFrequencyResidencyAbs(self, axes, residency, n_plots,
                                is_first, is_last, xmax, title=''):
        """
        Private method to generate frequency residency plots.

        :param axes: axes over which to generate the plot
        :type axes: matplotlib.axes.Axes

        :param residency: tuple of total and active time dataframes
        :type residency: namedtuple(ResidencyTime)

        :param n_plots: total number of plots
        :type n_plots: int

        :param is_first: if True this is the first plot
        :type is_first: bool

        :param is_first: if True this is the last plot
        :type is_first: bool

        :param xmax: x-axes higher bound
        :param xmax: double

        :param title: title of this subplot
        :type title: str
        """
        yrange = 0.4 * max(6, len(residency.total)) * n_plots
        residency.total.plot.barh(ax = axes, color='g',
                                  legend=False, figsize=(16,yrange))
        residency.active.plot.barh(ax = axes, color='r',
                                   legend=False, figsize=(16,yrange))

        axes.set_xlim(0, 1.05*xmax)
        axes.set_ylabel('Frequency [MHz]')
        axes.set_title(title)
        axes.grid(True)
        if is_last:
            axes.set_xlabel('Time [s]')
        else:
            axes.set_xticklabels([])

        if is_first:
            # Put title on top of the figure. As of now there is no clean way
            # to make the title appear always in the same position in the
            # figure because figure heights may vary between different
            # platforms (different number of OPPs). Hence, we use annotation
            legend_y = axes.get_ylim()[1]
            axes.annotate('OPP Residency Time', xy=(0, legend_y),
                          xytext=(-50, 45), textcoords='offset points',
                          fontsize=18)
            axes.annotate('GREEN: Total', xy=(0, legend_y),
                          xytext=(-50, 25), textcoords='offset points',
                          color='g', fontsize=14)
            axes.annotate('RED: Active', xy=(0, legend_y),
                          xytext=(50, 25), textcoords='offset points',
                          color='r', fontsize=14)

    def _plotFrequencyResidencyPct(self, axes, residency_df, label,
                                   n_plots, is_first, is_last, res_type):
        """
        Private method to generate PERCENTAGE frequency residency plots.

        :param axes: axes over which to generate the plot
        :type axes: matplotlib.axes.Axes

        :param residency_df: residency time dataframe
        :type residency_df: :mod:`pandas.DataFrame`

        :param label: label to be used for percentage residency dataframe
        :type label: str

        :param n_plots: total number of plots
        :type n_plots: int

        :param is_first: if True this is the first plot
        :type is_first: bool

        :param is_first: if True this is the last plot
        :type is_first: bool

        :param res_type: type of residency, either TOTAL or ACTIVE
        :type title: str
        """
        # Compute sum of the time intervals
        duration = residency_df.time.sum()
        residency_pct = pd.DataFrame(
            {label : residency_df.time.apply(lambda x: x*100/duration)},
            index=residency_df.index
        )
        yrange = 3 * n_plots
        residency_pct.T.plot.barh(ax=axes, stacked=True, figsize=(16, yrange))

        axes.legend(loc='lower center', ncol=7)
        axes.set_xlim(0, 100)
        axes.grid(True)
        if is_last:
            axes.set_xlabel('Residency [%]')
        else:
            axes.set_xticklabels([])
        if is_first:
            legend_y = axes.get_ylim()[1]
            axes.annotate('OPP {} Residency Time'.format(res_type),
                          xy=(0, legend_y), xytext=(-50, 35),
                          textcoords='offset points', fontsize=18)

    def _plotFrequencyResidency(self, residencies, entity_name, xmax,
                                pct, active):
        """
        Generate Frequency residency plots for the given entities.

        :param residencies: list of residencies to be plotted
        :type residencies: list(namedtuple(ResidencyData)) - each tuple
            contains:

            - a label to be used as subplot title
            - a namedtuple(ResidencyTime)

        :param entity_name: name of the entity ('cpu' or 'cluster') used in the
            figure name
        :type entity_name: str

        :param xmax: upper bound of x-axes
        :type xmax: double

        :param pct: plot residencies in percentage
        :type pct: bool

        :param active: for percentage plot specify whether to plot active or
            total time. Default is TOTAL time
        :type active: bool
        """
        n_plots = len(residencies)
        gs = gridspec.GridSpec(n_plots, 1)
        fig = plt.figure()

        figtype = ""
        for idx, data in enumerate(residencies):
            label = data[0]
            r = data[1]
            if r is None:
                plt.close(fig)
                return

            axes = fig.add_subplot(gs[idx])
            is_first = idx == 0
            is_last = idx+1 == n_plots
            if pct and active:
                self._plotFrequencyResidencyPct(axes, data.residency.active,
                                                data.label, n_plots,
                                                is_first, is_last,
                                                'ACTIVE')
                figtype = "_pct_active"
                continue
            if pct:
                self._plotFrequencyResidencyPct(axes, data.residency.total,
                                                data.label, n_plots,
                                                is_first, is_last,
                                                'TOTAL')
                figtype = "_pct_total"
                continue

            self._plotFrequencyResidencyAbs(axes, data.residency,
                                            n_plots, is_first,
                                            is_last, xmax,
                                            title=data.label)

        figname = '{}/{}{}_freq_residency{}.png'\
                  .format(self.plotsdir, self.prefix, entity_name, figtype)

        pl.savefig(figname, bbox_inches='tight')

    def plotCPUFrequencyResidency(self, cpus=None, pct=False, active=False):
        """
        Plot per-CPU frequency residency. big CPUs are plotted first and then
        LITTLEs.

        Requires the following trace events:
            - cpu_frequency
            - cpu_idle

        :param cpus: list of CPU IDs. By default plot all CPUs
        :type cpus: list(int) or int

        :param pct: plot residencies in percentage
        :type pct: bool

        :param active: for percentage plot specify whether to plot active or
            total time. Default is TOTAL time
        :type active: bool
        """
        if not self.trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, plot DISABLED!')
            return

        if cpus is None:
            # Generate plots only for available CPUs
            cpufreq_data = self.trace.df('cpu_frequency')
            _cpus = range(cpufreq_data.cpu.max()+1)
        else:
            _cpus = listify(cpus)

        # Split between big and LITTLE CPUs ordered from higher to lower ID
        _cpus.reverse()
        big_cpus = [c for c in _cpus if c in self.platform['clusters']['big']]
        little_cpus = [c for c in _cpus if c in
                       self.platform['clusters']['little']]
        _cpus = big_cpus + little_cpus

        # Precompute active and total time for each CPU
        residencies = []
        xmax = 0.0
        for c in _cpus:
            r = self.getCPUFrequencyResidency(c)
            residencies.append(ResidencyData('CPU{}'.format(c), r))

            max_time = r.total.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plotFrequencyResidency(residencies, 'cpu', xmax, pct, active)

    def plotClusterFrequencyResidency(self, clusters=None,
                                      pct=False, active=False):
        """
        Plot the frequency residency in a given cluster, i.e. the amount of
        time cluster `cluster` spent at frequency `f_i`. By default, both 'big'
        and 'LITTLE' clusters data are plotted.

        Requires the following trace events:
            - cpu_frequency
            - cpu_idle

        :param clusters: name of the clusters to be plotted (all of them by
            default)
        :type clusters: str ot list(str)

        :param pct: plot residencies in percentage
        :type pct: bool

        :param active: for percentage plot specify whether to plot active or
            total time. Default is TOTAL time
        :type active: bool
        """
        if not self.trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, plot DISABLED!')
            return

        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU
        if not self.trace.freq_coherency:
            logging.warn('Cluster frequency is not coherent, plot DISABLED!')
            return

        # Sanitize clusters
        if clusters is None:
            _clusters = self.platform['clusters'].keys()
        else:
            _clusters = listify(clusters)

        # Precompute active and total time for each cluster
        residencies = []
        xmax = 0.0
        for c in _clusters:
            r = self.getClusterFrequencyResidency(
                    self.platform['clusters'][c.lower()])
            residencies.append(ResidencyData('{} Cluster'.format(c), r))

            max_time = r.total.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plotFrequencyResidency(residencies, 'cluster', xmax, pct, active)

    def _getIdleStateResidency(self, entity):
        """
        Compute time spent by a given entity in each idle state.

        :param entity: cpu ID or cluster name or list of CPU IDs
        :type entity: int or str or list(int)

        :returns: :mod:`pandas.DataFrame` - idle time dataframe
        """
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'idle state residency computation not possible!')
            return None

        if isinstance(entity, str):
            try:
                _entity = self.platform['clusters'][entity.lower()]
            except KeyError:
                logging.warn('%s entity not found!', entity)
                return None
        else:
            _entity = listify(entity)

        idle_df = self.trace.df('cpu_idle')
        entity_idle_df = idle_df[idle_df.cpu_id == _entity[0]]

        # Build cpu_idle, a square wave of the form:
        #     entity_idle[t] == 1 if all CPUs in the entity are reported to be
        #                       idle by cpufreq at time t
        #     entity_idle[t] == 0 otherwise
        entity_idle = self.getClusterActiveSignal(_entity) ^ 1

        # In order to compute the time spent in each idle statefrequency we
        # multiply 2 square waves:
        # - entity_idle
        # - idle_state, square wave of the form:
        #     idle_state[t] == 1 if at time t entity is in idle state i
        #     idle_state[t] == 0 otherwise
        available_idles = sorted(idle_df.state.unique())
        # Remove non-idle state from availables
        available_idles.pop()
        new_idx = sorted(entity_idle_df.index.tolist() + \
                         entity_idle.index.tolist())
        entity_idle_df = entity_idle_df.reindex(new_idx, method='ffill')
        entity_idle = entity_idle.reindex(new_idx, method='ffill')
        idle_time = []
        for i in available_idles:
            idle_state = entity_idle_df.state.apply(
                lambda x: 1 if x == i else 0
            )
            idle_t = entity_idle * idle_state
            # Compute total time by integrating the square wave
            idle_time.append(self._integrate_square_wave(idle_t))

        idle_time_df = pd.DataFrame({'time' : idle_time}, index=available_idles)
        idle_time_df.index.name = 'idle_state'
        return idle_time_df

    @memoized
    def getCPUIdleStateResidency(self, cpu):
        """
        Compute time spent by a given CPU in each idle state.

        :param cpu: CPU ID
        :type cpu: int

        :returns: :mod:`pandas.DataFrame` - idle time dataframe
        """
        return self._getIdleStateResidency(cpu)

    @memoized
    def getClusterIdleStateResidency(self, cluster):
        """
        Compute time spent by a given cluster in each idle state.

        :param cluster: cluster name or list of CPUs ID
        :type cluster: str or list(int)

        :returns: :mod:`pandas.DataFrame` - idle time dataframe
        """
        return self._getIdleStateResidency(cluster)

    def _plotIdleStateResidency(self, residencies, entity_name, xmax):
        """
        Generate Idle state residency plots for the given entities.

        :param residencies: list of residencies to be plot
        :type residencies: list(namedtuple(ResidencyData)) - each tuple
            contains:

            - a label to be used as subplot title
            - a dataframe with residency for each idle state

        :param entity_name: name of the entity ('cpu' or 'cluster') used in the
            figure name
        :type entity_name: str

        :param xmax: upper bound of x-axes
        :type xmax: double
        """
        n_plots = len(residencies)
        gs = gridspec.GridSpec(n_plots, 1)
        fig = plt.figure()

        for idx, data in enumerate(residencies):
            r = data.residency
            if r is None:
                plt.close(fig)
                return

            axes = fig.add_subplot(gs[idx])
            is_first = idx == 0
            is_last = idx+1 == n_plots
            yrange = 0.4 * max(6, len(r)) * n_plots
            r.plot.barh(ax = axes, color='g',
                        legend=False, figsize=(16,yrange))

            axes.set_xlim(0, 1.05*xmax)
            axes.set_ylabel('Idle State')
            axes.set_title(data.label)
            axes.grid(True)
            if is_last:
                axes.set_xlabel('Time [s]')
            else:
                axes.set_xticklabels([])

            if is_first:
                legend_y = axes.get_ylim()[1]
                axes.annotate('Idle State Residency Time', xy=(0, legend_y),
                              xytext=(-50, 45), textcoords='offset points',
                              fontsize=18)

        figname = '{}/{}{}_idle_state_residency.png'\
                  .format(self.plotsdir, self.prefix, entity_name)

        pl.savefig(figname, bbox_inches='tight')


    def plotCPUIdleStateResidency(self, cpus=None):
        """
        Plot per-CPU idle state residency. big CPUs are plotted first and then
        LITTLEs.

        Requires cpu_idle trace events.

        :param cpus: list of CPU IDs. By default plot all CPUs
        :type cpus: list(int) or int
        """
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'plot DISABLED!')
            return

        if cpus is None:
            # Generate plots only for available CPUs
            cpuidle_data = self.trace.df('cpu_idle')
            _cpus = range(cpuidle_data.cpu_id.max()+1)
        else:
            _cpus = listify(cpus)

        # Split between big and LITTLE CPUs ordered from higher to lower ID
        _cpus.reverse()
        big_cpus = [c for c in _cpus if c in self.platform['clusters']['big']]
        little_cpus = [c for c in _cpus if c in
                       self.platform['clusters']['little']]
        _cpus = big_cpus + little_cpus

        residencies = []
        xmax = 0.0
        for cpu in _cpus:
            r = self.getCPUIdleStateResidency(cpu)
            residencies.append(ResidencyData('CPU{}'.format(cpu), r))

            max_time = r.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plotIdleStateResidency(residencies, 'cpu', xmax)

    def plotClusterIdleStateResidency(self, clusters=None):
        """

        Plot per-cluster idle state residency in a given cluster, i.e. the
        amount of time cluster `cluster` spent in idle state `i`. By default,
        both 'big' and 'LITTLE' clusters data are plotted.

        Requires cpu_idle following trace events.
        :param clusters: name of the clusters to be plotted (all of them by
            default)
        :type clusters: str ot list(str)
        """
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, plot DISABLED!')
            return

        # Sanitize clusters
        if clusters is None:
            _clusters = self.platform['clusters'].keys()
        else:
            _clusters = listify(clusters)

        # Precompute residencies for each cluster
        residencies = []
        xmax = 0.0
        for c in _clusters:
            r = self.getClusterIdleStateResidency(
                    self.platform['clusters'][c.lower()])
            residencies.append(ResidencyData('{} Cluster'.format(c), r))

            max_time = r.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plotIdleStateResidency(residencies, 'cluster', xmax)

