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

# Configure logging
import logging

NON_IDLE_STATE = 4294967295

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
        self.setXTimeRange()

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

    def _plotTaskSignals(self, df, axes, task_name, tid, signals, is_last=False):
        axes.set_title('Task [{0:d}:{1:s}] Signals'.format(tid, task_name));

        # Plot load and util
        signals_to_plot = list({'load_avg', 'util_avg'}.intersection(signals))
        if len(signals_to_plot):
            data = df[df.comm == task_name][signals_to_plot]
            data.plot(ax=axes, drawstyle='steps-post');

        # Plot boost utilization if available
        if 'boosted_util' in signals and \
            self.trace.hasEvents('sched_boost_task'):
            df2 = self.trace.df('sched_boost_task')
            data = df2[df2.comm == task_name][['boosted_util']]
            if len(data):
                data.plot(ax=axes, style=['y-'], drawstyle='steps-post');

        # Add Capacities data if avilable
        if 'nrg_model' in self.trace.platform:
            nrg_model = self.trace.platform['nrg_model']
            max_lcap = nrg_model['little']['cpu']['cap_max']
            max_bcap = nrg_model['big']['cpu']['cap_max']
            tip_lcap = 0.8 * max_lcap
            tip_bcap = 0.8 * max_bcap
            logging.info('%d %d %d %d', tip_lcap, max_lcap, tip_bcap, max_bcap)
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

    def _plotTaskResidencies(self, df, axes, task_name, signals, is_last=False):
        axes.set_title('CPUs residency (green: LITTLE, red: big)');
        axes.set_xlim(0, self.trace.time_range);
        data = df[df.comm == task_name][['cluster', 'cpu']]
        for ccolor, clabel in zip('gr', ['LITTLE', 'big']):
            cdata = data[data.cluster == clabel]
            if (len(cdata) > 0):
                cdata.plot(ax=axes, style=[ccolor+'+'], legend=False);
        axes.set_ylim(-1, self.platform['cpus_count']+1)
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_ylabel('CPUs')
        axes.grid(True);
        if not is_last:
            axes.set_xticklabels([])
            axes.set_xlabel('')
        if 'sched_overutilized' in signals:
            self.plotOverutilized(axes)

    def _plotTaskPelt(self, df, axes, task_name, signals):
        axes.set_title('PELT Signals');
        data = df[df.comm == task_name][['load_sum', 'util_sum', 'period_contrib']]
        data.plot(ax=axes, drawstyle='steps-post');
        axes.set_xlim(self.x_min, self.x_max);
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

        Args:
            tasks   (list): the list of tasks to plot
                            default: all tasks defined at TraceAnalysis
                            creation time are plotted
            signals (list): list of signals (and thus plots) to generate
                            default: all the plots and signals available in the
                            current trace
        """
        if not signals:
            signals = ['load_avg', 'util_avg', 'boosted_util',
                       'sched_overutilized',
                       'load_sum', 'util_sum', 'period_contrib',
                       'residencies']
        if not self.trace.hasEvents('sched_load_avg_task'):
            logging.warn('Events [sched_load_avg_task] not found, '\
                    'plot DISABLED!')
            return
        df = self.trace.df('sched_load_avg_task')
        self.trace.getTasks(df, tasks)
        tasks_to_plot = sorted(self.tasks)
        if tasks:
            tasks_to_plot = tasks

        # Compute number of plots to produce
        plots_count = 0
        plots_signals = [
                {'load_avg', 'util_avg', 'boosted_util'},
                {'residencies'},
                {'load_sum', 'util_sum', 'period_contrib'}
        ]
        for signals_to_plot in plots_signals:
            signals_to_plot = signals_to_plot.intersection(signals)
            if len(signals_to_plot):
                plots_count = plots_count + 1

        # Grid
        gs = gridspec.GridSpec(plots_count, 1, height_ratios=[2,1,1]);
        gs.update(wspace=0.1, hspace=0.1);

        for task_name in tasks_to_plot:
            logging.debug('Plotting [%s]', task_name)
            tid = self.trace.getTaskByName(task_name)[0]
            plot_id = 0

            # Figure
            plt.figure(figsize=(16, 2*6+3));
            plt.suptitle("Task Signals",
                         y=.94, fontsize=16, horizontalalignment='center');

            # Plot load and utilization
            signals_to_plot = {'load_avg', 'util_avg',
                               'boosted_util', 'sched_overutilized'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 1:
                axes = plt.subplot(gs[plot_id,0]);
                plot_id = plot_id + 1
                is_last = (plot_id == plots_count)
                self._plotTaskSignals(df, axes, task_name, tid,
                                      signals_to_plot, is_last)

            # Plot CPUs residency
            signals_to_plot = {'residencies', 'sched_overutilized'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 1:
                axes = plt.subplot(gs[plot_id,0]);
                plot_id = plot_id + 1
                is_last = (plot_id == plots_count)
                self._plotTaskResidencies(df, axes, task_name,
                                          signals_to_plot, is_last)

            # Plot PELT signals
            signals_to_plot = {
                'load_sum', 'util_sum',
                'period_contrib', 'sched_overutilized'}
            signals_to_plot = list(signals_to_plot.intersection(signals))
            if len(signals_to_plot) > 1:
                axes = plt.subplot(gs[plot_id,0]);
                self._plotTaskPelt(df, axes, task_name, signals_to_plot)
                plot_id = plot_id + 1

            # Save generated plots into datadir
            figname = '{}/{}task_util_{}.png'.format(self.plotsdir, self.prefix, task_name)
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

    def _computeNonIdleTime(self, t_prev, t, tasks_time_intervals):
        """
        Compute the non-idle time spent by tasks within the time interval
        [t_prev. t].

        :param t_prev: Time interval init boundary
        :type t_prev: double

        :param t: Time interval end boundary
        :type t: double

        :param tasks_time_intervals: DataFrame containing start and end times
            of tasks
        :type tasks_time_intervals: :mod:`pandas.DataFrame`
        """
        nonidle_time = 0.0
        for _, row in tasks_time_intervals.iterrows():
            start = row['start']
            end = row['end']
            if start > t:
                # Task not running at f_prev
                continue
            elif start >= t_prev and start <= t:
                # Task starting at instant of time where frequency is f_prev
                if end <= t:
                    # Task runs at f_prev for the whole interval
                    nonidle_time += end - start
                else:
                    # Task continues running at different frequency
                    nonidle_time += t - start
            else:
                # Task started with previous frequency
                if end > t_prev:
                    # Task is still running now, at f_prev
                    if end <= t:
                        nonidle_time += end - t_prev
                    else:
                        # Task will continue running at different frequency
                        nonidle_time += t - t_prev

        return nonidle_time

    def getCPUFrequencyResidency(self, cpu):
        """
        Get a DataFrame with per-CPU frequency residency, i.e. amount of
        time CPU `cpu` spent at each frequency. Both total and active times
        will be computed.

        :param cpu: CPU ID
        :type cpu: int
        """

        if not self.trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, '\
                         'plot DISABLED!')
            return None
        if not self.trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'plot DISABLED!')
            return None

        freq_dfr = self.trace.df('cpu_frequency')
        cpu_freqs = freq_dfr[freq_dfr.cpu == cpu]
        idle_dfr = self.trace.df('cpu_idle')
        cpu_idle = idle_dfr[idle_dfr.cpu_id == cpu]

        ### Compute TOTAL time spent at each frequency ###
        time_intervals = cpu_freqs.index[1:] - cpu_freqs.index[:-1]
        total_time = pd.DataFrame({
            'TOTAL Time' : time_intervals,
            'Frequency [MHz]' : [f/1000 for f in cpu_freqs.iloc[:-1].frequency]
        })
        total_time = total_time.groupby(['Frequency [MHz]']).sum()

        ### Compute ACTIVE time spent at each frequency ###
        # time instants where CPU is idle
        idle_times = cpu_idle[cpu_idle['state'] != NON_IDLE_STATE]
        # time instants where CPU is not idle
        non_idle_times = cpu_idle[cpu_idle['state'] == NON_IDLE_STATE]

        if idle_times.index[0] < non_idle_times.index[0]:
            idle_times = idle_times.iloc[1:]

        time_intervals = [t2 - t1 for t1, t2 in zip(non_idle_times.index,
                                                    idle_times.index)]

        times_len = min(len(idle_times), len(non_idle_times))
        tasks_time_intervals = pd.DataFrame({
            "start": non_idle_times.index[:times_len],
            "end": idle_times.index[:times_len]
        })

        # Compute the non-IDLE time spent at a certain frequency
        time_intervals = []
        frequencies = []
        row_iterator = cpu_freqs.iterrows()
        # Take first item from row_iterator
        t_prev, f_prev = row_iterator.next()
        for t, f in row_iterator:
            nonidle_time = self._computeNonIdleTime(t_prev,
                                                    t,
                                                    tasks_time_intervals)
            freq = f_prev['frequency']
            time_intervals.append(nonidle_time)
            frequencies.append(freq)
            t_prev = t
            f_prev = f

        active_time = pd.DataFrame({
            'ACTIVE Time' : time_intervals,
            'Frequency [MHz]' : [f/1000 for f in frequencies]
        })
        # Sum the time intervals with same frequncy
        active_time = active_time.groupby(['Frequency [MHz]']).sum()
        return total_time, active_time

    def plotCPUFrequencyResidency(self, cpus=None):
        """
        Plot per-CPU frequency residency.

        :param cpus: List of cpus. By default plot all CPUs
        :type cpus: list(str)
        """

        if not cpus:
            cpus = sorted(self.platform['clusters']['little'] + self.platform['clusters']['big'])
        else:
            cpus.sort()

        n_cpus = len(cpus)
        if n_cpus == 1:
            gs = gridspec.GridSpec(1, 1)
        else:
            gs = gridspec.GridSpec(n_cpus/2 + n_cpus%2, 2)

        fig = plt.figure()

        axis = None
        for idx, cpu in enumerate(cpus):
            total_time, active_time = self.getCPUFrequencyResidency(cpu)

            axis = fig.add_subplot(gs[idx])
            total_time.plot.barh(ax=axis,
                                 color='g',
                                 legend=False,
                                 figsize=(16,8),
                                 fontsize=16);
            active_time.plot.barh(ax=axis,
                                  color='r',
                                  legend=False,
                                  figsize=(16,8),
                                  fontsize=16);
            axis.set_title('CPU{}'.format(cpu))
            axis.set_xlabel('Time [s]')
            axis.set_xlim(0, self.x_max)

        if axis:
            handles, labels = axis.get_legend_handles_labels()
            plt.figlegend(handles,
                          labels,
                          loc='lower right')
            plt.subplots_adjust(hspace=0.5)
