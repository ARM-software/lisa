
# import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# import pylab as pl
# import re
# import sys
# import trappy

# Configure logging
import logging

class Filters(object):

    def __init__(self, trace, tasks=None):
        self.trace = trace
        self.tasks = tasks

        self.big_tasks = {}

        self.big_frequent_tasks_pids = None
        self.wkp_frequent_tasks_pids = None

        self.big_cap = self.trace.platform['nrg_model']['big']['cpu']['cap_max']
        self.little_cap = self.trace.platform['nrg_model']['little']['cpu']['cap_max']

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


    def topBigTasks(self, max_tasks=10, min_samples=100, min_utilization=None):
        """
        Tasks which had a 'utilization' bigger than the specified threshold
        """

        if min_utilization is None:
            min_utilization = self.little_cap

        df = self.trace.df('tload')
        big_tasks_events = df[df.utilization > min_utilization]
        big_tasks = big_tasks_events.pid.unique()

        big_tasks_count = big_tasks.size
        print 'Total {} tasks with at least {} "utilization" samples > {}'\
                .format(big_tasks_count, min_samples, min_utilization)

        big_tasks_stats = big_tasks_events.groupby('pid')\
                            .describe(include=['object']);
        big_tasks_pids = big_tasks_stats.unstack()['comm']\
                            .sort(columns=['count'], ascending=False)
        big_tasks_pids = big_tasks_pids[big_tasks_pids['count'] > min_samples]

        big_topmost = big_tasks_pids.head(max_tasks)
        print 'Top {} "big" tasks:'.format(max_tasks)
        print big_topmost

        self.big_frequent_tasks_pids = list(big_topmost.index)

        # Keep track of big tasks tload events
        self.big_tasks['tload'] = big_tasks_events
        return self.big_frequent_tasks_pids

    def _taskIsBig(self, utilization):
        if utilization > self.little_cap:
            return self.big_cap
        return self.little_cap

    def plotBigTasks(self, max_tasks=10, min_samples=100, min_utilization=None):

        # Get the list of big and frequent tasks
        if self.big_frequent_tasks_pids is None:
            pids = self.topBigTasks(max_tasks, min_samples, min_utilization)

        big_frequent_tasks_count = len(self.big_frequent_tasks_pids)
        if big_frequent_tasks_count == 0:
            print "No big/frequent tasks to plot"
            return

        # Get the list of events for all big frequent tasks
        df = self.trace.df('tload')
        big_frequent_tasks_events = df[df.pid.isin(self.big_frequent_tasks_pids)]

        # Add a column to represent big status
        big_frequent_tasks_events['isbig'] = big_frequent_tasks_events['utilization'].map(self._taskIsBig)


        # Define axes for side-by-side plottings
        fig, axes = plt.subplots(big_frequent_tasks_count, 1,
                                 figsize=(14, big_frequent_tasks_count*5));
        plt.subplots_adjust(wspace=0.1, hspace=0.2);

        plot_idx = 0
        for i, group in big_frequent_tasks_events.groupby('pid'):

            # Build task names (there could be multiple, during the task lifetime)
            big_frequent_task_i = big_frequent_tasks_events[big_frequent_tasks_events['pid'] == i]
            task_names = big_frequent_task_i.comm.unique()
            task_name = 'PID: ' + str(i)
            for s in task_names:
                task_name += ' | ' + s

            # Plot title
            if (big_frequent_tasks_count == 1):
                ax_ratio = axes
            else:
                 ax_ratio = axes[plot_idx]
            ax_ratio.set_title(task_name);

            # Left axis: utilization
            ax_ratio = group.plot(y=['utilization', 'isbig'],
                            style=['r.', '-b'],
                            drawstyle='steps-post',
                            linewidth=1,
                            ax=ax_ratio)
            ax_ratio.set_xlim(self.x_min, self.x_max);
            ax_ratio.set_ylim(0, 1100)
            ax_ratio.set_ylabel('utilization')

            plot_idx+=1

        print 'Tasks which have been a "utilization" of {0:d} for at least {1:d} samples'\
            .format(self.little_cap, min_samples)

    def topWakeupTasks(self, max_tasks=10, min_wakeups=100):
        """
        Tasks which wakeups more frequent than a specified threshold
        """

        df = self.trace.df('swkp')

        wkp_tasks_stats = df.groupby('pid').describe(include=['object'])
        wkp_tasks_pids = wkp_tasks_stats.unstack()['comm']\
                        .sort(columns=['count'], ascending=False)
        wkp_tasks_pids = wkp_tasks_pids[wkp_tasks_pids['count'] > min_wakeups]

        wkp_topmost = wkp_tasks_pids.head(max_tasks)
        print 'Top {} "big" tasks:'.format(max_tasks)
        print wkp_topmost

        self.wkp_frequent_tasks_pids = list(wkp_topmost.index)
        return self.wkp_frequent_tasks_pids

    def plotWakeupTasks(self, max_tasks=10, min_wakeups=100, per_cluster=False):

        # Get the list of big and frequent tasks
        if self.wkp_frequent_tasks_pids is None:
            pids = self.topWakeupTasks(max_tasks, min_wakeups)

        wkp_frequent_tasks_count = len(self.wkp_frequent_tasks_pids)
        if wkp_frequent_tasks_count == 0:
            print "No big/frequent wakeups tasks to plot"
            return

        # Define axes for side-by-side plottings
        fig, axes = plt.subplots(2, 1, figsize=(14, 5));
        plt.subplots_adjust(wspace=0.2, hspace=0.3);

        if per_cluster:

            # Get list of big/LITTLE CPUs
            big_cpus = self.trace.platform['clusters']['big']
            little_cpus = self.trace.platform['clusters']['little']

            # Get per cluster wakeup events
            df = self.trace.df('swkpn')
            ntbc = df[df.target_cpu.astype(int).isin(big_cpus)];
            ntlc = df[df.target_cpu.astype(int).isin(little_cpus)];

            ax = axes[0]
            ax.set_title('Tasks Forks on big CPUs');
            ax.set_xlim(self.x_min, self.x_max);
            ntbc.pid.astype(int).plot(style=['g.'], ax=ax);

            ax = axes[1]
            ax.set_title('Tasks Forks on LITTLE CPUs');
            ax.set_xlim(self.x_min, self.x_max);
            ntlc.pid.astype(int).plot(style=['g.'], ax=ax);

        else:

            ax = axes[0]
            ax.set_title('Tasks WakeUps Events');
            df = self.trace.df('swkp')
            df.pid.astype(int).plot(style=['b.'], ax=ax);
            ax.set_xlim(self.x_min, self.x_max);
            ax.xaxis.set_visible(False);

            ax = axes[1]
            ax.set_title('Tasks Forks Events');
            df = self.trace.df('swkpn')
            df.pid.astype(int).plot(style=['r.'], ax=ax);
            ax.set_xlim(self.x_min, self.x_max);
            ax.xaxis.set_visible(False);
