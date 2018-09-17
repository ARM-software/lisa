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

""" Latency Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import re
import os

from collections import namedtuple
from lisa.analysis.base import AnalysisBase
from devlib.utils.misc import memoized
from trappy.utils import listify

# Tuple representing all IDs data of a Task
TaskData = namedtuple('TaskData', ['pid', 'names', 'label'])

CDF = namedtuple('CDF', ['df', 'threshold', 'above', 'below'])

class LatencyAnalysis(AnalysisBase):
    """
    Support for plotting Latency Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'latency'

    def __init__(self, trace):
        super(LatencyAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @memoized
    def df_latency(self, task):
        """
        DataFrame of task's wakeup/suspend events

        The returned DataFrame index is the time, in seconds, an event related
        to `task` happened.
        The DataFrame has these columns:
        - target_cpu: the CPU where the task has been scheduled
                      reported only for wakeup events
        - curr_state: the current task state:
            A letter which corresponds to the standard events reported by the
            prev_state field of a sched_switch event.
            Only exception is 'A', which is used to represent active tasks,
            i.e. tasks RUNNING on a CPU
        - next_state: the next status for the task
        - t_start: the time when the current status started, it matches Time
        - t_delta: the interval of time after witch the task will switch to the
                   next_state

        :param task: the task to report wakeup latencies for
        :type task: int or str
        """

        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Events [sched_wakeup] not found, '
                              'cannot compute CPU active signal!')
            return None
        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Events [sched_switch] not found, '
                              'cannot compute CPU active signal!')
            return None

        # Get task data
        td = self._get_task_data(task)
        if not td:
            return None

        wk_df = self._trace.df_events('sched_wakeup')
        sw_df = self._trace.df_events('sched_switch')

        # Filter Task's WAKEUP events
        task_wakeup = wk_df[wk_df.pid == td.pid][['target_cpu', 'pid']]

        # Filter Task's START events
        task_events = (sw_df.prev_pid == td.pid) | (sw_df.next_pid == td.pid)
        task_switches_df = sw_df[task_events]\
            [['__cpu', 'prev_pid', 'next_pid', 'prev_state']]

        # Unset prev_state for switch_in events, i.e.
        # we don't care about the status of a task we are replacing
        task_switches_df.prev_state = task_switches_df.apply(
            lambda r : np.nan if r['prev_pid'] != td.pid
                              else self._task_state(r['prev_state']),
            axis=1)

        # Rename prev_state
        task_switches_df.rename(columns={'prev_state' : 'curr_state'}, inplace=True)

        # Fill in Running status
        # We've just set curr_state (a.k.a prev_state) to nan where td.pid was
        # switching in, so set the state to 'A' ("active") in those places.
        task_switches_df.curr_state = task_switches_df.curr_state.fillna(value='A')

        # Join Wakeup and SchedSwitch events
        task_latency_df = task_wakeup.join(task_switches_df, how='outer',
                                             lsuffix='_wkp', rsuffix='_slp')
        # Remove not required columns
        task_latency_df = task_latency_df[['target_cpu', '__cpu', 'curr_state']]
        # Set Wakeup state on each Wakeup event
        task_latency_df.curr_state = task_latency_df.curr_state.fillna(value='W')

        # Sanity check for all task states to be mapped to a char
        numbers = 0
        for value in task_switches_df.curr_state.unique():
            if type(value) is not str:
                self._log.warning('The [sched_switch] events contain "prev_state" value [%s]',
                                  value)
                numbers += 1
        if numbers:
            verb = 'is' if numbers == 1 else 'are'
            self._log.warning('  which %s not currently mapped into a task state.',
                              verb)
            self._log.warning('Check mappings in:')
            self._log.warning(' %s::%s _task_state()',
                              __file__, self.__class__.__name__)

        # Forward annotate task state
        task_latency_df['next_state'] = task_latency_df.curr_state.shift(-1)

        # Forward account for previous state duration
        task_latency_df['t_start'] =  task_latency_df.index
        task_latency_df['t_delta'] = (
              task_latency_df['t_start'].shift(-1)
            - task_latency_df['t_start']
        )

        # Fix the last entry, which will have a NaN state duration
        # Set duration to trace_end - last_event
        task_latency_df.loc[task_latency_df.index[-1], 't_delta'] = (
            self._trace.start_time +
            self._trace.time_range -
            task_latency_df.index[-1]
        )

        return task_latency_df


    # Select Wakeup latency
    def df_latency_wakeup(self, task):
        """
        DataFrame of task's wakeup latencies

        The returned DataFrame index is the time, in seconds, `task` waken-up.
        The DataFrame has just one column:
        - wakeup_latency: the time the task waited before getting a CPU

        :param task: the task to report wakeup latencies for
        :type task: int or str
        """

        task_latency_df = self.df_latency(task)
        if task_latency_df is None:
            return None
        df = task_latency_df[
                    (task_latency_df.curr_state == 'W') &
                    (task_latency_df.next_state == 'A')][['t_delta']]
        df.rename(columns={'t_delta' : 'wakeup_latency'}, inplace=True)
        return df

    # Select Wakeup latency
    def df_latency_preemption(self, task):
        """
        DataFrame of task's preemption latencies

        The returned DataFrame index is the time, in seconds, `task` has been
        preempted.
        The DataFrame has just one column:
        - preemption_latency: the time the task waited before getting again a CPU

        :param task: the task to report wakeup latencies for
        :type task: int or str
        """
        task_latency_df = self.df_latency(task)
        if task_latency_df is None:
            return None
        df = task_latency_df[
                    (task_latency_df.curr_state.isin([0, 'R', 'R+'])) &
                    (task_latency_df.next_state == 'A')][['t_delta']]
        df.rename(columns={'t_delta' : 'preempt_latency'}, inplace=True)
        return df

    @memoized
    def df_activations(self, task):
        """
        DataFrame of task's wakeup intrvals

        The returned DataFrame index is the time, in seconds, `task` has
        waken-up.
        The DataFrame has just one column:
        - activation_interval: the time since the previous wakeup events

        :param task: the task to report runtimes for
        :type task: int or str
        """
        # Select all wakeup events
        wkp_df = self.df_latency(task)
        wkp_df = wkp_df[wkp_df.curr_state == 'W'].copy()
        # Compute delta between successive wakeup events
        wkp_df['activation_interval'] = (
                wkp_df['t_start'].shift(-1) - wkp_df['t_start'])
        wkp_df['activation_interval'] = wkp_df['activation_interval'].shift(1)
        # Return the activation period each time the task wakeups
        wkp_df = wkp_df[['activation_interval']].shift(-1)
        return wkp_df

    @memoized
    def df_runtimes(self, task):
        """
        DataFrame of task's runtime each time the task blocks

        The returned DataFrame index is the time, in seconds, `task` completed
        an activation (i.e. sleep or exit)
        The DataFrame has just one column:
        - running_time: the time the task spent RUNNING since its last wakeup

        :param task: the task to report runtimes for
        :type task: int or str
        """
        # Select all wakeup events
        run_df = self.df_latency(task)

        # Filter function to add up RUNNING intervals of each activation
        def cr(row):
            if row['curr_state'] in ['S']:
                return cr.runtime
            if row['curr_state'] in ['W']:
                if cr.spurious_wkp:
                        cr.runtime += row['t_delta']
                        cr.spurious_wkp = False
                        return cr.runtime
                cr.runtime = 0
                return cr.runtime
            if row['curr_state'] != 'A':
                return cr.runtime
            if row['next_state'] in ['R', 'R+', 'S', 'x', 'D']:
                cr.runtime += row['t_delta']
                return cr.runtime
            # This is required to capture strange trace sequences where
            # a switch_in event is follower by a wakeup_event.
            # This sequence is not expected, but we found it in some traces.
            # Possible reasons could be:
            # - misplaced sched_wakeup events
            # - trace buffer artifacts
            # TO BE BETTER investigated in kernel space.
            # For the time being, we account this interval as RUNNING time,
            # which is what kernelshark does.
            if row['next_state'] in ['W']:
                cr.runtime += row['t_delta']
                cr.spurious_wkp = True
                return cr.runtime
            if row['next_state'] in ['n']:
                return cr.runtime
            self._log.warning("Unexpected next state: %s @ %f",
                              row['next_state'], row['t_start'])
            return 0
        # cr's static variables intialization
        cr.runtime = 0
        cr.spurious_wkp = False

        # Add up RUNNING intervals of each activation
        run_df['running_time'] = run_df.apply(cr, axis=1)
        # Return RUNTIME computed for each activation,
        # each time the task blocks or terminate
        run_df = run_df[run_df.next_state.isin(['S', 'x'])][['running_time']]
        return run_df

    @memoized
    def df_task_residency(self, task):
        """
        DataFrame of a task's execution time on each CPU

        The returned DataFrame index is the CPU indexes
        The DataFrame has just one column:
        - runtime: the time the task spent being active on a given CPU,
          in seconds.

        :param task: the task to report runtimes for
        :type task: int or str
        """
        cpus = list(range(self._trace.platform['cpus_count']))
        runtimes = {cpu : 0.0 for cpu in cpus}

        df = self.df_latency(task)

        # Exclude sleep time
        df = df[df.curr_state != 'S']

        for time, data in df.iterrows():
            cpu = data['__cpu']

            # When waking up, '__cpu' is NaN but 'target_cpu' is populated instead
            if np.isnan(cpu):
                if data['curr_state'] == 'W':
                    cpu = data['target_cpu']
                else:
                    raise RuntimeError('No CPU data for latency_df @{}'.format(time))

            runtimes[cpu] += data['t_delta']

        data = [(cpu, time) for  cpu, time in runtimes.items()]
        return pd.DataFrame(data, columns=['CPU', 'runtime']).set_index('CPU')

    @memoized
    def _get_latency_df(self, task, kind='all', threshold_ms=1):
        """
        Compute statistics on latencies of the specified task.

        :param task: the task to report latencies for
        :type task: int or list(str)

        :param kind: the kind of latencies to report (WAKEUP and/or PREEMPT")
        :type kind: str

        :param threshold_ms: the minimum acceptable [ms] value to report
                             graphically in the generated plots
        :type threshold_ms: int or float

        :returns: a DataFrame with statistics on task latencies
        """

        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Event [sched_switch] not found, '
                              'plot DISABLED!')
            return
        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Event [sched_wakeup] not found, '
                              'plot DISABLED!')
            return

        # Get task data
        td = self._get_task_data(task)
        if not td:
            return None

        # Load wakeup latencies (if required)
        wkp_df = None
        if 'all' in kind or 'wakeup' in kind:
            wkp_df = self.df_latency_wakeup(td.pid)
        if wkp_df is not None:
            wkp_df.rename(columns={'wakeup_latency' : 'latency'}, inplace=True)
            self._log.info('Found: %5d WAKEUP latencies', len(wkp_df))

        # Load preempt latencies (if required)
        prt_df = None
        if 'all' in kind or 'preempt' in kind:
            prt_df = self.df_latency_preemption(td.pid)
        if prt_df is not None:
            prt_df.rename(columns={'preempt_latency' : 'latency'}, inplace=True)
            self._log.info('Found: %5d PREEMPT latencies', len(prt_df))

        if wkp_df is None and prt_df is None:
            self._log.warning('No Latency info for task [%s]', td.label)
            return

        # Join the two data frames
        df = wkp_df.append(prt_df)
        cdf = self._get_cdf(df.latency, (threshold_ms / 1000.))

        return df, cdf

    @memoized
    def df_latency_stats(self, task, kind='all', threshold_ms=1):
        """
        Compute statistics on latencies of the specified task.

        :param task: the task to report latencies for
        :type task: int or list(str)

        :param kind: the kind of latencies to report (WAKEUP and/or PREEMPT")
        :type kind: str

        :param threshold_ms: the minimum acceptable [ms] value to report
                             graphically in the generated plots
        :type threshold_ms: int or float

        :returns: a DataFrame with statistics on task latencies
        """

        # Get latency events
        df, cdf = self._get_latency_df(task, kind, threshold_ms)

        # Return statistics
        stats_df = df.describe(percentiles=[0.95, 0.99])
        label = '{:.1f}%'.format(100. * cdf.below)
        stats = { label : cdf.threshold }
        return stats_df.append(pd.DataFrame(
            list(stats.values()), columns=['latency'], index=list(stats.keys())))


###############################################################################
# Plotting Methods
###############################################################################

    def plot_latency(self, task, kind='all', tag=None, threshold_ms=1, bins=64):
        """
        Generate a set of plots to report the WAKEUP and PREEMPT latencies the
        specified task has been subject to. A WAKEUP latencies is the time from
        when a task becomes RUNNABLE till the first time it gets a CPU.
        A PREEMPT latencies is the time from when a RUNNING task is suspended
        because of the CPU is assigned to another task till when the task
        enters the CPU again.

        :param task: the task to report latencies for
        :type task: int or list(str)

        :param kind: the kind of latencies to report (WAKEUP and/or PREEMPT")
        :type kind: str

        :param tag: a string to add to the plot title
        :type tag: str

        :param threshold_ms: the minimum acceptable [ms] value to report
                             graphically in the generated plots
        :type threshold_ms: int or float

        :param bins: number of bins to be used for the runtime's histogram
        :type bins: int

        :returns: a DataFrame with statistics on ploted latencies
        """

        # Get latency events
        df, cdf = self._get_latency_df(task, kind, threshold_ms)
        self._log.info('Total: %5d latency events', len(df))
        self._log.info('%.1f %% samples below %d [ms] threshold',
                       100. * cdf.below, threshold_ms)

        # Get task data
        td = self._get_task_data(task)
        if not td:
            return None

        # Setup plots
        gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,1])
        plt.figure(figsize=(16, 8))

        plot_title = "[{}]: {} latencies".format(td.label, kind.upper())
        if tag:
            plot_title = "{} [{}]".format(plot_title, tag)
        plot_title = "{}, threshold @ {} [ms]".format(plot_title, threshold_ms)

        # Latency events duration over time
        axes = plt.subplot(gs[0,0:2])
        axes.set_title(plot_title)
        try:
            wkp_df.rename(columns={'latency': 'wakeup'}, inplace=True)
            wkp_df.plot(style='b+', logy=True, ax=axes)
        except: pass
        try:
            prt_df.rename(columns={'latency' : 'preempt'}, inplace=True)
            prt_df.plot(style='r+', logy=True, ax=axes)
        except: pass
        axes.axhline(threshold_ms / 1000., linestyle='--', color='g')
        self._trace.analysis.status.plot_overutilized(axes)
        axes.legend(loc='lower center', ncol=2)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)

        # Cumulative distribution of latencies samples
        axes = plt.subplot(gs[1,0])
        cdf.df.plot(ax=axes, legend=False, xlim=(0,None),
                    title='Latencies CDF ({:.1f}% within {} [ms] threshold)'\
                          .format(100. * cdf.below, threshold_ms))
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);
        axes.axhline(y=cdf.below, linewidth=1, color='r', linestyle='--')

        # Histogram of all latencies
        axes = plt.subplot(gs[1,1])
        ymax = 1.1 * df.latency.max()
        df.latency.plot(kind='hist', bins=bins, ax=axes,
                        xlim=(0,ymax), legend=False,
                        title='Latency histogram ({} bins, {} [ms] green threshold)'\
                        .format(bins, threshold_ms));
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);

        # Save generated plots into datadir
        task_name = re.sub('[\ :/]', '_', td.label)
        figname = '{}/{}task_latencies_{}_{}.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix,
                          td.pid, task_name)
        pl.savefig(figname, bbox_inches='tight')


    def plot_latency_bands(self, task, axes=None):
        """
        Draw a plot that shows intervals of time when the execution of a
        RUNNABLE task has been delayed. The plot reports:
          WAKEUP     lantecies as RED colored bands
          PREEMPTION lantecies as BLUE colored bands

        The optional axes parameter allows to plot the signal on an existing
        graph.

        :param task: the task to report latencies for
        :type task: str

        :param axes: axes on which to plot the signal
        :type axes: :mod:`matplotlib.axes.Axes`
        """
        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Event [sched_switch] not found, '
                              'plot DISABLED!')
            return
        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Event [sched_wakeup] not found, '
                              'plot DISABLED!')
            return

        # Get task PID
        td = self._get_task_data(task)
        if not td:
            return None

        wkl_df = self.df_latency_wakeup(td.pid)
        prt_df = self.df_latency_preemption(td.pid)

        if wkl_df is None and prt_df is None:
            self._log.warning('No task with name [%s]', td.label)
            return

        # If not axis provided: generate a standalone plot
        if not axes:
            gs = gridspec.GridSpec(1, 1)
            plt.figure(figsize=(16, 2))
            axes = plt.subplot(gs[0, 0])
            axes.set_title('Latencies on [{}] '
                           '(red: WAKEUP, blue: PREEMPT)'\
                          .format(td.label))
            axes.set_xlim(self._trace.x_min, self._trace.x_max)
            axes.set_yticklabels([])
            axes.set_xlabel('Time [s]')
            axes.grid(True)

        # Draw WAKEUP latencies
        try:
            bands = [(t, wkl_df['wakeup_latency'][t]) for t in wkl_df.index]
            for (start, duration) in bands:
                end = start + duration
                axes.axvspan(start, end, facecolor='r', alpha=0.1)
                axes.set_xlim(self._trace.x_min, self._trace.x_max)
        except: pass

        # Draw PREEMPTION latencies
        try:
            bands = [(t, prt_df['preempt_latency'][t]) for t in prt_df.index]
            for (start, duration) in bands:
                end = start + duration
                axes.axvspan(start, end, facecolor='b', alpha=0.1)
                axes.set_xlim(self._trace.x_min, self._trace.x_max)
        except: pass

    def plot_activations(self, task, tag=None, threshold_ms=16, bins=64):
        """
        Plots "activation intervals" for the specified task

        An "activation interval" is time incurring between two consecutive
        wakeups of a task. A set of plots is generated to report:
        - Activations interval at wakeup time: every time a task wakeups a
          point is plotted to represent the time interval since the previous
          wakeup.
        - Activations interval cumulative function: reports the cumulative
          function of the activation intervals.
        - Activations intervals histogram: reports a 64 bins histogram of
          the activation intervals.

        All plots are parameterized based on the value of threshold_ms, which
        can be used to filter activations intervals bigger than 2 times this
        value.
        Such a threshold is useful to filter out from the plots outliers thus
        focusing the analysis in the most critical periodicity under analysis.
        The number and percentage of discarded samples is reported in output.
        A default threshold of 16 [ms] is used, which is useful for example
        to analyze a 60Hz rendering pipelines.

        A PNG of the generated plots is generated and saved in the same folder
        where the trace is.

        :param task: the task to report latencies for
        :type task: int or list(str)

        :param tag: a string to add to the plot title
        :type tag: str

        :param threshold_ms: the minimum acceptable [ms] value to report
                             graphically in the generated plots
        :type threshold_ms: int or float

        :param bins: number of bins to be used for the runtime's histogram
        :type bins: int

        :returns: a DataFrame with statistics on ploted activation intervals
        """

        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Event [sched_switch] not found, '
                              'plot DISABLED!')
            return
        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Event [sched_wakeup] not found, '
                              'plot DISABLED!')
            return

        # Get task data
        td = self._get_task_data(task)
        if not td:
            return None

        # Load activation data
        wkp_df = self.df_activations(td.pid)
        if wkp_df is None:
            return None
        self._log.info('Found: %5d activations for [%s]',
                       len(wkp_df), td.label)

        # Disregard data above two time the specified threshold
        y_max = (2 * threshold_ms) / 1000.
        len_tot = len(wkp_df)
        wkp_df = wkp_df[wkp_df.activation_interval <= y_max]
        len_plt = len(wkp_df)
        if len_plt < len_tot:
            len_dif = len_tot - len_plt
            len_pct = 100. * len_dif / len_tot
            self._log.warning('Discarding {} activation intervals (above 2 x threshold_ms, '
                              '{:.1f}% of the overall activations)'\
                              .format(len_dif, len_pct))
        ymax = 1.1 * wkp_df.activation_interval.max()

        # Build the series for the CDF
        cdf = self._get_cdf(wkp_df.activation_interval, (threshold_ms / 1000.))
        self._log.info('%.1f %% samples below %d [ms] threshold',
                       100. * cdf.below, threshold_ms)

        # Setup plots
        gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,1])
        plt.figure(figsize=(16, 8))

        plot_title = "[{}]: activaton intervals (@ wakeup time)".format(td.label)
        if tag:
            plot_title = "{} [{}]".format(plot_title, tag)
        plot_title = "{}, threshold @ {} [ms]".format(plot_title, threshold_ms)

        # Activations intervals over time
        axes = plt.subplot(gs[0,0:2])
        axes.set_title(plot_title)
        wkp_df.plot(style='g+', logy=False, ax=axes)

        axes.axhline(threshold_ms / 1000., linestyle='--', color='g')
        self._trace.analysis.status.plot_overutilized(axes)
        axes.legend(loc='lower center', ncol=2)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)

        # Cumulative distribution of all activations intervals
        axes = plt.subplot(gs[1,0])
        cdf.df.plot(ax=axes, legend=False, xlim=(0,None),
                    title='Activations CDF ({:.1f}% within {} [ms] threshold)'\
                          .format(100. * cdf.below, threshold_ms))
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);
        axes.axhline(y=cdf.below, linewidth=1, color='r', linestyle='--')

        # Histogram of all activations intervals
        axes = plt.subplot(gs[1,1])
        wkp_df.plot(kind='hist', bins=bins, ax=axes,
                        xlim=(0,ymax), legend=False,
                        title='Activation intervals histogram ({} bins, {} [ms] green threshold)'\
                        .format(bins, threshold_ms));
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);

        # Save generated plots into datadir
        task_name = re.sub('[\ :/]', '_', td.label)
        figname = '{}/{}task_activations_{}_{}.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix,
                          td.pid, task_name)
        pl.savefig(figname, bbox_inches='tight')

        # Return statistics
        stats_df = wkp_df.describe(percentiles=[0.95, 0.99])
        label = '{:.1f}%'.format(100. * cdf.below)
        stats = { label : cdf.threshold }
        return stats_df.append(pd.DataFrame(
            list(stats.values()), columns=['activation_interval'], index=list(stats.keys())))


    def plot_runtimes(self, task, tag=None, threshold_ms=8, bins=64):
        """
        Plots "running times" for the specified task

        A "running time" is the sum of all the time intervals a task executed
        in between a wakeup and the next sleep (or exit).
        A set of plots is generated to report:
        - Running times at block time: every time a task blocks a
          point is plotted to represent the cumulative time the task has be
          running since its last wakeup
        - Running time cumulative function: reports the cumulative
          function of the running times.
        - Running times histogram: reports a 64 bins histogram of
          the running times.

        All plots are parameterized based on the value of threshold_ms, which
        can be used to filter running times bigger than 2 times this value.
        Such a threshold is useful to filter out from the plots outliers thus
        focusing the analysis in the most critical periodicity under analysis.
        The number and percentage of discarded samples is reported in output.
        A default threshold of 16 [ms] is used, which is useful for example to
        analyze a 60Hz rendering pipelines.

        A PNG of the generated plots is generated and saved in the same folder
        where the trace is.

        :param task: the task to report latencies for
        :type task: int or list(str)

        :param tag: a string to add to the plot title
        :type tag: str

        :param threshold_ms: the minimum acceptable [ms] value to report
                             graphically in the generated plots
        :type threshold_ms: int or float

        :param bins: number of bins to be used for the runtime's histogram
        :type bins: int

        :returns: a DataFrame with statistics on ploted running times
        """

        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Event [sched_switch] not found, '
                              'plot DISABLED!')
            return
        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Event [sched_wakeup] not found, '
                              'plot DISABLED!')
            return

        # Get task data
        td = self._get_task_data(task)
        if not td:
            return None

        # Load runtime data
        run_df = self.df_runtimes(td.pid)
        if run_df is None:
            return None
        self._log.info('Found: %5d activations for [%s]',
                       len(run_df), td.label)

        # Disregard data above two time the specified threshold
        y_max = (2 * threshold_ms) / 1000.
        len_tot = len(run_df)
        run_df = run_df[run_df.running_time <= y_max]
        len_plt = len(run_df)
        if len_plt < len_tot:
            len_dif = len_tot - len_plt
            len_pct = 100. * len_dif / len_tot
            self._log.warning('Discarding {} running times (above 2 x threshold_ms, '
                              '{:.1f}% of the overall activations)'\
                              .format(len_dif, len_pct))
        ymax = 1.1 * run_df.running_time.max()

        # Build the series for the CDF
        cdf = self._get_cdf(run_df.running_time, (threshold_ms / 1000.))
        self._log.info('%.1f %% samples below %d [ms] threshold',
                       100. * cdf.below, threshold_ms)

        # Setup plots
        gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,1])
        plt.figure(figsize=(16, 8))

        plot_title = "[{}]: running times (@ block time)".format(td.label)
        if tag:
            plot_title = "{} [{}]".format(plot_title, tag)
        plot_title = "{}, threshold @ {} [ms]".format(plot_title, threshold_ms)

        # Running time over time
        axes = plt.subplot(gs[0,0:2])
        axes.set_title(plot_title)
        run_df.plot(style='g+', logy=False, ax=axes)

        axes.axhline(threshold_ms / 1000., linestyle='--', color='g')
        self._trace.analysis.status.plot_overutilized(axes)
        axes.legend(loc='lower center', ncol=2)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)

        # Cumulative distribution of all running times
        axes = plt.subplot(gs[1,0])
        cdf.df.plot(ax=axes, legend=False, xlim=(0,None),
                    title='Runtime CDF ({:.1f}% within {} [ms] threshold)'\
                          .format(100. * cdf.below, threshold_ms))
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);
        axes.axhline(y=cdf.below, linewidth=1, color='r', linestyle='--')

        # Histogram of all running times
        axes = plt.subplot(gs[1,1])
        run_df.plot(kind='hist', bins=bins, ax=axes,
                        xlim=(0,ymax), legend=False,
                        title='Latency histogram ({} bins, {} [ms] green threshold)'\
                        .format(bins, threshold_ms));
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);

        # Save generated plots into datadir
        task_name = re.sub('[\ :/]', '_', td.label)
        figname = '{}/{}task_runtimes_{}_{}.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix,
                          td.pid, task_name)
        pl.savefig(figname, bbox_inches='tight')

        # Return statistics
        stats_df = run_df.describe(percentiles=[0.95, 0.99])
        label = '{:.1f}%'.format(100. * cdf.below)
        stats = { label : cdf.threshold }
        return stats_df.append(pd.DataFrame(
            list(stats.values()), columns=['running_time'], index=list(stats.keys())))

    def plot_task_residency(self, task):
        """
        Plot CPU residency of the specified task
        This will show an overview of how much time that task spent being
        active on each available CPU, in seconds.

        :param task: the task to report runtimes for
        :type task: int or str
        """
        df = self.df_task_residency(task)

        ax = df.plot(kind='bar', figsize=(16, 6))
        ax.set_title('CPU residency of task {}'.format(task))

        figname = os.path.join(
            self._trace.plots_dir,
            '{}task_cpu_residency_{}.png'.format(
                self._trace.plots_prefix, task
            )
        )

        pl.savefig(figname, bbox_inches='tight')

###############################################################################
# Utility Methods
###############################################################################

    @memoized
    def _get_task_data(self, task):

        # Get task PID
        if isinstance(task, str):
            task_pids = self._trace.getTaskByName(task)
            if len(task_pids) == 0:
                self._log.warning('No tasks found with name [%s]', task)
                return None

            task_pid = task_pids[0]
            if len(task_pids) > 1:
                self._log.warning('Multiple PIDs for task named [%s]', task)
                for pid in task_pids:
                    self._log.warning('  %5d :  %s', pid,
                                      ','.join(self._trace.getTaskByPid(pid)))
                self._log.warning('Returning stats only for PID: %d',
                                  task_pid)
            task_name = self._trace.getTaskByPid(task_pid)

        # Get task name
        elif isinstance(task, int):
            task_pid = task
            task_name = self._trace.getTaskByPid(task_pid)
            if task_name is None:
                self._log.warning('No tasks found with name [%s]', task)
                return None

        else:
            raise ValueError("Task must be either an int or str")

        task_label = "{}: {}".format(task_pid, task_name)
        return TaskData(task_pid, task_name, task_label)

    @memoized
    def _task_state(self, state):
        try:
            state = int(state)
        except ValueError:
            # State already converted to symbol
            return state

        # Tasks STATE flags (Linux 3.18)
        TASK_STATES = {
              0: "R", # TASK_RUNNING
              1: "S", # TASK_INTERRUPTIBLE
              2: "D", # TASK_UNINTERRUPTIBLE
              4: "T", # __TASK_STOPPED
              8: "t", # __TASK_TRACED
             16: "X", # EXIT_DEAD
             32: "Z", # EXIT_ZOMBIE
             64: "x", # TASK_DEAD
            128: "K", # TASK_WAKEKILL
            256: "W", # TASK_WAKING
            512: "P", # TASK_PARKED
           1024: "N", # TASK_NOLOAD
        }
        if self._trace.kernel_version >= (4, 8):
            TASK_STATES[2048] = "n" # TASK_NEW
        TASK_MAX_STATE = 2 * max(TASK_STATES)

        res = "R"
        if state & (TASK_MAX_STATE - 1) != 0:
            res = ""
        for key in list(TASK_STATES.keys()):
            if key & state:
                res += TASK_STATES[key]
        if state & TASK_MAX_STATE:
            res += "+"
        else:
            res = '|'.join(res)
        return res


    def _get_cdf(self, data, threshold):
        """
        Build the "Cumulative Distribution Function" (CDF) for the given data
        """

        # Build the series of sorted values
        ser = data.sort_values()
        if len(ser) < 1000:
            # Append again the last (and largest) value.
            # This step is important especially for small sample sizes
            # in order to get an unbiased CDF
            ser = ser.append(pd.Series(ser.iloc[-1]))
        df = pd.Series(np.linspace(0., 1., len(ser)), index=ser)

        # Compute percentage of samples above/below the specified threshold
        below = float(max(df[:threshold]))
        above = 1 - below
        return CDF(df, threshold, above, below)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
