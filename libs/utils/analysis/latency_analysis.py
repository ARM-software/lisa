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

from collections import namedtuple
from analysis_module import AnalysisModule
from devlib.utils.misc import memoized
from trappy.utils import listify

# Tuple representing all IDs data of a Task
TaskData = namedtuple('TaskData', ['pid', 'names', 'label'])

class LatencyAnalysis(AnalysisModule):
    """
    Support for plotting Latency Analysis data

    :param trace: input Trace object
    :type trace: :mod:`libs.utils.Trace`
    """

    def __init__(self, trace):
        super(LatencyAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @memoized
    def _dfg_latency_df(self, task):

        if not self._trace.hasEvents('sched_wakeup'):
            self._log.warning('Events [sched_wakeup] not found, '
                              'cannot compute CPU active signal!')
            return None
        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Events [sched_switch] not found, '
                              'cannot compute CPU active signal!')
            return None

        # Get task data
        td = self._getTaskData(task)
        if not td:
            return None

        wk_df = self._dfg_trace_event('sched_wakeup')
        sw_df = self._dfg_trace_event('sched_switch')

        # Filter Task's WAKEUP events
        task_wakeup = wk_df[wk_df.pid == td.pid][['success', 'target_cpu', 'pid']]

        # Filter Task's START events
        task_events = (sw_df.prev_pid == td.pid) | (sw_df.next_pid == td.pid)
        task_switches_df = sw_df[task_events]\
            [['__cpu', 'prev_pid', 'next_pid', 'prev_state']]

        # Unset prev_state for switch_in events, i.e.
        # we don't care about the status of a task we are replacing
        task_switches_df.prev_state = task_switches_df.apply(
            lambda r : np.nan if r['prev_pid'] != td.pid
                              else self._taskState(int(r['prev_state'])),
            axis=1)

        # Rename prev_state
        task_switches_df.rename(columns={'prev_state' : 'curr_state'}, inplace=True)

        # Fill in Running status
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
            self._log.warning(' %s::%s _taskState()',
                              __file__, self.__class__.__name__)

        # Forward annotate task state
        task_latency_df['next_state'] = task_latency_df.curr_state.shift(-1)

        # Forward account for previous state duration
        task_latency_df['t_start'] =  task_latency_df.index
        task_latency_df['t_delta'] = (
              task_latency_df['t_start'].shift(-1)
            - task_latency_df['t_start']
        )

        return task_latency_df


    # Select Wakeup latency
    def _dfg_latency_wakeup_df(self, task):
        task_latency_df = self._dfg_latency_df(task)
        if task_latency_df is None:
            return None
        df = task_latency_df[
                    (task_latency_df.curr_state == 'W') &
                    (task_latency_df.next_state == 'A')][['t_delta']]
        df.rename(columns={'t_delta' : 'wakeup_latency'}, inplace=True)
        return df

    # Select Wakeup latency
    def _dfg_latency_preemption_df(self, task):
        task_latency_df = self._dfg_latency_df(task)
        if task_latency_df is None:
            return None
        df = task_latency_df[
                    (task_latency_df.curr_state.isin([0, 'R', 'R+'])) &
                    (task_latency_df.next_state == 'A')][['t_delta']]
        df.rename(columns={'t_delta' : 'preempt_latency'}, inplace=True)
        return df


###############################################################################
# Plotting Methods
###############################################################################

    def plotLatency(self, task, kind='all', tag=None, threshold_ms=1):
        """
        Generate a set of plots to report the WAKEUP and PREEMPT latencies the
        specified task has been subject to. A WAKEUP latencies is the time from
        when a task becomes RUNNABLE till the first time it gets a CPU.
        A PREEMPT latencies is the time from when a RUNNABLE task is suspended
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
        td = self._getTaskData(task)
        if not td:
            return None

        # Load wakeup latencies (if required)
        wkp_df = None
        if 'all' in kind or 'wakeup' in kind:
            wkp_df = self._dfg_latency_wakeup_df(td.pid)
        if wkp_df is not None:
            wkp_df.rename(columns={'wakeup_latency' : 'latency'}, inplace=True)
            self._log.info('Found: %5d WAKEUP latencies', len(wkp_df))

        # Load preempt latencies (if required)
        prt_df = None
        if 'all' in kind or 'preempt' in kind:
            prt_df = self._dfg_latency_preemption_df(td.pid)
        if prt_df is not None:
            prt_df.rename(columns={'preempt_latency' : 'latency'}, inplace=True)
            self._log.info('Found: %5d PREEMPT latencies', len(prt_df))

        if wkp_df is None and prt_df is None:
            self._log.warning('No Latency info for task [%s]', td.label)
            return

        # Join the two data frames
        df = wkp_df.append(prt_df)
        ymax = 1.1 * df.latency.max()
        self._log.info('Total: %5d latency events', len(df))

        df = pd.DataFrame(sorted(df.latency), columns=['latency'])

        # Setup plots
        gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,1])
        plt.figure(figsize=(16, 8))

        plot_title = "Task [{}] latencies".format(kind.upper())
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
        self._trace.analysis.status.plotOverutilized(axes)
        axes.legend(loc='lower center', ncol=2)

        # Cumulative distribution of all latencies
        axes = plt.subplot(gs[1,0])
        df.latency.plot(ax=axes, logy=True, legend=False,
                        title='Latencies cumulative distribution [{}]'\
                              .format(td.label))
        axes.axhline(y=threshold_ms / 1000., linewidth=2,
                     color='r', linestyle='--')

        # Histogram of all latencies
        axes = plt.subplot(gs[1,1])
        df.latency.plot(kind='hist', bins=64, ax=axes,
                        xlim=(0,ymax), legend=False,
                        title='Latency histogram (64 bins, {} [ms] green threshold)'\
                        .format(threshold_ms));
        axes.axvspan(0, threshold_ms / 1000., facecolor='g', alpha=0.5);

        # Save generated plots into datadir
        task_name = re.sub('[\ :/]', '_', td.label)
        figname = '{}/{}task_latencies_{}_{}.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix,
                          td.pid, task_name)
        pl.savefig(figname, bbox_inches='tight')

        # Return statistics
        return df.describe(percentiles=[0.95, 0.99])

    def plotLatencyBands(self, task, axes=None):
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
        td = self._getTaskData(task)
        if not td:
            return None

        wkl_df = self._dfg_latency_wakeup_df(td.pid)
        prt_df = self._dfg_latency_preemption_df(td.pid)

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
        except: pass

        # Draw PREEMPTION latencies
        try:
            bands = [(t, prt_df['preempt_latency'][t]) for t in prt_df.index]
            for (start, duration) in bands:
                end = start + duration
                axes.axvspan(start, end, facecolor='b', alpha=0.1)
        except: pass


###############################################################################
# Utility Methods
###############################################################################

    @memoized
    def _getTaskData(self, task):

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
            task_names = self._trace.getTaskByPid(task_pid)

        # Get task name
        elif isinstance(task, int):
            task_pid = task
            task_names = self._trace.getTaskByPid(task_pid)
            if len(task_names) == 0:
                self._log.warning('No tasks found with name [%s]', task)
                return None

        else:
            raise ValueError("Task must be either an int or str")

        task_label = "{}: {}".format(task_pid, ', '.join(task_names))
        return TaskData(task_pid, task_names, task_label)

    @memoized
    def _taskState(self, state):

        # Tasks STATE flags (Linux 4.8)
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
           2048: "n", # TASK_NEW
        }
        TASK_MAX_STATE = 4096

        res = "R"
        if state & (TASK_MAX_STATE - 1) != 0:
            res = ""
        for key in TASK_STATES.keys():
            if key & state:
                res += TASK_STATES[key]
        if state & TASK_MAX_STATE:
            res += "+"
        else:
            res = '|'.join(res)
        return res

# vim :set tabstop=4 shiftwidth=4 expandtab
