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

""" CPUs Analysis Module """

import pandas as pd
import holoviews as hv

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events, CPU
from lisa.datautils import df_window


class CpusAnalysis(TraceAnalysisBase):
    """
    Support for CPUs signals analysis
    """

    name = 'cpus'

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.cache
    @requires_events('sched_switch')
    def df_context_switches(self):
        """
        Compute number of context switches on each CPU.

        :returns: A :class:`pandas.DataFrame` with:

          * A ``context_switch_cnt`` column (the number of context switch per CPU)
        """
        # Since we want to count the number of context switches, we don't want
        # all tasks to appear
        sched_df = self.trace.df_event('sched_switch', signals_init=False)
        # Make sure to only get the switches inside the window
        sched_df = df_window(
            sched_df,
            method='exclusive',
            window=self.trace.window,
            clip_window=False,
        )
        cpus = list(range(self.trace.cpus_count))
        ctx_sw_df = pd.DataFrame(
            [len(sched_df[sched_df['__cpu'] == cpu]) for cpu in cpus],
            index=cpus,
            columns=['context_switch_cnt']
        )
        ctx_sw_df.index.name = 'cpu'

        return ctx_sw_df

    @TraceAnalysisBase.cache
    @requires_events('sched_switch')
    def df_states(self):
        """
        Compute the state intervals on each CPU.

        :returns: A :class:`pandas.DataFrame` with:

          * A ``cpu`` column (the CPU the state refers to)
          * A ``state`` column (the BUSY/IDLE state the CPU is into)
          * A ``duration`` column (the time the CPU is in this state)
          * A ``end_time`` column (the time the CPU will exit this state)
        """
        # Start from sched_switch events
        sched_df = self.trace.df_event('sched_switch')
        
        # Keep only CPU transition events (IDLE to/from BUSY)
        def switch_cpu_state(row):
            if row.prev_pid != 0 and row.next_pid != 0:
                return False
            return True
        states_df = sched_df[sched_df.apply(switch_cpu_state, axis=1)]
        
        # Reset index and use the event timestamp to compute deltas
        states_df.reset_index(inplace=True)
        
        # Compute next transition time and deltas (by grouping events by CPU)
        grouped = states_df.groupby('__cpu', observed=True, sort=False)
        new_columns = dict(
            end_time=grouped['Time'].shift(-1),
            # GroupBy.transform() will run the function on each group, and
            # concatenate the resulting series to create a new column.
            # Note: We actually need transform() to chain 2 operations on
            # the group, otherwise the first operation returns a final
            # Series, and the 2nd is not applied on groups
            duration=grouped['Time'].transform(lambda time: time.diff().shift(-1)),
        )
        states_df = states_df.assign(**new_columns)[:-1]
        
        # Back annotate the CPU state on each period
        def cpu_state(prev_pid):
            # Idle entry event
            if prev_pid:
                return 'IDLE'
            # Idle exit event
            return 'BUSY'
        states_df['state'] = states_df['prev_pid'].apply(lambda prev_pid: cpu_state(prev_pid))
        
        # Reset the index and return ordered minimal set of columns
        states_df.set_index('Time', inplace=True)
        states_df.rename({'__cpu': 'cpu'}, axis=1, inplace=True)
        return states_df[['cpu', 'state', 'duration', 'end_time']]

    @TraceAnalysisBase.cache
    @requires_events('sched_switch')
    @df_states.used_events
    def df_utils(self):
        """
        Compute stats on utilization levels of each CPU.

        :returns: A :class:`pandas.DataFrame` indexed on CPUs with:

          * A ``busy`` column (the total time a CPU has been BUSY)
          * An ``idle`` column (the total time a CPU has been IDLE)
          * An ``unacc`` colums (the total time not accounted as BUSY/IDLE)
          * An ``unacc_pct`` colums (the percentage of accounted time)
          * A ``util`` colums (the fraction of time the CPU has been BUSY)
        """
        states_df = self.df_states()

        # Busy time for each CPU
        grouped = states_df[states_df.state == 'BUSY'].groupby('cpu')
        stats_df = grouped[['duration']].sum()
        stats_df.rename({'duration': 'busy'}, axis=1, inplace=True)
        
        # Idle time for each CPU
        grouped = states_df[states_df.state == 'IDLE'].groupby('cpu')
        stats_df = stats_df.join(grouped[['duration']].sum())
        stats_df.rename({'duration': 'idle'}, axis=1, inplace=True)
        
        # Measure of the not accounted BUSY/IDLE time intervals
        # i.e. due to missing initial events
        stats_df['unacc'] = self.trace.time_range - (stats_df['busy'] + stats_df['idle'])
        stats_df['unacc_pct'] = 100 * stats_df['unacc'] / self.trace.time_range
        
        # CPU utilization
        stats_df['util'] = stats_df['busy'] / self.trace.time_range
        
        return stats_df

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method
    @df_context_switches.used_events
    def plot_context_switches(self):
        """
        Plot histogram of context switches on each CPU.
        """
        ctx_sw_df = self.df_context_switches()
        return hv.Bars(
            ctx_sw_df["context_switch_cnt"]
        ).options(
            title='Per-CPU Task Context Switches',
            xlabel='CPU',
            ylabel='Number of context switches',
            invert_axes=True,
        )

    @TraceAnalysisBase.plot_method
    @df_utils.used_events
    def plot_states(self):
        """
        Plot stacked histogram of BUSY/IDLE states of each CPU.
        """
        utils_df = self.df_utils()[['busy', 'idle', 'unacc']]
        utils_df = utils_df.reset_index().melt(id_vars=['cpu'])

        def set_color(state):
            if state == 'busy':
                return 'red'
            if state == 'idle':
                return 'green'
            if state == 'unacc':
                return 'yellow'  
        utils_df['color'] = utils_df['variable'].apply(
            lambda state: set_color(state))
        return hv.Bars(utils_df, kdims=['cpu', 'variable']).options(
            title='Per-CPU busy/idle time breakdown',
            ylabel='cpu status time',
            stacked=True,
            invert_axes=True,
            color='color',
        )

    @TraceAnalysisBase.plot_method
    def plot_orig_capacity(self, cpu: CPU):
        """
        Plot the orig capacity of a CPU onto a given axis

        :param cpu: The CPU
        :type cpu: int
        """
        orig_capacities = self.trace.plat_info['cpu-capacities']['orig']
        return hv.HLine(
            orig_capacities[cpu],
            label='orig capacity'
        ).options(
            backend='matplotlib',
            linestyle='--',
        ).options(
            backend='bokeh',
            line_dash='dashed',
        )

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
