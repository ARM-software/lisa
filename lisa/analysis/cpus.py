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

import operator
from functools import reduce

import pandas as pd

from trappy.utils import handle_duplicate_index

from lisa.utils import memoized
from lisa.analysis.base import AnalysisBase, requires_events


class CpusAnalysis(AnalysisBase):
    """
    Support for CPUs Signals Analysis

    :param trace: input Trace object
    :type trace: :class:`Trace`
    """

    name = 'cpus'

    def __init__(self, trace):
        super(CpusAnalysis, self).__init__(trace)


###############################################################################
# DataFrame Getter Methods
###############################################################################

    @requires_events(['sched_switch'])
    def df_context_switches(self):
        """
        Compute number of context switches on each CPU.

        :returns: :mod:`pandas.DataFrame`
        """
        sched_df = self._trace.df_events('sched_switch')
        cpus = list(range(self._trace.cpus_count))
        ctx_sw_df = pd.DataFrame(
            [len(sched_df[sched_df['__cpu'] == cpu]) for cpu in cpus],
            index=cpus,
            columns=['context_switch_cnt']
        )
        ctx_sw_df.index.name = 'cpu'

        return ctx_sw_df

    @requires_events(['cpu_idle'])
    def df_cpu_wakeups(self, cpus=None):
        """"
        Get a DataFrame showing when a CPU was woken from idle

        :param cpus: List of CPUs to find wakeups for. If None, all CPUs.
        :type cpus: list(int) or None

        :returns: :mod:`pandas.DataFrame` with one column ``cpu``, where each
                  row shows a time when the given ``cpu`` was woken up from
                  idle.
        """
        cpus = cpus or list(range(self._trace.cpus_count))

        sr = pd.Series()
        for cpu in cpus:
            cpu_sr = self._trace.getCPUActiveSignal(cpu)
            cpu_sr = cpu_sr[cpu_sr == 1]
            cpu_sr = cpu_sr.replace(1, cpu)
            sr = sr.append(cpu_sr)

        return pd.DataFrame({'cpu': sr}).sort_index()

    @memoized
    @requires_events(['cpu_idle'])
    def signal_cpu_active(self, cpu):
        """
        Build a square wave representing the active (i.e. non-idle) CPU time,
        i.e.:

          cpu_active[t] == 1 if the CPU is reported to be non-idle by cpuidle at
          time t
          cpu_active[t] == 0 otherwise

        :param cpu: CPU ID
        :type cpu: int

        :returns: A :class:`pandas.Series` or ``None`` if the trace contains no
                  "cpu_idle" events
        """
        idle_df = self._trace.df_events('cpu_idle')
        cpu_df = idle_df[idle_df.cpu_id == cpu]

        cpu_active = cpu_df.state.apply(
            lambda s: 1 if s == -1 else 0
        )

        start_time = 0.0
        if not self._trace.ftrace.normalized_time:
            start_time = self._trace.ftrace.basetime

        if cpu_active.empty:
            cpu_active = pd.Series([0], index=[start_time])
        elif cpu_active.index[0] != start_time:
            entry_0 = pd.Series(cpu_active.iloc[0] ^ 1, index=[start_time])
            cpu_active = pd.concat([entry_0, cpu_active])

        # Fix sequences of wakeup/sleep events reported with the same index
        return handle_duplicate_index(cpu_active)

    @requires_events(signal_cpu_active.required_events)
    def signal_cluster_active(self, cluster):
        """
        Build a square wave representing the active (i.e. non-idle) cluster
        time, i.e.:

          cluster_active[t] == 1 if at least one CPU is reported to be non-idle
          by CPUFreq at time t
          cluster_active[t] == 0 otherwise

        :param cluster: list of CPU IDs belonging to a cluster
        :type cluster: list(int)

        :returns: A :class:`pandas.Series` or ``None`` if the trace contains no
                  "cpu_idle" events
        """
        active = self.signal_cpu_active(cluster[0]).to_frame(name=cluster[0])
        for cpu in cluster[1:]:
            active = active.join(
                self.signal_cpu_active(cpu).to_frame(name=cpu),
                how='outer'
            )

        active.fillna(method='ffill', inplace=True)
        # There might be NaNs in the signal where we got data from some CPUs
        # before others. That will break the .astype(int) below, so drop rows
        # with NaN in them.
        active.dropna(inplace=True)

        # Cluster active is the OR between the actives on each CPU
        # belonging to that specific cluster
        cluster_active = reduce(
            operator.or_,
            [cpu_active.astype(int) for _, cpu_active in
             active.items()]
        )

        return cluster_active



###############################################################################
# Plotting Methods
###############################################################################

    @requires_events(df_context_switches.required_events)
    def plot_context_switch(self, filepath=None):
        """
        Plot histogram of context switches on each CPU.
        """
        fig, axis = self.setup_plot(height=8)

        ctx_sw_df = self.df_context_switches()
        ctx_sw_df.plot.bar(title="Per-CPU Task Context Switches",
                           legend=False,
                           ax=axis)
        axis.grid()

        self.save_plot(fig, filepath)
        return axis

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
