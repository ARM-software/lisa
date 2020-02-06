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

from functools import reduce
import operator

import pandas as pd
import numpy as np

from lisa.datautils import series_integrate, df_split_signals, series_combine, df_add_delta, df_refit_index
from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events, CPU


class IdleAnalysis(TraceAnalysisBase):
    """
    Support for plotting Idle Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'idle'

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @requires_events('cpu_idle')
    def signal_cpu_active(self, cpu):
        """
        Build a square wave representing the active (i.e. non-idle) CPU time

        :param cpu: CPU ID
        :type cpu: int

        :returns: A :class:`pandas.Series` that equals 1 at timestamps where the
          CPU is reported to be non-idle, 0 otherwise
        """
        idle_df = self.trace.df_events('cpu_idle')
        cpu_df = idle_df[idle_df.cpu_id == cpu]

        # Turn -1 into 1 and everything else into 0
        cpu_active = cpu_df.state.map({-1: 1})
        cpu_active.fillna(value=0, inplace=True)

        return cpu_active

    @signal_cpu_active.used_events
    def signal_cluster_active(self, cluster):
        """
        Build a square wave representing the active (i.e. non-idle) cluster time

        :param cluster: list of CPU IDs belonging to a cluster
        :type cluster: list(int)

        :returns: A :class:`pandas.Series` that equals 1 at timestamps where at
          least one CPU is reported to be non-idle, 0 otherwise
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

    @TraceAnalysisBase.cache
    @requires_events('cpu_idle')
    def df_cpus_wakeups(self):
        """"
        Get a DataFrame showing when CPUs have woken from idle

        :param cpus: List of CPUs to find wakeups for. If None, all CPUs.
        :type cpus: list(int) or None

        :returns: A :class:`pandas.DataFrame` with

          * A ``cpu`` column (the CPU that woke up at the row index)
        """
        cpus = list(range(self.trace.cpus_count))

        sr = pd.Series(dtype='float64')
        for cpu in cpus:
            cpu_sr = self.signal_cpu_active(cpu)
            cpu_sr = cpu_sr[cpu_sr == 1]
            cpu_sr = cpu_sr.replace(1, cpu)
            sr = sr.append(cpu_sr)

        return pd.DataFrame({'cpu': sr}).sort_index()

    @requires_events("cpu_idle")
    def df_cpu_idle_state_residency(self, cpu):
        """
        Compute time spent by a given CPU in each idle state.

        :param cpu: CPU ID
        :type cpu: int

        :returns: a :class:`pandas.DataFrame` with:

          * Idle states as index
          * A ``time`` column (The time spent in the idle state)
        """
        idle_df = self.trace.df_events('cpu_idle')
        idle_df = idle_df[idle_df['cpu_id'] == cpu]

        # Ensure accurate time-based sum of state deltas
        idle_df = df_refit_index(idle_df, window=self.trace.window)

        # For each state, sum the time spent in it
        idle_df = df_add_delta(idle_df)

        residency = {
            cols['state']: state_df['delta'].sum()
            for cols, state_df in df_split_signals(idle_df, ['state'])

        }
        df = pd.DataFrame.from_dict(residency, orient='index', columns=['time'])
        df.index.name = 'idle_state'
        return df

    @requires_events('cpu_idle')
    def df_cluster_idle_state_residency(self, cluster):
        """
        Compute time spent by a given cluster in each idle state.

        :param cluster: list of CPU IDs
        :type cluster: list(int)

        :returns: a :class:`pandas.DataFrame` with:

          * Idle states as index
          * A ``time`` column (The time spent in the idle state)
        """
        idle_df = self.trace.df_events('cpu_idle')

        # Create a dataframe with a column per CPU
        cols = {
            cpu: group['state']
            for cpu, group in idle_df.groupby('cpu_id', group_keys=False)
            if cpu in cluster
        }
        cpus_df = pd.DataFrame(cols, index=idle_df.index)
        cpus_df.fillna(method='ffill', inplace=True)

        # Ensure accurate time-based sum of state deltas. This will extrapolate
        # the known cluster_state both to the left and the right.
        cpus_df = df_refit_index(cpus_df, window=self.trace.window)

        # Each core in a cluster can be in a different idle state, but the
        # cluster lies in the idle state with lowest ID, that is the shallowest
        # idle state among the idle states of its CPUs
        cluster_state = cpus_df.min(axis='columns')
        cluster_state.name = 'cluster_state'
        df = cluster_state.to_frame()

        # For each state transition, sum the time spent in it
        df_add_delta(df, inplace=True)

        # For each cluster state, take the sum of the delta column.
        # The resulting dataframe is indexed by group keys (cluster_state).
        residency = df.groupby('cluster_state')['delta'].sum()
        residency.name = 'time'

        residency = residency.to_frame()
        residency.index.name = 'idle_state'
        return residency

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method()
    @df_cpu_idle_state_residency.used_events
    def plot_cpu_idle_state_residency(self, cpu: CPU, axis, local_fig, pct=False):
        """
        Plot the idle state residency of a CPU

        :param cpu: The CPU
        :type cpu: int

        :param pct: Plot residencies in percentage
        :type pct: bool
        """
        df = self.df_cpu_idle_state_residency(cpu)
        self._plot_idle_state_residency(df, axis, pct)
        axis.set_title("CPU{} idle state residency".format(cpu))

    @TraceAnalysisBase.plot_method()
    @df_cluster_idle_state_residency.used_events
    def plot_cluster_idle_state_residency(self, cluster, axis, local_fig, pct=False):
        """
        Plot the idle state residency of a cluster

        :param cluster: The cluster
        :type cpu: list(int)

        :param pct: Plot residencies in percentage
        :type pct: bool
        """

        df = self.df_cluster_idle_state_residency(cluster)

        self._plot_idle_state_residency(df, axis, pct)
        axis.set_title("CPUs {} idle state residency".format(cluster))

    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cluster_idle_state_residency.used_events
    def plot_clusters_idle_state_residency(self, pct=False, axis=None, **kwargs):
        """
        Plot the idle state residency of all clusters

        :param pct: Plot residencies in percentage
        :type pct: bool

        .. note:: This assumes clusters == frequency domains, which may
          not hold true...
        """
        clusters = self.trace.plat_info['freq-domains']

        def plotter(axes, local_fig):
            for axis, cluster in zip(axes, clusters):
                self.plot_cluster_idle_state_residency(cluster, pct=pct, axis=axis)

        return self.do_plot(plotter, nrows=len(clusters), sharex=True, axis=axis, **kwargs)

###############################################################################
# Utility Methods
###############################################################################

    def _plot_idle_state_residency(self, df, axis, pct):
        """
        A convenient helper to plot idle state residency
        """
        if pct:
            df = df * 100 / df.sum()

        df["time"].plot.barh(ax=axis)

        if pct:
            axis.set_xlabel("Time share (%)")
        else:
            axis.set_xlabel("Time (s)")

        axis.set_ylabel("Idle state")
        axis.grid(True)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
