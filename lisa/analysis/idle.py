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
import warnings
import typing

import pandas as pd
import holoviews as hv

from lisa.datautils import df_add_delta, df_refit_index, df_split_signals
from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events, CPU
from lisa.analysis.base import TraceAnalysisBase


class IdleAnalysis(TraceAnalysisBase):
    """
    Support for plotting Idle Analysis data

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'idle'

###############################################################################
# DataFrame Getter Methods
###############################################################################
    @TraceAnalysisBase.cache
    @requires_events('cpu_idle')
    def df_cpus_idle(self, cpus=None):
        """
        Dataframe of the ``cpu_idle`` event, with the following columns:

            * ``cpu``
            * ``state``: Instead of 4294967295, the -1 type independent value
              is used.

        :param cpus: Optionally, filter on that list of CPUs
        :type cpus: list(int) or None
        """
        df = self.trace.df_event('cpu_idle')
        # Filter before rename to avoid copying data we will ignore
        if cpus is not None:
            df = df[df['cpu_id'].isin(cpus)]

        df = df.rename({'cpu_id': 'cpu'}, axis=1)
        # The event uses an unsigned int even though the kernel uses -1, so use
        # -1 to avoid being tied to the event field type size
        non_idle = (2 ** 32) -1
        df['state'].replace(non_idle, -1, inplace=True)
        return df

    @TraceAnalysisBase.cache
    @df_cpus_idle.used_events
    def df_cpu_idle(self, cpu=None):
        """
        Same as :meth:`df_cpus_idle` but for one CPU.
        """
        if cpu is None:
            warnings.warn('cpu=None is deprecated, use df_cpus_idle() to get a dataframe for all CPUs', DeprecationWarning)
            cpus = None
        else:
            cpus = [cpu]

        return self.df_cpus_idle(cpus=cpus)

    @df_cpu_idle.used_events
    def signal_cpu_active(self, cpu):
        """
        Build a square wave representing the active (i.e. non-idle) CPU time

        :param cpu: CPU ID
        :type cpu: int

        :returns: A :class:`pandas.Series` that equals 1 at timestamps where the
          CPU is reported to be non-idle, 0 otherwise
        """
        cpu_df = self.df_cpu_idle(cpu)

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
    @signal_cpu_active.used_events
    def df_cpus_wakeups(self):
        """
        Get a DataFrame showing when CPUs have woken from idle

        :param cpus: List of CPUs to find wakeups for. If None, all CPUs.
        :type cpus: list(int) or None

        :returns: A :class:`pandas.DataFrame` with

          * A ``cpu`` column (the CPU that woke up at the row index)
        """
        cpus = list(range(self.trace.cpus_count))

        def make_series(cpu):
            series = self.signal_cpu_active(cpu)
            series = series[series == 1]
            return series.replace(1, cpu)

        return pd.DataFrame({
            'cpu': pd.concat(map(make_series, cpus))
        }).sort_index()

    @df_cpu_idle.used_events
    def df_cpu_idle_state_residency(self, cpu):
        """
        Compute time spent by a given CPU in each idle state.

        :param cpu: CPU ID
        :type cpu: int

        :returns: a :class:`pandas.DataFrame` with:

          * Idle states as index
          * A ``time`` column (The time spent in the idle state)
        """
        idle_df = self.df_cpu_idle(cpu)

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

    @df_cpu_idle.used_events
    def df_cluster_idle_state_residency(self, cluster):
        """
        Compute time spent by a given cluster in each idle state.

        :param cluster: list of CPU IDs
        :type cluster: list(int)

        :returns: a :class:`pandas.DataFrame` with:

          * Idle states as index
          * A ``time`` column (The time spent in the idle state)
        """
        idle_df = self.df_cpus_idle()

        # Create a dataframe with a column per CPU
        cols = {
            cpu: group['state']
            for cpu, group in idle_df.groupby(
                'cpu',
                sort=False,
                observed=True,
                group_keys=False,
            )
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
        residency = df.groupby('cluster_state', sort=False, observed=True, group_keys=False)['delta'].sum()
        residency.name = 'time'

        residency = residency.to_frame()
        residency.index.name = 'idle_state'
        return residency

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method
    @df_cpu_idle_state_residency.used_events
    def plot_cpu_idle_state_residency(self, cpu: CPU, pct: bool=False):
        """
        Plot the idle state residency of a CPU

        :param cpu: The CPU
        :type cpu: int

        :param pct: Plot residencies in percentage
        :type pct: bool
        """
        df = self.df_cpu_idle_state_residency(cpu)
        return self._plot_idle_state_residency(df, pct=pct).options(
            title=f"CPU{cpu} idle state residency",
        )

    @TraceAnalysisBase.plot_method
    @df_cluster_idle_state_residency.used_events
    def plot_cluster_idle_state_residency(self, cluster: typing.Sequence[CPU], pct: bool=False):
        """
        Plot the idle state residency of a cluster

        :param cluster: The cluster
        :type cpu: list(int)

        :param pct: Plot residencies in percentage
        :type pct: bool
        """

        df = self.df_cluster_idle_state_residency(cluster)
        return self._plot_idle_state_residency(df, pct=pct).options(
            title=f"CPUs {cluster} idle state residency",
        )

    @TraceAnalysisBase.plot_method
    @plot_cluster_idle_state_residency.used_events
    def plot_clusters_idle_state_residency(self, pct: bool=False):
        """
        Plot the idle state residency of all clusters

        :param pct: Plot residencies in percentage
        :type pct: bool

        .. note:: This assumes clusters == frequency domains, which may
          not hold true...
        """
        return reduce(
            operator.add,
            (
                self.plot_cluster_idle_state_residency(cluster, pct=pct)
                for cluster in self.trace.plat_info['freq-domains']
            )
        ).cols(1)

###############################################################################
# Utility Methods
###############################################################################

    def _plot_idle_state_residency(self, df, pct):
        """
        A convenient helper to plot idle state residency
        """
        if pct:
            df = df * 100 / df.sum()

        ylabel = 'Time share (%)' if pct else 'Time (s)'

        return hv.Bars(df['time']).options(
            ylabel=ylabel,
            xlabel='Idle state',
            invert_axes=True,
        )

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
