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

from trappy.utils import handle_duplicate_index

from lisa.utils import memoized
from lisa.datautils import series_integrate
from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events


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

    @memoized
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

        cpu_active = cpu_df.state.apply(
            lambda s: 1 if s == -1 else 0
        )

        start_time = self.trace.start

        if cpu_active.empty:
            cpu_active = pd.Series([0], index=[start_time])
        elif cpu_active.index[0] != start_time:
            entry_0 = pd.Series(cpu_active.iloc[0] ^ 1, index=[start_time])
            cpu_active = pd.concat([entry_0, cpu_active])

        # Fix sequences of wakeup/sleep events reported with the same index
        return handle_duplicate_index(cpu_active)

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

        sr = pd.Series()
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
        cpu_idle = idle_df[idle_df.cpu_id == cpu]

        cpu_is_idle = self.signal_cpu_active(cpu) ^ 1

        # In order to compute the time spent in each idle state we
        # multiply 2 square waves:
        # - cpu_idle
        # - idle_state, square wave of the form:
        #     idle_state[t] == 1 if at time t CPU is in idle state i
        #     idle_state[t] == 0 otherwise
        available_idles = sorted(idle_df.state.unique())
        # Remove non-idle state from availables
        available_idles = available_idles[1:]
        cpu_idle = cpu_idle.join(cpu_is_idle.to_frame(name='is_idle'),
                                 how='outer')
        cpu_idle.fillna(method='ffill', inplace=True)

        # Extend the last cpu_idle event to the end of the time window under
        # consideration
        final_entry = pd.DataFrame([cpu_idle.iloc[-1]], index=[self.trace.end])
        cpu_idle = cpu_idle.append(final_entry)

        idle_time = []
        for i in available_idles:
            idle_state = cpu_idle.state.apply(
                lambda x: 1 if x == i else 0
            )
            idle_t = cpu_idle.is_idle * idle_state
            # Compute total time by integrating the square wave
            idle_time.append(series_integrate(idle_t))

        idle_time_df = pd.DataFrame({'time': idle_time}, index=available_idles)
        idle_time_df.index.name = 'idle_state'
        return idle_time_df

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
        # Each core in a cluster can be in a different idle state, but the
        # cluster lies in the idle state with lowest ID, that is the shallowest
        # idle state among the idle states of its CPUs
        cl_idle = idle_df[idle_df.cpu_id == cluster[0]].state.to_frame(
            name=cluster[0])
        for cpu in cluster[1:]:
            cl_idle = cl_idle.join(
                idle_df[idle_df.cpu_id == cpu].state.to_frame(name=cpu),
                how='outer'
            )
        cl_idle.fillna(method='ffill', inplace=True)
        cl_idle = pd.DataFrame(cl_idle.min(axis=1), columns=['state'])

        # Build a square wave of the form:
        #     cl_is_idle[t] == 1 if all CPUs in the cluster are reported
        #                      to be idle by cpufreq at time t
        #     cl_is_idle[t] == 0 otherwise
        cl_is_idle = self.signal_cluster_active(cluster) ^ 1

        # In order to compute the time spent in each idle state frequency we
        # multiply 2 square waves:
        # - cluster_is_idle
        # - idle_state, square wave of the form:
        #     idle_state[t] == 1 if at time t cluster is in idle state i
        #     idle_state[t] == 0 otherwise
        available_idles = sorted(idle_df.state.unique())
        # Remove non-idle state from availables
        available_idles = available_idles[1:]
        cl_idle = cl_idle.join(cl_is_idle.to_frame(name='is_idle'),
                               how='outer')
        cl_idle.fillna(method='ffill', inplace=True)
        idle_time = []
        for i in available_idles:
            idle_state = cl_idle.state.apply(
                lambda x: 1 if x == i else 0
            )
            idle_t = cl_idle.is_idle * idle_state
            # Compute total time by integrating the square wave
            idle_time.append(series_integrate(idle_t))

        idle_time_df = pd.DataFrame({'time': idle_time}, index=available_idles)
        idle_time_df.index.name = 'idle_state'
        return idle_time_df


###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method()
    @df_cpu_idle_state_residency.used_events
    def plot_cpu_idle_state_residency(self, cpu, axis, local_fig, pct=False):
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
    def plot_clusters_idle_state_residency(self, pct=False, **kwargs):
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

        return self.do_plot(plotter, nrows=len(clusters), sharex=True, **kwargs)

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
