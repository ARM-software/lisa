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

""" Frequency Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import operator
from trappy.utils import listify
from devlib.utils.misc import memoized
from collections import namedtuple

from analysis_module import AnalysisModule

# Configure logging
import logging

NON_IDLE_STATE = 4294967295

ResidencyTime = namedtuple('ResidencyTime', ['total', 'active'])
ResidencyData = namedtuple('ResidencyData', ['label', 'residency'])


class FrequencyAnalysis(AnalysisModule):
    """
    Support for plotting Frequency Analysis data

    :param trace: input Trace object
    :type trace: :mod:`libs.utils.Trace`
    """

    def __init__(self, trace):
        super(FrequencyAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    def _dfg_cpu_frequency_residency(self, cpu, total=True):
        """
        Get per-CPU frequency residency, i.e. amount of
        time CPU `cpu` spent at each frequency.

        :param cpu: CPU ID
        :type cpu: int

        :param total: if true returns the "total" time, otherwise the "active"
                      time is returned
        :type total: bool

        :returns: :mod:`pandas.DataFrame` - "total" or "active" time residency
                  at each frequency.
        """
        residency = self._getCPUFrequencyResidency(cpu)
        if not residency:
            return None
        if total:
            return residency.total
        return residency.active

    def _dfg_cluster_frequency_residency(self, cluster, total=True):
        """
        Get per-Cluster frequency residency, i.e. amount of time CLUSTER
        `cluster` spent at each frequency.

        :param cluster: this can be either a single CPU ID or a list of CPU IDs
            belonging to a cluster or the cluster name as specified in the
            platform description
        :type cluster: str or int or list(int)

        :param total: if true returns the "total" time, otherwise the "active"
                      time is returned
        :type total: bool

        :returns: :mod:`pandas.DataFrame` - "total" or "active" time residency
                  at each frequency.
        """
        residency = self._getClusterFrequencyResidency(cluster)
        if not residency:
            return None
        if total:
            return residency.total
        return residency.active


###############################################################################
# Plotting Methods
###############################################################################

    def plotClusterFrequencies(self, title='Clusters Frequencies'):
        """
        Plot frequency trend for all clusters. If sched_overutilized events are
        available, the plots will also show the intervals of time where the
        cluster was overutilized.

        :param title: user-defined plot title
        :type title: str
        """
        if not self._trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return
        df = self._dfg_trace_event('cpu_frequency')

        pd.options.mode.chained_assignment = None

        # Extract LITTLE and big clusters frequencies
        # and scale them to [MHz]
        if len(self._platform['clusters']['little']):
            lfreq = df[df.cpu == self._platform['clusters']['little'][-1]]
            lfreq['frequency'] = lfreq['frequency']/1e3
        else:
            lfreq = []
        if len(self._platform['clusters']['big']):
            bfreq = df[df.cpu == self._platform['clusters']['big'][-1]]
            bfreq['frequency'] = bfreq['frequency']/1e3
        else:
            bfreq = []

        # Compute AVG frequency for LITTLE cluster
        avg_lfreq = 0
        if len(lfreq) > 0:
            lfreq['timestamp'] = lfreq.index
            lfreq['delta'] = (lfreq['timestamp'] -lfreq['timestamp'].shift()).fillna(0).shift(-1)
            lfreq['cfreq'] = (lfreq['frequency'] * lfreq['delta']).fillna(0)
            timespan = lfreq.iloc[-1].timestamp - lfreq.iloc[0].timestamp
            avg_lfreq = lfreq['cfreq'].sum()/timespan

        # Compute AVG frequency for big cluster
        avg_bfreq = 0
        if len(bfreq) > 0:
            bfreq['timestamp'] = bfreq.index
            bfreq['delta'] = (bfreq['timestamp'] - bfreq['timestamp'].shift()).fillna(0).shift(-1)
            bfreq['cfreq'] = (bfreq['frequency'] * bfreq['delta']).fillna(0)
            timespan = bfreq.iloc[-1].timestamp - bfreq.iloc[0].timestamp
            avg_bfreq = bfreq['cfreq'].sum()/timespan

        pd.options.mode.chained_assignment = 'warn'

        # Setup a dual cluster plot
        fig, pltaxes = plt.subplots(2, 1, figsize=(16, 8))
        plt.suptitle(title, y=.97, fontsize=16, horizontalalignment='center')

        # Plot Cluster frequencies
        axes = pltaxes[0]
        axes.set_title('big Cluster')
        if avg_bfreq > 0:
            axes.axhline(avg_bfreq, color='r', linestyle='--', linewidth=2)
        axes.set_ylim(
                (self._platform['freqs']['big'][0] - 100000)/1e3,
                (self._platform['freqs']['big'][-1] + 100000)/1e3
        )
        if len(bfreq) > 0:
            bfreq['frequency'].plot(style=['r-'], ax=axes,
                                    drawstyle='steps-post', alpha=0.4)
        else:
            logging.warn('NO big CPUs frequency events to plot')
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.set_ylabel('MHz')
        axes.grid(True)
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self._trace.analysis.status.plotOverutilized(axes)

        axes = pltaxes[1]
        axes.set_title('LITTLE Cluster')
        if avg_lfreq > 0:
            axes.axhline(avg_lfreq, color='b', linestyle='--', linewidth=2)
        axes.set_ylim(
                (self._platform['freqs']['little'][0] - 100000)/1e3,
                (self._platform['freqs']['little'][-1] + 100000)/1e3
        )
        if len(lfreq) > 0:
            lfreq['frequency'].plot(style=['b-'], ax=axes,
                                    drawstyle='steps-post', alpha=0.4)
        else:
            logging.warn('NO LITTLE CPUs frequency events to plot')
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.set_ylabel('MHz')
        axes.grid(True)
        self._trace.analysis.status.plotOverutilized(axes)

        # Save generated plots into datadir
        figname = '{}/{}cluster_freqs.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

        logging.info('LITTLE cluster average frequency: %.3f GHz',
                     avg_lfreq/1e3)
        logging.info('big    cluster average frequency: %.3f GHz',
                     avg_bfreq/1e3)

        return (avg_lfreq/1e3, avg_bfreq/1e3)

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
        if not self._trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return
        if not self._trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, plot DISABLED!')
            return

        if cpus is None:
            # Generate plots only for available CPUs
            cpufreq_data = self._dfg_trace_event('cpu_frequency')
            _cpus = range(cpufreq_data.cpu.max()+1)
        else:
            _cpus = listify(cpus)

        # Split between big and LITTLE CPUs ordered from higher to lower ID
        _cpus.reverse()
        big_cpus = [c for c in _cpus if c in self._platform['clusters']['big']]
        little_cpus = [c for c in _cpus if c in
                       self._platform['clusters']['little']]
        _cpus = big_cpus + little_cpus

        # Precompute active and total time for each CPU
        residencies = []
        xmax = 0.0
        for cpu in _cpus:
            res = self._getCPUFrequencyResidency(cpu)
            residencies.append(ResidencyData('CPU{}'.format(cpu), res))

            max_time = res.total.max().values[0]
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
        if not self._trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return
        if not self._trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, plot DISABLED!')
            return

        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU
        if not self._trace.freq_coherency:
            logging.warn('Cluster frequency is not coherent, plot DISABLED!')
            return

        # Sanitize clusters
        if clusters is None:
            _clusters = self._platform['clusters'].keys()
        else:
            _clusters = listify(clusters)

        # Precompute active and total time for each cluster
        residencies = []
        xmax = 0.0
        for cluster in _clusters:
            res = self._getClusterFrequencyResidency(
                self._platform['clusters'][cluster.lower()])
            residencies.append(ResidencyData('{} Cluster'.format(cluster),
                                             res))

            max_time = res.total.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plotFrequencyResidency(residencies, 'cluster', xmax, pct, active)

###############################################################################
# Utility Methods
###############################################################################

    @memoized
    def _getCPUActiveSignal(self, cpu):
        """
        Build a square wave representing the active (i.e. non-idle) CPU time,
        i.e.:
            cpu_active[t] == 1 if at least one CPU is reported to be
                               non-idle by CPUFreq at time t
            cpu_active[t] == 0 otherwise

        :param cpu: CPU ID
        :type cpu: int
        """
        if not self._trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '
                         'cannot compute CPU active signal!')
            return None

        idle_df = self._dfg_trace_event('cpu_idle')
        cpu_states = idle_df[idle_df.cpu_id == cpu].state

        cpu_active = cpu_states.apply(
            lambda s: 1 if s == NON_IDLE_STATE else 0
        )

        start_time = 0.0
        if not self._trace.ftrace.normalized_time:
            start_time = self._trace.ftrace.basetime
        if cpu_active.index[0] != start_time:
            entry_0 = pd.Series(cpu_active.iloc[0] ^ 1, index=[start_time])
            cpu_active = pd.concat([entry_0, cpu_active])

        return cpu_active

    @memoized
    def _getClusterActiveSignal(self, cluster):
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
            cpu_active[cpu] = self._getCPUActiveSignal(cpu)

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

    def _getClusterFrequencyResidency(self, cluster):
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
        if not self._trace.hasEvents('cpu_frequency'):
            logging.warn('Events [cpu_frequency] not found, '
                         'frequency residency computation not possible!')
            return None
        if not self._trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '
                         'frequency residency computation not possible!')
            return None

        if isinstance(cluster, str):
            try:
                _cluster = self._platform['clusters'][cluster.lower()]
            except KeyError:
                logging.warn('%s cluster not found!', cluster)
                return None
        else:
            _cluster = listify(cluster)

        freq_df = self._dfg_trace_event('cpu_frequency')
        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU. This assumption is verified
        # by the Trace module when parsing the trace.
        if len(_cluster) > 1 and not self._trace.freq_coherency:
            logging.warn('Cluster frequency is NOT coherent,'
                         'cannot compute residency!')
            return None
        cluster_freqs = freq_df[freq_df.cpu == _cluster[0]].frequency
        cluster_freqs = self._trace._cropToXTimeRange(cluster_freqs)

        # Compute TOTAL Time
        time_intervals = cluster_freqs.index[1:] - cluster_freqs.index[:-1]
        total_time = pd.DataFrame({
            'time': time_intervals,
            'frequency': [f/1000.0 for f in cluster_freqs.iloc[:-1]]
        })
        total_time = total_time.groupby(['frequency']).sum()

        # Compute ACTIVE Time
        cluster_active = self._getClusterActiveSignal(_cluster)
        cluster_active = self._trace._cropToXTimeRange(cluster_active)

        # In order to compute the active time spent at each frequency we
        # multiply 2 square waves:
        # - cluster_active, a square wave of the form:
        #     cluster_active[t] == 1 if at least one CPU is reported to be
        #                            non-idle by CPUFreq at time t
        #     cluster_active[t] == 0 otherwise
        # - freq_active, square wave of the form:
        #     freq_active[t] == 1 if at time t the frequency is f
        #     freq_active[t] == 0 otherwise
        available_freqs = sorted(cluster_freqs.unique())
        new_idx = sorted(cluster_freqs.index.tolist() +
                         cluster_active.index.tolist())
        cluster_freqs = cluster_freqs.reindex(new_idx, method='ffill')
        cluster_active = cluster_active.reindex(new_idx, method='ffill')
        nonidle_time = []
        for f in available_freqs:
            freq_active = cluster_freqs.apply(lambda x: 1 if x == f else 0)
            active_t = cluster_active * freq_active
            # Compute total time by integrating the square wave
            nonidle_time.append(self._trace.integrate_square_wave(active_t))

        active_time = pd.DataFrame({'time': nonidle_time},
                                   index=[f/1000.0 for f in available_freqs])
        active_time.index.name = 'frequency'
        return ResidencyTime(total_time, active_time)

    def _getCPUFrequencyResidency(self, cpu):
        """
        Get a DataFrame with per-CPU frequency residency, i.e. amount of
        time CPU `cpu` spent at each frequency. Both total and active times
        will be computed.

        :param cpu: CPU ID
        :type cpu: int

        :returns: namedtuple(ResidencyTime) - tuple of total and active time
            dataframes
        """
        return self._getClusterFrequencyResidency(cpu)

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

        :param is_last: if True this is the last plot
        :type is_last: bool

        :param xmax: x-axes higher bound
        :param xmax: double

        :param title: title of this subplot
        :type title: str
        """
        yrange = 0.4 * max(6, len(residency.total)) * n_plots
        residency.total.plot.barh(ax=axes, color='g',
                                  legend=False, figsize=(16, yrange))
        residency.active.plot.barh(ax=axes, color='r',
                                   legend=False, figsize=(16, yrange))

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
            {label: residency_df.time.apply(lambda x: x*100/duration)},
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
            if data.residency is None:
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
                  .format(self._trace.plots_dir,
                          self._trace.plots_prefix,
                          entity_name, figtype)
        pl.savefig(figname, bbox_inches='tight')

# vim :set tabstop=4 shiftwidth=4 expandtab
