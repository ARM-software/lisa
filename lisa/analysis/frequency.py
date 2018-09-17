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
import operator
import os
import pandas as pd
import pylab as pl

from lisa.analysis.base import AnalysisBase, ResidencyTime, ResidencyData
from bart.common.Utils import area_under_curve
from devlib.utils.misc import memoized
from matplotlib.ticker import FuncFormatter
from trappy.utils import listify

class FrequencyAnalysis(AnalysisBase):
    """
    Support for plotting Frequency Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'frequency'

    def __init__(self, trace):
        super(FrequencyAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    def df_cpu_frequency_residency(self, cpu, total=True):
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

        :raises: TypeError
        """
        if not isinstance(cpu, int):
            raise TypeError('Input CPU parameter must be an integer')

        residency = self._get_frequency_residency(cpu)
        if not residency:
            return None
        if total:
            return residency.total
        return residency.active

    def df_cluster_frequency_residency(self, cluster, total=True):
        """
        Get per-Cluster frequency residency, i.e. amount of time CLUSTER
        `cluster` spent at each frequency.

        :param cluster: this can be either a list of CPU IDs belonging to a
            cluster or the cluster name as specified in the platform
            description
        :type cluster: str or list(int)

        :param total: if true returns the "total" time, otherwise the "active"
                      time is returned
        :type total: bool

        :returns: :mod:`pandas.DataFrame` - "total" or "active" time residency
                  at each frequency.

        :raises: KeyError
        """
        if isinstance(cluster, str):
            try:
                residency = self._get_frequency_residency(
                    self._trace.platform['clusters'][cluster.lower()]
                )
            except KeyError:
                self._log.warning(
                    'Platform descriptor has not a cluster named [%s], '
                    'plot disabled!', cluster
                )
                return None
        else:
            residency = self._get_frequency_residency(cluster)
        if not residency:
            return None
        if total:
            return residency.total
        return residency.active

    def df_cpu_frequency_transitions(self, cpu):
        """
        Compute number of frequency transitions of a given CPU.

        Requires cpu_frequency events to be available in the trace.

        :param cpu: a CPU ID
        :type cpu: int

        :returns: :mod:`pandas.DataFrame` - number of frequency transitions
        """
        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warn('Events [cpu_frequency] not found, '
                           'frequency data not available')
            return None

        freq_df = self._trace.df_events('cpu_frequency')
        cpu_freqs = freq_df[freq_df.cpu == cpu].frequency

        # Remove possible duplicates (example: when devlib sets trace markers
        # a cpu_frequency event is triggered that can generate a duplicate)
        cpu_freqs = cpu_freqs.loc[cpu_freqs.shift(-1) != cpu_freqs]
        transitions = cpu_freqs.value_counts()
        # Convert frequencies to MHz
        transitions.index = transitions.index / 1000
        transitions.name = "transitions"
        transitions.sort_index(inplace=True)
        return pd.DataFrame(transitions)

    def df_cpu_frequency_transition_rate(self, cpu):
        """
        Compute frequency transition rate of a given CPU.
        Requires cpu_frequency events to be available in the trace.

        :param cpu: a CPU ID
        :type cpu: int

        :returns: :mod:`pandas.DataFrame - number of frequency transitions per
            second
        """
        transitions = self.df_cpu_frequency_transitions(cpu)
        if transitions is None:
            return None

        return transitions.apply(
            lambda x: x / (self._trace.x_max - self._trace.x_min)
        )

###############################################################################
# Plotting Methods
###############################################################################

    def plot_peripheral_clock(self, clk, title='Peripheral Frequency'):
        """
        Produce graph plotting the frequency of a particular peripheral clock

        :param title: The title for the chart
        :type  title: str

        :param clk: The clk name to chart
        :type  clk: str

        :raises: KeyError
        """
        freq = self._trace.getPeripheralClockEffectiveRate(clk)
        if freq is None or freq.empty:
            self._log.warning('no peripheral clock events found for clock')
            return

        fig = plt.figure(figsize=(16,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])
        freq_axis = plt.subplot(gs[0])
        state_axis = plt.subplot(gs[1])
        plt.suptitle(title, y=.97, fontsize=16, horizontalalignment='center')


        # Plot frequency information (set rate)
        freq_axis.set_title("Clock frequency for " + clk)
        set_rate = freq['rate'].dropna()

        rate_axis_lib = 0
        if len(set_rate) > 0:
            rate_axis_lib = set_rate.max()
            set_rate.plot(style=['b--'], ax=freq_axis, drawstyle='steps-post', alpha=0.4, label="clock_set_rate value")
            freq_axis.hlines(set_rate.iloc[-1], set_rate.index[-1], self._trace.x_max, linestyle='--', color='b', alpha=0.4)
        else:
            self._log.warning('No clock_set_rate events to plot')

        # Plot frequency information (effective rate)
        eff_rate = freq['effective_rate'].dropna()
        if len(eff_rate) > 0 and eff_rate.max() > 0:
            rate_axis_lib = max(rate_axis_lib, eff_rate.max())
            eff_rate.plot(style=['b-'], ax=freq_axis, drawstyle='steps-post', alpha=1.0, label="Effective rate (with on/off)")
            freq_axis.hlines(eff_rate.iloc[-1], eff_rate.index[-1], self._trace.x_max, linestyle='-', color='b', alpha=1.0)
        else:
            self._log.warning('No effective frequency events to plot')

        freq_axis.set_ylim(0, rate_axis_lib * 1.1)
        freq_axis.set_xlim(self._trace.x_min, self._trace.x_max)
        freq_axis.set_xlabel('')
        freq_axis.grid(True)
        freq_axis.legend()
        def mhz(x, pos):
            return '%1.2f MHz' % (x*1e-6)
        freq_axis.get_yaxis().set_major_formatter(FuncFormatter(mhz))

        on = freq[freq.state == 1]
        state_axis.hlines([0] * len(on),
                          on['start'], on['start'] + on['len'],
                          linewidth = 10.0, label='clock on', color='green')
        off = freq[freq.state == 0]
        state_axis.hlines([0] * len(off),
                          off['start'], off['start'] + off['len'],
                          linewidth = 10.0, label='clock off', color='red')


        # Plot time period that the clock state was unknown from the trace
        indeterminate = pd.concat([on, off]).sort_index()
        if indeterminate.empty:
            indet_range_max = self._trace.x_max
        else:
            indet_range_max = indeterminate.index[0]
        state_axis.hlines(0, 0, indet_range_max, linewidth = 1.0, label='indeterminate clock state', linestyle='--')
        state_axis.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=3, mode='expand')
        state_axis.set_yticks([])
        state_axis.set_xlabel('seconds')
        state_axis.set_xlim(self._trace.x_min, self._trace.x_max)

        figname = os.path.join(self._trace.plots_dir, '{}{}.png'.format(self._trace.plots_prefix, clk))
        pl.savefig(figname, bbox_inches='tight')

    def plot_cluster_frequencies(self, title='Clusters Frequencies'):
        """
        Plot frequency trend for all clusters. If sched_overutilized events are
        available, the plots will also show the intervals of time where the
        cluster was overutilized.

        :param title: user-defined plot title
        :type title: str
        """
        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warning('Events [cpu_frequency] not found, plot DISABLED!')
            return
        df = self._trace.df_events('cpu_frequency')

        pd.options.mode.chained_assignment = None

        # Extract LITTLE and big clusters frequencies
        # and scale them to [MHz]
        if self._little_cpus:
            lfreq = df[df.cpu == self._little_cpus[-1]]
            lfreq['frequency'] = lfreq['frequency']/1e3
        else:
            lfreq = []
        if self._big_cpus:
            bfreq = df[df.cpu == self._big_cpus[-1]]
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
                (self._trace.platform['freqs']['big'][0] - 100000)/1e3,
                (self._trace.platform['freqs']['big'][-1] + 100000)/1e3
        )
        if len(bfreq) > 0:
            bfreq['frequency'].plot(style=['r-'], ax=axes,
                                    drawstyle='steps-post', alpha=0.4)
        else:
            self._log.warning('NO big CPUs frequency events to plot')
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.set_ylabel('MHz')
        axes.grid(True)
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self._trace.analysis.status.plot_overutilized(axes)

        axes = pltaxes[1]
        axes.set_title('LITTLE Cluster')
        if avg_lfreq > 0:
            axes.axhline(avg_lfreq, color='b', linestyle='--', linewidth=2)
        axes.set_ylim(
                (self._trace.platform['freqs']['little'][0] - 100000)/1e3,
                (self._trace.platform['freqs']['little'][-1] + 100000)/1e3
        )
        if len(lfreq) > 0:
            lfreq['frequency'].plot(style=['b-'], ax=axes,
                                    drawstyle='steps-post', alpha=0.4)
        else:
            self._log.warning('NO LITTLE CPUs frequency events to plot')
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.set_ylabel('MHz')
        axes.grid(True)
        self._trace.analysis.status.plot_overutilized(axes)

        # Save generated plots into datadir
        figname = '{}/{}cluster_freqs.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

        self._log.info('LITTLE cluster average frequency: %.3f GHz',
                       avg_lfreq/1e3)
        self._log.info('big    cluster average frequency: %.3f GHz',
                       avg_bfreq/1e3)

        return (avg_lfreq/1e3, avg_bfreq/1e3)

    def plot_cpu_frequencies(self, cpus=None):
        """
        Plot frequency for the specified CPUs (or all if not specified).
        If sched_overutilized events are available, the plots will also show the
        intervals of time where the system was overutilized.

        The generated plots are also saved as PNG images under the folder
        specified by the `plots_dir` parameter of :class:`Trace`.

        :param cpus: the list of CPUs to plot, if None it generate a plot
                     for each available CPU
        :type cpus: int or list(int)

        :return: a dictionary of average frequency for each CPU.
        """
        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warning('Events [cpu_frequency] not found, plot DISABLED!')
            return
        df = self._trace.df_events('cpu_frequency')

        if cpus is None:
            # Generate plots only for available CPUs
            cpus = list(range(df.cpu.max()+1))
        else:
            # Generate plots only specified CPUs
            cpus = listify(cpus)

        chained_assignment = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None

        freq = {}
        for cpu_id in listify(cpus):
            # Extract CPUs' frequencies and scale them to [MHz]
            _df = df[df.cpu == cpu_id]
            if _df.empty:
                self._log.warning('No [cpu_frequency] events for CPU%d, '
                                  'plot DISABLED!', cpu_id)
                continue
            _df['frequency'] = _df.frequency / 1e3

            # Compute AVG frequency for this CPU
            avg_freq = 0
            if len(_df) > 1:
                timespan = _df.index[-1] - _df.index[0]
                avg_freq = area_under_curve(_df['frequency'], method='rect') / timespan

            # Store DF for plotting
            freq[cpu_id] = {
                'df'  : _df,
                'avg' : avg_freq,
            }

        pd.options.mode.chained_assignment = chained_assignment

        plots_count = len(freq)
        if not plots_count:
            return

        # Setup CPUs plots
        fig, pltaxes = plt.subplots(len(freq), 1, figsize=(16, 4 * plots_count))

        avg_freqs = {}
        for plot_idx, cpu_id in enumerate(freq):

            # CPU frequencies and average value
            _df = freq[cpu_id]['df']
            _avg = freq[cpu_id]['avg']

            # Plot average frequency
            try:
                axes = pltaxes[plot_idx]
            except TypeError:
                axes = pltaxes
            axes.set_title('CPU{:2d} Frequency'.format(cpu_id))
            axes.axhline(_avg, color='r', linestyle='--', linewidth=2)

            # Set plot limit based on CPU min/max frequencies
            if 'clusters' in self._trace.platform:
                for cluster,cpus in self._trace.platform['clusters'].items():
                    if cpu_id not in cpus:
                        continue
                    freqs = self._trace.platform['freqs'][cluster]
                    break
            else:
                freqs = df['frequency'].unique()

            axes.set_ylim((min(freqs) - 100000) / 1e3,
                          (max(freqs) + 100000) / 1e3)

            # Plot CPU frequency transitions
            _df['frequency'].plot(style=['r-'], ax=axes,
                                  drawstyle='steps-post', alpha=0.4)

            # Plot overutilzied regions (if signal available)
            self._trace.analysis.status.plot_overutilized(axes)

            # Finalize plot
            axes.set_xlim(self._trace.x_min, self._trace.x_max)
            axes.set_ylabel('MHz')
            axes.grid(True)
            if plot_idx + 1 < plots_count:
                axes.set_xticklabels([])
                axes.set_xlabel('')

            avg_freqs[cpu_id] = _avg/1e3
            self._log.info('CPU%02d average frequency: %.3f GHz',
                           cpu_id, avg_freqs[cpu_id])

        # Save generated plots into datadir
        figname = '{}/{}cpus_freqs.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

        return avg_freqs


    def plot_cpu_frequency_residency(self, cpus=None, pct=False, active=False):
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
            self._log.warning('Events [cpu_frequency] not found, plot DISABLED!')
            return
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, plot DISABLED!')
            return

        if cpus is None:
            # Generate plots only for available CPUs
            cpufreq_data = self._trace.df_events('cpu_frequency')
            _cpus = list(range(cpufreq_data.cpu.max()+1))
        else:
            _cpus = listify(cpus)

        # Split between big and LITTLE CPUs ordered from higher to lower ID
        _cpus.reverse()
        big_cpus = [c for c in _cpus if c in self._big_cpus]
        little_cpus = [c for c in _cpus if c in self._little_cpus]
        _cpus = big_cpus + little_cpus

        # Precompute active and total time for each CPU
        residencies = []
        xmax = 0.0
        for cpu in _cpus:
            res = self._get_frequency_residency(cpu)
            residencies.append(ResidencyData('CPU{}'.format(cpu), res))

            max_time = res.total.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plot_frequency_residency(residencies, 'cpu', xmax, pct, active)

    def plot_cluster_frequency_residency(self, clusters=None,
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
            self._log.warning('Events [cpu_frequency] not found, plot DISABLED!')
            return
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, plot DISABLED!')
            return
        if 'clusters' not in self._trace.platform:
            self._log.warning('No platform cluster info. Plot DISABLED!')
            return

        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU
        if not self._trace.freq_coherency:
            self._log.warning('Cluster frequency is not coherent, plot DISABLED!')
            return

        # Sanitize clusters
        if clusters is None:
            _clusters = list(self._trace.platform['clusters'].keys())
        else:
            _clusters = listify(clusters)

        # Precompute active and total time for each cluster
        residencies = []
        xmax = 0.0
        for cluster in _clusters:
            res = self._get_frequency_residency(
                self._trace.platform['clusters'][cluster.lower()])
            residencies.append(ResidencyData('{} Cluster'.format(cluster),
                                             res))

            max_time = res.total.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plot_frequency_residency(residencies, 'cluster', xmax, pct, active)

    def plot_cpu_frequency_transitions(self, cpus=None, pct=False):
        """
        Plot frequency transitions count of the specified CPUs (or all if not
        specified).

        Requires cpu_frequency events to be available in the trace.

        :param cpus: list of CPU IDs (all CPUs by default)
        :type clusters: int or list(int)

        :param pct: plot frequency transitions in percentage
        :type pct: bool
        """
        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return
        df = self._trace.df_events('cpu_frequency')

        if cpus is None:
            _cpus = list(range(df.cpu.max() + 1))
        else:
            _cpus = listify(cpus)

        n_plots = len(_cpus)
        gs = gridspec.GridSpec(n_plots, 1)
        fig = plt.figure()

        # Precompute frequency transitions
        transitions = {}
        xmax = 0
        for cpu_id in _cpus:
            t = self.df_cpu_frequency_transitions(cpu_id)

            if pct:
                tot = t.transitions.sum()
                t = t.apply(lambda x: x * 100.0 / tot)

            transitions[cpu_id] = t
            max_cnt = t.transitions.max()
            if xmax < max_cnt: xmax = max_cnt

        if pct:
            yrange = 0.4 * max(6, len(t)) * n_plots
            figtype = "_pct"
            labeltype = " [%]"
        else:
            yrange = 3 * n_plots
            figtype = ""
            labeltype = ""

        for idx, cpu_id in enumerate(_cpus):
            t = transitions[cpu_id]

            axes = fig.add_subplot(gs[idx])
            if pct:
                t.T.plot.barh(ax=axes, figsize=(16, yrange),
                              stacked=True, title='CPU{}'.format(cpu_id))
                axes.legend(loc='lower center', ncol=7)
                axes.set_xlim(0, 100)
                axes.set_yticklabels([])
            else:
                t.plot.barh(ax=axes, figsize=(16, yrange),
                            color='g', legend=False,
                            title='CPU{}'.format(cpu_id))
                axes.set_xlim(0, xmax*1.05)
                axes.grid(True)
                axes.set_ylabel('Frequency [MHz]')

            if idx+1 < n_plots:
                axes.set_xticklabels([])

        axes = fig.axes[0]
        legend_y = axes.get_ylim()[1]
        axes.annotate('OPP Transitions{}'.format(labeltype),
                      xy=(0, legend_y), xytext=(-50, 25),
                      textcoords='offset points', fontsize=18)
        fig.axes[-1].set_xlabel('Number of transitions{}'.format(labeltype))

        figname = '{}cpu_freq_transitions{}.png'.format(
            self._trace.plots_prefix, figtype)
        fig.savefig(os.path.join(self._trace.plots_dir, figname),
                    bbox_inches='tight')

    def plot_cluster_frequency_transitions(self, clusters=None, pct=False):
        """
        Plot frequency transitions count of the specified clusters (all of them
        is not specified).

        Requires cpu_frequency events to be available in the trace.

        Notice that we assume that frequency is
        scaled at cluster level, therefore we always consider the first CPU of
        a cluster for this computation.

        :param clusters: name of the clusters to be plotted (all of them by
            default)
        :type clusters: str or list(str)

        :param pct: plot frequency transitions in percentage
        :type pct: bool
        """
        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warn('Events [cpu_frequency] not found, plot DISABLED!')
            return

        if not self._trace.platform or 'clusters' not in self._trace.platform:
            self._log.warn('No platform cluster info, plot DISABLED!')
            return

        if clusters is None:
            _clusters = list(self._trace.platform['clusters'].keys())
        else:
            _clusters = listify(clusters)

        n_plots = len(_clusters)
        gs = gridspec.GridSpec(n_plots, 1)
        fig = plt.figure()

        # Precompute frequency transitions
        transitions = {}
        xmax = 0
        for c in _clusters:
            # We assume frequency is scaled at cluster level and we therefore
            # pick information from the first CPU in the cluster.
            cpu_id = self._trace.platform['clusters'][c.lower()][0]
            t = self.df_cpu_frequency_transitions(cpu_id)

            if pct:
                tot = t.transitions.sum()
                t = t.apply(lambda x: x * 100.0 / tot)

            transitions[c] = t
            max_cnt = t.transitions.max()
            if xmax < max_cnt: xmax = max_cnt

        if pct:
            yrange = 0.4 * max(6, len(t)) * n_plots
            figtype = "_pct"
            labeltype = " [%]"
        else:
            yrange = 3 * n_plots
            figtype = ""
            labeltype = ""

        for idx, c in enumerate(_clusters):
            t = transitions[c]

            axes = fig.add_subplot(gs[idx])
            if pct:
                t.T.plot.barh(ax=axes, figsize=(16, yrange),
                              stacked=True, title='{} Cluster'.format(c))
                axes.legend(loc='lower center', ncol=7)
                axes.set_xlim(0, 100)
                axes.set_yticklabels([])
            else:
                t.plot.barh(ax=axes, figsize=(16, yrange),
                            color='g', legend=False,
                            title='{} Cluster'.format(c))
                axes.set_xlim(0, xmax*1.05)
                axes.grid(True)
                axes.set_ylabel('Frequency [MHz]')

            if idx+1 < n_plots:
                axes.set_xticklabels([])

        axes = fig.axes[0]
        legend_y = axes.get_ylim()[1]
        axes.annotate('OPP Transitions{}'.format(labeltype),
                      xy=(0, legend_y), xytext=(-50, 25),
                      textcoords='offset points', fontsize=18)
        fig.axes[-1].set_xlabel('Number of transitions{}'.format(labeltype))

        figname = '{}cluster_freq_transitions{}.png'.format(
            self._trace.plots_prefix, figtype)
        fig.savefig(os.path.join(self._trace.plots_dir, figname),
                    bbox_inches='tight')

###############################################################################
# Utility Methods
###############################################################################

    @memoized
    def _get_frequency_residency(self, cluster):
        """
        Get a DataFrame with per cluster frequency residency, i.e. amount of
        time spent at a given frequency in each cluster.

        :param cluster: this can be either a single CPU ID or a list of CPU IDs
            belonging to a cluster
        :type cluster: int or list(int)

        :returns: namedtuple(ResidencyTime) - tuple of total and active time
            dataframes
        """
        if not self._trace.hasEvents('cpu_frequency'):
            self._log.warning('Events [cpu_frequency] not found, '
                              'frequency residency computation not possible!')
            return None
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, '
                              'frequency residency computation not possible!')
            return None

        _cluster = listify(cluster)

        freq_df = self._trace.df_events('cpu_frequency')
        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU. This assumption is verified
        # by the Trace module when parsing the trace.
        if len(_cluster) > 1 and not self._trace.freq_coherency:
            self._log.warning('Cluster frequency is NOT coherent,'
                              'cannot compute residency!')
            return None
        cluster_freqs = freq_df[freq_df.cpu == _cluster[0]]

        # Compute TOTAL Time
        time_intervals = cluster_freqs.index[1:] - cluster_freqs.index[:-1]
        total_time = pd.DataFrame({
            'time': time_intervals,
            'frequency': [f/1000.0 for f in cluster_freqs.iloc[:-1].frequency]
        })
        total_time = total_time.groupby(['frequency']).sum()

        # Compute ACTIVE Time
        cluster_active = self._trace.getClusterActiveSignal(_cluster)

        # In order to compute the active time spent at each frequency we
        # multiply 2 square waves:
        # - cluster_active, a square wave of the form:
        #     cluster_active[t] == 1 if at least one CPU is reported to be
        #                            non-idle by CPUFreq at time t
        #     cluster_active[t] == 0 otherwise
        # - freq_active, square wave of the form:
        #     freq_active[t] == 1 if at time t the frequency is f
        #     freq_active[t] == 0 otherwise
        available_freqs = sorted(cluster_freqs.frequency.unique())
        cluster_freqs = cluster_freqs.join(
            cluster_active.to_frame(name='active'), how='outer')
        cluster_freqs.fillna(method='ffill', inplace=True)
        nonidle_time = []
        for f in available_freqs:
            freq_active = cluster_freqs.frequency.apply(lambda x: 1 if x == f else 0)
            active_t = cluster_freqs.active * freq_active
            # Compute total time by integrating the square wave
            nonidle_time.append(self._trace.integrate_square_wave(active_t))

        active_time = pd.DataFrame({'time': nonidle_time},
                                   index=[f/1000.0 for f in available_freqs])
        active_time.index.name = 'frequency'
        return ResidencyTime(total_time, active_time)

    def _plot_frequency_residency_abs(self, axes, residency, n_plots,
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

    def _plot_frequency_residency_pct(self, axes, residency_df, label,
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

    def _plot_frequency_residency(self, residencies, entity_name, xmax,
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
                self._plot_frequency_residency_pct(axes, data.residency.active,
                                                data.label, n_plots,
                                                is_first, is_last,
                                                'ACTIVE')
                figtype = "_pct_active"
                continue
            if pct:
                self._plot_frequency_residency_pct(axes, data.residency.total,
                                                data.label, n_plots,
                                                is_first, is_last,
                                                'TOTAL')
                figtype = "_pct_total"
                continue

            self._plot_frequency_residency_abs(axes, data.residency,
                                            n_plots, is_first,
                                            is_last, xmax,
                                            title=data.label)

        figname = '{}/{}{}_freq_residency{}.png'\
                  .format(self._trace.plots_dir,
                          self._trace.plots_prefix,
                          entity_name, figtype)
        pl.savefig(figname, bbox_inches='tight')

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
