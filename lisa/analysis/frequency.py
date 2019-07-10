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

import os
import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import memoized
from lisa.trace import requires_events
from lisa.datautils import series_integrate

class FrequencyAnalysis(TraceAnalysisBase):
    """
    Support for plotting Frequency Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'frequency'

    def __init__(self, trace):
        super(FrequencyAnalysis, self).__init__(trace)

    @requires_events('cpu_frequency')
    def _check_freq_domain_coherency(self, cpus=None):
        """
        Check that all CPUs of a given frequency domain have the same frequency
        transitions.

        :param cpus: CPUs to take into account. All other CPUs are ignored.
            If `None`, all CPUs will be checked.
        :type cpus: list(int) or None
        """
        domains = self.trace.plat_info['freq-domains']
        if cpus is None:
            cpus = list(itertools.chain.from_iterable(domains))

        if len(cpus) < 2:
            return

        df = self.trace.df_events('cpu_frequency')

        for domain in domains:
            # restrict the domain to what we care. Other CPUs may have garbage
            # data, but the caller is not going to look at it anyway.
            domain = set(domain) - set(cpus)
            if len(domain) < 2:
                continue

            # Get the frequency column for each CPU in the domain
            freq_columns = [
                # drop the index since we only care about the transitions, and
                # not when they happened
                df[df['cpu'] == cpu]['frequency'].reset_index(drop=True)
                for cpu in domain
            ]

            # Check that all columns are equal. If they are not, that means that
            # at least one CPU has a frequency transition that is different
            # from another one in the same domain, which is highly suspicious
            ref = freq_columns[0]
            for col in freq_columns:
                # If the trace started in the middle of a group of transitions,
                # ignore that transition by shifting and re-test
                if not (ref.equals(col) or ref[:-1].equals(col.shift()[1:])):
                    raise ValueError('Frequencies of CPUs in the freq domain {} are not coherent'.format(cpus))

    @memoized
    @requires_events('cpu_frequency', 'cpu_idle')
    def _get_frequency_residency(self, cpus):
        """
        Get a DataFrame with per cluster frequency residency, i.e. amount of
        time spent at a given frequency in each cluster.

        :param cpus: A tuple of CPU IDs
        :type cpus: tuple(int)

        :returns: A :class:`pandas.DataFrame` with:

          * A ``total_time`` column (the total time spent at a frequency)
          * A ``active_time`` column (the non-idle time spent at a frequency)
        """
        freq_df = self.trace.df_events('cpu_frequency')
        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU.
        self._check_freq_domain_coherency(cpus)

        cluster_freqs = freq_df[freq_df.cpu == cpus[0]]

        # Compute TOTAL Time
        cluster_freqs = self.trace.add_events_deltas(
            cluster_freqs, col_name="total_time", inplace=False)
        time_df = cluster_freqs[["total_time", "frequency"]].groupby(["frequency"]).sum()

        # Compute ACTIVE Time
        cluster_active = self.trace.analysis.idle.signal_cluster_active(cpus)

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
        for freq in available_freqs:
            freq_active = cluster_freqs.frequency.apply(lambda x: 1 if x == freq else 0)
            active_t = cluster_freqs.active * freq_active
            # Compute total time by integrating the square wave
            nonidle_time.append(series_integrate(active_t))

        time_df["active_time"] = pd.DataFrame(index=available_freqs, data=nonidle_time)
        return time_df

    @_get_frequency_residency.used_events
    def df_cpu_frequency_residency(self, cpu):
        """
        Get per-CPU frequency residency, i.e. amount of
        time CPU `cpu` spent at each frequency.

        :param cpu: CPU ID
        :type cpu: int

        :returns: A :class:`pandas.DataFrame` with:

          * A ``total_time`` column (the total time spent at a frequency)
          * A ``active_time`` column (the non-idle time spent at a frequency)
        """
        if not isinstance(cpu, int):
            raise TypeError('Input CPU parameter must be an integer')

        return self._get_frequency_residency((cpu,))

    @_get_frequency_residency.used_events
    def df_domain_frequency_residency(self, cpu):
        """
        Get per-frequency-domain frequency residency, i.e. amount of time each
        domain at each frequency.

        :param cpu: Any CPU of the domain to analyse
        :type cpu: int

        :returns: A :class:`pandas.DataFrame` with:

          * A ``total_time`` column (the total time spent at a frequency)
          * A ``active_time`` column (the non-idle time spent at a frequency)
        """
        domains = self.trace.plat_info['freq-domains']
        for domain in domains:
            if cpu in domain:
                return self._get_frequency_residency(tuple(domain))

    @requires_events('cpu_frequency')
    def df_cpu_frequency_transitions(self, cpu):
        """
        Compute number of frequency transitions of a given CPU.

        :param cpu: a CPU ID
        :type cpu: int

        :returns: A :class:`pandas.DataFrame` with:

          * A ``transitions`` column (the number of frequency transitions)
        """

        freq_df = self.trace.df_events('cpu_frequency')
        cpu_freqs = freq_df[freq_df.cpu == cpu].frequency

        # Remove possible duplicates (example: when devlib sets trace markers
        # a cpu_frequency event is triggered that can generate a duplicate)
        cpu_freqs = cpu_freqs.loc[cpu_freqs.shift(-1) != cpu_freqs]
        transitions = cpu_freqs.value_counts()

        transitions.name = "transitions"
        transitions.sort_index(inplace=True)

        return pd.DataFrame(transitions)

    @df_cpu_frequency_transitions.used_events
    def df_cpu_frequency_transition_rate(self, cpu):
        """
        Compute frequency transition rate of a given CPU.

        :param cpu: a CPU ID
        :type cpu: int

        :returns: A :class:`pandas.DataFrame` with:

          * A ``transitions`` column (the number of frequency transitions per second)
        """
        transitions = self.df_cpu_frequency_transitions(cpu)
        if transitions is None:
            return None

        return transitions.apply(
            lambda x: x / (self.trace.end - self.trace.start)
        )

    @requires_events('cpu_frequency')
    def get_average_cpu_frequency(self, cpu):
        """
        Get the average frequency for a given CPU

        :param cpu: The CPU to analyse
        :type cpu: int
        """
        df = self.trace.df_events('cpu_frequency')
        df = df[df.cpu == cpu]

        # We can't use the pandas average because it's not weighted by
        # time spent in each frequency, so we have to craft our own.
        df = self.trace.add_events_deltas(df, inplace=False)
        timespan = self.trace.end - self.trace.start

        return (df['frequency'] * df['delta']).sum() / timespan

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
        freq = self.trace.get_peripheral_clock_effective_rate(clk)
        if freq is None or freq.empty:
            self.get_logger().warning('no peripheral clock events found for clock')
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
            freq_axis.hlines(set_rate.iloc[-1], set_rate.index[-1], self.trace.end, linestyle='--', color='b', alpha=0.4)
        else:
            self.get_logger().warning('No clock_set_rate events to plot')

        # Plot frequency information (effective rate)
        eff_rate = freq['effective_rate'].dropna()
        if len(eff_rate) > 0 and eff_rate.max() > 0:
            rate_axis_lib = max(rate_axis_lib, eff_rate.max())
            eff_rate.plot(style=['b-'], ax=freq_axis, drawstyle='steps-post', alpha=1.0, label="Effective rate (with on/off)")
            freq_axis.hlines(eff_rate.iloc[-1], eff_rate.index[-1], self.trace.end, linestyle='-', color='b', alpha=1.0)
        else:
            self.get_logger().warning('No effective frequency events to plot')

        freq_axis.set_ylim(0, rate_axis_lib * 1.1)
        freq_axis.set_xlim(self.trace.start, self.trace.end)
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
            indet_range_max = self.trace.end
        else:
            indet_range_max = indeterminate.index[0]
        state_axis.hlines(0, 0, indet_range_max, linewidth = 1.0, label='indeterminate clock state', linestyle='--')
        state_axis.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=3, mode='expand')
        state_axis.set_yticks([])
        state_axis.set_xlabel('seconds')
        state_axis.set_xlim(self.trace.start, self.trace.end)

        figname = os.path.join(self.trace.plots_dir, '{}{}.png'.format(self.trace.plots_prefix, clk))
        plt.savefig(figname, bbox_inches='tight')


    @TraceAnalysisBase.plot_method()
    @requires_events('cpu_frequency')
    def plot_cpu_frequencies(self, cpu, axis, local_fig, average=True):
        """
        Plot frequency for the specified CPU

        :param cpu: The CPU for which to plot frequencies
        :type cpus: int

        :param average: If ``True``, add a horizontal line which is the
            frequency average.
        :type average: bool

        If ``sched_overutilized`` events are available, the plots will also
        show the intervals of time where the system was overutilized.
        """
        logger = self.get_logger()

        df = self.trace.df_events('cpu_frequency')
        df = df[df.cpu == cpu]

        if "freqs" in self.trace.plat_info:
            frequencies = self.trace.plat_info['freqs'][cpu]
        else:
            logger.info("Estimating CPU%s frequencies from trace", cpu)
            frequencies = sorted(list(df.frequency.unique()))
            logger.debug("Estimated frequencies: %s", frequencies)

        avg = self.get_average_cpu_frequency(cpu)
        logger.info(
            "Average frequency for CPU{} : {:.3f} GHz".format(cpu, avg/1e6))

        df['frequency'].plot(ax=axis, drawstyle='steps-post')

        if average and avg > 0:
            axis.axhline(avg, color=self.get_next_color(axis), linestyle='--',
                         label="average")

        plot_overutilized = self.trace.analysis.status.plot_overutilized
        if self.trace.has_events(plot_overutilized.used_events):
            plot_overutilized(axis=axis)

        axis.set_ylim(frequencies[0] * 0.9, frequencies[-1] * 1.1)
        axis.set_xlim(self.trace.start, self.trace.end)

        axis.set_ylabel('Frequency (Hz)')
        axis.set_xlabel('Time')

        axis.set_title('Frequency of CPU{}'.format(cpu))
        axis.grid(True)
        axis.legend()

    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cpu_frequencies.used_events
    def plot_domain_frequencies(self, **kwargs):
        """
        Plot frequency trend for all frequency domains.

        If ``sched_overutilized`` events are available, the plots will also show
        the intervals of time where the cluster was overutilized.
        """
        domains = self.trace.plat_info['freq-domains']

        def plotter(axes, local_fig):
            for idx, domain in enumerate(domains):
                axis = axes[idx] if len(domains) > 1 else axes

                self.plot_cpu_frequencies(domain[0], axis=axis)

                axis.set_title('Frequencies of CPUS {}'.format(domain))
                axis.set_xlim(self.trace.start, self.trace.end)

        return self.do_plot(plotter, nrows=len(domains), sharex=True, **kwargs)

    @TraceAnalysisBase.plot_method(return_axis=True)
    @df_cpu_frequency_residency.used_events
    def plot_cpu_frequency_residency(self, cpu, pct=False, **kwargs):
        """
        Plot per-CPU frequency residency.

        :param cpu: The CPU to generate the plot for
        :type cpu: int

        :param pct: Plot residencies in percentage
        :type pct: bool
        """

        residency_df = self.df_cpu_frequency_residency(cpu)

        total_df = residency_df.total_time
        active_df = residency_df.active_time

        if pct:
            total_df = total_df * 100 / total_df.sum()
            active_df = active_df * 100 / active_df.sum()

        def plotter(axes, local_fig):
            total_df.plot.barh(ax=axes[0])
            axes[0].set_title("CPU{} total frequency residency".format(cpu))

            active_df.plot.barh(ax=axes[1])
            axes[1].set_title("CPU{} active frequency residency".format(cpu))

            for axis in axes:
                if pct:
                    axis.set_xlabel("Time share (%)")
                else:
                    axis.set_xlabel("Time (s)")

                axis.set_ylabel("Frequency (Hz)")
                axis.grid(True)

        return self.do_plot(plotter, nrows=2, **kwargs)

    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cpu_frequency_residency.used_events
    def plot_domain_frequency_residency(self, pct=False, filepath=None, **kwargs):
        """
        Plot the frequency residency for all frequency domains.

        :param pct: Plot residencies in percentage
        :type pct: bool
        """
        domains = self.trace.plat_info['freq-domains']

        def plotter(axes, local_fig):
            for idx, domain in enumerate(domains):
                local_axes = axes[2 * idx : 2 * (idx + 1)]

                self.plot_cpu_frequency_residency(domain[0], filepath, pct,
                                                                    local_axes)
                for axis in local_axes:
                    title = axis.get_title()
                    axis.set_title(title.replace(
                        "CPU{}".format(domain[0]), "CPUs {}".format(domain)))

        return self.do_plot(plotter, nrows=2*len(domains), sharex=True, filepath=filepath, **kwargs)

    @TraceAnalysisBase.plot_method()
    @df_cpu_frequency_transitions.used_events
    def plot_cpu_frequency_transitions(self, cpu, axis, local_fig, pct=False):
        """
        Plot frequency transitions count of the specified CPU

        :param cpu: The CPU to genererate the plot for
        :type cpu: int

        :param pct: Plot frequency transitions in percentage
        :type pct: bool
        """

        df = self.df_cpu_frequency_transitions(cpu)

        if pct:
            df = df * 100 / df.sum()

        df["transitions"].plot.barh(ax=axis)
        axis.set_title('Frequency transitions of CPU{}'.format(cpu))

        if pct:
            axis.set_xlabel("Transitions share (%)")
        else:
            axis.set_xlabel("Transition count")

        axis.set_ylabel("Frequency (Hz)")
        axis.grid(True)


    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cpu_frequency_transitions.used_events
    def plot_domain_frequency_transitions(self, pct=False, **kwargs):
        """
        Plot frequency transitions count for all frequency domains

        :param pct: Plot frequency transitions in percentage
        :type pct: bool
        """
        domains = self.trace.plat_info['freq-domains']

        def plotter(axes, local_fig):
            for domain, axis in zip(domains, axes):
                self.plot_cpu_frequency_transitions(
                    cpu=domain[0],
                    pct=pct,
                    axis=axis,
                    always_save=False,
                )

                title = axis.get_title()
                axis.set_title(title.replace("CPU{}".format(domain[0]),
                                                    "CPUs {}".format(domain)))


        return self.do_plot(plotter, nrows=len(domains), **kwargs)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
