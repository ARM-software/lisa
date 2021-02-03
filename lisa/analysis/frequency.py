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

import itertools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import memoized
from lisa.trace import requires_events, requires_one_event_of, CPU, MissingTraceEventError
from lisa.datautils import series_integrate, df_refit_index, series_refit_index, series_deduplicate, df_add_delta, series_mean, df_window


class FrequencyAnalysis(TraceAnalysisBase):
    """
    Support for plotting Frequency Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'frequency'

    @requires_one_event_of('cpu_frequency', 'userspace@cpu_frequency_devlib')
    def df_cpus_frequency(self, signals_init=True):
        """
        Similar to ``trace.df_event('cpu_frequency')``, with
        ``userspace@cpu_frequency_devlib`` support.

        :param signals_init: If ``True``, and initial value for signals will be
            provided. This includes initial value taken outside window
            boundaries and devlib-provided events.

        The ``userspace@cpu_frequency_devlib`` user event is merged in the dataframe if
        it provides earlier values for a CPU.
        """
        def rename(df):
            return df.rename(
                {
                    'cpu_id': 'cpu',
                    'state': 'frequency',
                },
                axis=1,
            )

        def check_empty(df, excep):
            if df.empty:
                raise excep
            else:
                return df

        try:
            df = self.trace.df_event('cpu_frequency', signals_init=signals_init)
        except MissingTraceEventError as e:
            excep = e
            df = pd.DataFrame(columns=['cpu', 'frequency'])
        else:
            excep = None
            df = rename(df)

        if not signals_init:
            return check_empty(df, excep)

        try:
            devlib_df = self.trace.df_event('userspace@cpu_frequency_devlib')
        except MissingTraceEventError as e:
            return check_empty(df, e)
        else:
            devlib_df = rename(devlib_df)

        def groupby_cpu(df):
            return df.groupby('cpu', observed=True, sort=False)

        # Get the initial values for each CPU
        def init_freq(df, from_devlib):
            df = groupby_cpu(df).head(1).copy()
            df['from_devlib'] = from_devlib
            return df

        init_df = init_freq(df, False)
        init_devlib_df = init_freq(devlib_df, True)

        # Get the first frequency for each CPU as given by devlib and cpufreq.
        init_df = pd.concat([init_df, init_devlib_df])
        init_df.sort_index(inplace=True)
        # Get the first value for each CPU
        first_df = groupby_cpu(init_df).head(1)
        # Only keep the ones coming from devlib, as the other ones are already
        # in the cpufreq df
        first_df = first_df[first_df['from_devlib'] == True]
        del first_df['from_devlib']

        df = pd.concat([first_df, df])
        df.sort_index(inplace=True)
        return check_empty(df, None)

    @df_cpus_frequency.used_events
    def df_cpu_frequency(self, cpu, **kwargs):
        """
        Same as :meth:`df_cpus_frequency` but for a single CPU.

        :param cpu: CPU ID to get the frequency of.
        :type cpu: int

        :Variable keyword arguments: Forwarded to :meth:`df_cpus_frequency`.
        """
        df = self.df_cpus_frequency(**kwargs)
        return df[df['cpu'] == cpu]

    @df_cpus_frequency.used_events
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

        df = self.df_cpus_frequency()

        for domain in domains:
            # restrict the domain to what we care. Other CPUs may have garbage
            # data, but the caller is not going to look at it anyway.
            domain = set(domain) & set(cpus)
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
                    raise ValueError(f'Frequencies of CPUs in the freq domain {cpus} are not coherent')

    @TraceAnalysisBase.cache
    @df_cpus_frequency.used_events
    @requires_events('cpu_idle')
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
        freq_df = self.df_cpus_frequency()
        # Assumption: all CPUs in a cluster run at the same frequency, i.e. the
        # frequency is scaled per-cluster not per-CPU. Hence, we can limit the
        # cluster frequencies data to a single CPU.
        self._check_freq_domain_coherency(cpus)

        cluster_freqs = freq_df[freq_df.cpu == cpus[0]]

        # Compute TOTAL Time
        cluster_freqs = df_add_delta(cluster_freqs, col="total_time", window=self.trace.window)
        time_df = cluster_freqs[["total_time", "frequency"]].groupby('frequency', observed=True, sort=False).sum()

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
        cluster_freqs = cluster_freqs.join(
            cluster_active.to_frame(name='active'), how='outer')
        cluster_freqs.fillna(method='ffill', inplace=True)

        # Compute total time by integrating the square wave
        time_df['active_time'] = pd.Series({
            freq: series_integrate(
                cluster_freqs['active'] * (cluster_freqs['frequency'] == freq)
            )
            for freq in cluster_freqs['frequency'].unique()
        })

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
        domains = [
            domain
            for domain in self.trace.plat_info['freq-domains']
            if cpu in domain
        ]

        if not domains:
            raise ValueError(f'The given CPU "{cpu}" does not belong to any domain')
        else:
            domain, = domains
            return self._get_frequency_residency(tuple(domain))

    @TraceAnalysisBase.cache
    @df_cpu_frequency.used_events
    def df_cpu_frequency_transitions(self, cpu):
        """
        Compute number of frequency transitions of a given CPU.

        :param cpu: a CPU ID
        :type cpu: int

        :returns: A :class:`pandas.DataFrame` with:

          * A ``transitions`` column (the number of frequency transitions)
        """

        freq_df = self.df_cpu_frequency(cpu, signals_init=False)
        # Since we want to count the number of events appearing inside the
        # window, make sure we don't get anything outside it
        freq_df = df_window(
            freq_df,
            window=self.trace.window,
            method='exclusive',
            clip_window=False,
        )
        cpu_freqs = freq_df['frequency']

        # Remove possible duplicates (example: when devlib sets trace markers
        # a cpu_frequency event is triggered that can generate a duplicate)
        cpu_freqs = series_deduplicate(cpu_freqs, keep='first', consecutives=True)
        transitions = cpu_freqs.value_counts()

        transitions.name = "transitions"
        transitions.sort_index(inplace=True)

        return pd.DataFrame(transitions)

    @TraceAnalysisBase.cache
    @df_cpu_frequency_transitions.used_events
    def df_cpu_frequency_transition_rate(self, cpu):
        """
        Compute frequency transition rate of a given CPU.

        :param cpu: a CPU ID
        :type cpu: int

        :returns: A :class:`pandas.DataFrame` with:

          * A ``transitions`` column (the number of frequency transitions per second)
        """
        transitions = self.df_cpu_frequency_transitions(cpu)['transitions']
        return pd.DataFrame(dict(
            transitions=transitions / self.trace.time_range,
        ))

    @df_cpu_frequency.used_events
    def get_average_cpu_frequency(self, cpu):
        """
        Get the average frequency for a given CPU

        :param cpu: The CPU to analyse
        :type cpu: int
        """
        df = self.df_cpu_frequency(cpu)
        freq = series_refit_index(df['frequency'], window=self.trace.window)
        return series_mean(freq)

    @TraceAnalysisBase.cache
    @requires_events('clock_set_rate', 'clock_enable', 'clock_disable')
    def df_peripheral_clock_effective_rate(self, clk_name):
        rate_df = self.trace.df_event('clock_set_rate')
        enable_df = self.trace.df_event('clock_enable')
        disable_df = self.trace.df_event('clock_disable')

        freq = rate_df[rate_df.clk_name == clk_name]
        enables = enable_df[enable_df.clk_name == clk_name]
        disables = disable_df[disable_df.clk_name == clk_name]

        freq = pd.concat([freq, enables, disables], sort=False).sort_index()
        freq['start'] = freq.index
        freq['len'] = (freq.start - freq.start.shift()).fillna(0).shift(-1)
        # The last value will be NaN, fix to be appropriate length
        freq.loc[freq.index[-1], 'len'] = self.trace.end - freq.index[-1]

        freq.ffill(inplace=True)
        freq['effective_rate'] = np.where(
            freq['state'] == 0, 0,
            np.where(freq['state'] == 1, freq['state'], float('nan'))
        )
        return freq

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method(return_axis=True)
    @df_peripheral_clock_effective_rate.used_events
    def plot_peripheral_clock(self, clk, axis=None, **kwargs):
        """
        Plot the frequency of a particular peripheral clock

        :param clk: The clk name to chart
        :type clk: str
        """

        logger = self.get_logger()
        window = self.trace.window
        start, end = window

        def plotter(axis, local_fig):
            freq_axis, state_axis = axis
            freq_axis.get_figure().suptitle('Peripheral frequency', y=.97, fontsize=16, horizontalalignment='center')

            freq = self.df_peripheral_clock_effective_rate(clk)
            freq = df_refit_index(freq, window=window)

            # Plot frequency information (set rate)
            freq_axis.set_title("Clock frequency for " + clk)
            set_rate = freq['state'].dropna()

            rate_axis_lib = 0
            if len(set_rate) > 0:
                rate_axis_lib = set_rate.max()
                set_rate.plot(style=['b--'], ax=freq_axis, drawstyle='steps-post', alpha=0.4, label="clock_set_rate value")
                freq_axis.hlines(set_rate.iloc[-1], set_rate.index[-1], end, linestyle='--', color='b', alpha=0.4)
            else:
                logger.warning('No clock_set_rate events to plot')

            # Plot frequency information (effective rate)
            eff_rate = freq['effective_rate'].dropna()
            eff_rate = series_refit_index(eff_rate, window=window)
            if len(eff_rate) > 0 and eff_rate.max() > 0:
                rate_axis_lib = max(rate_axis_lib, eff_rate.max())
                eff_rate.plot(style=['b-'], ax=freq_axis, drawstyle='steps-post', alpha=1.0, label="Effective rate (with on/off)")
                freq_axis.hlines(eff_rate.iloc[-1], eff_rate.index[-1], end, linestyle='-', color='b', alpha=1.0)
            else:
                logger.warning('No effective frequency events to plot')

            freq_axis.set_ylim(0, rate_axis_lib * 1.1)
            freq_axis.set_xlabel('')
            freq_axis.grid(True)
            freq_axis.legend()

            def mhz(x, pos):
                return '{:1.2f} MHz'.format(x * 1e-6)
            freq_axis.get_yaxis().set_major_formatter(FuncFormatter(mhz))

            on = freq[freq.state == 1]
            state_axis.hlines([0] * len(on),
                              on['start'], on['start'] + on['len'],
                              linewidth=10.0, label='clock on', color='green')
            off = freq[freq.state == 0]
            state_axis.hlines([0] * len(off),
                              off['start'], off['start'] + off['len'],
                              linewidth=10.0, label='clock off', color='red')

            # Plot time period that the clock state was unknown from the trace
            indeterminate = pd.concat([on, off]).sort_index()
            if indeterminate.empty:
                indet_range_max = end
            else:
                indet_range_max = indeterminate.index[0]
            state_axis.hlines(0, 0, indet_range_max, linewidth=1.0, label='indeterminate clock state', linestyle='--')
            state_axis.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=3, mode='expand')
            state_axis.set_yticks([])
            state_axis.set_xlabel('seconds')
            state_axis.set_xlim(start, end)

        return self.do_plot(plotter, height=8, nrows=2, axis=axis, **kwargs)

    @TraceAnalysisBase.plot_method()
    @df_cpu_frequency.used_events
    def plot_cpu_frequencies(self, cpu: CPU, axis, local_fig, average: bool=True):
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
        df = self.df_cpu_frequency(cpu)

        if "freqs" in self.trace.plat_info:
            frequencies = self.trace.plat_info['freqs'][cpu]
        else:
            logger.info(f"Estimating CPU{cpu} frequencies from trace")
            frequencies = sorted(list(df.frequency.unique()))
            logger.debug(f"Estimated frequencies: {frequencies}")

        avg = self.get_average_cpu_frequency(cpu)
        logger.info(
            "Average frequency for CPU{} : {:.3f} GHz".format(cpu, avg / 1e6))

        df = df_refit_index(df, window=self.trace.window)
        df['frequency'].plot(ax=axis, drawstyle='steps-post')

        if average and avg > 0:
            axis.axhline(avg, color=self.get_next_color(axis), linestyle='--',
                         label="average")

        plot_overutilized = self.trace.analysis.status.plot_overutilized
        if self.trace.has_events(plot_overutilized.used_events):
            plot_overutilized(axis=axis)

        axis.set_ylabel('Frequency (Hz)')
        axis.set_ylim(frequencies[0] * 0.9, frequencies[-1] * 1.1)
        axis.legend()
        if local_fig:
            axis.set_xlabel('Time')
            axis.set_title(f'Frequency of CPU{cpu}')
            axis.grid(True)

    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cpu_frequencies.used_events
    def plot_domain_frequencies(self, axis=None, **kwargs):
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

                axis.set_title(f'Frequencies of CPUS {domain}')

        return self.do_plot(plotter, nrows=len(domains), sharex=True, axis=axis, **kwargs)

    @TraceAnalysisBase.plot_method(return_axis=True)
    @df_cpu_frequency_residency.used_events
    def plot_cpu_frequency_residency(self, cpu: CPU, pct: bool=False, axis=None, **kwargs):
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
            axes[0].set_title(f"CPU{cpu} total frequency residency")

            active_df.plot.barh(ax=axes[1])
            axes[1].set_title(f"CPU{cpu} active frequency residency")

            for axis in axes:
                if pct:
                    axis.set_xlabel("Time share (%)")
                else:
                    axis.set_xlabel("Time (s)")

                axis.set_ylabel("Frequency (Hz)")
                axis.grid(True)

        return self.do_plot(plotter, nrows=2, axis=axis, **kwargs)

    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cpu_frequency_residency.used_events
    def plot_domain_frequency_residency(self, pct: bool=False, axis=None, **kwargs):
        """
        Plot the frequency residency for all frequency domains.

        :param pct: Plot residencies in percentage
        :type pct: bool
        """
        domains = self.trace.plat_info['freq-domains']

        def plotter(axes, local_fig):
            for idx, domain in enumerate(domains):
                local_axes = axes[2 * idx: 2 * (idx + 1)]

                self.plot_cpu_frequency_residency(domain[0],
                    pct=pct,
                    axis=local_axes,
                )
                for axis in local_axes:
                    title = axis.get_title()
                    axis.set_title(title.replace(f'CPU{domain[0]}', f"CPUs {domain}"))

        return self.do_plot(plotter, nrows=2 * len(domains), sharex=True, axis=axis, **kwargs)

    @TraceAnalysisBase.plot_method()
    @df_cpu_frequency_transitions.used_events
    def plot_cpu_frequency_transitions(self, cpu: CPU, axis, local_fig, pct: bool=False):
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

        if not df.empty:
            df["transitions"].plot.barh(ax=axis)

        axis.set_title(f'Frequency transitions of CPU{cpu}')

        if pct:
            axis.set_xlabel("Transitions share (%)")
        else:
            axis.set_xlabel("Transition count")

        axis.set_ylabel("Frequency (Hz)")
        axis.grid(True)

    @TraceAnalysisBase.plot_method(return_axis=True)
    @plot_cpu_frequency_transitions.used_events
    def plot_domain_frequency_transitions(self, pct: bool=False, axis=None, **kwargs):
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
                )

                title = axis.get_title()
                axis.set_title(title.replace(f'CPU{domain[0]}', f"CPUs {domain}"))

        return self.do_plot(plotter, nrows=len(domains), axis=axis, **kwargs)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
