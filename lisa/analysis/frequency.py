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
import functools
import operator

import pandas as pd
import polars as pl
import numpy as np
import holoviews as hv

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events, requires_one_event_of, CPU, MissingTraceEventError
from lisa.datautils import series_integrate, df_refit_index, series_refit_index, series_deduplicate, df_add_delta, series_mean, df_window, df_merge, SignalDesc
from lisa.notebook import plot_signal, _hv_neutral


class FrequencyAnalysis(TraceAnalysisBase):
    """
    Support for plotting Frequency Analysis data

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'frequency'

    @TraceAnalysisBase.df_method
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
            df = self.trace.df_event(
                'cpu_frequency',
                signals=(
                    [SignalDesc('cpu_frequency', ['cpu_id'])]
                    if signals_init else
                    []
                )
            )
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
            return df.groupby('cpu', observed=True, sort=False, group_keys=False)

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
        df.index.name = 'Time'
        return check_empty(df, None)

    @TraceAnalysisBase.df_method
    @df_cpus_frequency.used_events
    def df_cpu_frequency(self, cpu, **kwargs):
        """
        Same as :meth:`df_cpus_frequency` but for a single CPU.

        :param cpu: CPU ID to get the frequency of.
        :type cpu: int

        :Variable keyword arguments: Forwarded to :meth:`df_cpus_frequency`.
        """
        view = self.trace.get_view(df_fmt='polars-lazyframe')
        ana = view.ana.frequency
        df = ana.df_cpus_frequency(**kwargs)
        return df.filter(pl.col('cpu') == cpu)

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
        time_df = cluster_freqs[["total_time", "frequency"]].groupby('frequency', observed=True, sort=False, group_keys=False).sum()

        # Compute ACTIVE Time
        cluster_active = self.ana.idle.signal_cluster_active(cpus)

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
        cluster_freqs.ffill(inplace=True)

        # Compute total time by integrating the square wave
        time_df['active_time'] = pd.Series({
            freq: series_integrate(
                cluster_freqs['active'] * (cluster_freqs['frequency'] == freq)
            )
            for freq in cluster_freqs['frequency'].unique()
        })

        return time_df


    @TraceAnalysisBase.df_method
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
        if isinstance(cpu, int):
            return self._get_frequency_residency((cpu,))
        else:
            raise TypeError('Input CPU parameter must be an integer')

    @TraceAnalysisBase.df_method
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

    @TraceAnalysisBase.df_method
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
        )
        cpu_freqs = freq_df['frequency']

        # Remove possible duplicates (example: when devlib sets trace markers
        # a cpu_frequency event is triggered that can generate a duplicate)
        cpu_freqs = series_deduplicate(cpu_freqs, keep='first', consecutives=True)
        transitions = cpu_freqs.value_counts()

        transitions.name = "transitions"
        transitions.sort_index(inplace=True)

        return pd.DataFrame(transitions)

    @TraceAnalysisBase.df_method
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

    @TraceAnalysisBase.df_method
    @requires_events('clk_set_rate', 'clk_enable', 'clk_disable')
    def df_peripheral_clock_effective_rate(self, clk_name):
        """
        Dataframe of peripheral clock frequencies.
        """

        # Note: the kernel still defines a "clock_*" variant for each of these,
        # but it's not actually used anywhere in the code. The new "clk_*"
        # events are the ones we are interested about.
        rate_df = self.trace.df_event('clk_set_rate')
        enable_df = self.trace.df_event('clk_enable').copy()
        disable_df = self.trace.df_event('clk_disable').copy()

        # Add 'state' for enable and disable events
        enable_df['state'] = 1
        disable_df['state'] = 0

        freq = rate_df[rate_df['name'] == clk_name]
        enables = enable_df[enable_df['name'] == clk_name]
        disables = disable_df[disable_df['name'] == clk_name]

        freq = df_merge((freq, enables, disables)).ffill()
        freq['start'] = freq.index
        df_add_delta(
            freq,
            col='len',
            src_col='start',
            window=self.trace.window,
            inplace=True
        )
        freq['effective_rate'] = np.where(
            freq['state'] == 0, 0, freq['rate']
        )
        return freq

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method
    @df_cpu_frequency.used_events
    def plot_cpu_frequencies(self, cpu: CPU, average: bool=True):
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
        logger = self.logger
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
        fig = plot_signal(df['frequency'], name=f'Frequency of CPU{cpu} (Hz)')

        if average and avg > 0:
            fig *= hv.HLine(avg, group='average').opts(color='red')

        plot_overutilized = self.ana.status.plot_overutilized
        if self.trace.has_events(plot_overutilized.used_events):
            fig *= plot_overutilized()

        return fig

    @TraceAnalysisBase.plot_method
    @plot_cpu_frequencies.used_events
    def plot_domain_frequencies(self):
        """
        Plot frequency trend for all frequency domains.

        If ``sched_overutilized`` events are available, the plots will also show
        the intervals of time where the cluster was overutilized.
        """
        return functools.reduce(
            operator.add,
            (
                self.plot_cpu_frequencies(domain[0]).relabel(
                    f'Frequencies of domain CPUS {", ".join(map(str, domain))}'
                )
                for domain in self.trace.plat_info['freq-domains']
            )
        ).cols(1)

    @TraceAnalysisBase.plot_method
    @df_cpu_frequency_residency.used_events
    def plot_cpu_frequency_residency(self, cpu: CPU, pct: bool=False, domain_label: bool=False):
        """
        Plot per-CPU frequency residency.

        :param cpu: The CPU to generate the plot for
        :type cpu: int

        :param pct: Plot residencies in percentage
        :type pct: bool

        :param domain_label: If ``True``, the labels will mention all CPUs in
            the domain, rather than the CPU passed.
        :type domain_label: bool
        """

        residency_df = self.df_cpu_frequency_residency(cpu)

        total_df = residency_df.total_time
        active_df = residency_df.active_time

        if pct:
            total_df = total_df * 100 / total_df.sum()
            active_df = active_df * 100 / active_df.sum()

        ylabel = 'Time share (%)' if pct else 'Time (s)'
        opts = dict(
            xlabel='Frequency (Hz)',
            ylabel=ylabel,
            # Horizontal bar plots
            invert_axes=True,
        )

        if domain_label:
            domains = self.trace.plat_info['freq-domains']
            rev_domains = {
                cpu: sorted(domain)
                for domain in domains
                for cpu in domain
            }
            def make_label(kind):
                name = ', '.join(map(str, rev_domains[cpu]))
                return f'CPUs {name} {kind}frequency residency'
        else:
            def make_label(kind):
                return f'CPU{cpu} {kind}frequency residency'

        return (
            hv.Bars(total_df, label=make_label('total ')).opts(**opts) +
            hv.Bars(active_df, label=make_label('active ')).opts(**opts)
        ).cols(1).options(
            title=make_label('')
        )

    @TraceAnalysisBase.plot_method
    @plot_cpu_frequency_residency.used_events
    def plot_domain_frequency_residency(self, pct: bool=False):
        """
        Plot the frequency residency for all frequency domains.

        :param pct: Plot residencies in percentage
        :type pct: bool
        """
        return functools.reduce(
            operator.add,
            (
                self.plot_cpu_frequency_residency(
                    domain[0],
                    domain_label=True,
                    pct=pct,
                )
                for domain in self.trace.plat_info['freq-domains']
            )
        ).cols(1)

    @TraceAnalysisBase.plot_method
    @df_cpu_frequency_transitions.used_events
    def plot_cpu_frequency_transitions(self, cpu: CPU, pct: bool=False, domain_label: bool=False):
        """
        Plot frequency transitions count of the specified CPU

        :param cpu: The CPU to genererate the plot for
        :type cpu: int

        :param pct: Plot frequency transitions in percentage
        :type pct: bool

        :param domain_label: If ``True``, the labels will mention all CPUs in
            the domain, rather than the CPU passed.
        :type domain_label: bool
        """

        df = self.df_cpu_frequency_transitions(cpu)

        if pct:
            df = df * 100 / df.sum()

        ylabel = 'Transitions share (%)' if pct else 'Transition count'

        if domain_label:
            domains = self.trace.plat_info['freq-domains']
            rev_domains = {
                cpu: sorted(domain)
                for domain in domains
                for cpu in domain
            }
            name = ', '.join(map(str, rev_domains[cpu]))
            title = f'Frequency transitions of CPUs {name}'
        else:
            title = f'Frequency transitions of CPU{cpu}'

        if not df.empty:
            return hv.Bars(df['transitions']).options(
                title=title,
                xlabel='Frequency (Hz)',
                ylabel=ylabel,
                invert_axes=True,
            )
        else:
            return _hv_neutral()

    @TraceAnalysisBase.plot_method
    @plot_cpu_frequency_transitions.used_events
    def plot_domain_frequency_transitions(self, pct: bool=False):
        """
        Plot frequency transitions count for all frequency domains

        :param pct: Plot frequency transitions in percentage
        :type pct: bool
        """
        return functools.reduce(
            operator.add,
            (
                self.plot_cpu_frequency_transitions(
                    cpu=domain[0],
                    domain_label=True,
                    pct=pct,
                )
                for domain in self.trace.plat_info['freq-domains']
            )
        ).cols(1)

    @TraceAnalysisBase.plot_method
    @df_peripheral_clock_effective_rate.used_events
    def plot_peripheral_frequency(self, clk_name: str, average: bool=True):
        """
        Plot frequency for the specified peripheral clock frequency

        :param clk_name: The clock name for which to plot frequency
        :type clk_name: str

        :param average: If ``True``, add a horizontal line which is the
            frequency average.
        :type average: bool

        """
        df = self.df_peripheral_clock_effective_rate(clk_name)
        freq = df['effective_rate']
        freq = series_refit_index(freq, window=self.trace.window)

        fig = plot_signal(freq, name=f'Frequency of {clk_name} (Hz)')

        if average:
            avg = series_mean(freq)
            if avg > 0:
                fig *= hv.HLine(avg, group='average').opts(color='red')

        return fig

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
