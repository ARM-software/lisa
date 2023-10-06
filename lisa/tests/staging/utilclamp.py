# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020, Arm Limited and contributors.
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

import functools
from operator import itemgetter

import numpy as np
import pandas as pd
import holoviews as hv

from lisa.analysis.frequency import FrequencyAnalysis
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.datautils import df_refit_index, series_mean
from lisa.pelt import PELT_SCALE
from lisa.tests.base import ResultBundle, TestBundle, RTATestBundle, TestMetric
from lisa.wlgen.rta import RTAPhase, PeriodicWload
from lisa.notebook import plot_signal


class UtilClamp(RTATestBundle, TestBundle):
    """
    Validate that UtilClamp min values are honoured properly by the kernel.

    The test is split into 8 phases. For each phase, a UtilClamp value is set
    for a task, whose duty cycle would generate a lower utilization. Then the
    actual capacity, allocated to the task during its activation is checked.

    The 8 phases UtilClamp values are picked to cover the entire SoC's CPU
    scale. (usually between 0 and 1024)

    .. code-block:: text

                   |<-- band 0 -->|<-- band 1 -->|<-- band 2 -->|<-- ...
      capacities:  0      |      128     |      256            512
                          |              |
      --------------------|--------------|-------------------------------
      phase 1:       uclamp_val          |
                                         |
      -----------------------------------|-------------------------------
      phase 2:                       uclamp_val
         ...

      phase 8:

    """

    NR_PHASES = 8
    CAPACITY_MARGIN = 0.8  # kernel task placement a 80% capacity margin

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)
        kconfig = target.plat_info['kernel']['config']
        if not kconfig.get('UCLAMP_TASK'):
            ResultBundle.raise_skip("The target's kernel needs CONFIG_UCLAMP_TASK=y kconfig enabled")

    @classmethod
    def _collect_capacities(cls, plat_info):
        """
        Returns, for each CPU a mapping frequency / capacity:

        dict(cpu, dict(freq, capacity))

        where capacity = max_cpu_capacity * freq / max_cpu_frequency.
        """

        max_capacities = plat_info['cpu-capacities']['rtapp']

        return {
            cpu: {
                freq: int(max_capacities[cpu] * freq / max(freqs))
                for freq in freqs
            }
            for cpu, freqs in plat_info['freqs'].items()
        }

    @classmethod
    def _collect_capacities_flatten(cls, plat_info):
        capacities = [
            capa
            for freq_capas in cls._collect_capacities(plat_info).values()
            for capa in freq_capas.values()
        ]

        # Remove the duplicates from the list
        return sorted(set(capacities))

    @classmethod
    def _get_bands(cls, capacities):
        bands = list(zip(capacities, capacities[1:]))

        # Only keep a number of bands
        nr_bands = cls.NR_PHASES
        if len(bands) > nr_bands:
            # Pick the bands covering the widest range of util, since they
            # are easier to test
            bands = sorted(
                bands,
                key=lambda band: band[1] - band[0],
                reverse=True
            )
            bands = bands[:nr_bands]
            bands = sorted(bands, key=itemgetter(0))

        return bands

    @classmethod
    def _get_phases(cls, plat_info):
        """
        Returns a list of phases. Each phase being described by a tuple:

          (uclamp_val, util)
        """

        capacities = cls._collect_capacities_flatten(plat_info)
        bands = cls._get_bands(capacities)

        def band_mid(band):
            return int((band[1] + band[0]) / 2)

        def make_phase(band):
            uclamp = band_mid(band)
            util = uclamp / 2
            name = f'uclamp-{uclamp}'
            return (name, (uclamp, util))

        return dict(map(make_phase, bands))

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        periods = [
            RTAPhase(
                prop_name=name,
                prop_wload=PeriodicWload(
                    duty_cycle_pct=(util / PELT_SCALE) * 100,  # util to pct
                    duration=5,
                    period=cls.TASK_PERIOD,
                ),
                prop_uclamp=(uclamp_val, uclamp_val),
                prop_meta={'uclamp_val': uclamp_val},
            )
            for name, (uclamp_val, util) in cls._get_phases(plat_info).items()
        ]

        return {'task': functools.reduce(lambda a, b: a + b, periods)}

    def _get_trace_df(self):
        task = self.rtapp_task_ids_map['task'][0]

        # There is no CPU selection when we're going back from preemption.
        # Setting preempted_value=1 ensures that it won't count as a new
        # activation.
        df = self.trace.ana.tasks.df_task_activation(task,
                                                          preempted_value=1)
        df = df_refit_index(df, window=self.trace.window)
        df = df[['active', 'cpu']]
        df['activation_start'] = df['active'] == 1

        df_freq = self.trace.ana.frequency.df_cpus_frequency()
        df_freq = df_freq[['cpu', 'frequency']]
        df_freq = df_freq.pivot(columns='cpu', values='frequency')
        df_freq.reset_index(inplace=True)
        df_freq.set_index('Time', inplace=True)

        df = df.merge(df_freq, how='outer', left_index=True, right_index=True)

        # Merge with df_freq will bring NaN in the activation column. We do not
        # want to ffill() them.
        df['activation_start'].fillna(value=False, inplace=True)

        # Ensures that frequency values are propogated through the entire
        # DataFrame, as it is possible that no frequency event occur
        # during a phase.
        df.ffill(inplace=True)

        return df

    def _get_phases_df(self):
        task = self.rtapp_task_ids_map['task'][0]

        df = self.trace.ana.rta.df_phases(task, wlgen_profile=self.rtapp_profile)
        df = df.copy()
        df = df[df['properties'].apply(lambda props: props['meta']['from_test'])]
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'start'}, inplace=True)
        df['end'] = df['start'].shift(-1)
        df['uclamp_val'] = df['properties'].apply(lambda row: row['meta']['uclamp_val'])
        return df

    def _for_each_phase(self, callback):
        df_phases = self._get_phases_df()
        df_trace = self._get_trace_df()

        def parse_phase(phase):
            start = phase['start']
            end = phase['end']
            df = df_trace

            # During a phase change, rt-app will wakeup and then change
            # UtilClamp value will be changed. We then need to wait for the
            # second wakeup for the kernel to apply the most recently set
            # UtilClamp value.
            start = df[(df.index >= start) &
                       (df['active'] == 1)].first_valid_index()

            end = end if not np.isnan(end) else df.last_valid_index()

            if (start > end):
                raise ValueError('Phase ends before it has even started')

            df = df_trace[start:end].copy()

            return callback(df, phase)

        return df_phases.apply(parse_phase, axis=1)

    def _plot_phases(self, test, failures, signals=None):
        task, = self.rtapp_task_ids
        ana = self.trace.ana(
            task=task,
            tasks=[task],
        )
        figs = [
            (
                ana.tasks.plot_tasks_activation(
                    overlay=True,
                    which_cpu=True
                ) *
                ana.rta.plot_phases(wlgen_profile=self.rtapp_profile) *
                hv.Overlay(
                    [
                        hv.VLine(failure).options(
                            alpha=0.5,
                            color='red'
                        )
                        for failure in failures
                    ]
                )
            ),
        ]
        if signals is not None:
            figs.append(
                hv.Overlay([
                    plot_signal(signals[signal]).opts(responsive=True, height=400)
                    for signal in signals.columns
                ])
            )

        fig = hv.Layout(figs).cols(1)

        self._save_debug_plot(fig, name=f'utilclamp_{test}')
        return fig

    @FrequencyAnalysis.df_cpus_frequency.used_events
    @LoadTrackingAnalysis.df_tasks_signal.used_events
    def test_placement(self) -> ResultBundle:
        """
        For each phase, checks if the task placement is compatible with
        UtilClamp requirements. This is done by comparing the maximum capacity
        of the CPU on which the task has been placed, with the UtilClamp
        value.
        """

        metrics = {}
        test_failures = []
        cpu_max_capacities = self.plat_info['cpu-capacities']['rtapp']

        def parse_phase(df, phase):
            # Only keep the activations
            df = df[df['activation_start']]

            uclamp_val = phase['uclamp_val']
            num_activations = len(df.index)
            cpus = set(map(int, df.cpu.dropna().unique()))
            fitting_cpus = {
                cpu
                for cpu, cap in cpu_max_capacities.items()
                if (cap == PELT_SCALE) or cap > uclamp_val
            }

            failures = df[(df['cpu'].isin(cpus - fitting_cpus))].index.tolist()
            num_failures = len(failures)
            test_failures.extend(failures)

            metrics[phase['phase']] = {
                'uclamp-min': TestMetric(uclamp_val),
                'cpu-placements': TestMetric(cpus),
                'expected-cpus': TestMetric(fitting_cpus),
                'bad-activations': TestMetric(
                    num_failures * 100 / num_activations, "%"),
            }

            return cpus.issubset(fitting_cpus)

        res = ResultBundle.from_bool(self._for_each_phase(parse_phase).all())
        res.add_metric('Phases', metrics)

        self._plot_phases('test_placement', test_failures)

        return res

    @FrequencyAnalysis.df_cpus_frequency.used_events
    @LoadTrackingAnalysis.df_tasks_signal.used_events
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    def test_freq_selection(self) -> ResultBundle:
        """
        For each phase, checks if the task placement and frequency selection
        is compatible with UtilClamp requirements. This is done by comparing
        the current CPU capacity on which the task has been placed, with the
        UtilClamp value.

        The expected capacity is the schedutil projected frequency selection
        for the given uclamp value.
        """

        metrics = {}
        test_failures = []
        capacity_dfs = []
        # (
        #    # schedutil factor that converts util to a frequency for a
        #    # given CPU:
        #    #
        #    #   next_freq = max_freq * C * util / max_cap
        #    #
        #    #   where C = 1.25
        #    schedutil_factor,
        #
        #    # list of frequencies available for a given CPU.
        #    frequencies,
        # )
        cpu_frequencies = {
            cpu: (
                (max(capacities) * (1 / self.CAPACITY_MARGIN)) / max(capacities.values()),
                sorted(capacities)
            )
            for cpu, capacities in
            self._collect_capacities(self.plat_info).items()
        }
        cpu_capacities = self._collect_capacities(self.plat_info)

        @functools.lru_cache(maxsize=4096)
        def schedutil_map_util_cap(cpu, util):
            """
            Returns, for a given util on a given CPU, the capacity that
            schedutil would select.
            """

            schedutil_factor, frequencies = cpu_frequencies[cpu]
            schedutil_freq = schedutil_factor * util

            # Find the first available freq that meet the schedutil freq
            # requirement.
            for freq in frequencies:
                if freq >= schedutil_freq:
                    break

            return cpu_capacities[cpu][freq]

        def parse_phase(df, phase):
            uclamp_val = phase['uclamp_val']
            num_activations = df['activation_start'].sum()

            df['expected_capacity'] = df.apply(lambda line: schedutil_map_util_cap(line['cpu'], uclamp_val), axis=1)

            # Activations numbering
            df['activation'] = df['activation_start'].cumsum()

            # Only keep the activations
            df = df[df['activation_start']]

            # Actual capacity at which the task is running
            for cpu, freq_to_capa in cpu_capacities.items():
                df[cpu] = df[cpu].map(freq_to_capa)
            df['capacity'] = df.apply(lambda line: line[line.cpu], axis=1)

            failures = df[df['capacity'] != df['expected_capacity']]
            num_failures = failures['activation'].nunique()

            test_failures.extend(failures.index.tolist())
            capacity_dfs.append(df[['capacity', 'expected_capacity']])

            metrics[phase['phase']] = {
                'uclamp-min': TestMetric(uclamp_val),
                'expected-mean-capacity': TestMetric(series_mean(df['expected_capacity'])),
                'bad-activations': TestMetric(
                    num_failures * 100 / num_activations, "%"),
            }

            return failures.empty

        res = ResultBundle.from_bool(self._for_each_phase(parse_phase).all())
        res.add_metric('Phases', metrics)

        self._plot_phases(
            'test_frequency',
            test_failures,
            signals=pd.concat(capacity_dfs)
        )

        return res
