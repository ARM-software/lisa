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
import os
import pandas as pd

from lisa.analysis.frequency import FrequencyAnalysis
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.datautils import df_add_delta, series_mean, df_window
from lisa.pelt import PELT_SCALE
from lisa.tests.base import ResultBundle, RTATestBundle, TestMetric
from lisa.wlgen.rta import Periodic


class UtilClamp(RTATestBundle):
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

        return [
            (
                band_mid(band),
                band_mid(band) / 2
            )
            for band in bands
        ]

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        periods = [
            Periodic(
                duty_cycle_pct=(util / PELT_SCALE) * 100,  # util to pct
                duration_s=5,
                period_ms=cls.TASK_PERIOD_MS,
                uclamp_min=uclamp_val,
                uclamp_max=uclamp_val
            )
            for uclamp_val, util in cls._get_phases(plat_info)
        ]

        return {'task': functools.reduce(lambda a, b: a + b, periods)}

    def _get_trace_df(self):
        task = self.rtapp_task_ids_map['task'][0]

        # There is no CPU selection when we're going back from preemption.
        # Setting preempted_value=1 ensures that it won't count as a new
        # activation.
        df = self.trace.analysis.tasks.df_task_activation(task,
                                                          preempted_value=1)
        df = df[['active', 'cpu']]

        df_freq = self.trace.analysis.frequency.df_cpus_frequency()
        df_freq = df_freq[['cpu', 'frequency']]
        df_freq = df_freq.pivot(index=None, columns='cpu', values='frequency')
        df_freq.reset_index(inplace=True)
        df_freq.set_index('Time', inplace=True)

        df = df.merge(df_freq, how='outer', left_index=True, right_index=True)

        # Ensures that frequency values are propogated through the entire
        # DataFrame, as it is possible that no frequency event occur
        # during a phase.
        for cpu in self.plat_info['cpu-capacities']['rtapp']:
            df[cpu].ffill(inplace=True)

        return df

    def _get_phases_df(self):
        task = self.rtapp_task_ids_map['task'][0]

        df = self.trace.analysis.rta.df_phases(task).copy()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'start'}, inplace=True)
        df['end'] = df['start'].shift(-1)

        phases = [(0, 0, 0)] + self._get_phases(self.plat_info)
        df['uclamp_val'] = df.apply(
                lambda x: phases[int(x.phase)][0],
                axis=1)

        # TODO: To remove once we have named phases.
        df = df[df.phase > 0]

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

            if (start >= end):
                raise ValueError('Phase ends before it has even started')

            df = df_trace[start:end].copy()

            return callback(df, phase)

        return df_phases.apply(parse_phase, axis=1)

    def _plot_phases(self, test, failures, signal=None):
        task = self.rtapp_task_ids_map['task'][0]
        ax = self.trace.analysis.tasks.plot_task_activation(task,
                                                            which_cpu=True)
        ax = self.trace.analysis.rta.plot_phases(task, axis=ax)
        for failure in failures:
            ax.axvline(failure, alpha=0.5, color='r')
        if signal is not None:
            signal.plot(ax=ax.twinx(), drawstyle='steps-post')
        filepath = os.path.join(self.res_dir, f'utilclamp_{test}.png')
        self.trace.analysis.rta.save_plot(ax.figure, filepath=filepath)

        return ax

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
        capacity_margin = self.CAPACITY_MARGIN
        cpu_max_capacities = self.plat_info['cpu-capacities']['rtapp']

        def parse_phase(df, phase):
            uclamp_val = phase['uclamp_val']
            num_activations = df['active'][df['active'] == 1].count()
            cpus = set(map(int, df.cpu.dropna().unique()))
            fitting_cpus = {
                cpu
                for cpu, cap in cpu_max_capacities.items()
                if (cap == PELT_SCALE) or (cap * capacity_margin) > uclamp_val
            }

            failures = df[(
                df['active'] == 1) & (df['cpu'].isin(cpus - fitting_cpus))
            ].index.tolist()
            num_failures = len(failures)
            test_failures.extend(failures)

            phase_str = f"Phase-{int(phase['phase'])}"
            metrics[phase_str] = {
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
            num_activations = df['active'][df['active'] == 1].count()
            expected = schedutil_map_util_cap(df['cpu'].unique()[0],
                                              uclamp_val)

            # Activations numbering
            df['activation'] = df['active'].cumsum()

            # Only keep the activations
            df.ffill(inplace=True)
            df = df[df['active'] == 1]

            # Actual capacity at which the task is running
            for cpu, freq_to_capa in cpu_capacities.items():
                df[cpu] = df[cpu].map(freq_to_capa)
            df['capacity'] = df.apply(lambda line: line[line.cpu], axis=1)

            failures = df[df['capacity'] != expected]
            num_failures = failures['activation'].nunique()

            test_failures.extend(failures.index.tolist())
            capacity_dfs.append(df[['capacity']])

            phase_str = f"Phase-{phase['phase']}"
            metrics[phase_str] = {
                'uclamp-min': TestMetric(uclamp_val),
                'expected-capacity': TestMetric(expected),
                'bad-activations': TestMetric(
                    num_failures * 100 / num_activations, "%"),
            }

            return failures.empty

        res = ResultBundle.from_bool(self._for_each_phase(parse_phase).all())
        res.add_metric('Phases', metrics)

        self._plot_phases('test_frequency', test_failures,
                          pd.concat(capacity_dfs))

        return res
