# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
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

from math import ceil
import itertools

import pandas as pd
import holoviews as hv

from lisa.wlgen.rta import DutyCycleSweepPhase
from lisa.tests.base import ResultBundle, Result, TestBundle, RTATestBundle
from lisa.target import Target
from lisa.trace import requires_events
from lisa.datautils import df_merge, series_mean
from lisa.utils import ArtifactPath

from lisa.notebook import plot_signal
from lisa.analysis.frequency import FrequencyAnalysis
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.analysis.rta import RTAEventsAnalysis
from lisa.analysis.tasks import TasksAnalysis, TaskState


class RampBoostTestBase(RTATestBundle, TestBundle):
    """
    Test schedutil's ramp boost feature.
    """

    def __init__(self, res_dir, plat_info, cpu, rtapp_profile_kwargs=None):
        super().__init__(res_dir, plat_info, rtapp_profile_kwargs=rtapp_profile_kwargs)
        self.cpu = cpu

    @requires_events('cpu_idle', 'cpu_frequency', 'sched_wakeup')
    def estimate_nrg(self):
        return self.plat_info['nrg-model'].estimate_from_trace(self.trace).sum(axis=1)

    def get_avg_slack(self, only_negative=False):
        analysis = self.trace.ana.rta

        def get_slack(task):
            series = analysis.df_rtapp_stats(task)['slack']
            if only_negative:
                series = series[series < 0]

            if series.empty:
                return 0
            else:
                # average negative slack across all activations
                return series.mean()

        return {
            task: get_slack(task)
            for task in self.trace.ana.rta.rtapp_tasks
        }

    @LoadTrackingAnalysis.df_cpus_signal.used_events
    @requires_events('schedutil_em')
    def df_ramp_boost(self):
        """
        Return a dataframe with schedutil-related signals, sampled at the
        frequency decisions timestamps for the CPU this bundle was executed on.

        .. note:: The computed columns only take into account the CPU the test
            was executing on. It currently does not handle multi-task workloads.
        """
        trace = self.trace
        cpu = self.cpu
        task = self.rtapp_task_ids[0]

        # schedutil_df also has a 'util' column that would conflict
        schedutil_df = trace.df_event('schedutil_em')[['cpu', 'cost_margin', 'base_freq']]
        schedutil_df = schedutil_df.copy()
        schedutil_df['from_schedutil'] = True

        def compute_base_cost(row):
            freq = row['base_freq']
            cpu = row['cpu']

            em = self.plat_info['nrg-model']
            active_states = em.cpu_nodes[cpu].active_states
            freqs = sorted(active_states.keys())
            max_freq = max(freqs)

            def cost(freq):
                higher_freqs = list(itertools.dropwhile(lambda f: f < freq, freqs))
                freq = freqs[-1] if not higher_freqs else higher_freqs[0]
                active_state = active_states[freq]
                return active_state.power * max_freq / freq

            max_cost = max(
                cost(freq)
                for freq in active_states.keys()
            )

            return cost(freq) / max_cost * 100

        schedutil_df['base_cost'] = schedutil_df.apply(compute_base_cost, axis=1)

        task_active = trace.ana.tasks.df_task_states(task)['curr_state']
        task_active = task_active.apply(lambda state: int(state == TaskState.TASK_ACTIVE))
        task_active = task_active.reindex(schedutil_df.index, method='ffill')
        # Assume task active == CPU active, since there is only one task
        assert len(self.rtapp_task_ids) == 1
        cpu_active_df = pd.DataFrame({'cpu_active': task_active})
        cpu_active_df['cpu'] = cpu
        cpu_active_df.dropna(inplace=True)

        df_list = [
            schedutil_df,
            trace.ana.load_tracking.df_cpus_signal('util'),
            trace.ana.load_tracking.df_cpus_signal('enqueued'),
            cpu_active_df,
        ]

        df = df_merge(df_list, filter_columns={'cpu': cpu})
        df['from_schedutil'].fillna(value=False, inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        # Reconstitute how schedutil sees signals by subsampling the
        # "main" dataframe, so we can look at signals coming from other
        # dataframes
        df = df[df['from_schedutil'] == True] # pylint: disable=singleton-comparison
        df.drop(columns=['from_schedutil'], inplace=True)

        # If there are some NaN at the beginning, just drop some data rather
        # than using fake data
        df.dropna(inplace=True)

        boost_points = (
            # util_est_enqueued is the same as last freq update
            (df['enqueued'].diff() == 0) &

            # util_avg is increasing
            (df['util'].diff() >= 0) &

            # util_avg > util_est_enqueued
            (df['util'] > df['enqueued']) &

            # CPU is not idle
            (df['cpu_active'])
        )
        df['boost_points'] = boost_points

        df['expected_cost_margin'] = (df['util'] - df['enqueued']).where(
            cond=boost_points,
            other=0,
        )

        # cost_margin values range from 0 to 1024
        ENERGY_SCALE = 1024

        for col in ('expected_cost_margin', 'cost_margin'):
            df[col] *= 100 / ENERGY_SCALE

        df['allowed_cost'] = df['base_cost'] + df['cost_margin']

        # We cannot know if the first row is supposed to be boosted or not
        # because we lack history, so we just drop it
        return df.iloc[1:]

    @FrequencyAnalysis.plot_cpu_frequencies.used_events
    @TasksAnalysis.plot_tasks_activation.used_events
    @LoadTrackingAnalysis.plot_task_signals.used_events
    def _plot_test_boost(self, df):
        task, = self.rtapp_tasks
        ana = self.trace.ana(
            task=task,
        )

        fig = hv.Layout(
            [
                (
                    plot_signal(df['cost_margin']).options(
                        'Curve',
                        color='red'
                    ) *
                    plot_signal(df['boost_points'].astype(int)).options(
                        'Curve',
                        color='black'
                    ) *
                    plot_signal(df['expected_cost_margin']).options(
                        'Curve',
                        color='blue'
                    ) *
                    plot_signal(df['base_cost']).options(
                        'Curve',
                        color='orange'
                    ) *
                    plot_signal(df['allowed_cost']).options(
                        'Curve',
                        color='green'
                    ) *
                    ana.tasks.plot_tasks_activation(overlay=True)
                ).options(
                    title='Ramp boost for 5% => 75% util step',
                    ylabel='Cost (% of max cost)',
                ),

                ana.frequency.plot_cpu_frequencies(cpu=self.cpu, average=False),

                (
                    ana.load_tracking.plot_task_signals(
                        signals=['util', 'enqueued'],
                        colors=['orange', 'red']
                    ) *
                    ana.tasks.plot_tasks_activation(overlay=True)
                ),
            ]
        ).cols(1)

        self._save_debug_plot(fig, name=f'ramp_boost')
        return fig

    @RTAEventsAnalysis.plot_slack_histogram.used_events
    @RTAEventsAnalysis.plot_perf_index_histogram.used_events
    @RTAEventsAnalysis.plot_latency.used_events
    @df_ramp_boost.used_events
    @_plot_test_boost.used_events
    def test_ramp_boost(self, cost_threshold_pct=0.1, bad_samples_threshold_pct=0.1) -> ResultBundle:
        """
        Test that the energy boost feature is triggering as expected.
        """
        # If there was no cost_margin sample to look at, that means boosting
        # was not exhibited by that test so we cannot conclude anything
        df = self.df_ramp_boost()
        self._plot_test_boost(df)

        if df.empty:
            return ResultBundle(Result.UNDECIDED)

        # Make sure the boost is always positive (negative cannot really happen
        # since the kernel is using unsigned arithmetic, but still check in
        # case there are some dataframe handling issues)
        assert not (df['expected_cost_margin'] < 0).any()
        assert not (df['cost_margin'] < 0).any()

        # "rect" method is accurate here since the signal is really following
        # "post" steps
        expected_boost_cost = series_mean(df['expected_cost_margin'])
        actual_boost_cost = series_mean(df['cost_margin'])
        boost_overhead = series_mean(df['cost_margin'] / df['base_cost'] * 100)

        # Check that the total amount of boost is close to expectations
        lower = max(0, expected_boost_cost - cost_threshold_pct)
        higher = expected_boost_cost
        passed_overhead = lower <= actual_boost_cost <= higher

        # Check the shape of the signal: actual boost must be lower or equal
        # than the expected one.
        good_shape_nr = (df['cost_margin'] <= df['expected_cost_margin']).sum()

        df_len = len(df)
        bad_shape_nr = df_len - good_shape_nr
        bad_shape_pct = bad_shape_nr / df_len * 100

        # Tolerate a few bad samples that added too much boost
        passed_shape = bad_shape_pct < bad_samples_threshold_pct

        passed = passed_overhead and passed_shape
        res = ResultBundle.from_bool(passed)
        res.add_metric('expected boost cost', expected_boost_cost, '%')
        res.add_metric('boost cost', actual_boost_cost, '%')
        res.add_metric('boost overhead', boost_overhead, '%')
        res.add_metric('bad boost samples', bad_shape_pct, '%')

        # Add some slack metrics and plots
        analysis = self.trace.ana.rta
        for task in self.rtapp_tasks:
            analysis.plot_slack_histogram(task)
            analysis.plot_perf_index_histogram(task)
            analysis.plot_latency(task)

        res.add_metric('avg slack', self.get_avg_slack(), 'us')
        res.add_metric('avg negative slack', self.get_avg_slack(only_negative=True), 'us')

        return res


class LargeStepUp(RampBoostTestBase):
    """
    A single task whose utilization rises extremely quickly
    """
    task_name = "step_up"

    def __init__(self, res_dir, plat_info, cpu, nr_steps):
        rtapp_profile_kwargs = dict(
            cpu=cpu,
            nr_steps=nr_steps,
        )
        super().__init__(
            res_dir,
            plat_info,
            cpu=cpu,
            rtapp_profile_kwargs=rtapp_profile_kwargs,
        )
        self.nr_steps = nr_steps

    @classmethod
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath = None, collector=None, cpu=None, nr_steps=1) -> 'LargeStepUp':
        plat_info = target.plat_info

        # Use a big CPU by default to allow maximum range of utilization
        cpu = cpu if cpu is not None else plat_info["capacity-classes"][-1][0]

        rtapp_profile = cls.get_rtapp_profile(plat_info, cpu=cpu, nr_steps=nr_steps)

        # Ensure accurate duty cycle and idle state misprediction on some
        # boards. This helps having predictable execution.
        with target.disable_idle_states():
            with target.cpufreq.use_governor("schedutil"):
                cls.run_rtapp(target, res_dir, rtapp_profile, collector=collector)

        return cls(res_dir, plat_info, cpu, nr_steps)

    @classmethod
    def _get_rtapp_profile(cls, plat_info, cpu, nr_steps, min_util=5, max_util=75):
        start_pct = cls.unscaled_utilization(plat_info, cpu, min_util)
        end_pct = cls.unscaled_utilization(plat_info, cpu, max_util)

        delta_pct = ceil((end_pct - start_pct) / nr_steps)

        return {
            cls.task_name: 20 * DutyCycleSweepPhase(
                start=start_pct,
                stop=end_pct,
                step=delta_pct,
                duration=0.3,
                duration_of='step',
                period=cls.TASK_PERIOD,
                # Make sure we run on one CPU only, so that we only stress
                # frequency scaling and not placement.
                prop_cpus=[cpu],
            )
        }
