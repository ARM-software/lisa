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

import os
import functools

from devlib.target import KernelVersion

from lisa.tests.base import ResultBundle, RTATestBundle
from lisa.target import Target
from lisa.utils import ArtifactPath, memoized, namedtuple
from lisa.wlgen.rta import Periodic, Ramp
from lisa.trace import FtraceCollector, requires_events
from lisa.analysis.rta import RTAEventsAnalysis
from lisa.analysis.tasks import TaskState, TasksAnalysis
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.datautils import df_window, df_refit_index, series_mean, df_filter_task_ids

from lisa.tests.scheduler.load_tracking import LoadTrackingHelpers


class UtilTrackingBase(RTATestBundle, LoadTrackingHelpers):
    """
    Base class for shared functionality of utilization tracking tests
    """

    @classmethod
    def _from_target(cls, target: Target, *,
                     res_dir: ArtifactPath = None,
                     ftrace_coll: FtraceCollector = None) -> 'UtilTrackingBase':
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info)

        # After a bit of experimenting, it turns out that on some platforms
        # misprediction of the idle time (which leads to a shallow idle state,
        # a wakeup and another idle nap) can mess up the duty cycle of the
        # rt-app task we're running. In our case, a 50% duty cycle, 16ms period
        # task would always be active for 8ms, but it would sometimes sleep for
        # only 5 or 6 ms.
        # This is fine to do this here, as we only care about the proper
        # behaviour of the signal on running/not-running tasks.
        with target.disable_idle_states():
            with target.cpufreq.use_governor('performance'):
                cls.run_rtapp(target, res_dir, rtapp_profile, ftrace_coll)

        return cls(res_dir, plat_info)


PhaseStats = namedtuple("PhaseStats",
    ['start', 'end', 'mean_util', 'mean_enqueued', 'mean_ewma', 'issue'],
    module=__name__,
)


ActivationSignals = namedtuple("ActivationSignals", [
    'time', 'util', 'enqueued', 'ewma', 'issue'],
    module=__name__,
)


class UtilConvergence(UtilTrackingBase):
    """
    Basic checks for estimated utilization signals

    **Expected Behaviour:**

    The estimated utilization of a task is properly computed starting form its
    `util` value at the end of each activation.

    Two signals composes the estimated utlization of a task:

    * `enqueued` : is expected to match the max between `util` and
      `ewma` at the end of the previous activation

    * `ewma` : is expected to track an Exponential Weighted Moving
      Average of the `util` signal sampled at the end of each activation.

    Based on these two invariant, this class provides a set of tests to verify
    these conditions using different methods and sampling points.
    """

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        big_cpu = plat_info["capacity-classes"][-1][0]

        task = (
            # Big task
            Periodic(
                duty_cycle_pct=75,
                duration_s=5,
                period_ms=200,
                cpus=[big_cpu]) +
            # Ramp Down
            Ramp(
                start_pct=50,
                end_pct=5,
                delta_pct=20,
                time_s=1,
                period_ms=200,
                cpus=[big_cpu]) +
            # Ramp Up
            Ramp(
                start_pct=10,
                end_pct=60,
                delta_pct=20,
                time_s=1,
                period_ms=200,
                cpus=[big_cpu])
        )
        return {'test': task}

    @property
    @memoized
    def fast_ramp(self):
        min_kernel = KernelVersion('5.5.0').parts
        cur_kernel = self.plat_info['kernel']['version'].parts
        return cur_kernel >= min_kernel

    def _plot_signals(self, task, test, failures):
        signals = ['util', 'enqueued', 'ewma']
        ax = self.trace.analysis.load_tracking.plot_task_signals(task, signals=signals)
        ax = self.trace.analysis.rta.plot_phases(task, axis=ax)
        for start in failures:
            ax.axvline(start, alpha=0.5, color='r')
        filepath = os.path.join(self.res_dir, 'util_est_{}.png'.format(test))
        self.trace.analysis.rta.save_plot(ax.figure, filepath=filepath)

    @requires_events('sched_util_est_se')
    @LoadTrackingAnalysis.df_tasks_signal.used_events
    @RTAEventsAnalysis.task_phase_windows.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_means(self) -> ResultBundle:
        """
        Test signals are properly "dominated".

        The mean of `enqueued` is expected to be always not
        smaller than that of `util`, since this last is subject to decays
        while the first not.

        The mean of `enqueued` is expected to be always greater or
        equal than the mean of `util`, since this `util` is subject
        to decays while `enqueued` not.

        On fast-ramp systems, the `ewma` signal is never smaller then
        the `enqueued`, thus his mean is expected to be bigger.

        On non fast-ramp systems instead, the `ewma` is expected to be
        smaller then `enqueued` in ramp-up phases, or bigger in
        ramp-down phases.

        Those conditions are checked on a single execution of a task which has
        three main behaviours:

            * STABLE: periodic big task running for a relatively long period to
              ensure `util` saturation.
            * DOWN: periodic ramp-down task, to slowly decay `util`
            * UP: periodic ramp-up task, to slowly increase `util`

        """
        failure_reasons = {}
        metrics = {}

        task = self.rtapp_task_ids_map['test'][0]

        ue_df = self.trace.df_event('sched_util_est_se')
        ue_df = df_filter_task_ids(ue_df, [task])
        ua_df = self.trace.analysis.load_tracking.df_task_signal(task, 'util')

        failures = []
        for phase in self.trace.analysis.rta.task_phase_windows(task):
            # TODO: remove that once we have named phases to skip the buffer phase
            if phase.id == 0:
                continue

            apply_phase_window = functools.partial(df_refit_index, window=(phase.start, phase.end))

            ue_phase_df = apply_phase_window(ue_df)
            mean_enqueued = series_mean(ue_phase_df['enqueued'])
            mean_ewma = series_mean(ue_phase_df['ewma'])

            ua_phase_df = apply_phase_window(ua_df)
            mean_util = series_mean(ua_phase_df['util'])

            def make_issue(msg):
                return msg.format(
                    util='util={}'.format(mean_util),
                    enq='enqueud={}'.format(mean_enqueued),
                    ewma='ewma={}'.format(mean_ewma),
                )

            issue = None
            if mean_enqueued < mean_util:
                issue = make_issue('{enq} smaller than {util}')

            # Running on FastRamp kernels:
            elif self.fast_ramp:

                # STABLE, DOWN and UP:
                if mean_ewma < mean_enqueued:
                    issue = make_issue('no fast ramp: {ewma} smaller than {enq}')

            # Running on (legacy) non FastRamp kernels:
            else:

                # STABLE: ewma ramping up
                if phase.id == 1 and mean_ewma > mean_enqueued:
                    issue = make_issue('fast ramp, stable: {ewma} bigger than {enq}')

                # DOWN: ewma ramping down
                elif phase.id <= 5 and mean_ewma < mean_enqueued:
                    issue = make_issue('fast ramp, down: {ewma} smaller than {enq}')

                # UP: ewma ramping up
                elif phase.id >= 6 and mean_ewma > mean_enqueued:
                    issue = make_issue('fast ramp, up: {ewma} bigger than {enq}')

            metrics[phase.id] = PhaseStats(
                phase.start, phase.end, mean_util, mean_enqueued, mean_ewma, issue
            )

        failures = [
            (phase, stat)
            for phase, stat in metrics.items()
            if stat.issue
        ]

        # Plot signals to support debugging analysis
        self._plot_signals(task, 'means', sorted(stat.start for phase, stat in failures))

        bundle = ResultBundle.from_bool(not failures)
        bundle.add_metric("fast ramp", self.fast_ramp)
        bundle.add_metric("phases", metrics)
        bundle.add_metric("failures", sorted(phase for phase, stat in failures))
        return bundle

    @requires_events('sched_util_est_se')
    @TasksAnalysis.df_task_states.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_activations(self) -> ResultBundle:
        """
        Test signals are properly "aggregated" at enqueue/dequeue time.

        On fast-ramp systems, `enqueud` is expected to be always
        smaller than `ewma`.

        On non fast-ramp systems, the `enqueued` is expected to be
        smaller then `ewma` in ramp-down phases, or bigger in ramp-up
        phases.

        Those conditions are checked on a single execution of a task which has
        three main behaviours:

            * STABLE: periodic big task running for a relatively long period to
              ensure `util` saturation.
            * DOWN: periodic ramp-down task, to slowly decay `util`
            * UP: periodic ramp-up task, to slowly increase `util`

        """
        metrics = {}

        task = self.rtapp_task_ids_map['test'][0]

        # Get list of task's activations
        df = self.trace.analysis.tasks.df_task_states(task)
        activations = df[
            (df.curr_state == TaskState.TASK_WAKING) &
            (df.next_state == TaskState.TASK_ACTIVE)
        ].index

        # Check task signals at each activation
        df = self.trace.df_event('sched_util_est_se')
        df = df_filter_task_ids(df, [task])


        for idx, activation in enumerate(activations):
            # If we are outside a phase, ignore the activation
            try:
                phase = self.trace.analysis.rta.task_phase_at(task, activation)
            except KeyError:
                continue

            # Get the value of signals at their first update after the activation
            row = df_window(df, (activation, None), method='post').iloc[0]
            util = row['util']
            enq = row['enqueued']
            ewma = row['ewma']
            def make_issue(msg):
                return msg.format(
                    util='util={}'.format(util),
                    enq='enqueud={}'.format(enq),
                    ewma='ewma={}'.format(ewma),
                )

            issue = None

            # UtilEst is not updated when within 1% of previous activation
            if 1.01 * enq < util:
                issue = make_issue('{enq} smaller than {util}')

            # Running on FastRamp kernels:
            elif self.fast_ramp:

                # ewma stable, down and up
                if enq > ewma:
                    issue = make_issue('{enq} bigger than {ewma}')

            # Running on (legacy) non FastRamp kernels:
            else:
                # TODO: remove that once we have named phases to skip the buffer phase
                if phase.id == 0:
                    continue

                # ewma stable
                if phase.id == 1 and enq < ewma:
                    issue = make_issue('stable: {enq} smaller than {ewma}')

                # ewma ramping down
                elif phase.id <= 5 and enq > ewma:
                    issue = make_issue('ramp down: {enq} bigger than {ewma}')

                # ewma ramping up
                elif phase.id >= 6 and enq < ewma:
                    issue = make_issue('ramp up: {enq} smaller than {ewma}')

            metrics[idx] = ActivationSignals(activation, util, enq, ewma, issue)

        failures = [
            (idx, activation_signals)
            for idx, activation_signals in metrics.items()
            if activation_signals.issue
        ]

        bundle = ResultBundle.from_bool(not failures)
        bundle.add_metric("failures", sorted(idx for idx, activation in failures))
        bundle.add_metric("activations", metrics)

        failures_time = [activation.time for idx, activation in failures]
        self._plot_signals(task, 'activations', failures_time)
        return bundle
