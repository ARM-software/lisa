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
from collections import OrderedDict
import itertools
from statistics import mean

import matplotlib.pyplot as plt
import pylab as pl

from devlib.target import KernelVersion

from lisa.tests.base import (TestMetric, Result, ResultBundle, TestBundle,
                             RTATestBundle, CannotCreateError)
from lisa.target import Target
from lisa.utils import ArtifactPath, memoized, namedtuple
from lisa.wlgen.rta import Periodic, Ramp
from lisa.trace import FtraceCollector, requires_events
from lisa.analysis.rta import RTAEventsAnalysis
from lisa.analysis.tasks import TaskState, TasksAnalysis
from lisa.datautils import series_integrate, df_filter_task_ids

from lisa.tests.scheduler.load_tracking import LoadTrackingHelpers

class UtilTrackingBase(RTATestBundle, LoadTrackingHelpers):
    """
    Base class for shared functionality of utilization tracking tests
    """

    @classmethod
    def _from_target(cls, target:Target, *,
                     res_dir:ArtifactPath=None,
                     ftrace_coll:FtraceCollector=None) -> 'UtilTrackingBase':
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


PhaseStats = namedtuple("PhaseStats", [
    'start', 'end', 'area_util', 'area_enqueued', 'area_ewma'],
    module=__name__,
)


ActivationSignals = namedtuple("ActivationSignals", [
    'time', 'util_avg', 'util_est_enqueued', 'util_est_ewma'],
    module=__name__,
)


class UtilConvergence(UtilTrackingBase):
    """
    Basic checks for estimated utilization signals

    **Expected Behaviour:**

    The estimated utilization of a task is properly computed starting form its
    `util_avg` value at the end of each activation.

    Two signals composes the estimated utlization of a task:

    * `util_est_enqueued` : is expected to match the max between `util_avg` and
      `util_est_ewma` at the end of the previous activation

    * `util_est_ewma` : is expected to track an Exponential Weighted Moving
      Average of the `util_avg` signal sampled at the end of each activation.

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
        return {'test' : task}

    @property
    @memoized
    def fast_ramp(self):
        # The EWMA "fast ramp up" feature has been merged only in kernel TBD
        # TODO fix kernel version
        return False

        min_kernel = KernelVersion('5.4-rc1').parts
        cur_kernel = self.plat_info['kernel']['version'].parts
        return cur_kernel < min_kernel

    def _plot_signals(self, task, test, failures):
        signals = ['util', 'util_est_enqueued', 'util_est_ewma']
        ax = self.trace.analysis.load_tracking.plot_task_signals(task, signals=signals, interactive=False)
        ax = self.trace.analysis.rta.plot_phases(task, axis=ax, interactive=False);
        for start in failures:
            ax.axvline(start, alpha=0.5, color='r')
        filepath = os.path.join(self.res_dir, 'util_est_{}.png'.format(test))
        self.trace.analysis.rta.save_plot(ax.figure, filepath=filepath)

    @requires_events('sched_util_est_task', 'sched_load_se')
    @RTAEventsAnalysis.task_phase_windows.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_areas(self) -> ResultBundle:
        """
        Test signals are properly "dominated".

        The integral of `util_est_enqueued` is expected to be always not
        smaller than that of `util_avg`, since this last is subject to decays
        while the first not.

        The integral of `util_est_enqueued` is expected to be always greater or
        equal than the integral of `util_avg`, since this `util_avg` is subject
        to decays while `util_est_enqueued` not.

        On fast-ramp systems, the `util_est_ewma` signal is never smaller then
        the `util_est_enqueued`, thus his integral is expected to be bigger.

        On non fast-ramp systems instead, the `util_est_ewma` is expected to be
        smaller then `util_est_enqueued` in ramp-up phases, or bigger in
        ramp-down phases.

        Those conditions are checked on a single execution of a task which has
        three main behaviours:

            * STABLE: periodic big task running for a relatively long period to
              ensure `util_avg` saturation.
            * DOWN: periodic ramp-down task, to slowly decay `util_avg`
            * UP: periodic ramp-up task, to slowly increase `util_avg`

        """
        failure_reasons = {}
        metrics = {}

        # We have only two task: the main 'rt-app' task and our 'test_task'
        test_task = self.trace.analysis.rta.rtapp_tasks[-1]

        ue_df = self.trace.df_events('sched_util_est_task')
        ue_df = df_filter_task_ids(ue_df, [test_task])
        ua_df = self.trace.df_events('sched_load_se')
        ua_df = df_filter_task_ids(ua_df, [test_task])

        failures = []
        for phase in self.trace.analysis.rta.task_phase_windows(test_task):
            phase_df = ue_df[phase.start:phase.end]
            area_enqueued = series_integrate(phase_df.util_est_enqueued)
            area_ewma = series_integrate(phase_df.util_est_ewma)

            phase_df = ua_df[phase.start:phase.end]
            area_util = series_integrate(phase_df.util)

            metrics[phase.id] = PhaseStats(phase.start, phase.end,
                                                area_util, area_enqueued, area_ewma)

            phase_name = "phase {}".format(phase.id)
            if area_enqueued < area_util:
                failure_reasons[phase_name] = 'Enqueued smaller then Util Average'
                failures.append(phase.start)
                continue

            # Running on FastRamp kernels:
            if self.fast_ramp:

                # STABLE, DOWN and UP:
                if area_ewma < area_enqueued:
                    failure_reasons[phase_name] = 'NO_FAST_RAMP: EWMA smaller then Enqueued'
                    failures.append(phase.start)
                    continue

            # Running on (legacy) non FastRamp kernels:
            else:

                # STABLE: ewma ramping up
                if phase.id == 0 and area_ewma > area_enqueued:
                    failure_reasons[phase_name] = 'FAST_RAMP(STABLE): EWMA bigger then Enqueued'
                    failures.append(phase.start)
                    continue

                # DOWN: ewma ramping down
                if 0 < phase.id < 5 and area_ewma < area_enqueued:
                    failure_reasons[phase_name] = 'FAST_RAMP(DOWN): EWMA smaller then Enqueued'
                    failures.append(phase.start)
                    continue

                # UP: ewma ramping up
                if phase.id > 4 and area_ewma > area_enqueued:
                    failure_reasons[phase_name] = 'FAST_RAMP(UP): EWMA bigger then Enqueued'
                    failures.append(phase.start)
                    continue

        bundle = ResultBundle.from_bool(failure_reasons)
        bundle.add_metric("fast ramp", self.fast_ramp)
        bundle.add_metric("phases stats", metrics)
        if not failure_reasons:
            return bundle

        # Plot signals to support debugging analysis
        self._plot_signals(test_task, 'areas', failures)
        bundle.add_metric("failure reasons", failure_reasons)

        return bundle

    @requires_events('sched_util_est_task')
    @TasksAnalysis.df_task_states.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_activations(self) -> ResultBundle:
        """
        Test signals are properly "aggregated" at enqueue/dequeue time.

        On fast-ramp systems, `util_est_enqueud` is expected to be always
        smaller than `util_est_ewma`.

        On non fast-ramp systems, the `util_est_enqueued` is expected to be
        smaller then `util_est_ewma` in ramp-down phases, or bigger in ramp-up
        phases.

        Those conditions are checked on a single execution of a task which has
        three main behaviours:

            * STABLE: periodic big task running for a relatively long period to
              ensure `util_avg` saturation.
            * DOWN: periodic ramp-down task, to slowly decay `util_avg`
            * UP: periodic ramp-up task, to slowly increase `util_avg`

        """
        failure_reasons = {}
        metrics = {}

        # We have only two task: the main 'rt-app' task and our 'test_task'
        test_task = self.trace.analysis.rta.rtapp_tasks[-1]

        # Get list of task's activations
        df = self.trace.analysis.tasks.df_task_states(test_task)
        activations = df[(df.curr_state == TaskState.TASK_WAKING) &
                         (df.next_state == TaskState.TASK_ACTIVE)].index

        # Check task signals at each activation
        df = self.trace.df_events('sched_util_est_task')
        df = df_filter_task_ids(df, [test_task])

        # Define a time interval to correlate relative trace events.
        def restrict(df, time, delta=1e-3):
            return df[time-delta:time+delta]

        failures = []
        for idx, activation in enumerate(activations):
            avg, enq, ewma = restrict(df, activation)[[
                'util_avg', 'util_est_enqueued', 'util_est_ewma']].iloc[-1]

            metrics[idx+1] = ActivationSignals(activation, avg, enq, ewma)

            # UtilEst is not updated when within 1% of previous activation
            if 1.01 * enq < avg:
                failure_reasons[idx] = 'enqueued({}) smaller than util_avg({}) @ {}'\
                                        .format(enq, avg, activation)
                failures.append(activation)
                continue

            # Running on FastRamp kernels:
            if self.fast_ramp:

                # STABLE, DOWN and UP:
                if enq > ewma:
                    failure_reasons[idx] = 'enqueued({}) bigger than ewma({}) @ {}'\
                                            .format(enq, ewma, activation)
                    failures.append(activation)
                    continue

            # Running on (legacy) non FastRamp kernels:
            else:

                phase = self.trace.analysis.rta.task_phase_at(test_task, activation)

                # STABLE: ewma ramping up
                if phase.id == 0 and enq < ewma:
                    failure_reasons[idx] = 'enqueued({}) smaller than ewma({}) @ {}'\
                                            .format(enq, ewma, activation)
                    failures.append(activation)
                    continue

                # DOWN: ewma ramping down
                if 0 < phase.id < 5 and enq > ewma:
                    failure_reasons[idx] = 'enqueued({}) bigger than ewma({}) @ {}'\
                                            .format(enq, ewma, activation)
                    failures.append(activation)
                    continue

                # UP: ewma ramping up
                if phase.id > 4 and enq < ewma:
                    failure_reasons[idx] = 'enqueued({}) smaller than ewma({}) @ {}'\
                                            .format(enq, ewma, activation)
                    failures.append(activation)
                    continue

        self._plot_signals(test_task, 'activations', failures)

        bundle = ResultBundle.from_bool(not failure_reasons)
        bundle.add_metric("signals", metrics)
        bundle.add_metric("failure reasons", failure_reasons)
        return bundle
