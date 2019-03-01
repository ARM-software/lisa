# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
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

from bart.common.Utils import select_window, area_under_curve
from bart.sched import pelt
from bart.sched.SchedAssert import SchedAssert

from trappy.stats.Topology import Topology

from lisa.tests.base import (
    TestMetric, Result, ResultBundle, TestBundle, RTATestBundle
)
from lisa.target import Target
from lisa.utils import ArtifactPath, groupby
from lisa.wlgen.rta import Periodic
from lisa.trace import FtraceConf, FtraceCollector

UTIL_SCALE = 1024
"""
PELT utilization values scale
"""

HALF_LIFE_MS = 32
"""
PELT half-life value in ms
"""

UTIL_AVG_CONVERGENCE_TIME_S = 0.3
"""
Time in seconds for util_avg to converge (i.e. ignored time)
"""

class LoadTrackingHelpers:
    """
    Common bunch of helpers for load tracking tests.
    """

    MAX_RTAPP_CALIB_DEVIATION = 3/100
    """
    Blacklist CPUs that have a RTapp calibration value that deviates too much
    from the average calib value in their capacity class.
    """

    @classmethod
    def _get_blacklisted_cpus(cls, plat_info):
        """
        Consider some CPUs as blacklisted when the load would not be
        proportionnal to utilization on them.

        That happens for CPUs that are busy executing other code than the test
        workload, like handling interrupts. It is detect that by looking at the
        RTapp calibration value and we blacklist outliers.
        """
        rtapp_calib = plat_info['rtapp']['calib']
        blacklisted = set()
        # For each class of CPUs, get the average rtapp calibration value
        # and blacklist the ones that are deviating too much from that
        for cpu_class in plat_info['capacity-classes']:
            calib_mean = mean(rtapp_calib[cpu] for cpu in cpu_class)
            calib_max = (1 + cls.MAX_RTAPP_CALIB_DEVIATION) * calib_mean
            blacklisted.update(
                cpu
                for cpu in cpu_class
                # exclude outliers that are too slow (i.e. calib value too small)
                if rtapp_calib[cpu] > calib_max
            )
        return sorted(blacklisted)

    @classmethod
    def filter_capacity_classes(cls, plat_info):
        """
        Filter out capacity-classes key of ``plat_info`` to remove blacklisted CPUs.
        .. seealso:: :meth:`_get_blacklisted_cpus`
        """
        blacklisted_cpus = set(cls._get_blacklisted_cpus(plat_info))
        return [
            sorted(set(cpu_class) - blacklisted_cpus)
            for cpu_class in plat_info['capacity-classes']
        ]

    @classmethod
    def get_max_capa_cpu(cls, plat_info):
        """
        :returns: A CPU with the highest capacity value that is not blacklisted.
        """
        # capacity-classes is sorted by capacity, last class therefore contains
        # the biggest CPUs
        candidates = cls.filter_capacity_classes(plat_info)[-1]

        if not candidates:
            raise RuntimeError('All CPUs of that class have been blacklisted: {}'.format(
                plat_info['capacity-class'][-1]
            ))
        return candidates[0]

    @classmethod
    def get_task_duty_cycle_pct(cls, trace, task_name, cpu):
        window = cls.get_task_window(trace, task_name, cpu)

        top = Topology()
        top.add_to_level('cpu', [[cpu]])
        return SchedAssert(trace.ftrace, top, execname=task_name).getDutyCycle(window)

    @staticmethod
    def get_task_window(trace, task_name, cpu=None):
        """
        Get the execution window of a given task

        :param trace: The trace to look at
        :type trace: Trace

        :param task_name: The name of the task
        :type task_name: str

        :param cpu: If specified, limit the window to times where the task was
          running on that particular CPU
        :type cpu: int

        :returns: tuple(int, int)
        """
        sw_df = trace.df_events('sched_switch')

        start_df = sw_df[(sw_df["next_comm"] == task_name)]
        end_df = sw_df[(sw_df["prev_comm"] == task_name)]

        if not cpu is None:
            start_df = start_df[(start_df["__cpu"] == cpu)]
            end_df = end_df[(end_df["__cpu"] == cpu)]

        return (start_df.index[0], end_df.index[-1])


class LoadTrackingBase(RTATestBundle, LoadTrackingHelpers):
    """
    Base class for shared functionality of load tracking tests
    """

    ftrace_conf = FtraceConf({
        "events" : [
            "sched_switch",
            "sched_load_avg_task",
            "sched_load_avg_cpu",
            "sched_pelt_se",
            "sched_load_se",
            "sched_load_cfs_rq",
        ],
    }, __qualname__)

    cpufreq_conf = {
        "governor" : "performance"
    }
    """
    The cpufreq configuration used while the synthetic workload is being run.
    Items are arguments to :meth:`devlib.cpufreq.use_governor`.
    """

    @classmethod
    def _from_target(cls, target, res_dir, ftrace_coll):
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
            with target.cpufreq.use_governor(**cls.cpufreq_conf):
                cls._run_rtapp(target, res_dir, rtapp_profile, ftrace_coll)

        return cls(res_dir, plat_info, rtapp_profile)

    def get_task_sched_signals(self, trace, cpu, task_name):
        """
        Get a :class:`pandas.DataFrame` with the sched signals for the workload task

        This examines scheduler load tracking trace events, supporting either
        sched_load_avg_task or sched_pelt_se. You will need a target kernel that
        includes these events.

        :returns: :class:`pandas.DataFrame` with a column for each signal for
          the workload task
        """
        df = trace.analysis.load_tracking.df_tasks_signals()
        df = df[df['comm'] == task_name]
        window = self.get_task_window(trace, task_name, cpu)
        df = select_window(df, window)

        # Normalize the signal with the detected task execution start
        df.index -= window[0]

        return df

    @staticmethod
    def is_almost_equal(target, value, allowed_delta_pct):
        """
        Verify that ``value``` is reasonably close to ``target```
        """
        delta = target * allowed_delta_pct / 100
        return target - delta <= value <= target + delta

class InvarianceItem(LoadTrackingBase):
    """
    Basic check for CPU and frequency invariant load and utilization tracking

    **Expected Behaviour:**

    Load tracking signals are scaled so that the workload results in
    roughly the same util & load values regardless of compute power of the
    CPU used and its frequency.
    """
    task_prefix = 'invariance'
    cpufreq_conf = {
        "governor" : "userspace"
    }

    def __init__(self, res_dir, plat_info, rtapp_profile, cpu, freq, freq_list):
        super().__init__(res_dir, plat_info, rtapp_profile)

        self.freq = freq
        self.freq_list = freq_list
        self.cpu = cpu

    @classmethod
    def get_rtapp_profile(cls, plat_info, cpu):
        """
        Get a specification for a rt-app workload with the specificied duty
        cycle, pinned to the given CPU.
        """
        rtapp_profile = {}
        rtapp_profile["{}_cpu{}".format(cls.task_prefix, cpu)] = Periodic(
            duty_cycle_pct=10,
            duration_s=2,
            period_ms=cls.TASK_PERIOD_MS,
            cpus=[cpu],
        )

        return rtapp_profile

    @classmethod
    # Not annotated, to prevent exekall from picking it up. See
    # Invariance.from_target
    def from_target(cls, target, cpu, freq, freq_list, res_dir=None, ftrace_coll=None):
        """
        .. warning:: `res_dir` is at the end of the parameter list, unlike most
            other `from_target` where it is the second one.
        """
        return super().from_target(target, res_dir,
            cpu=cpu, freq=freq, freq_list=freq_list, ftrace_coll=ftrace_coll)

    @classmethod
    def _from_target(cls, target, res_dir, cpu, freq, freq_list, ftrace_coll):
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info, cpu)
        logger = cls.get_logger()

        with target.cpufreq.use_governor(**cls.cpufreq_conf):
            target.cpufreq.set_frequency(cpu, freq)
            logger.debug('CPU{} frequency: {}'.format(cpu, target.cpufreq.get_frequency(cpu)))
            cls._run_rtapp(target, res_dir, rtapp_profile, ftrace_coll)

        return cls(res_dir, plat_info, rtapp_profile, cpu, freq, freq_list)

    def get_expected_util_avg(self, trace, cpu, task_name, capacity):
        """
        Examine trace to figure out an expected mean for util_avg

        Assumes an RT-App workload with a single phase
        """
        # Find duty cycle of the workload task
        duty_cycle_pct = self.get_task_duty_cycle_pct(trace, task_name, cpu)

        # Scale the relative CPU/freq capacity
        return (duty_cycle_pct / 100) * capacity

    def _test_task_signal(self, signal_name, allowed_error_pct,
                          trace, cpu, task_name, capacity):
        # Use utilization signal for both load and util, since they should be
        # proportionnal in the test environment we setup
        exp_signal = self.get_expected_util_avg(trace, cpu, task_name, capacity)
        signal_df = self.get_task_sched_signals(trace, cpu, task_name)
        signal = signal_df[UTIL_AVG_CONVERGENCE_TIME_S:][signal_name]

        signal_mean = area_under_curve(signal) / (signal.index[-1] - signal.index[0])

        if signal_name == 'load':
            # Load isn't CPU invariant
            exp_signal /= (self.plat_info['cpu-capacities'][cpu] / UTIL_SCALE)

        ok = self.is_almost_equal(exp_signal, signal_mean, allowed_error_pct)

        return ok, exp_signal, signal_mean

    def _test_signal(self, signal_name, allowed_error_pct):
        passed = True
        expected_data = {}
        trace_data = {}

        freq_str = '@{}'.format(self.freq) if self.freq is not None else ''

        capacity = self.plat_info['cpu-capacities'][self.cpu]

        # Scale the capacity linearly according to the frequency
        max_freq = max(self.plat_info['freqs'][self.cpu])
        capacity *= (self.freq / max_freq)

        for name, task in self.rtapp_profile.items():
            ok, exp_util, signal_mean = self._test_task_signal(
                signal_name, allowed_error_pct, self.trace, self.cpu, name, capacity)

            if not ok:
                passed = False

            metric_name = 'cpu{}{}'.format(self.cpu, freq_str)
            expected_data[metric_name] = TestMetric(exp_util)
            trace_data[metric_name] = TestMetric(signal_mean)

        bundle = ResultBundle.from_bool(passed)
        bundle.add_metric("Expected signals", expected_data)
        bundle.add_metric("Trace signals", trace_data)
        return bundle

    def test_task_util_avg(self, allowed_error_pct=15) -> ResultBundle:
        """
        Test that the mean of the util_avg signal matched the expected value

        The trace is examined to estimate an expected mean value for util_avg
        for each CPU's workload by combining the known period and the average
        activation length of the workload. The util_avg value is extracted from
        scheduler trace events and its mean is compared with the expected value
        (ignoring the first 300ms so that the signal can stabilize).

        The test fails if the observed mean is beyond a certain error margin
        from the expected one.

        :param allowed_error_pct: How much the real signal can stray from the
          expected values
        :type allowed_error_pct: float
        """
        return self._test_signal('util', allowed_error_pct)

    def test_task_load_avg(self, allowed_error_pct=15) -> ResultBundle:
        """
        Test that the mean of the load_avg signal matched the expected value.

        Assuming that the system was under little stress (so the task was
        RUNNING whenever it was RUNNABLE) and that the task was run with a
        'nice' value of 0, the load_avg should be similar to the util_avg. So,
        this test does the same as test_task_util_avg but for load_avg.

        For asymmetric systems, this is only true for tasks run on the
        biggest CPUs.

        :param allowed_error_pct: How much the real signal can stray from the
          expected values
        :type allowed_error_pct: float
        """
        return self._test_signal('load', allowed_error_pct)

class Invariance(TestBundle, LoadTrackingHelpers):
    """
    Basic check for frequency invariant load and utilization tracking

    This test runs the same workload on the most capable CPU on the system at a
    cross section of available frequencies.

    This class is mostly a wrapper around :class:`InvarianceItem`,
    providing a way to build a list of those for a few frequencies, and
    providing aggregated versions of the tests. Calling the tests methods on
    the items directly is recommended to avoid the unavoidable loss of
    information when aggregating the
    :class:`~lisa.tests.base.Result` of each item.

    `invariance_items` instance attribute is a list of instances of
    :class:`InvarianceItem`.
    """

    # Make sure ftrace_conf is available so exekall can find the right settings
    # when building the FtraceCollector
    ftrace_conf = InvarianceItem.ftrace_conf

    def __init__(self, res_dir, plat_info, invariance_items):
        super().__init__(res_dir, plat_info)

        self.invariance_items = invariance_items

    @classmethod
    def _build_invariance_items(cls, target, res_dir, ftrace_coll):
        """
        Yield a :class:`InvarianceItem` for a subset of target's
        frequencies, for one CPU of each capacity class.

        This is a generator function.

        :rtype: Iterator[:class:`InvarianceItem`]
        """
        plat_info = target.plat_info

        def pick_cpu(filtered_class, cpu_class):
            try:
                return filtered_class[0]
            except IndexError:
                raise RuntimeError('All CPUs of one capacity class have been blacklisted: {}'.format(cpu_class))

        # pick one CPU per class of capacity
        cpus = [
            pick_cpu(filtered_class, cpu_class)
            for cpu_class, filtered_class
            in zip(
                plat_info['capacity-classes'],
                cls.filter_capacity_classes(plat_info)
            )
        ]

        logger = cls.get_logger()
        logger.info('Selected one CPU of each capacity class: {}'.format(cpus))
        for cpu in cpus:
            all_freqs = target.cpufreq.list_frequencies(cpu)
            # If we have loads of frequencies just test a cross-section so it
            # doesn't take all day
            freq_list = all_freqs[::len(all_freqs) // 8 + (1 if len(all_freqs) % 2 else 0)]

            # Make sure we have increasing frequency order, to make the logs easier
            # to navigate
            freq_list.sort()

            for freq in freq_list:
                item_dir = os.path.join(res_dir, "{prefix}_{cpu}@{freq}".format(
                    prefix=InvarianceItem.task_prefix,
                    cpu=cpu,
                    freq=freq,
                ))
                os.makedirs(item_dir)

                logger.info('Running experiment for CPU {}@{}'.format(cpu, freq))
                yield InvarianceItem.from_target(
                    target, cpu, freq, all_freqs, res_dir=item_dir,
                    ftrace_coll=ftrace_coll,
                )

    def iter_invariance_items(self) -> InvarianceItem:
        yield from self.invariance_items

    @classmethod
    def from_target(cls, target:Target, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'Invariance':
        return super().from_target(target, res_dir, ftrace_coll=ftrace_coll)

    @classmethod
    def _from_target(cls, target, res_dir, ftrace_coll):
        return cls(res_dir, target.plat_info,
            list(cls._build_invariance_items(target, res_dir, ftrace_coll))
        )

    def get_trace(self, cpu, freq):
        """
        :returns: The trace generated when running at a given frequency
        """
        for item in self.invariance_items:
            if item.cpu == cpu and item.freq == freq:
                return item
        raise ValueError('No invariance item matching {cpu}@{freq}'.format(cpu, freq))

    # Combined version of some other tests, applied on all available
    # InvarianceItem with the result merged.

    def test_task_util_avg(self, allowed_error_pct=15) -> ResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_task_util_avg`
        """
        def item_test(test_item):
            return test_item.test_task_util_avg(
                allowed_error_pct=allowed_error_pct
            )
        return self._test_all_freq(item_test)

    def test_task_load_avg(self, allowed_error_pct=15) -> ResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_task_load_avg`
        """
        def item_test(test_item):
            return test_item.test_task_load_avg(allowed_error_pct=allowed_error_pct)
        return self._test_all_freq(item_test)

    def _test_all_freq(self, item_test):
        """
        Apply the `item_test` function on all instances of
        :class:`InvarianceItem` and aggregate the returned
        :class:`~lisa.tests.base.ResultBundle` into one.

        :attr:`~lisa.tests.base.Result.UNDECIDED` is ignored.
        """
        item_res_bundles = {
            '{}@{}'.format(item.cpu, item.freq): item_test(item)
            for item in self.invariance_items
        }

        overall_bundle = ResultBundle.from_bool(all(item_res_bundles.values()))
        for name, bundle in item_res_bundles.items():
            overall_bundle.add_metric(name, bundle.metrics)

        overall_bundle.add_metric('failed cpu@freq', [
            name for name, bundle in item_res_bundles.items()
            if bundle.result is Result.FAILED
        ])

        return overall_bundle

    def test_cpu_invariance(self) -> ResultBundle:
        """
        Check that items using the max freq on each CPU is passing util avg test.

        There could be false positives, but they are expected to be relatively
        rare.

        .. seealso:: :class:`InvarianceItem.test_task_util_avg`
        """
        metrics = {}
        passed = True
        for cpu, item_group in groupby(self.invariance_items, key=lambda x: x.cpu):
            item_group = list(item_group)
            # combine all frequencies of that CPU class, although they should
            # all be the same
            max_freq = max(itertools.chain.from_iterable(
                x.freq_list for x in item_group
            ))
            max_freq_items = [
                item
                for item in item_group
                if item.freq == max_freq
            ]
            for item in max_freq_items:
                # Only test util, as it should be more robust
                res = item.test_task_util_avg()
                passed &= bool(res)
                metrics.setdefault(cpu, []).append(res.metrics)

        res = ResultBundle.from_bool(passed)
        for cpu, submetrics in metrics.items():
            for submetric in submetrics:
                res.add_metric(cpu, submetric)

        return res

    def test_freq_invariance(self) -> ResultBundle:
        """
        Check that at least one CPU has items passing for all tested frequencies.

        .. seealso:: :class:`InvarianceItem.test_task_util_avg`
        """
        logger = self.get_logger()
        metrics = {}
        passed = False
        for cpu, item_group in groupby(self.invariance_items, key=lambda x: x.cpu):
            group_passed = True
            freq_list = []
            for item in item_group:
                freq_list.append(item.freq)
                # Only test util, as it should be more robust
                res = item.test_task_util_avg()
                passed &= bool(res)
                name = '{}@{}'.format(cpu, item.freq)
                metrics[name] = res.metrics

            logger.info('Util avg invariance {res} for CPU {cpu} at frequencies: {freq_list}'.format(
                res='passed' if group_passed else 'failed',
                cpu=cpu,
                freq_list=freq_list,
            ))

            # At least one group must pass
            passed |= group_passed

        res = ResultBundle.from_bool(passed)
        for cpu, submetric in metrics.items():
            res.add_metric(cpu, submetric)

        return res


class PELTTask(LoadTrackingBase):
    """
    Basic checks for task related PELT signals behaviour

    This test runs a synthetic periodic task on a CPU in the system and
    collects a trace from the target device. The util_avg values are extracted
    from scheduler trace events and the behaviour of the signal is compared
    against a simulated value of PELT.
    This class runs the following tests:

    - test_util_avg_range: test that util_avg's stable range matches with the
        stable range of the simulated signal. In particular, this test compares
        min, max and mean values of the two signals.

    - test_util_avg_behaviour: check behaviour of util_avg against the simualted
        PELT signal. This test assumes that PELT is configured with 32 ms half
        life time and the samples are 1024 us. Also, it assumes that the first
        trace event related to the task used for testing is generated 'after'
        the task starts (hence, we compute the initial PELT value when the task
        started).

    **Expected Behaviour:**

    Simulated PELT signal and the signal extracted from the trace should have
    very similar min, max and mean values in the stable range and the behaviour of
    the signal should be very similar to simulated one.
    """

    task_prefix = 'pelt_behv'

    @classmethod
    def from_target(cls, target:Target, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'PELTTask':
        """
        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in :meth:`get_rtapp_profile`
        """
        return super().from_target(target, res_dir, ftrace_coll=ftrace_coll)

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        # Run the 50% workload on a CPU with highest capacity
        cpu = cls.get_max_capa_cpu(plat_info)

        rtapp_profile = {}
        rtapp_profile["{}_cpu{}".format(cls.task_prefix, cpu)] = Periodic(
            duty_cycle_pct=50,
            duration_s=2,
            period_ms=cls.TASK_PERIOD_MS,
            cpus=[cpu]
        )

        return rtapp_profile

    @property
    def task_name(self):
        """
        The name of the only task this test uses
        """
        return list(self.rtapp_profile.keys())[0]

    def get_task_sched_signals(self, cpu):
        # We only have one task and one trace, simplify this method a bit
        return super().get_task_sched_signals(self.trace, cpu, self.task_name)

    def get_simulated_pelt(self, cpu, signal_name):
        """
        Get simulated PELT signal and the periodic task used to model it.

        :returns: tuple of
          - :class:`bart.sched.pelt.Simulator.Simulator` the PELT simulator object
          - :class:`bart.sched.pelt.PeriodicTask` simulated periodic task
          - :class:`pandas.DataFrame` instance which reports the computed
          PELT values at each PELT sample interval.
        """
        signal_df = self.get_task_sched_signals(cpu)

        init_value = pelt.Simulator.estimateInitialPeltValue(
            signal_df[signal_name].iloc[0], signal_df.index[0],
            0, HALF_LIFE_MS
        )

        phase = self.rtapp_profile[self.task_name].phases[0]
        pelt_task = pelt.PeriodicTask(period_samples=phase.period_ms,
                                      duty_cycle_pct=phase.duty_cycle_pct)
        peltsim = pelt.Simulator(init_value=init_value,
                                 half_life_ms=HALF_LIFE_MS)
        df = peltsim.getSignal(pelt_task, 0, phase.duration_s)

        return peltsim, pelt_task, df

    def _test_range(self, signal_name, allowed_error_pct):
        res = ResultBundle.from_bool(True)
        task = self.rtapp_profile[self.task_name]
        cpu = task.phases[0].cpus[0]

        peltsim, pelt_task, sim_df = self.get_simulated_pelt(cpu, signal_name)
        signal_df = self.get_task_sched_signals(cpu)

        sim_range = peltsim.stableRange(pelt_task)
        stable_time = peltsim.stableTime(pelt_task)
        window = (stable_time, stable_time + 0.5)

        # Get signal statistics in a period of time where the signal is
        # supposed to be stable
        signal_stats = signal_df[window[0]:window[1]][signal_name].describe()
        # Narrow down simulated PELT signal to stable period
        sim_df = sim_df[window[0]:window[1]].pelt_value

        expected_data = {}
        trace_data = {}

        for stat in ['min', 'max', 'mean']:
            if stat == 'mean':
                stat_value = sim_df.mean()
            else:
                stat_value = getattr(sim_range, '{}_value'.format(stat))

            if not self.is_almost_equal(stat_value, signal_stats[stat], allowed_error_pct):
                res.result = Result.FAILED

            trace_data[stat] = TestMetric(signal_stats[stat])
            expected_data[stat] = TestMetric(stat_value)

        res.add_metric("Trace signal", trace_data)
        res.add_metric("Expected signal", expected_data)

        return res

    def _plot_behaviour(self, ax, df, title, task_duty_cycle_pct):
        df.plot(ax=ax, title=title)
        avg = task_duty_cycle_pct * 1024 / 100
        ax.axhline(avg, label="duty-cycle based average", linestyle="--", color="orange")
        ax.legend()

    def _test_behaviour(self, signal_name, error_margin_pct, allowed_error_pct):
        res = ResultBundle.from_bool(True)
        task = self.rtapp_profile[self.task_name]
        phase = task.phases[0]
        cpu = phase.cpus[0]

        peltsim, _, sim_df = self.get_simulated_pelt(cpu, signal_name)
        signal_df = self.get_task_sched_signals(cpu)

        trace_duty_cycle = self.get_task_duty_cycle_pct(
            self.trace, self.task_name, cpu)
        requested_duty_cycle = phase.duty_cycle_pct

        # Do a bit of plotting
        fig, axes = plt.subplots(2, 1, figsize=(32, 10), sharex=True)
        self._plot_behaviour(axes[0], signal_df[signal_name], "Trace signal", trace_duty_cycle)
        self._plot_behaviour(axes[1], sim_df.pelt_value, "Expected signal",
                             requested_duty_cycle)

        figname = os.path.join(self.res_dir, '{}_behaviour.png'.format(signal_name))
        pl.savefig(figname, bbox_inches='tight')
        plt.close()

        # Compare actual PELT signal with the simulated one
        period_s = phase.period_ms / 1e3
        sim_period_ms = phase.period_ms * (peltsim._sample_us / 1e6)
        errors = 0

        for entry in signal_df.iterrows():
            trace_val = entry[1][signal_name]
            timestamp = entry[0]
            # Next two instructions map the trace timestamp to a simulated
            # signal timestamp. This is due to the fact that the 1 ms is
            # actually 1024 us in the simulated signal.
            n_periods = timestamp / period_s
            nearest_timestamp = n_periods * sim_period_ms
            sim_val_loc = sim_df.index.get_loc(nearest_timestamp,
                                               method='nearest')
            sim_val = sim_df.pelt_value.iloc[sim_val_loc]

            if not self.is_almost_equal(trace_val, sim_val, error_margin_pct):
                errors += 1

        error_pct = (errors / len(signal_df)) * 100
        res.add_metric("Error stats",
                       {"total" : TestMetric(errors), "pct" : TestMetric(error_pct)})
        # Exclude possible outliers (these may be due to a kernel thread that
        # for some reason gets coscheduled with our workload).
        if error_pct > allowed_error_pct:
            res.result = Result.FAILED

        return res

    def test_util_avg_range(self, allowed_error_pct=15) -> ResultBundle:
        """
        Test that the util_avg value ranges (min, mean, max) are sane

        :param allowed_error_pct: The allowed range difference
        """
        return self._test_range('util', allowed_error_pct)

    def test_load_avg_range(self, allowed_error_pct=15) -> ResultBundle:
        """
        Test that the load_avg value ranges (min, mean, max) are sane

        :param allowed_error_pct: The allowed range difference
        """
        return self._test_range('load', allowed_error_pct)

    def test_util_avg_behaviour(self, error_margin_pct=5, allowed_error_pct=5)\
        -> ResultBundle:
        """
        Validate every utilization signal event

        :param error_margin_pct: How much the actual signal can stray from the
          simulated signal

        :param allowed_error_pct: How many PELT errors (determined by
          ``error_margin_pct```) are allowed
        """
        return self._test_behaviour('util', error_margin_pct, allowed_error_pct)

    def test_load_avg_behaviour(self, error_margin_pct=5, allowed_error_pct=5)\
        -> ResultBundle:
        """
        Validate every load signal event

        :param error_margin_pct: How much the actual signal can stray from the
          simulated signal

        :param allowed_error_pct: How many PELT errors (determined by
          ``error_margin_pct```) are allowed
        """
        return self._test_behaviour('load', error_margin_pct, allowed_error_pct)
