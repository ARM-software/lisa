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

import abc
import os
import matplotlib.pyplot as plt
import pylab as pl
from collections import OrderedDict

from bart.common.Utils import select_window
from bart.sched import pelt
from bart.sched.SchedAssert import SchedAssert

from trappy.stats.Topology import Topology

from lisa.tests.kernel.test_bundle import (
    TestMetric, Result, ResultBundle, TestBundle, RTATestBundle
)
from lisa.env import TestEnv
from lisa.utils import ArtifactPath
from lisa.wlgen.rta import Periodic
from lisa.trace import Trace

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
    @staticmethod
    def get_max_capa_cpu(te):
        """
        :returns: A CPU with the highest capacity value
        """
        cpu_capacities = te.plat_info['cpu-capacities']
        return max(cpu_capacities.keys(), key=lambda cpu: cpu_capacities[cpu])

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

    ftrace_conf = {
        "events" : [
            "sched_switch",
            "sched_load_avg_task",
            "sched_load_avg_cpu",
            "sched_pelt_se",
            "sched_load_se",
            "sched_load_cfs_rq",
        ],
    }

    cpufreq_conf = {
        "governor" : "performance"
    }
    """
    The cpufreq configuration used while the synthetic workload is being run.
    Items are arguments to :meth:`devlib.cpufreq.use_governor`.
    """

    @classmethod
    def _from_testenv(cls, te, res_dir):
        rtapp_profile = cls.get_rtapp_profile(te)

        # After a bit of experimenting, it turns out that on some platforms
        # misprediction of the idle time (which leads to a shallow idle state,
        # a wakeup and another idle nap) can mess up the duty cycle of the
        # rt-app task we're running. In our case, a 50% duty cycle, 16ms period
        # task would always be active for 8ms, but it would sometimes sleep for
        # only 5 or 6 ms.
        # This is fine to do this here, as we only care about the proper
        # behaviour of the signal on running/not-running tasks.
        with te.disable_idle_states():
            with te.target.cpufreq.use_governor(**cls.cpufreq_conf):
                cls._run_rtapp(te, res_dir, rtapp_profile)

        return cls(res_dir, te.plat_info, rtapp_profile)

    def get_task_sched_signals(self, trace, cpu, task_name, signals):
        """
        Get a :class:`pandas.DataFrame` with the sched signals for the workload task

        This examines scheduler load tracking trace events, supporting either
        sched_load_avg_task or sched_pelt_se. You will need a target kernel that
        includes these events.

        :param signals: List of load tracking signals to extract. Probably a
          subset of ``['util_avg', 'load_avg']``
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

class InvarianceBase(LoadTrackingBase):
    """
    Basic check for CPU and frequency invariant load and utilization tracking

    **Expected Behaviour:**

    Load tracking signals are scaled so that the workload results in
    roughly the same util & load values regardless of compute power of the
    CPU used and its frequency.
    """
    task_prefix = 'cie'

    def get_expected_util_avg(self, trace, cpu, task_name, capacity=None):
        """
        Examine trace to figure out an expected mean for util_avg

        Assumes an RT-App workload with a single phase
        """
        # Find duty cycle of the workload task
        duty_cycle_pct = self.get_task_duty_cycle_pct(trace, task_name, cpu)
        cpu_capacities = self.plat_info['cpu-capacities']
        capacity = capacity or cpu_capacities[cpu]

        # Scale the relative CPU/freq capacity into the range 0..1
        scaling_factor = capacity / max(cpu_capacities.values())
        return UTIL_SCALE * (duty_cycle_pct / 100) * scaling_factor

    def _test_task_signal(self, signal_name, allowed_error_pct,
                          trace, cpu, task_name, capacity=None):
        exp_util = self.get_expected_util_avg(trace, cpu, task_name, capacity)
        signal_df = self.get_task_sched_signals(trace, cpu, task_name, [signal_name])
        signal_mean = signal_df[UTIL_AVG_CONVERGENCE_TIME_S:][signal_name].describe()['mean']

        ok = self.is_almost_equal(exp_util, signal_mean, allowed_error_pct)

        return ok, exp_util, signal_mean

    def _test_signal(self, signal_name, allowed_error_pct, freq=None, capacity_scale=1):
        passed = True
        expected_data = {}
        trace_data = {}

        freq_str = '@{}'.format(freq) if freq is not None else ''

        for name, task in self.rtapp_profile.items():
            cpu = task.phases[0].cpus[0]
            capacity = self.plat_info['cpu-capacities'][cpu] * capacity_scale

            ok, exp_util, signal_mean = self._test_task_signal(
                signal_name, allowed_error_pct, self.trace, cpu, name, capacity)

            if not ok:
                passed = False

            metric_name = 'cpu{}{}'.format(cpu, freq_str)
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

class CpuInvariance(InvarianceBase):
    """
    Basic check for CPU invariant load and utilization tracking

    This class runs the same workload on one CPU of each type in the system.
    """

    @classmethod
    def from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None) -> 'CpuInvariance':
        """
        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in :meth:`get_rtapp_profile`
        """
        return super().from_testenv(te, res_dir)

    @classmethod
    def get_rtapp_profile(cls, te):
        # Run the 10% workload on one CPU in each capacity group
        cpu_capacities = te.target.sched.get_capacities()
        capa_cpus = {capa : [] for capa in cpu_capacities.values()}
        for cpu in range(te.target.number_of_cpus):
            capa_cpus[cpu_capacities[cpu]].append(cpu)

        rtapp_profile = {}
        for cpus in capa_cpus.values():
            # No need to test on every CPU, just one for each capacity value
            cpu = cpus[0]
            rtapp_profile["{}_cpu{}".format(cls.task_prefix, cpu)] = Periodic(
                duty_cycle_pct=10,
                duration_s=2,
                period_ms=cls.TASK_PERIOD_MS,
                cpus=[cpu]
            )

        return rtapp_profile

class FreqInvarianceItem(InvarianceBase):
    """
    Frequency invariance test bundle item

    This bundle holds the trace for a given frequency.
    """
    cpufreq_conf = {
        "governor" : "userspace"
    }

    task_prefix = 'fie'

    def __init__(self, res_dir, plat_info, rtapp_profile, freq, freq_list, cpu=None):
        super().__init__(res_dir, plat_info, rtapp_profile)

        self.freq = freq
        self.freq_list = freq_list
        self.cpu = cpu

    @classmethod
    def get_rtapp_profile(cls, te):
        """
        Get a specification for a rt-app workload with the specificied duty
        cycle, pinned to the given CPU.
        """
        cpu = cls.get_max_capa_cpu(te)

        rtapp_profile = {}
        rtapp_profile["{}_cpu{}".format(cls.task_prefix, cpu)] = Periodic(
            duty_cycle_pct=10,
            duration_s=2,
            period_ms=cls.TASK_PERIOD_MS,
            cpus=[cpu]
        )

        return rtapp_profile

    @classmethod
    # Not annotated, to prevent exekall from picking it up. See
    # FreqInvariance.from_testenv
    def from_testenv(cls, te, freq, freq_list, res_dir=None):
        """
        .. warning:: `res_dir` is the last parameter, unlike most other
            `from_testenv` where it is the second one.
        """
        return super().from_testenv(te, res_dir, freq=freq, freq_list=freq_list)

    @classmethod
    def _from_testenv(cls, te, res_dir, freq, freq_list):
        rtapp_profile = cls.get_rtapp_profile(te)
        cpu = cls.get_max_capa_cpu(te)
        logger = cls.get_logger()

        with te.target.cpufreq.use_governor(**cls.cpufreq_conf):
            te.target.cpufreq.set_frequency(cpu, freq)
            logger.debug('CPU{} frequency: {}'.format(cpu, te.target.cpufreq.get_frequency(cpu)))
            cls._run_rtapp(te, res_dir, rtapp_profile)

        return cls(res_dir, te.plat_info, rtapp_profile, freq, freq_list, cpu)

    def _test_signal(self, signal_name, allowed_error_pct):
        # Scale the capacity linearly according to the frequency
        capacity_scale = self.freq / max(self.freq_list)

        return super()._test_signal(
            signal_name, allowed_error_pct,
            freq=self.freq,
            capacity_scale=capacity_scale
        )

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


class FreqInvariance(TestBundle, LoadTrackingHelpers):
    """
    Basic check for frequency invariant load and utilization tracking

    This test runs the same workload on the most capable CPU on the system at a
    cross section of available frequencies.

    This class is mostly a wrapper around :class:`FreqInvarianceItem`,
    providing a way to build a list of those for a few frequencies, and
    providing aggregated versions of the tests. Calling the tests methods on
    the items directly is recommended to avoid the unavoidable loss of
    information when aggregating the
    :class:`~lisa.tests.kernel.test_bundle.Result` of each item.

    `freq_items` instance attribute is a mapping of frequency values in
    Hz to an instance of :class:`FreqInvarianceItem`.
    """

    def __init__(self, res_dir, plat_info, freq_items):
        super().__init__(res_dir, plat_info)

        self.freq_items = freq_items

    @classmethod
    def iter_freq_items(cls, te:TestEnv, res_dir:ArtifactPath) -> FreqInvarianceItem:
        """
        Yield a :class:`FreqInvarianceItem` for a subset of target's
        frequencies.

        This is a generator function.

        :rtype: Iterator[:class:`FreqInvarianceItem`]
        """
        cpu = cls.get_max_capa_cpu(te)
        all_freqs = te.target.cpufreq.list_frequencies(cpu)
        # If we have loads of frequencies just test a cross-section so it
        # doesn't take all day
        freq_list = all_freqs[::len(all_freqs) // 8 + (1 if len(all_freqs) % 2 else 0)]

        # Make sure we have increasing frequency order, to make the logs easier
        # to navigate
        freq_list.sort()

        for freq in freq_list:
            item_dir = os.path.join(res_dir, "{}_{}".format(
                FreqInvarianceItem.task_prefix, freq))
            os.makedirs(item_dir)

            yield FreqInvarianceItem.from_testenv(
                te, freq, all_freqs, res_dir=item_dir,
            )

    @classmethod
    def from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None) -> 'FreqInvariance':
        return super().from_testenv(te, res_dir)

    @classmethod
    def _from_testenv(cls, te, res_dir):
        return cls(res_dir, te.plat_info, OrderedDict(
            (item.freq, item)
            for item in cls.iter_freq_items(te, res_dir)
        ))

    def get_trace(self, freq):
        """
        :returns: The trace generated when running at a given frequency
        """
        return self.freq_items[freq].trace

    # Combined version of some other tests, applied on all available
    # FreqInvarianceItem with the result merged.

    def test_task_util_avg(self, allowed_error_pct=15) -> ResultBundle:
        """
        Aggregated version of :meth:`InvarianceBase.test_task_util_avg`
        """
        def item_test(test_item):
            return test_item.test_task_util_avg(
                allowed_error_pct=allowed_error_pct
            )
        return self._test_all_freq(item_test)

    def test_task_load_avg(self, allowed_error_pct=15) -> ResultBundle:
        """
        Aggregated version of :meth:`FreqInvarianceItem.test_task_load_avg`
        """
        def item_test(test_item):
            return test_item.test_task_load_avg(allowed_error_pct=allowed_error_pct)
        return self._test_all_freq(item_test)

    def _test_all_freq(self, item_test):
        """
        Apply the `test_item` function on all instances of
        :class:`FreqInvarianceItem` and aggregate the returned
        :class:`~lisa.tests.kernel.test_bundle.ResultBundle` into one.

        :attr:`~lisa.tests.kernel.test_bundle.Result.UNDECIDED` is ignored.
        """
        item_res_bundles = {
            freq: item_test(test_item)
            for freq, test_item in self.freq_items.items()
        }

        overall_bundle = ResultBundle.from_bool(all(item_res_bundles.values()))
        for freq, bundle in item_res_bundles.items():
            overall_bundle.add_metric(freq, bundle.metrics)

        overall_bundle.add_metric('failed freqs', [
            freq for freq, bundle in item_res_bundles.items()
            if bundle.result is Result.FAILED
        ])

        return overall_bundle

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
    def from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None) -> 'PELTTask':
        """
        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in :meth:`get_rtapp_profile`
        """
        return super().from_testenv(te, res_dir)

    @classmethod
    def get_rtapp_profile(cls, te):
        # Run the 50% workload on a CPU with highest capacity
        cpu = cls.get_max_capa_cpu(te)

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

    def get_task_sched_signals(self, cpu, signals):
        # We only have one task and one trace, simplify this method a bit
        return super().get_task_sched_signals(self.trace, cpu, self.task_name, signals)

    def get_simulated_pelt(self, cpu, signal_name):
        """
        Get simulated PELT signal and the periodic task used to model it.

        :returns: tuple of
          - :class:`bart.sched.pelt.Simulator.Simulator` the PELT simulator object
          - :class:`bart.sched.pelt.PeriodicTask` simulated periodic task
          - :class:`pandas.DataFrame` instance which reports the computed
          PELT values at each PELT sample interval.
        """
        signal_df = self.get_task_sched_signals(cpu, [signal_name])

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
        signal_df = self.get_task_sched_signals(cpu, [signal_name])

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
        signal_df = self.get_task_sched_signals(cpu, [signal_name])

        trace_duty_cycle = self.get_task_duty_cycle_pct(
            self.trace, self.task_name, cpu)
        requested_duty_cycle = phase.duty_cycle_pct

        # Do a bit of plotting
        fig, axes = plt.subplots(2, 1, figsize=(32, 10), sharex=True)
        self._plot_behaviour(axes[0], signal_df, "Trace signal", trace_duty_cycle)
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
