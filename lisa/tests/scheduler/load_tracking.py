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
import itertools
from statistics import mean

from lisa.tests.base import (
    TestMetric, Result, ResultBundle, AggregatedResultBundle, TestBundle,
    RTATestBundle, CannotCreateError
)
from lisa.target import Target
from lisa.utils import ArtifactPath, groupby, ExekallTaggable
from lisa.datautils import series_mean, df_window, df_filter_task_ids
from lisa.wlgen.rta import Periodic, RTATask
from lisa.trace import FtraceCollector, requires_events
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.pelt import simulate_pelt, pelt_settling_time


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

        df = trace.analysis.tasks.df_task_total_residency(task_name)
        run_time = df['runtime'][cpu]

        return (run_time * 100) / trace.time_range

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

    cpufreq_conf = {
        "governor" : "performance"
    }
    """
    The cpufreq configuration used while the synthetic workload is being run.
    Items are arguments to :meth:`devlib.cpufreq.use_governor`.
    """

    @classmethod
    def _from_target(cls, target:Target, *, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'LoadTrackingBase':
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
                cls.run_rtapp(target, res_dir, rtapp_profile, ftrace_coll)

        return cls(res_dir, plat_info)

    @LoadTrackingAnalysis.df_tasks_signal.used_events
    @requires_events('sched_switch')
    def get_task_sched_signal(self, trace, cpu, task_name, signal):
        """
        Get a :class:`pandas.DataFrame` with the sched signals for the workload task

        This examines scheduler load tracking trace events. You will need a
        target kernel that includes the required events.

        :returns: :class:`pandas.DataFrame` with a column for each signal for
          the workload task
        """
        df = trace.analysis.load_tracking.df_tasks_signal(signal)
        df = df[df['comm'] == task_name]
        window = self.get_task_window(trace, task_name, cpu)
        df = df_window(df, window, method='exclusive')

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

class InvarianceItem(LoadTrackingBase, ExekallTaggable):
    """
    Basic check for CPU and frequency invariant load and utilization tracking

    **Expected Behaviour:**

    Load tracking signals are scaled so that the workload results in
    roughly the same util & load values regardless of compute power of the
    CPU used and its frequency.
    """
    task_prefix = 'invar'
    cpufreq_conf = {
        "governor" : "userspace"
    }

    def __init__(self, res_dir, plat_info, cpu, freq, freq_list):
        super().__init__(res_dir, plat_info)

        self.freq = freq
        self.freq_list = freq_list
        self.cpu = cpu

    @property
    def rtapp_profile(self):
        return self.get_rtapp_profile(self.plat_info, cpu=self.cpu, freq=self.freq)

    def get_tags(self):
        return {'cpu': '{}@{}'.format(self.cpu, self.freq)}

    @classmethod
    def get_rtapp_profile(cls, plat_info, cpu, freq):
        """
        Get a specification for a rt-app workload with the specificied duty
        cycle, pinned to the given CPU.
        """
        freq_capa = cls._get_freq_capa(cpu, freq, plat_info)
        duty_cycle_pct = freq_capa/UTIL_SCALE * 100
        # Use half of the capacity at that OPP, so we are sure that the
        # task will fit even at the lowest OPP
        duty_cycle_pct //= 2

        rtapp_profile = {}
        rtapp_profile["{}{}".format(cls.task_prefix, cpu)] = Periodic(
            duty_cycle_pct=duty_cycle_pct,
            duration_s=2,
            period_ms=cls.TASK_PERIOD_MS,
            cpus=[cpu],
        )

        return rtapp_profile

    @classmethod
    def _from_target(cls, target:Target, *, cpu:int, freq:int, freq_list=None, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'InvarianceItem':
        """
        :param cpu: CPU to use, or ``None`` to automatically choose an
            appropriate set of CPUs.
        :type cpu: int or None

        :param freq: Frequency to run at in kHz. It is only relevant in
            combination with ``cpu``.
        :type freq: int or None
        """
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info, cpu, freq)
        logger = cls.get_logger()

        with target.cpufreq.use_governor(**cls.cpufreq_conf):
            target.cpufreq.set_frequency(cpu, freq)
            logger.debug('CPU{} frequency: {}'.format(cpu, target.cpufreq.get_frequency(cpu)))
            cls.run_rtapp(target, res_dir, rtapp_profile, ftrace_coll)

        freq_list = freq_list or [freq]
        return cls(res_dir, plat_info, cpu, freq, freq_list)

    @requires_events('sched_switch')
    def get_expected_util_avg(self, trace, cpu, task_name, capacity):
        """
        Examine trace to figure out an expected mean for util_avg

        Assumes an RT-App workload with a single phase
        """
        # Find duty cycle of the workload task
        duty_cycle_pct = self.get_task_duty_cycle_pct(trace, task_name, cpu)

        # Scale the relative CPU/freq capacity
        return (duty_cycle_pct / 100) * capacity

    @LoadTrackingBase.get_task_sched_signal.used_events
    @get_expected_util_avg.used_events
    def _test_task_signal(self, signal_name, allowed_error_pct,
                          trace, cpu, task_name, capacity):
        # Use utilization signal for both load and util, since they should be
        # proportionnal in the test environment we setup
        exp_signal = self.get_expected_util_avg(trace, cpu, task_name, capacity)
        signal_df = self.get_task_sched_signal(trace, cpu, task_name, signal_name)
        signal = signal_df[UTIL_AVG_CONVERGENCE_TIME_S:][signal_name]
        signal_mean = series_mean(signal)

        # Since load is now CPU invariant in recent kernel versions, we don't
        # rescale it back. To match the old behavior, that line is
        # needed:
        #  exp_signal /= (self.plat_info['cpu-capacities'][cpu] / UTIL_SCALE)
        kernel_version = self.plat_info['kernel']['version']
        if (
            signal_name == 'load'
            and kernel_version.parts[:2] < (5, 1)
        ):
            self.logger.warning('Load signal is assumed to be CPU invariant, which is true for recent mainline kernels, but may be wrong for {}'.format(
                kernel_version,
            ))

        ok = self.is_almost_equal(exp_signal, signal_mean, allowed_error_pct)

        return ok, exp_signal, signal_mean

    @staticmethod
    def _get_freq_capa(cpu, freq, plat_info):
        capacity = plat_info['cpu-capacities'][cpu]
        # Scale the capacity linearly according to the frequency
        max_freq = max(plat_info['freqs'][cpu])
        capacity *= freq / max_freq

        return capacity

    @_test_task_signal.used_events
    def _test_signal(self, signal_name, allowed_error_pct):
        passed = True
        expected_data = {}
        trace_data = {}

        capacity = self._get_freq_capa(self.cpu, self.freq, self.plat_info)

        for name in self.rtapp_tasks:
            ok, exp_util, signal_mean = self._test_task_signal(
                signal_name, allowed_error_pct, self.trace, self.cpu, name, capacity)

            if not ok:
                passed = False

            expected_data[name] = TestMetric(exp_util)
            trace_data[name] = TestMetric(signal_mean)

        freq_str = '@{}'.format(self.freq) if self.freq is not None else ''
        bundle = ResultBundle.from_bool(passed)
        bundle.add_metric("cpu", '{}{}'.format(self.cpu, freq_str))
        bundle.add_metric("Expected signals", expected_data)
        bundle.add_metric("Trace signals", trace_data)
        return bundle

    @_test_signal.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
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

    @_test_signal.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
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
            # Make sure the last one is the max freq
            freq_list[-1] = all_freqs[-1]

            # Make sure we have increasing frequency order, to make the logs easier
            # to navigate
            freq_list.sort()

            for freq in freq_list:
                item_dir = ArtifactPath.join(res_dir, "{prefix}_{cpu}@{freq}".format(
                    prefix=InvarianceItem.task_prefix,
                    cpu=cpu,
                    freq=freq,
                ))
                os.makedirs(item_dir)

                logger.info('Running experiment for CPU {}@{}'.format(cpu, freq))
                yield InvarianceItem.from_target(
                    target, cpu=cpu, freq=freq, freq_list=all_freqs, res_dir=item_dir,
                    ftrace_coll=ftrace_coll,
                )

    def iter_invariance_items(self) -> InvarianceItem:
        yield from self.invariance_items

    @classmethod
    def _from_target(cls, target:Target, *, res_dir:ArtifactPath=None, ftrace_coll:FtraceCollector=None) -> 'Invariance':
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

    @InvarianceItem.test_task_util_avg.used_events
    def test_task_util_avg(self, allowed_error_pct=15) -> AggregatedResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_task_util_avg`
        """
        def item_test(test_item):
            return test_item.test_task_util_avg(
                allowed_error_pct=allowed_error_pct
            )
        return self._test_all_freq(item_test)

    @InvarianceItem.test_task_load_avg.used_events
    def test_task_load_avg(self, allowed_error_pct=15) -> AggregatedResultBundle:
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
        item_res_bundles = [
            item_test(item)
            for item in self.invariance_items
        ]
        return AggregatedResultBundle(item_res_bundles, 'cpu')

    @InvarianceItem.test_task_util_avg.used_events
    def test_cpu_invariance(self) -> AggregatedResultBundle:
        """
        Check that items using the max freq on each CPU is passing util avg test.

        There could be false positives, but they are expected to be relatively
        rare.

        .. seealso:: :class:`InvarianceItem.test_task_util_avg`
        """
        res_list = []
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
                res_list.append(res)

        return AggregatedResultBundle(res_list, 'cpu')

    @InvarianceItem.test_task_util_avg.used_events
    def test_freq_invariance(self) -> ResultBundle:
        """
        Check that at least one CPU has items passing for all tested frequencies.

        .. seealso:: :class:`InvarianceItem.test_task_util_avg`
        """

        logger = self.logger

        def make_group_bundle(cpu, item_group):
            bundle = AggregatedResultBundle(
                [
                    # Only test util, as it should be more robust
                    item.test_task_util_avg()
                    for item in item_group
                ],
                # each item's "cpu" metric also contains the frequency
                name_metric='cpu',
            )
            # At that level, we only report the CPU, since nested bundles cover
            # different frequencies
            bundle.add_metric('cpu', cpu)

            logger.info('Util avg invariance {res} for CPU {cpu}'.format(
                res=bundle.result.lower_name,
                cpu=cpu,
            ))
            return bundle

        group_result_bundles = [
            make_group_bundle(cpu, item_group)
            for cpu, item_group in groupby(self.invariance_items, key=lambda x: x.cpu)
        ]

        # The combination differs from the AggregatedResultBundle default one:
        # we consider as passed as long as at least one of the group has
        # passed, instead of forcing all of them to pass.
        if any(result_bundle.result is Result.PASSED for result_bundle in group_result_bundles):
            overall_result = Result.PASSED
        elif all(result_bundle.result is Result.UNDECIDED for result_bundle in group_result_bundles):
            overall_result = Result.UNDECIDED
        else:
            overall_result = Result.FAILED

        return AggregatedResultBundle(
            group_result_bundles,
            name_metric='cpu',
            result=overall_result
        )

class PELTTask(LoadTrackingBase):
    """
    Basic checks for task related PELT signals behaviour

    This test runs a synthetic periodic task on a CPU in the system and
    collects a trace from the target device. The util_avg values are extracted
    from scheduler trace events and the behaviour of the signal is compared
    against a simulated value of PELT.
    """

    task_prefix = 'pelt'

    TASK_PERIOD_MS = 16
    """
    Force a small enough period so that average util signal is close to the
    duty cycle. This is not generally true, since the signal is not modified
    when the task is sleeping, leading to a higher average. If the period is
    small enough, the difference stays small.
    """

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        # Run the 50% workload on a CPU with highest capacity
        cpu = cls.get_max_capa_cpu(plat_info)

        rtapp_profile = {}
        rtapp_profile["{}{}".format(cls.task_prefix, cpu)] = Periodic(
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
        tasks = self.rtapp_tasks
        assert len(tasks) == 1
        return tasks[0]

    @property
    def wlgen_task(self):
        """
        The :class:`lisa.wlgen.rta.RTATask` description of the only rt-app
        task, as specified in the profile.
        """
        tasks = list(self.rtapp_profile.values())
        assert len(tasks) == 1
        return tasks[0]

    @LoadTrackingBase.get_task_sched_signal.used_events
    def get_task_sched_signal(self, cpu, signal):
        # We only have one task and one trace, simplify this method a bit
        return super().get_task_sched_signal(self.trace, cpu, self.task_name, signal)

    @requires_events('sched_switch')
    def get_simulated_pelt(self, task, signal_name):
        """
        Simulate a PELT signal for a given task.

        :param task: task to look for in the trace.
        :type task: int or str or tuple(int, str)

        :param signal_name: Name of the PELT signal to simulate.
        :type signal_name: str

        :return: A :class:`pandas.DataFrame` with a ``simulated`` column
            containing the simulated signal, along with the column of the
            signal as found in the trace.
        """
        logger = self.logger
        trace = self.trace
        task = trace.get_task_id(task)
        cpus = trace.analysis.tasks.cpus_of_tasks([task])

        # Capacity lower than 1024 will create some time-scaling artifacts that
        # are not currently simulated
        assert all(
            self.plat_info["cpu-capacities"][cpu] == UTIL_SCALE
            for cpu in cpus
        )

        df_activation = trace.analysis.tasks.df_task_activation(task)
        df = trace.analysis.load_tracking.df_tasks_signal(signal_name)
        df = df_filter_task_ids(df, [task])

        # Ignore the first activation, as its signals are incorrect
        df_activation = df_activation.iloc[2:]

        # Make sure the activation df does not start before the dataframe of
        # signal values, otherwise we cannot provide a sensible init value
        df_activation = df_activation[df.index[0]:]

        # Get the initial signal value matching the first activation we will care about
        init_iloc = df.index.get_loc(df_activation.index[0], method='ffill')
        init = df[signal_name].iloc[init_iloc]

        try:
            # PELT clock in nanoseconds
            clock = df['update_time'] * 1e-9
        except KeyError:
            logger.warning('PELT clock is not available, ftrace timestamp will be used at the expense of accuracy')
            clock = None

        df['simulated'] = simulate_pelt(df_activation['active'], index=df.index, init=init, clock=clock)
        df['error'] = df[signal_name] - df['simulated']

        df = df.dropna()
        return df

    def _plot_pelt(self, task, signal_name, simulated, test_name):
        trace = self.trace

        kwargs = dict(always_save=False, interactive=False)

        axis = trace.analysis.load_tracking.plot_task_signals(task, signals=[signal_name], **kwargs)
        simulated.plot(ax=axis, drawstyle='steps-post', label='simulated {}'.format(signal_name))
        trace.analysis.tasks.plot_task_activation(task, alpha=0.2, axis=axis, **kwargs)

        axis.legend()

        path = ArtifactPath.join(self.res_dir, '{}_{}.png'.format(test_name, signal_name))
        trace.analysis.load_tracking.save_plot(axis.get_figure(), filepath=path)

    @get_simulated_pelt.used_events
    def _test_behaviour(self, signal_name, error_margin_pct):

        task = self.task_name
        phase = self.wlgen_task.phases[0]
        df = self.get_simulated_pelt(task, signal_name)

        expected_duty_cycle_pct = phase.duty_cycle_pct
        expected_final_util = expected_duty_cycle_pct / 100 * UTIL_SCALE
        settling_time = pelt_settling_time(1, init=0, final=expected_final_util)
        settling_time += df.index[0]

        df = df[settling_time:]

        settled_signal_mean = series_mean(df[signal_name])
        expected_signal_mean = expected_final_util

        signal_mean_error_pct = abs(expected_signal_mean - settled_signal_mean) / UTIL_SCALE * 100
        res = ResultBundle.from_bool(signal_mean_error_pct < error_margin_pct)

        res.add_metric('expected mean', expected_signal_mean)
        res.add_metric('settled mean', settled_signal_mean)
        res.add_metric('settled mean error', signal_mean_error_pct, '%')

        self._plot_pelt(task, signal_name, df['simulated'], 'behaviour')

        return res

    @get_simulated_pelt.used_events
    def _test_correctness(self, signal_name, mean_error_margin_pct, max_error_margin_pct):

        task = self.task_name
        df = self.get_simulated_pelt(task, signal_name)

        abs_error = df['error'].abs()
        mean_error_pct = series_mean(abs_error) / UTIL_SCALE * 100
        max_error_pct = abs_error.max() / UTIL_SCALE * 100

        mean_ok = mean_error_pct <= mean_error_margin_pct
        max_ok = max_error_pct <= max_error_margin_pct

        res = ResultBundle.from_bool(mean_ok and max_ok)

        res.add_metric('actual mean', series_mean(df[signal_name]))
        res.add_metric('simulated mean', series_mean(df['simulated']))
        res.add_metric('mean error', mean_error_pct, '%')

        res.add_metric('actual max', df[signal_name].max())
        res.add_metric('simulated max', df['simulated'].max())
        res.add_metric('max error', max_error_pct, '%')

        self._plot_pelt(task, signal_name, df['simulated'], 'correctness')

        return res

    @_test_correctness.used_events
    def test_util_correctness(self, mean_error_margin_pct=2, max_error_margin_pct=2) -> ResultBundle:
        """
        Check that the utilization signal is as expected.

        :param mean_error_margin_pct: Maximum allowed difference in the mean of
            the actual signal and the simulated one, as a percentage of utilization
            scale.
        :type mean_error_margin_pct: float

        :param max_error_margin_pct: Maximum allowed difference between samples
            of the actual signal and the simulated one, as a percentage of
            utilization scale.
        :type max_error_margin_pct: float
        """
        return self._test_correctness(
            signal_name='util',
            mean_error_margin_pct=mean_error_margin_pct,
            max_error_margin_pct=max_error_margin_pct,
        )

    @_test_correctness.used_events
    def test_load_correctness(self, mean_error_margin_pct=2, max_error_margin_pct=2) -> ResultBundle:
        """
        Same as :meth:`test_util_correctness` but checking the load.
        """
        return self._test_correctness(
            signal_name='load',
            mean_error_margin_pct=mean_error_margin_pct,
            max_error_margin_pct=max_error_margin_pct,
        )


    @_test_behaviour.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_util_avg_behaviour(self, error_margin_pct=5) -> ResultBundle:
        """
        Check the utilization mean is linked to the task duty cycle.


        .. note:: That is not really the case, as the util of a task is not
            updated when the task is sleeping, but is fairly close to reality
            as long as the task period is small enough.

        :param error_margin_pct: Allowed difference in percentage of
            utilization scale.
        :type error_margin_pct: float

        """
        return self._test_behaviour('util', error_margin_pct)

    @_test_behaviour.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_load_avg_behaviour(self, error_margin_pct=5) -> ResultBundle:
        """
        Same as :meth:`test_util_avg_behaviour` but checking the load.
        """
        return self._test_behaviour('load', error_margin_pct)


class CPUMigrationBase(LoadTrackingBase):
    """
    Base class for migration-related load tracking tests

    The idea here is to run several rt-app tasks and to have them pinned to
    a single CPU for a single phase. They can change CPUs in a new phase,
    and we can then inspect the CPU utilization - it should match the
    sum of the utilization of all the tasks running on it.

    **Design notes:**

    Since we sum up the utilization of each task, make sure not to overload the
    CPU - IOW, there should always be some idle cycles.

    The code assumes all tasks have the same number of phases, and that those
    phases are all aligned.
    """

    PHASE_DURATION_S = 3 * UTIL_AVG_CONVERGENCE_TIME_S
    """
    The duration of a single phase
    """

    TASK_PERIOD_MS = 16
    """
    The average value of the runqueue PELT signals is very dependent on the task
    period, so it's important to set it to a known validate value in that class.
    """

    @abc.abstractmethod
    def get_nr_required_cpu(cls, plat_info):
        """
        The number of CPUs of same capacity involved in the test
        """
        pass

    @classmethod
    def run_rtapp(cls, target, res_dir, profile, ftrace_coll, cgroup=None):
        # Just do some validation on the profile
        for name, task in profile.items():
            for phase in task.phases:
                if len(phase.cpus) != 1:
                    raise RuntimeError("Each phase must be tied to a single CPU. "
                                       "Task \"{}\" violates this".format(name))

        super().run_rtapp(target, res_dir, profile, ftrace_coll, cgroup)

    @property
    def cpus(self):
        """
        All CPUs used by RTapp workload.
        """
        return set(itertools.chain.from_iterable(
            phase.cpus
            for task in self.rtapp_profile.values()
            for phase in task.phases
        ))

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)

        try:
            target.plat_info["cpu-capacities"]
        except KeyError as e:
            raise CannotCreateError(str(e))

        # Check that there are enough CPUs of the same capacity
        cls.get_migration_cpus(target.plat_info)

    @classmethod
    def get_migration_cpus(cls, plat_info):
        """
        :returns: N CPUs of same capacity, with N set by :meth:`get_nr_required_cpu`.
        """
        # Iterate over descending CPU capacity groups
        nr_required_cpu = cls.get_nr_required_cpu(plat_info)
        for cpus in reversed(plat_info["capacity-classes"]):
            if len(cpus) >= nr_required_cpu:
                return cpus[:nr_required_cpu]

        raise CannotCreateError(
            "This workload requires {} CPUs of identical capacity".format(
                nr_required_cpu))

    def get_expected_cpu_util(self):
        """
        Get the per-phase average CPU utilization expected from the rtapp profile

        :returns: A dict of the shape {cpu : {phase_id : expected_util}}
        """
        cpu_util = {}
        for task in self.rtapp_profile.values():
            for phase_id, phase in enumerate(task.phases):
                cpu = phase.cpus[0]
                cpu_util.setdefault(cpu, {}).setdefault(phase_id, 0)
                cpu_util[cpu][phase_id] += UTIL_SCALE * (phase.duty_cycle_pct / 100)

        return cpu_util

    @property
    def reference_task(self):
        return list(self.rtapp_profile.values())[0]

    @LoadTrackingAnalysis.df_cpus_signal.used_events
    def get_trace_cpu_util(self):
        """
        Get the per-phase average CPU utilization read from the trace

        :returns: A dict of the shape {cpu : {phase_id : trace_util}}
        """
        df = self.trace.analysis.load_tracking.df_cpus_signal('util')
        phase_start = self.trace.start
        cpu_util = {}

        for i, phase in enumerate(self.reference_task.phases):
            # Start looking at signals once they should've converged
            start = phase_start + UTIL_AVG_CONVERGENCE_TIME_S
            # Trim the end a bit, otherwise we could have one or two events
            # from the next phase
            end = phase_start + phase.duration_s * .9
            phase_df = df[start:end]

            for cpu in self.cpus:
                util = phase_df[phase_df.cpu == cpu].util
                # The runqueue util signal's average does not match the duty
                # cycle of the task, since it "decays instantly" at next task
                # wakeup, but stays at its previous value when the task sleeps.
                # This means that rq PELT signal average is higher than the
                # idealized PELT signal. Using trapz integration allows to
                # lower the contribution of the sleep-time util, since it links
                # with a straight line the point when task goes to sleep with
                # the wakeup util point.
                cpu_util.setdefault(cpu, {})[i] = series_mean(util, method='trapz')

            phase_start += phase.duration_s

        return cpu_util

    @get_trace_cpu_util.used_events
    @RTATestBundle.check_noisy_tasks(noise_threshold_pct=1)
    def test_util_task_migration(self, allowed_error_pct=5) -> ResultBundle:
        """
        Test that a migrated task properly propagates its utilization at the CPU level

        :param allowed_error_pct: How much the trace averages can stray from the
          expected values
        :type allowed_error_pct: float
        """
        expected_cpu_util = self.get_expected_cpu_util()
        trace_cpu_util = self.get_trace_cpu_util()

        passed = True

        expected_metrics = {}
        trace_metrics = {}
        deltas = {}

        for cpu in self.cpus:
            cpu_str = "cpu{}".format(cpu)

            expected_metrics[cpu_str] = TestMetric({})
            trace_metrics[cpu_str] = TestMetric({})
            deltas[cpu_str] = TestMetric({})

            for i, phase in enumerate(self.reference_task.phases):
                if not self.is_almost_equal(
                        trace_cpu_util[cpu][i],
                        expected_cpu_util[cpu][i],
                        allowed_error_pct):
                    passed = False

                # Just some verbose metric collection...
                phase_str = "phase{}".format(i)

                expected = expected_cpu_util[cpu][i]
                trace = trace_cpu_util[cpu][i]
                delta = 100 * (trace - expected) / expected

                expected_metrics[cpu_str].data[phase_str] = TestMetric(expected)
                trace_metrics[cpu_str].data[phase_str] = TestMetric(trace)
                deltas[cpu_str].data[phase_str] = TestMetric(delta, "%")

        res = ResultBundle.from_bool(passed)
        res.add_metric("Expected utilization", expected_metrics)
        res.add_metric("Trace utilization", trace_metrics)
        res.add_metric("Utilization deltas", deltas)

        return res

class OneTaskCPUMigration(CPUMigrationBase):
    """
    Some tasks on two big CPUs, one of them migrates in its second phase.
    """

    @classmethod
    def get_nr_required_cpu(cls, plat_info):
        return 2

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        profile = {}
        cpus = cls.get_migration_cpus(plat_info)

        for task in ["migr", "static0", "static1"]:
            # An empty RTATask just to sum phases up
            profile[task] = RTATask()

        common_phase_settings = dict(
            duration_s=cls.PHASE_DURATION_S,
            period_ms=cls.TASK_PERIOD_MS,
        )

        for cpu in cpus:
            # A task that will migrate to another CPU
            profile["migr"] += Periodic(
                duty_cycle_pct=cls.unscaled_utilization(plat_info, cpu, 20),
                cpus=[cpu], **common_phase_settings)

            # Just some tasks that won't move to get some background utilization
            profile["static0"] += Periodic(
                duty_cycle_pct=cls.unscaled_utilization(plat_info, cpus[0], 30),
                cpus=[cpus[0]], **common_phase_settings)

            profile["static1"] += Periodic(
                duty_cycle_pct=cls.unscaled_utilization(plat_info, cpus[1], 20),
                cpus=[cpus[1]], **common_phase_settings)

        return profile

class NTasksCPUMigrationBase(CPUMigrationBase):
    """
    N tasks on N CPUs, with all the migration permutations.
    """

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        cpus = cls.get_migration_cpus(plat_info)
        make_name = lambda i: 'migr{}'.format(i)

        nr_tasks = len(cpus)
        profile = {
            make_name(i): RTATask()
            for i in range(nr_tasks)
        }

        # Define one task per CPU, and create all the possible migrations by
        # shuffling around these tasks
        for cpus_combi in itertools.permutations(cpus, r=nr_tasks):
            for i, cpu in enumerate(cpus_combi):
                profile[make_name(i)] += Periodic(
                    duty_cycle_pct=cls.unscaled_utilization(plat_info, cpu, 50),
                    duration_s=cls.PHASE_DURATION_S,
                    period_ms=cls.TASK_PERIOD_MS,
                    cpus=[cpu],
                )

        return profile


class TwoTasksCPUMigration(NTasksCPUMigrationBase):
    """
    Two tasks on two big CPUs, swap their CPU in the second phase
    """
    @classmethod
    def get_nr_required_cpu(cls, plat_info):
        return 2


class NTasksCPUMigration(NTasksCPUMigrationBase):
    """
    N tasks on N CPUs, and try all permutations of tasks and CPUs.
    """

    @classmethod
    def get_nr_required_cpu(cls, plat_info):
        """
        Select the maximum number of CPUs the tests can handle.
        """
        return max(len(cpus) for cpus in plat_info["capacity-classes"])

    def test_util_task_migration(self, allowed_error_pct=8) -> ResultBundle:
        """
        Relax the margins compared to the super-class version.
        """
        return super().test_util_task_migration(
            allowed_error_pct=allowed_error_pct,
        )
 # vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
