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

import pandas as pd

from lisa.tests.base import (
    TestMetric, Result, ResultBundle, AggregatedResultBundle, TestBundleBase,
    TestBundle, RTATestBundle, CannotCreateError
)
from lisa.target import Target
from lisa.utils import ArtifactPath, groupby, ExekallTaggable, add, memoized, kwargs_forwarded_to
from lisa.datautils import series_mean, df_window, df_filter_task_ids, series_refit_index, df_split_signals, df_refit_index, series_dereference
from lisa.wlgen.rta import RTA, RTAPhase, PeriodicWload
from lisa.trace import requires_events, may_use_events, MissingTraceEventError
from lisa.analysis.load_tracking import LoadTrackingAnalysis
from lisa.analysis.tasks import TasksAnalysis
from lisa.analysis.rta import RTAEventsAnalysis
from lisa.analysis.frequency import FrequencyAnalysis
from lisa.pelt import PELT_SCALE, simulate_pelt, pelt_settling_time, kernel_util_mean

UTIL_SCALE = PELT_SCALE

UTIL_CONVERGENCE_TIME_S = pelt_settling_time(1, init=0, final=1024)
"""
Time in seconds for util_avg to converge (i.e. ignored time)
"""


class LoadTrackingHelpers:
    """
    Common bunch of helpers for load tracking tests.
    """

    MAX_RTAPP_CALIB_DEVIATION = 3 / 100
    """
    Blacklist CPUs that have a RTapp calibration value that deviates too much
    from the average calib value in their capacity class.
    """

    @classmethod
    def _get_blacklisted_cpus(cls, plat_info):
        """
        :meta public:

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
        Filter out capacity-classes key of ``plat_info`` to remove blacklisted
        CPUs provided by:
        """
        blacklisted_cpus = set(cls._get_blacklisted_cpus(plat_info))
        return [
            sorted(set(cpu_class) - blacklisted_cpus)
            for cpu_class in plat_info['capacity-classes']
        ]

    @classmethod
    def correct_expected_pelt(cls, plat_info, cpu, signal_value):
        """
        Correct an expected PELT signal from ``rt-app`` based on the calibration
        values.

        Since the instruction mix of ``rt-app`` might not be the same as the
        benchmark that was used to establish CPU capacities, the duty cycle of
        ``rt-app`` will only be accurate on big CPUs. When we know on which CPU
        the task actually executed, we can correct the expected value based on
        the ratio of calibration values and CPU capacities.
        """

        calib = plat_info['rtapp']['calib']
        rtapp_capacities = plat_info['cpu-capacities']['rtapp']
        orig_capacities = plat_info['cpu-capacities']['orig']

        # Correct the signal mean to what it should have been if rt-app
        # workload was exactly the same as the one used to establish CPU
        # capacities
        return signal_value * orig_capacities[cpu] / rtapp_capacities[cpu]


class LoadTrackingBase(RTATestBundle, LoadTrackingHelpers, TestBundle):
    """
    Base class for shared functionality of load tracking tests
    """

    cpufreq_conf = {
        "governor": "performance"
    }
    """
    The cpufreq configuration used while the synthetic workload is being run.
    Items are arguments to
    :meth:`devlib.module.cpufreq.CpufreqModule.use_governor`.
    """

    @classmethod
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath = None, collector=None) -> 'LoadTrackingBase':
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
                cls.run_rtapp(
                    target=target,
                    res_dir=res_dir,
                    profile=rtapp_profile,
                    collector=collector
                )

        return cls(res_dir, plat_info)

    @staticmethod
    def is_almost_equal(target, value, allowed_delta_pct):
        """
        Verify that ``value``` is reasonably close to ``target```

        :returns: A tuple (bool, delta_pct)
        """
        delta = value - target
        delta_pct = delta / target * 100
        equal = abs(delta_pct) <= allowed_delta_pct

        return (equal, delta_pct)


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
        "governor": "userspace"
    }

    def __init__(self, res_dir, plat_info, cpu, freq, freq_list):
        super().__init__(res_dir, plat_info)

        self.freq = freq
        self.freq_list = freq_list
        self.cpu = cpu

    @property
    def rtapp_profile(self):
        return self.get_rtapp_profile(self.plat_info, cpu=self.cpu, freq=self.freq)

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

    def get_tags(self):
        return {'cpu': f'{self.cpu}@{self.freq}'}

    @classmethod
    def _get_rtapp_profile(cls, plat_info, cpu, freq):
        """
        :meta public:

        Get a specification for a rt-app workload with the specificied duty
        cycle, pinned to the given CPU.
        """
        freq_capa = cls._get_freq_capa(cpu, freq, plat_info)
        duty_cycle_pct = freq_capa / UTIL_SCALE * 100
        # Use half of the capacity at that OPP, so we are sure that the
        # task will fit even at the lowest OPP
        duty_cycle_pct //= 2

        return {
            f"{cls.task_prefix}{cpu}": RTAPhase(
                prop_wload=PeriodicWload(
                    duty_cycle_pct=duty_cycle_pct,
                    duration=2,
                    period=cls.TASK_PERIOD,
                ),
                prop_cpus=[cpu],
            )
        }

    @classmethod
    def _from_target(cls, target: Target, *, cpu: int, freq: int, freq_list=None, res_dir: ArtifactPath = None, collector=None) -> 'InvarianceItem':
        """
        :meta public:

        :param cpu: CPU to use, or ``None`` to automatically choose an
            appropriate set of CPUs.
        :type cpu: int or None

        :param freq: Frequency to run at in kHz. It is only relevant in
            combination with ``cpu``.
        :type freq: int or None
        """
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info, cpu=cpu, freq=freq)
        logger = cls.get_logger()

        with target.cpufreq.use_governor(**cls.cpufreq_conf):
            target.cpufreq.set_frequency(cpu, freq)
            logger.debug(f'CPU{cpu} frequency: {target.cpufreq.get_frequency(cpu)}')
            cls.run_rtapp(
                target=target,
                res_dir=res_dir,
                profile=rtapp_profile,
                collector=collector
            )

        freq_list = freq_list or [freq]
        return cls(res_dir, plat_info, cpu, freq, freq_list)

    @staticmethod
    def _get_freq_capa(cpu, freq, plat_info):
        capacity = plat_info['cpu-capacities']['rtapp'][cpu]
        # Scale the capacity linearly according to the frequency
        max_freq = max(plat_info['freqs'][cpu])
        capacity *= freq / max_freq

        return capacity

    @LoadTrackingAnalysis.df_task_signal.used_events
    @LoadTrackingAnalysis.df_cpus_signal.used_events
    @TasksAnalysis.df_task_activation.used_events
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
        logger = self.get_logger()
        trace = self.trace
        task = trace.get_task_id(task)
        cpus = trace.analysis.tasks.cpus_of_tasks([task])

        df_activation = trace.analysis.tasks.df_task_activation(
            task,
            # Util only takes into account times where the task is actually
            # executing
            preempted_value=0,
        )
        df = trace.analysis.load_tracking.df_task_signal(task, signal_name)
        df = df.copy(deep=False)

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
            if any(
                self.plat_info['cpu-capacities']['rtapp'][cpu] != UTIL_SCALE
                for phase in self.wlgen_task.phases
                for cpu in phase['cpus']
            ):
                raise CannotCreateError('PELT time scaling can only be simulated when the PELT clock is available from the trace')

            logger.warning('PELT clock is not available, ftrace timestamp will be used at the expense of accuracy')
            clock = None

        try:
            capacity = trace.analysis.load_tracking.df_cpus_signal('capacity', cpus)
        except MissingTraceEventError:
            capacity = None
        else:
            # Reshape the capacity dataframe so that we get one column per CPU
            capacity = capacity.pivot(columns=['cpu'])
            capacity.columns = capacity.columns.droplevel(0)
            capacity.ffill(inplace=True)
            # Make sure we end up with the timestamp at which the capacity
            # changes, rather than the timestamps at which the task is enqueued
            # or dequeued.
            activation_cpu = df_activation['cpu'].reindex(capacity.index, method='ffill')
            capacity = series_dereference(activation_cpu, capacity)

        df['simulated'] = simulate_pelt(
            df_activation['active'],
            index=df.index,
            init=init,
            clock=clock,
            capacity=capacity,
        )

        # Since load is now CPU invariant in recent kernel versions, we don't
        # rescale it back. To match the old behavior, that line is
        # needed:
        #  df['simulated'] /= self.plat_info['cpu-capacities']['rtapp'][cpu] / UTIL_SCALE
        kernel_version = self.plat_info['kernel']['version']
        if (
            signal_name == 'load'
            and kernel_version.parts[:2] < (5, 1)
        ):
            logger().warning(f'Load signal is assumed to be CPU invariant, which is true for recent mainline kernels, but may be wrong for {kernel_version}')

        df['error'] = df[signal_name] - df['simulated']
        df = df.dropna()
        return df

    def _plot_pelt(self, task, signal_name, simulated, test_name):
        trace = self.trace

        axis = trace.analysis.load_tracking.plot_task_signals(task, signals=[signal_name])
        simulated.plot(ax=axis, drawstyle='steps-post', label=f'simulated {signal_name}')

        activation_axis = axis.twinx()
        trace.analysis.tasks.plot_task_activation(task, alpha=0.2, axis=activation_axis, duration=True)

        axis.legend()

        path = ArtifactPath.join(self.res_dir, f'{test_name}_{signal_name}.png')
        trace.analysis.load_tracking.save_plot(axis.get_figure(), filepath=path)

    def _add_cpu_metric(self, res_bundle):
        freq_str = f'@{self.freq}' if self.freq is not None else ''
        res_bundle.add_metric("cpu", f'{self.cpu}{freq_str}')
        return res_bundle

    @memoized
    @get_simulated_pelt.used_events
    def _test_behaviour(self, signal_name, error_margin_pct):

        task = self.task_name
        phase = self.wlgen_task.phases[0]
        df = self.get_simulated_pelt(task, signal_name)

        cpus = sorted(phase['cpus'])
        assert len(cpus) == 1
        cpu = cpus[0]

        expected_duty_cycle_pct = phase['wload'].unscaled_duty_cycle_pct(self.plat_info)
        expected_final_util = expected_duty_cycle_pct / 100 * UTIL_SCALE
        settling_time = pelt_settling_time(10, init=0, final=expected_final_util)
        settling_time += df.index[0]

        df = df[settling_time:]

        # Instead of taking the mean, take the average between the min and max
        # values of the settled signal. This avoids the bias introduced by the
        # fact that the util signal stays high while the task sleeps
        settled_signal_mean = kernel_util_mean(df[signal_name], plat_info=self.plat_info)
        expected_signal_mean = expected_final_util

        signal_mean_error_pct = abs(expected_signal_mean - settled_signal_mean) / UTIL_SCALE * 100
        res = ResultBundle.from_bool(signal_mean_error_pct < error_margin_pct)

        res.add_metric('expected mean', expected_signal_mean)
        res.add_metric('settled mean', settled_signal_mean)
        res.add_metric('settled mean error', signal_mean_error_pct, '%')

        self._plot_pelt(task, signal_name, df['simulated'], 'behaviour')

        res = self._add_cpu_metric(res)
        return res

    @memoized
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

        res = self._add_cpu_metric(res)
        return res

    @_test_correctness.used_events
    def test_util_correctness(self, mean_error_margin_pct=2, max_error_margin_pct=5) -> ResultBundle:
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
    def test_load_correctness(self, mean_error_margin_pct=2, max_error_margin_pct=5) -> ResultBundle:
        """
        Same as :meth:`test_util_correctness` but checking the load.
        """
        return self._test_correctness(
            signal_name='load',
            mean_error_margin_pct=mean_error_margin_pct,
            max_error_margin_pct=max_error_margin_pct,
        )

    @_test_behaviour.used_events
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    def test_util_behaviour(self, error_margin_pct=5) -> ResultBundle:
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
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    def test_load_behaviour(self, error_margin_pct=5) -> ResultBundle:
        """
        Same as :meth:`test_util_behaviour` but checking the load.
        """
        return self._test_behaviour('load', error_margin_pct)


class Invariance(TestBundleBase, LoadTrackingHelpers):
    """
    Basic check for frequency invariant load and utilization tracking

    This test runs the same workload on one CPU of each capacity available in
    the system at a cross section of available frequencies.

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
    ftrace_conf = InvarianceItem.FTRACE_CONF

    NR_FREQUENCIES = 8
    """
    Maximum number of tested frequencies.
    """

    def __init__(self, res_dir, plat_info, invariance_items):
        super().__init__(res_dir, plat_info)

        self.invariance_items = invariance_items

    @classmethod
    def _build_invariance_items(cls, target, res_dir, **kwargs):
        """
        Yield a :class:`InvarianceItem` for a subset of target's
        frequencies, for one CPU of each capacity class.

        This is a generator function.

        :Variable keyword arguments: Forwarded to :meth:`InvarianceItem.from_target`

        :rtype: Iterator[:class:`InvarianceItem`]
        """
        plat_info = target.plat_info

        def pick_cpu(filtered_class, cpu_class):
            try:
                return filtered_class[0]
            except IndexError:
                raise RuntimeError(f'All CPUs of one capacity class have been blacklisted: {cpu_class}')

        # pick one CPU per class of capacity
        cpus = [
            pick_cpu(filtered_class, cpu_class)
            for cpu_class, filtered_class
            in zip(
                plat_info['capacity-classes'],
                cls.filter_capacity_classes(plat_info)
            )
        ]

        def select_freqs(cpu):
            all_freqs = plat_info['freqs'][cpu]

            def interpolate(start, stop, nr):
                step = (stop - start) / (nr - 1)
                return [start + i * step for i in range(nr)]

            # Select the higher freq no matter what
            selected_freqs = {max(all_freqs)}

            available_freqs = set(all_freqs) - selected_freqs
            nr_freqs = cls.NR_FREQUENCIES - len(selected_freqs)
            for ideal_freq in interpolate(min(all_freqs), max(all_freqs), nr_freqs):

                if not available_freqs:
                    break

                # Select the freq closest to ideal
                selected_freq = min(available_freqs, key=lambda freq: abs(freq - ideal_freq))
                available_freqs.discard(selected_freq)
                selected_freqs.add(selected_freq)

            return all_freqs, sorted(selected_freqs)

        cpu_freqs = {
            cpu: select_freqs(cpu)
            for cpu in cpus
        }

        logger = cls.get_logger()
        logger.info('Will run on: {}'.format(
            ', '.join(
                f'CPU{cpu}@{freq}'
                for cpu, (all_freqs, freq_list) in sorted(cpu_freqs.items())
                for freq in freq_list
            )
        ))

        for cpu, (all_freqs, freq_list) in sorted(cpu_freqs.items()):
            for freq in freq_list:
                item_dir = ArtifactPath.join(res_dir, f"{InvarianceItem.task_prefix}_{cpu}@{freq}")
                os.makedirs(item_dir)

                logger.info(f'Running experiment for CPU {cpu}@{freq}')
                yield InvarianceItem.from_target(
                    target,
                    cpu=cpu,
                    freq=freq,
                    freq_list=all_freqs,
                    res_dir=item_dir,
                    **kwargs,
                )

    def iter_invariance_items(self) -> InvarianceItem:
        yield from self.invariance_items

    @classmethod
    @kwargs_forwarded_to(
        InvarianceItem._from_target,
        ignore=[
            'cpu',
            'freq',
            'freq_list',
        ]
    )
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath = None, collector=None, **kwargs) -> 'Invariance':
        return cls(res_dir, target.plat_info,
            list(cls._build_invariance_items(target, res_dir, **kwargs))
        )

    def get_item(self, cpu, freq):
        """
        :returns: The
            :class:`~lisa.tests.scheduler.load_tracking.InvarianceItem`
            generated when running at a given frequency
        """
        for item in self.invariance_items:
            if item.cpu == cpu and item.freq == freq:
                return item
        raise ValueError('No invariance item matching {cpu}@{freq}'.format(cpu, freq))

    # Combined version of some other tests, applied on all available
    # InvarianceItem with the result merged.

    @InvarianceItem.test_util_correctness.used_events
    def test_util_correctness(self, mean_error_margin_pct=2, max_error_margin_pct=5) -> AggregatedResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_util_correctness`
        """
        def item_test(test_item):
            return test_item.test_util_correctness(
                mean_error_margin_pct=mean_error_margin_pct,
                max_error_margin_pct=max_error_margin_pct,
            )
        return self._test_all_items(item_test)

    @InvarianceItem.test_load_correctness.used_events
    def test_load_correctness(self, mean_error_margin_pct=2, max_error_margin_pct=5) -> AggregatedResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_load_correctness`
        """
        def item_test(test_item):
            return test_item.test_load_correctness(
                mean_error_margin_pct=mean_error_margin_pct,
                max_error_margin_pct=max_error_margin_pct,
            )
        return self._test_all_items(item_test)

    @InvarianceItem.test_util_behaviour.used_events
    def test_util_behaviour(self, error_margin_pct=5) -> AggregatedResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_util_behaviour`
        """
        def item_test(test_item):
            return test_item.test_util_behaviour(
                error_margin_pct=error_margin_pct,
            )
        return self._test_all_items(item_test)

    @InvarianceItem.test_load_behaviour.used_events
    def test_load_behaviour(self, error_margin_pct=5) -> AggregatedResultBundle:
        """
        Aggregated version of :meth:`InvarianceItem.test_load_behaviour`
        """
        def item_test(test_item):
            return test_item.test_load_behaviour(
                error_margin_pct=error_margin_pct,
            )
        return self._test_all_items(item_test)

    def _test_all_items(self, item_test):
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

    @InvarianceItem.test_util_behaviour.used_events
    def test_cpu_invariance(self) -> AggregatedResultBundle:
        """
        Check that items using the max freq on each CPU is passing util avg test.

        There could be false positives, but they are expected to be relatively
        rare.

        .. seealso:: :class:`InvarianceItem.test_util_behaviour`
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
                res = item.test_util_behaviour()
                res_list.append(res)

        return AggregatedResultBundle(res_list, 'cpu')

    @InvarianceItem.test_util_behaviour.used_events
    def test_freq_invariance(self) -> ResultBundle:
        """
        Check that at least one CPU has items passing for all tested frequencies.

        .. seealso:: :class:`InvarianceItem.test_util_behaviour`
        """

        logger = self.get_logger()

        def make_group_bundle(cpu, item_group):
            bundle = AggregatedResultBundle(
                [
                    # Only test util, as it should be more robust
                    item.test_util_behaviour()
                    for item in item_group
                ],
                # each item's "cpu" metric also contains the frequency
                name_metric='cpu',
            )
            # At that level, we only report the CPU, since nested bundles cover
            # different frequencies
            bundle.add_metric('cpu', cpu)

            logger.info(f'Util avg invariance {bundle.result.lower_name} for CPU {cpu}')
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

    PHASE_DURATION = 3 * UTIL_CONVERGENCE_TIME_S
    """
    The duration of a single phase
    """

    TASK_PERIOD = 16e-3
    """
    The average value of the runqueue PELT signals is very dependent on the
    task period, so it's important to set it to a known validated value in that
    class.
    """

    @abc.abstractmethod
    def get_nr_required_cpu(cls, plat_info):
        """
        The number of CPUs of same capacity involved in the test
        """
        pass

    @classmethod
    def run_rtapp(cls, *, profile, **kwargs):
        # Just do some validation on the profile
        for name, task in profile.items():
            for phase in task.phases:
                if len(phase['cpus']) != 1:
                    raise RuntimeError(f"Each phase must be tied to a single CPU. Task \"{name}\" violates this")

        super().run_rtapp(profile=profile, **kwargs)

    @property
    def cpus(self):
        """
        All CPUs used by RTapp workload.
        """
        return set(itertools.chain.from_iterable(
            phase['cpus']
            for task in self.rtapp_profile.values()
            for phase in task.phases
        ))

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)

        try:
            target.plat_info["cpu-capacities"]['rtapp']
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
        cpu_classes = plat_info["capacity-classes"]

        # If the CPU capacities are writeable, it's better to give priority to
        # LITTLE cores as they will be less prone to thermal capping.
        # Otherwise, it's better to pick big cores as they will not be affected
        # by CPU invariance issues.
        if not plat_info['cpu-capacities']['writeable']:
            cpu_classes = reversed(cpu_classes)

        for cpus in cpu_classes:
            if len(cpus) >= nr_required_cpu:
                return cpus[:nr_required_cpu]

        raise CannotCreateError(
            f"This workload requires {nr_required_cpu} CPUs of identical capacity")

    # Don't strictly check for cpu_frequency, since there might be no occurence
    # of the event.
    @may_use_events(FrequencyAnalysis.df_cpus_frequency.used_events)
    @TasksAnalysis.df_task_activation.used_events
    @RTAEventsAnalysis.df_phases.used_events
    def get_expected_cpu_util(self):
        """
        Get the per-phase average CPU utilization expected from the duty cycle
        of the tasks found in the trace.

        :returns: A dict of the shape {cpu : {phase_id : expected_util}}

        .. note:: This is more robust than just looking at the duty cycle in
            the task profile, since rtapp might not reproduce accurately the
            duty cycle it was asked.
        """
        cpu_capacities = self.plat_info['cpu-capacities']['rtapp']
        cpu_util = {}
        cpu_freqs = self.plat_info['freqs']

        try:
            freq_df = self.trace.analysis.frequency.df_cpus_frequency()
        except MissingTraceEventError:
            cpus_rel_freq = None
        else:
            cpus_rel_freq = {
                # Frequency, normalized according to max frequency on that CPU
                cols['cpu']: df['frequency'] / max(cpu_freqs[cols['cpu']])
                for cols, df in df_split_signals(freq_df, ['cpu'])
            }

        for task in self.rtapp_task_ids:
            df = self.trace.analysis.tasks.df_task_activation(task)

            for row in self.trace.analysis.rta.df_phases(task, wlgen_profile=self.rtapp_profile).itertuples():
                if not row.properties['meta']['from_test']:
                    continue

                phase = row.phase
                duration = row.duration
                start = row.Index
                end = start + duration
                # Ignore the first quarter of the util signal of each phase, since
                # it's impacted by the phase change, and util can be affected
                # (rtapp does some bookkeeping at the beginning of phases)
                # start += duration / 4

                # readjust the duration to take into account the modification of start
                duration = end - start
                window = (start, end)
                phase_df = df_window(df, window, clip_window=True)

                for cpu in self.cpus:

                    if cpus_rel_freq is None:
                        rel_freq_mean = 1
                    else:
                        phase_freq_series = df_window(cpus_rel_freq[cpu], window=window, clip_window=True)
                        # # We might not have frequency data at the beginning of the
                        # # trace, or if not frequency transition happened at all.
                        if phase_freq_series.empty:
                            rel_freq_mean = 1
                        else:
                            # If we lack freq data at the beginning of the
                            # window, assume the frequency was right.
                            if phase_freq_series.index[0] > start:
                                phase_freq_series = pd.concat([pd.Series([1.0], index=[start]), phase_freq_series])

                            # Extend the frequency to the right so that the mean
                            # takes into account all the data we have
                            freq_window = (phase_freq_series.index[0], end)
                            rel_freq_mean = series_mean(series_refit_index(phase_freq_series, window=freq_window))

                    cpu_phase_df = phase_df[phase_df['cpu'] == cpu].dropna()
                    if cpu_phase_df.empty:
                        duty_cycle = 0
                        cpu_residency = 0
                    else:
                        duty_cycle = series_mean(df_refit_index(cpu_phase_df['duty_cycle'], window=window))
                        cpu_residency = end - max(cpu_phase_df.index[0], start)

                    phase_util = UTIL_SCALE * duty_cycle * (cpu_capacities[cpu] / UTIL_SCALE)
                    # Pro-rata with the time spent on that CPU, so we get
                    # the correct average.
                    phase_util *= cpu_residency / duration

                    # We might not have run at max freq, e.g. because of
                    # thermal capping, so take that into account
                    phase_util *= rel_freq_mean

                    cpu_util.setdefault(cpu, {}).setdefault(phase, 0)
                    cpu_util[cpu][phase] += phase_util

        return cpu_util

    @LoadTrackingAnalysis.df_cpus_signal.used_events
    def get_trace_cpu_util(self):
        """
        Get the per-phase average CPU utilization read from the trace

        :returns: A dict of the shape {cpu : {phase_id : trace_util}}
        """
        df = self.trace.analysis.load_tracking.df_cpus_signal('util')
        tasks = self.rtapp_task_ids_map.keys()
        task = sorted(task for task in tasks if task.startswith('migr'))[0]
        task = self.rtapp_task_ids_map[task][0]

        cpu_util = {}
        for row in self.trace.analysis.rta.df_phases(task, wlgen_profile=self.rtapp_profile).itertuples():
            if not row.properties['meta']['from_test']:
                continue

            phase = row.phase
            duration = row.duration
            start = row.Index
            end = start + duration
            # Ignore the first quarter of the util signal of each phase, since
            # it's impacted by the phase change, and util can be affected
            # (rtapp does some bookkeeping at the beginning of phases)
            start += duration / 4
            phase_df = df_window(df, (start, end), method='pre', clip_window=True)

            for cpu in self.cpus:
                util = phase_df[phase_df['cpu'] == cpu]['util']
                cpu_util.setdefault(cpu, {})[phase] = kernel_util_mean(util, plat_info=self.plat_info)

        return cpu_util

    @LoadTrackingAnalysis.plot_task_signals.used_events
    def _plot_util(self):
        trace = self.trace
        analysis = trace.analysis.load_tracking
        fig, axes = analysis.setup_plot(nrows=len(self.rtapp_tasks))
        for task, axis in zip(self.rtapp_tasks, axes):
            analysis.plot_task_signals(task, signals=['util'], axis=axis)
            trace.analysis.rta.plot_phases(task, axis=axis, wlgen_profile=self.rtapp_profile)

            activation_axis = axis.twinx()
            trace.analysis.tasks.plot_task_activation(task, duty_cycle=True, overlay=True, alpha=0.2, axis=activation_axis)

            df_activations = trace.analysis.tasks.df_task_activation(task)
            df_util = analysis.df_task_signal(task, 'util')
            def compute_means(row):
                start = row.name
                end = start + row['duration']
                phase_activations = df_window(df_activations, (start, end))
                phase_util = df_window(df_util, (start, end))
                series = pd.Series({
                    'Phase duty cycle average': series_mean(phase_activations['duty_cycle']),
                    'Phase util tunnel average': kernel_util_mean(
                        phase_util['util'],
                        plat_info=self.plat_info,
                    ),
                })
                return series

            df_means = trace.analysis.rta.df_phases(task).apply(compute_means, axis=1)
            df_means = series_refit_index(df_means, window=trace.window)
            df_means['Phase duty cycle average'].plot(drawstyle='steps-post', ax=activation_axis)
            df_means['Phase util tunnel average'].plot(drawstyle='steps-post', ax=axis)
            activation_axis.legend()
            axis.legend()


        filepath = ArtifactPath.join(self.res_dir, 'tasks_util.png')
        analysis.save_plot(fig, filepath=filepath)

        filepath = ArtifactPath.join(self.res_dir, 'cpus_util.png')
        cpus = sorted(self.cpus)
        analysis.plot_cpus_signals(cpus, signals=['util'], filepath=filepath)

    @get_trace_cpu_util.used_events
    @get_expected_cpu_util.used_events
    @_plot_util.used_events
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    def test_util_task_migration(self, allowed_error_pct=3) -> ResultBundle:
        """
        Test that a migrated task properly propagates its utilization at the CPU level

        :param allowed_error_pct: How much the trace averages can stray from the
          expected values
        :type allowed_error_pct: float
        """
        expected_util = self.get_expected_cpu_util()
        trace_util = self.get_trace_cpu_util()

        passed = True

        expected_metrics = {}
        trace_metrics = {}
        deltas = {}

        for cpu in self.cpus:
            expected_cpu_util = expected_util[cpu]
            trace_cpu_util = trace_util[cpu]

            cpu_str = f"cpu{cpu}"
            expected_metrics[cpu_str] = TestMetric({})
            trace_metrics[cpu_str] = TestMetric({})
            deltas[cpu_str] = TestMetric({})

            for phase in sorted(trace_cpu_util.keys() & expected_cpu_util.keys()):
                expected_phase_util = expected_cpu_util[phase]
                trace_phase_util = trace_cpu_util[phase]
                is_equal, delta = self.is_almost_equal(
                        expected_phase_util,
                        trace_phase_util,
                        allowed_error_pct)

                if not is_equal:
                    passed = False

                # Just some verbose metric collection...
                phase_str = f"phase{phase}"
                expected_metrics[cpu_str].data[phase] = TestMetric(expected_phase_util)
                trace_metrics[cpu_str].data[phase] = TestMetric(trace_phase_util)
                deltas[cpu_str].data[phase] = TestMetric(delta, "%")

        res = ResultBundle.from_bool(passed)
        res.add_metric("Expected utilization", expected_metrics)
        res.add_metric("Trace utilization", trace_metrics)
        res.add_metric("Utilization deltas", deltas)

        self._plot_util()

        return res


class OneTaskCPUMigration(CPUMigrationBase):
    """
    Some tasks on two big CPUs, one of them migrates in its second phase.
    """

    @classmethod
    def get_nr_required_cpu(cls, plat_info):
        return 2

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        cpus = cls.get_migration_cpus(plat_info)
        nr_cpus = len(cpus)

        periodic_settings = dict(
            duration=cls.PHASE_DURATION,
            period=cls.TASK_PERIOD,
        )

        return {
            # A task that will migrate to another CPU
            'migr': add(
                RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=20,
                        scale_for_cpu=cpu,
                        **periodic_settings,
                    ),
                    prop_cpus=[cpu],
                )
                for cpu in cpus
            ),
            **{
                # Just some tasks that won't move to get some background utilization
                f"static{i}": nr_cpus * RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=30,
                        scale_for_cpu=cpus[i],
                        **periodic_settings,
                    ),
                    prop_cpus=[cpus[i]]
                )
                for i in range(min(2, nr_cpus))
            }
        }


class NTasksCPUMigrationBase(CPUMigrationBase):
    """
    N tasks on N CPUs, with all the migration permutations.
    """

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        cpus = cls.get_migration_cpus(plat_info)
        def make_name(i):
            return f'migr{i}'
        nr_tasks = len(cpus)
        # Define one task per CPU, and create all the possible migrations by
        # shuffling around these tasks
        profile = {}
        for cpus_combi in itertools.permutations(cpus, r=nr_tasks):
            for i, cpu in enumerate(cpus_combi):
                task_name = make_name(i)
                task = profile.setdefault(task_name, RTAPhase())
                profile[task_name] = task + RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=50,
                        scale_for_cpu=cpu,
                        duration=cls.PHASE_DURATION,
                        period=cls.TASK_PERIOD,
                    ),
                    prop_cpus=[cpu],
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
