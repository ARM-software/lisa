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
import pandas as pd
import logging
import re
import json
from collections import OrderedDict
from functools import reduce

from bart.common.Utils import select_window, area_under_curve
from bart.sched import pelt
from bart.sched.SchedAssert import SchedAssert
from devlib.utils.misc import memoized
from env import TestEnv
from test_workload import Metric, ResultBundle, TestBundle
from time import sleep
from trace import Trace
from trappy.stats.grammar import Parser
from wlgen.rta import Periodic, RTA
from wlgen.utils import SchedEntity, Task, Taskgroup
from trappy.stats.Topology import Topology

UTIL_SCALE = 1024
# Time in seconds to allow for util_avg to converge (i.e. ignored time)
UTIL_AVG_CONVERGENCE_TIME = 0.3
# Allowed margin between expected and observed util_avg value
ERROR_MARGIN_PCT = 15
# PELT half-life value in ms
HALF_LIFE_MS = 32

class LoadTrackingTestBundle(TestBundle):
    """Base class for shared functionality of load tracking tests"""

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
    """
    The FTrace configuration used to record a trace while the synthetic workload
    is being run.
    """

    @property
    def trace(self):
        """

        :returns: a Trace

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`to_path`
        """
        if not self._trace:
            self._trace = Trace(self.res_dir, events=self.ftrace_conf["events"])

        return self._trace

    def __init__(self, res_dir, nrg_model, rtapp_params, cpufreq_params, caps):
        super(LoadTrackingTestBundle, self).__init__(res_dir)

        # self.trace = Trace(res_dir, events=self.ftrace_conf["events"])
        #EnergyModel.from_path(os.path.join(res_dir, "nrg_model.yaml"))
        self._trace = None
        self.nrg_model = nrg_model
        self.rtapp_params = rtapp_params
        self.cpufreq_params = cpufreq_params
        self.caps = caps

    @classmethod
    def create_rtapp_params(cls, te):
        """
        :returns: a :class:`dict` with task names as keys and :class:`RTATask` as values

        This is the method you want to override to specify what is
        your synthetic workload.
        """
        raise NotImplementedError()

    @classmethod
    def create_cpufreq_params(cls, te):
        """
        :returns: a list of :class:`dict` each containing a unique tag,
        the cpufreq governor and any additional cpufreq params

        This is the method you want to override to specify what is
        your cpufreq configurations for the workload run.
        """
        raise NotImplementedError()

    @classmethod
    def _from_target(cls, te, res_dir):
        rtapp_params = cls.create_rtapp_params(te)
        cpufreq_params = cls.create_cpufreq_params(te)
        for cpufreq in cpufreq_params:
            iter_res_dir = os.path.join(res_dir, cpufreq['tag'])
            os.makedirs(iter_res_dir)

            wload = RTA(te.target, "rta_{}".format(cls.__name__.lower()), te.calibration())
            wload.conf(kind='profile', params=rtapp_params, work_dir=iter_res_dir)

            trace_path = os.path.join(iter_res_dir, "trace.dat")
            te.configure_ftrace(**cls.ftrace_conf)

            # te.target.cpufreq.use_governor(cpufreq['governor'], **cpufreq['params'])
            te.target.cpufreq.set_all_governors(cpufreq['governor'])

            if 'freqs' in cpufreq:
                if cpufreq['governor'] != 'userspace':
                    raise ValueError('Must use userspace governor to set CPU freqs')
                for cpu, freq in cpufreq['freqs'].iteritems():
                    te.target.cpufreq.set_frequency(cpu, freq)

            if 'params' in cpufreq:
                for cpu in te.target.list_online_cpus():
                    te.target.cpufreq.set_governor_tunables(
                            cpu,
                            cpufreq['governor'],
                            **cpufreq['params'])

            with te.record_ftrace(trace_path):
                with te.freeze_userspace():
                    wload.run(out_dir=iter_res_dir)

        caps = [cls._get_cpu_capacity(te, cpu)
                for cpu in range(te.target.number_of_cpus)]

        return cls(res_dir, te.nrg_model, rtapp_params, cpufreq_params, caps)

    @memoized
    @staticmethod
    def _get_cpu_capacity(te, cpu):
        return te.target.sched.get_capacity(cpu)

    def get_sched_assert(self, cpufreq, cpu):
        task = self.task_name
        d = os.path.join(self.res_dir, cpufreq['tag'])
        iter_trace = Trace(d, events=self.ftrace_conf["events"])
        t = Topology()
        t.add_to_level('cpu', [[cpu]])
        return SchedAssert(iter_trace.ftrace, t, execname=task)

    def get_window(self, cpufreq, cpu):
        sched_assert = self.get_sched_assert(cpufreq, cpu)
        start_time = sched_assert.getStartTime()
        end_time = sched_assert.getEndTime()
        return (start_time, end_time)

    def get_expected_util_avg(self, cpufreq, cpu, cap):
        """
        Examine trace to figure out an expected mean for util_avg

        Assumes an RT-App workload with a single task with a single phase
        per each CPU
        """
        # Find duty cycle of the workload task
        sched_assert = self.get_sched_assert(cpufreq, cpu)
        window = self.get_window(cpufreq, cpu)
        duty_cycle_pct = sched_assert.getDutyCycle(window)

        cpu_capacity = cap

        # Scale the capacity linearly according to the frequency the workload
        # was run at
        if cpufreq['governor'] == 'userspace':
            freq = cpufreq['freqs'][cpu]
            max_freq = max(self.all_freqs)
            cpu_capacity *= float(freq) / max_freq
        else:
            assert cpufreq['governor'] == 'performance'

        # Scale the relative CPU/freq capacity into the range 0..1
        scale = max(self.caps)
        scaling_factor = float(cpu_capacity) / scale

        return UTIL_SCALE * (duty_cycle_pct / 100.) * scaling_factor

    def get_sched_task_signals(self, cpufreq, cpu, signals):
        """
        Get a pandas.DataFrame with the sched signals for the workload task

        This examines scheduler load tracking trace events, supporting either
        sched_load_avg_task or sched_pelt_se. You will need a target kernel that
        includes these events.

        :param cpufreq: Cpufreq conf for the run to get trace for
        :param signals: List of load tracking signals to extract. Probably a
                        subset of ``['util_avg', 'load_avg']``
        :returns: :class:`pandas.DataFrame` with a column for each signal for
                  the workload task
        """
        task = self.task_name
        d = os.path.join(self.res_dir, cpufreq['tag'])
        trace = Trace(d, events=self.ftrace_conf["events"])

        # There are two different scheduler trace events that expose the load
        # tracking signals. Neither of them is in mainline. Eventually they
        # should be unified but for now we'll just check for both types of
        # event.
        # TODO: Add support for this parsing in Trappy and/or tasks_analysis
        signal_fields = signals
        if 'sched_load_avg_task' in trace.available_events:
            event = 'sched_load_avg_task'
        elif 'sched_load_se' in trace.available_events:
            event = 'sched_load_se'
            # sched_load_se uses 'util' and 'load' instead of 'util_avg' and
            # 'load_avg'
            signal_fields = [s.replace('_avg', '') for s in signals]
        elif 'sched_pelt_se' in trace.available_events:
            event = 'sched_pelt_se'
        else:
            raise ValueError('No sched_load_avg_task or sched_load_se or sched_pelt_se events. '
                             'Does the kernel support them?')

        df = getattr(trace.ftrace, event).data_frame
        df = df[df['comm'] == task][signal_fields]
        df = select_window(df, self.get_window(cpufreq, cpu))
        return df.rename(columns=dict(zip(signal_fields, signals)))

    def get_signal_mean(self, cpufreq, cpu, signal,
                        ignore_first_s=UTIL_AVG_CONVERGENCE_TIME):
        """
        Get the mean of a scheduler signal for the experiment's task

        Ignore the first `ignore_first_s` seconds of the signal.
        """
        (wload_start, wload_end) = self.get_window(cpufreq, cpu)
        window = (wload_start + ignore_first_s, wload_end)

        signal = self.get_sched_task_signals(cpufreq, cpu, [signal])[signal]
        signal = select_window(signal, window)
        return area_under_curve(signal) / (window[1] - window[0])

    def isAlmostEqual(self, target, value, delta):
        return (target - delta < value) and (value < target + delta)

    def _test_signal(self, signal_name):
        res = ResultBundle()
        passed = True
        for (cpu, cpu_cap) in zip(self.target_cpus, self.target_cpus_capacity):
            for cpufreq in self.cpufreq_params:
                exp_util = self.get_expected_util_avg(cpufreq, cpu, cpu_cap)
                signal_mean = self.get_signal_mean(cpufreq, cpu, signal_name)

                error_margin = exp_util * (ERROR_MARGIN_PCT / 100.)

                passed = passed and \
                        self.isAlmostEqual(exp_util, signal_mean, error_margin)

                res.add_metric(
                    Metric("expected_util_avg, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), exp_util)
                )
                res.add_metric(
                    Metric("signal_mean, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), signal_mean)
                )

        res.passed = passed
        return res

    def test_task_util_avg(self):
        """
        Test that the mean of the util_avg signal matched the expected value
        """
        return self._test_signal('util_avg')

    def test_task_load_avg(self):
        """
        Test that the mean of the load_avg signal matched the expected value

        Assuming that the system was under little stress (so the task was
        RUNNING whenever it was RUNNABLE) and that the task was run with a
        'nice' value of 0, the load_avg should be similar to the util_avg. So,
        this test does the same as test_task_util_avg but for load_avg.
        """
        return self._test_signal('load_avg')

class FreqInvarianceTest(LoadTrackingTestBundle):
    """
    **Goal**
    Basic check for frequency invariant load tracking

    **Detailed Description**
    This test runs the same workload on the most capable CPU on the system at a
    cross section of available frequencies. The trace is then examined to find
    the average activation length of the workload, which is combined with the
    known period to estimate an expected mean value for util_avg for each
    frequency. The util_avg value is extracted from scheduler trace events and
    its mean is compared with the expected value (ignoring the first 300ms so
    that the signal can stabilize). The test fails if the observed mean is
    beyond a certain error margin from the expected one. load_avg is then
    similarly compared with the expected util_avg mean, under the assumption
    that load_avg should equal util_avg when system load is light.

    **Expected Behaviour**
    Load tracking signals are scaled so that the workload results in roughly the
    same util & load values regardless of frequency.
    """
    task_name = 'fie_10pct'

    @classmethod
    def create_rtapp_params(cls, te):
        """
        Get a specification for a rt-app workload with the specificied duty
        cycle, pinned to the given CPU.
        """
        # Run on one of the CPUs with highest capacity
        cpu = max(range(te.target.number_of_cpus),
                key=lambda c: cls._get_cpu_capacity(te, c))
        cls.target_cpus = [cpu]
        cls.target_cpus_capacity = [cls._get_cpu_capacity(te, cpu)]

        rtapp_params = {}
        rtapp_params[cls.task_name] = Periodic(
            duty_cycle_pct=10,
            duration_s=2,
            period_ms=16,
            cpus=[cpu]
        )

        return rtapp_params

    @classmethod
    def create_cpufreq_params(cls, te):
        # Create a set of confs with different frequencies
        # We'll run the workload under each conf (i.e. at each frequency)
        confs = []
        cpu = max(range(te.target.number_of_cpus),
                key=lambda c: cls._get_cpu_capacity(te, c))
        cls.all_freqs = te.target.cpufreq.list_frequencies(cpu)
        # If we have loads of frequencies just test a cross-section so it
        # doesn't take all day
        cls.freqs = cls.all_freqs[::len(cls.all_freqs) / 8 + 1]
        for freq in cls.freqs:
            confs.append({
                'tag' : 'freq_{}'.format(freq),
                'freqs' : {cpu: freq},
                'governor' : 'userspace',
            })

        return confs

class CpuInvarianceTest(LoadTrackingTestBundle):
    """
    **Goal**
    Basic check for CPU invariant load and utilization tracking

    **Detailed Description**
    This test runs the same workload on one CPU of each type in the system. The
    trace is then examined to estimate an expected mean value for util_avg for
    each CPU's workload. The util_avg value is extracted from scheduler trace
    events and its mean is compared with the expected value (ignoring the first
    300ms so that the signal can stabilize). The test fails if the observed mean
    is beyond a certain error margin from the expected one. load_avg is then
    similarly compared with the expected util_avg mean, under the assumption
    that load_avg should equal util_avg when system load is light.

    **Expected Behaviour**
    Load tracking signals are scaled so that the workload results in roughly the
    same util & load values regardless of compute power of the CPU
    used. Moreover, assuming that the extraneous system load is negligible, the
    load signal is similar to the utilization signal.
    """
    task_name = 'cie_10pct'

    @classmethod
    def create_rtapp_params(cls, te):
        """
        Get a specification for a rt-app workload with the specificied duty
        cycle, pinned to the given CPU.
        """
        # Run the 10% workload on one CPU in each capacity group
        cls.target_cpus = []
        cls.target_cpus_capacity = []
        phases = []
        tested_caps = set()
        for cpu in range(te.target.number_of_cpus):
            cap = cls._get_cpu_capacity(te, cpu)
            # No need to test on every CPU, just one for each capacity value
            if cap not in tested_caps:
                tested_caps.add(cap)
                cls.target_cpus.append(cpu)
                cls.target_cpus_capacity.append(cap)
                phase = Periodic(
                    duty_cycle_pct=10,
                    duration_s=2,
                    period_ms=16,
                    cpus=[cpu]
                )
                phases.append(phase)

        rtapp_params = {}
        rtapp_params[cls.task_name] = reduce((lambda x, y: x + y), phases)

        return rtapp_params

    @classmethod
    def create_cpufreq_params(cls, te):
        return [{
            'tag' : 'cie_conf',
            'governor' : 'performance',
        }]

class PELTTaskTest(LoadTrackingTestBundle):
    """
    **Goal**
    Basic checks for tasks related PELT signals behaviour.

    **Detailed Description**
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

    **Expected Behaviour**
    Simulated PELT signal and the signal extracted from the trace should have
    very similar min, max and mean values in the stable range and the behaviour of
    the signal should be very similar to simulated one.
    """
    task_name = 'pelt_behv'

    @classmethod
    def create_rtapp_params(cls, te):
        # Run the 50% workload on a CPU with highest capacity
        cpu = max(range(te.target.number_of_cpus),
                key=lambda c: cls._get_cpu_capacity(te, c))
        cls.target_cpus = [cpu]
        cls.target_cpus_capacity = [cls._get_cpu_capacity(te, cpu)]

        rtapp_params = {}
        rtapp_params[cls.task_name] = Periodic(
            duty_cycle_pct=50,
            duration_s=2,
            period_ms=16,
            cpus=[cpu]
        )

        return rtapp_params

    @classmethod
    def create_cpufreq_params(cls, te):
        return [{
            'tag' : 'pelt_behv_conf',
            'governor' : 'performance',
        }]

    def get_simulated_pelt(self, task, init_value):
        """
        Get simulated PELT signal and the periodic task used to model it.

        :returns: tuple of
            - :mod:`bart.sched.pelt.Simulator` the PELT simulator object
            - :mod:`bart.sched.pelt.PeriodicTask` simulated periodic task
            - :mod:`pandas.DataFrame` instance which reports the computed
                    PELT values at each PELT sample interval.
        """
        phase = self.rtapp_params[self.task_name].phases[0]
        pelt_task = pelt.PeriodicTask(period_samples=phase.period_ms,
                                      duty_cycle_pct=phase.duty_cycle_pct)
        peltsim = pelt.Simulator(init_value=init_value,
                                 half_life_ms=HALF_LIFE_MS)
        df = peltsim.getSignal(pelt_task, 0, phase.duration_s + 1)
        return peltsim, pelt_task, df

    def _test_range(self, signal_name):
        res = ResultBundle()
        passed = True
        task = self.task_name
        for cpu in self.target_cpus:
            for cpufreq in self.cpufreq_params:
                signal_df = self.get_sched_task_signals(cpufreq, cpu, [signal_name])
                # Get stats and stable range of the simulated PELT signal
                start_time = self.get_sched_assert(cpufreq, cpu).getStartTime()
                init_pelt = pelt.Simulator.estimateInitialPeltValue(
                    signal_df[signal_name].iloc[0], signal_df.index[0],
                    start_time, HALF_LIFE_MS
                )
                peltsim, pelt_task, sim_df = self.get_simulated_pelt(task, init_pelt)
                sim_range = peltsim.stableRange(pelt_task)
                stable_time = peltsim.stableTime(pelt_task)
                window = (start_time + stable_time,
                          start_time + stable_time + 0.5)
                # Get signal statistics in a period of time where the signal is
                # supposed to be stable
                signal_stats = signal_df[window[0]:window[1]][signal_name].describe()

                # Narrow down simulated PELT signal to stable period
                sim_df = sim_df[window[0]:window[1]].pelt_value

                # Check min
                error_margin = sim_range.min_value * (ERROR_MARGIN_PCT / 100.)
                passed = passed and self.isAlmostEqual(sim_range.min_value, signal_stats['min'], error_margin)
                res.add_metric(
                    Metric("min_signal_value, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), signal_stats['min'])
                )
                res.add_metric(
                    Metric("expected_min_signal_value, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), sim_range.min_value)
                )

                # Check max
                error_margin = sim_range.max_value * (ERROR_MARGIN_PCT / 100.)
                passed = passed and self.isAlmostEqual(sim_range.max_value, signal_stats['max'], error_margin)
                res.add_metric(
                    Metric("max_signal_value, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), signal_stats['max'])
                )
                res.add_metric(
                    Metric("expected_max_signal_value, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), sim_range.max_value)
                )

                # Check mean
                sim_mean = sim_df.mean()
                error_margin = sim_mean * (ERROR_MARGIN_PCT / 100.)
                passed = passed and self.isAlmostEqual(sim_mean, signal_stats['mean'], error_margin)
                res.add_metric(
                    Metric("mean_signal_value, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), signal_stats['mean'])
                )
                res.add_metric(
                    Metric("expected_mean_signal, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), sim_mean)
                )

        res.passed = passed
        return res

    def _test_behaviour(self, signal_name):
        res = ResultBundle()
        passed = True
        task = self.task_name

        for cpu in self.target_cpus:
            for cpufreq in self.cpufreq_params:
                signal_df = self.get_sched_task_signals(cpufreq, cpu, [signal_name])
                # Get instant of time when the task starts running
                start_time = self.get_sched_assert(cpufreq, cpu).getStartTime()

                # Get information about the task
                phase = self.rtapp_params[self.task_name].phases[0]

                # Create simulated PELT signal for a periodic task
                init_pelt = pelt.Simulator.estimateInitialPeltValue(
                    signal_df[signal_name].iloc[0], signal_df.index[0],
                    start_time, HALF_LIFE_MS
                )
                peltsim, pelt_task, sim_df = self.get_simulated_pelt(task, init_pelt)

                # Compare actual PELT signal with the simulated one
                margin = 0.05
                period_s = phase.period_ms / 1e3
                sim_period_ms = phase.period_ms * (peltsim._sample_us / 1e6)
                n_errors = 0
                for entry in signal_df.iterrows():
                    trace_val = entry[1][signal_name]
                    timestamp = entry[0] - start_time
                    # Next two instructions map the trace timestamp to a simulated
                    # signal timestamp. This is due to the fact that the 1 ms is
                    # actually 1024 us in the simulated signal.
                    n_periods = timestamp / period_s
                    nearest_timestamp = n_periods * sim_period_ms
                    sim_val_loc = sim_df.index.get_loc(nearest_timestamp,
                                                       method='nearest')
                    sim_val = sim_df.pelt_value.iloc[sim_val_loc]
                    res.add_metric(
                        Metric("trace_val at {}".format(timestamp), trace_val)
                    )
                    res.add_metric(
                        Metric("sim_val at {}".format(timestamp), sim_val)
                    )
                    if trace_val > (sim_val * (1 + margin)) or \
                       trace_val < (sim_val * (1 - margin)):
                        n_errors += 1

                total_no_errors = n_errors / len(signal_df)
                res.add_metric(
                    Metric("total_no_errors, cpu {}, cpufreq conf {}"
                        .format(cpu, cpufreq['tag']), total_no_errors)
                )
                # Exclude possible outliers (these may be due to a kernel thread that
                # for some reason gets coscheduled with our workload).
                passed = passed and (total_no_errors < margin)

        res.passed = passed
        return res

    def test_util_avg_range(self):
        """
        Test util_avg stable range for a 50% periodic task
        """
        return self._test_range('util_avg')

    def test_util_avg_behaviour(self):
        """
        Test util_avg behaviour for a 50% periodic task
        """
        return self._test_behaviour('util_avg')

    def test_load_avg_range(self):
        """
        Test load_avg stable range for a 50% periodic task
        """
        return self._test_range('load_avg')

    def test_load_avg_behaviour(self, ):
        """
        Test load_avg behaviour for a 50% periodic task
        """
        return self._test_behaviour('load_avg')
