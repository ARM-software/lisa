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

import logging

from bart.common.Utils import select_window, area_under_curve
from devlib.utils.misc import memoized
from trappy.stats.grammar import Parser
import pandas as pd

from test import LisaTest, experiment_test
from trace import Trace

UTIL_SCALE = 1024
# Time in seconds to allow for util_avg to converge (i.e. ignored time)
UTIL_AVG_CONVERGENCE_TIME = 0.3
# Allowed margin between expected and observed util_avg value
ERROR_MARGIN_PCT = 15

class LoadTrackingTest(LisaTest):
    __test__ = False # This is a base class, don't try to run it as a test

    test_conf = {
        'tools'    : [ 'rt-app' ],
        'ftrace' : {
            'events' : [
                'sched_switch',
                'sched_load_avg_task',
                'sched_load_avg_cpu',
                'sched_pelt_se',
            ],
        },
        'modules': ['cpufreq', 'cgroups'],
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(LoadTrackingTest, cls)._init(*args, **kwargs)

    @classmethod
    def get_wload(cls, cpu):
        """
        get a specification for a 10% rt-app workload, pinned to the given cpu
        """
        return {
            'type' : 'rt-app',
                'conf' : {
                    'class' : 'periodic',
                    'params' : {
                        'duty_cycle_pct': 10,
                        'duration_s': 5,
                        'period_ms': 10,
                    },
                    'tasks' : 1,
                    'prefix' : 'lt_test',
                    'cpus' : [cpu]
                },
            }

    def get_task_activations(self, experiment):
        """
        Get a data frame with activation times of the experiment task

        Column 'length' contains the length of each task activation
        """
        [task] = experiment.wload.tasks.keys()
        sw = self.get_trace(experiment).ftrace.sched_switch.data_frame
        start_times = sw[sw['next_comm'] == task].index
        end_times = sw[(sw['prev_comm'] == task)
                       & (sw.index > start_times[0])].index
        lengths = (end_times - start_times).tolist()
        return pd.DataFrame(lengths, index=start_times, columns=['length'])

    def get_expected_util_avg(self, experiment):
        """
        Examine workload and trace to figure out an expected mean for util_avg

        Assumes an RT-App workload with a single task with a single phase
        """
        # Examine workload to find period
        [task] = experiment.wload.tasks.keys()
        [phase] = experiment.wload.params['profile'][task]['phases']
        task_period = phase.period_ms / 1000.

        # Examine trace to find observed duty cycle
        mean_runtime = self.get_task_activations(experiment).mean()['length']
        duty_cycle = min(mean_runtime / task_period, 1.0)

        # Find the capacity of the CPU the workload was run on
        [cpu] = experiment.wload.cpus
        cpufreq = experiment.conf['cpufreq']
        if cpufreq['governor'] == 'userspace':
            freq = cpufreq['freqs'][cpu]
            cpu_capacity = self.te.nrg_model.get_cpu_capacity(cpu, freq)
        else:
            assert cpufreq['governor'] == 'performance'
            cpu_capacity = self.te.nrg_model.get_cpu_capacity(cpu)

        scaling_factor = float(cpu_capacity) / self.te.nrg_model.capacity_scale

        return UTIL_SCALE * duty_cycle * scaling_factor

    def get_sched_signals(self, experiment, signals):
        """
        Get a pandas.DataFrame with the sched signals for the workload task

        This examines scheduler load tracking trace events, supporting either
        sched_load_avg_task or sched_pelt_se. You will need a target kernel that
        includes these events.
        """
        [task] = experiment.wload.tasks.keys()
        trace = self.get_trace(experiment)

        # There are two different scheduler trace events that expose the
        # util_avg signal. Neither of them is in mainline. Eventually they
        # should be unified but for now we'll just check for both types of
        # event.
        if 'sched_load_avg_task' in trace.available_events:
            event = 'sched_load_avg_task'
        elif 'sched_pelt_se' in trace.available_events:
            event = 'sched_pelt_se'
        else:
            raise ValueError('No sched_load_avg_task or sched_pelt_se events. '
                             'Does the kernel support them?')

        df = getattr(trace.ftrace, event).data_frame
        signals = df[df['comm'] == task][signals]
        return select_window(signals, self.get_window(experiment))

    def get_signal_mean(self, experiment, signal,
                        ignore_first_s=UTIL_AVG_CONVERGENCE_TIME):
        """
        Get the mean of a scheduler signal for the experiment's task

        Ignore the first `ignore_first_s` seconds of the signal.
        """
        (wload_start, wload_end) = self.get_window(experiment)
        window = (wload_start + ignore_first_s, wload_end)

        signal = self.get_sched_signals(experiment, [signal])[signal]
        signal = select_window(signal, window)
        return area_under_curve(signal) / (window[1] - window[0])

class FreqInvarianceTest(LoadTrackingTest):
    """
    Goal
    ====
    Basic check for frequency invariant load tracking

    Detailed Description
    ====================
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

    Expected Behaviour
    ==================
    Load tracking signals are scaled so that the workload results in roughly the
    same util & load values regardless of frequency.
    """

    @classmethod
    def _getExperimentsConf(cls, test_env):
        # Run on one of the CPUs with highest capacity
        cpu = test_env.nrg_model.biggest_cpus[0]

        wloads = {
            'fie_10pct' : cls.get_wload(cpu)
        }

        # Create a set of confs with different frequencies
        # We'll run the 10% workload under each conf (i.e. at each frequency)
        confs = []

        all_freqs = test_env.target.cpufreq.list_frequencies(cpu)
        # If we have loads of frequencies just test a cross-section so it
        # doesn't take all day
        cls.freqs = all_freqs[::len(all_freqs)/8 + 1]
        for freq in cls.freqs:
            confs.append({
                'tag' : 'freq_{}'.format(freq),
                'flags' : ['ftrace', 'freeze_userspace'],
                'cpufreq' : {
                    'freqs' : {cpu: freq},
                    'governor' : 'userspace',
                },
            })

        return {
            'wloads': wloads,
            'confs': confs,
        }

    def _test_signal(self, experiment, tasks, signal_name):
        [task] = tasks
        exp_util = self.get_expected_util_avg(experiment)
        signal_mean = self.get_signal_mean(experiment, signal_name)

        error_margin = exp_util * (ERROR_MARGIN_PCT / 100.)
        [freq] = experiment.conf['cpufreq']['freqs'].values()

        msg = 'Saw {} around {}, expected {} at freq {}'.format(
            signal_name, signal_mean, exp_util, freq)
        self.assertAlmostEqual(signal_mean, exp_util, delta=error_margin,
                               msg=msg)

    @experiment_test
    def test_task_util_avg(self, experiment, tasks):
        """
        Test that the mean of the util_avg signal matched the expected value
        """
        return self._test_signal(experiment, tasks, 'util_avg')

    @experiment_test
    def test_task_load_avg(self, experiment, tasks):
        """
        Test that the mean of the load_avg signal matched the expected value

        Assuming that the system was under little stress (so the task was
        RUNNING whenever it was RUNNABLE) and that the task was run with a
        'nice' value of 0, the load_avg should be similar to the util_avg. So,
        this test does the same as test_task_util_avg but for load_avg.
        """
        return self._test_signal(experiment, tasks, 'load_avg')

class CpuInvarianceTest(LoadTrackingTest):
    """
    Goal
    ====
    Basic check for CPU invariant load tracking

    Detailed Description
    ====================
    This test runs the same workload on each CPU in the system.  The trace is
    then examined to find the average activation length of the workload, which
    is combined with the known period to estimate an expected mean value for
    util_avg for each CPU. The util_avg value is extracted from scheduler trace
    events and its mean is compared with the expected value (ignoring the first
    300ms so that the signal can stabilize). The test fails if the observed mean
    is beyond a certain error margin from the expected one. load_avg is then
    similarly compared with the expected util_avg mean, under the assumption
    that load_avg should equal util_avg when system load is light.

    Expected Behaviour
    ==================
    Load tracking signals are scaled so that the workload results in roughly the
    same util & load values regardless of compute power of the CPU used.
    """

    @classmethod
    def _getExperimentsConf(cls, test_env):
        # Run the 10% workload on one CPU in each capacity group
        wloads = {}
        for group in test_env.nrg_model.cpu_groups:
            cpu = group[0]
            wloads['cie_cpu{}'.format(cpu)] = cls.get_wload(cpu)

        conf = {
            'tag' : 'cie_conf',
            'flags' : ['ftrace', 'freeze_userspace'],
            'cpufreq' : {'governor' : 'performance'},
        }

        return {
            'wloads': wloads,
            'confs': [conf],
        }

    def _test_signal(self, experiment, tasks, signal_name):
        [task] = tasks
        exp_util = self.get_expected_util_avg(experiment)
        signal_mean = self.get_signal_mean(experiment, signal_name)

        error_margin = exp_util * (ERROR_MARGIN_PCT / 100.)
        [cpu] = experiment.wload.cpus

        msg = 'Saw {} around {}, expected {} on cpu {}'.format(
            signal_name, signal_mean, exp_util, cpu)
        self.assertAlmostEqual(signal_mean, exp_util, delta=error_margin,
                               msg=msg)

    @experiment_test
    def test_task_util_avg(self, experiment, tasks):
        """
        Test that the mean of the util_avg signal matched the expected value
        """
        return self._test_signal(experiment, tasks, 'util_avg')

    @experiment_test
    def test_task_load_avg(self, experiment, tasks):
        """
        Test that the mean of the load_avg signal matched the expected value

        Assuming that the system was under little stress (so the task was
        RUNNING whenever it was RUNNABLE) and that the task was run with a
        'nice' value of 0, the load_avg should be similar to the util_avg. So,
        this test does the same as test_task_util_avg but for load_avg.
        """
        return self._test_signal(experiment, tasks, 'load_avg')

# nosetests sometimes tries to set up base test classes directly even though
# they don't have any test_* methods. That means it tries to call setUpClass on
# LoadTrackingTest, which doesn't work. To prevent that, we set __test__ = False
# in the base class and then override it in subclasses. Sorry.
for cls in LoadTrackingTest.__subclasses__():
    cls.__test__ = True
