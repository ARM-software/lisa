# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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

import json
import os
from unittest import TestCase

from trace import Trace

class TestTrace(TestCase):
    """Smoke tests for LISA's Trace class"""

    traces_dir = os.path.join(os.path.dirname(__file__), 'traces')
    events = [
        'sched_switch',
        'sched_overutilized',
        'cpu_idle',
    ]

    def __init__(self, *args, **kwargs):
        super(TestTrace, self).__init__(*args, **kwargs)

        self.test_trace = os.path.join(self.traces_dir, 'test_trace.txt')

        self.platform = self._get_platform()

        self.trace_path = os.path.join(self.traces_dir, 'trace.txt')
        self.trace = Trace(self.platform, self.trace_path, self.events)

    def make_trace(self, in_data):
        with open(self.test_trace, "w") as fout:
            fout.write(in_data)

        return Trace(self.platform, self.test_trace, self.events,
                     normalize_time=False)

    def _get_platform(self):
        with open(os.path.join(self.traces_dir, 'platform.json')) as f:
            return json.load(f)

    def test_getTaskByName(self):
        """TestTrace: getTaskByName() returns the list of PIDs for all tasks with the specified name"""
        for name, pids in [('watchdog/0', [12]),
                           ('sh', [1642, 1702, 1717, 1718]),
                           ('NOT_A_TASK', [])]:
            self.assertEqual(self.trace.getTaskByName(name), pids)

    def test_getTaskByPid(self):
        """TestTrace: getTaskByPid() returns the name of the task with the specified PID"""
        for pid, names in [(15, 'watchdog/1'),
                           (1639, 'sshd'),
                           (987654321, None)]:
            self.assertEqual(self.trace.getTaskByPid(pid), names)

    def test_getTasks(self):
        """TestTrace: getTasks() returns a dictionary mapping PIDs to a single task name"""
        tasks_dict = self.trace.getTasks()
        for pid, name in [(1, 'init'),
                          (9, 'rcu_sched'),
                          (1383, 'jbd2/sda2-8')]:
            self.assertEqual(tasks_dict[pid], name)

    def test_setTaskName(self):
        """TestTrace: getTaskBy{Pid,Name}() properly track tasks renaming"""
        in_data = """
          father-1234  [002] 18765.018235: sched_switch:          prev_comm=father prev_pid=1234 prev_prio=120 prev_state=0 next_comm=father next_pid=5678 next_prio=120
           child-5678  [002] 18766.018236: sched_switch:          prev_comm=child prev_pid=5678 prev_prio=120 prev_state=1 next_comm=sh next_pid=3367 next_prio=120
        """
        trace = self.make_trace(in_data)

        self.assertEqual(trace.getTaskByPid(1234), 'father')
        self.assertEqual(trace.getTaskByPid(5678), 'child')
        self.assertEqual(trace.getTaskByName('father'), [1234])

        os.remove(self.test_trace)

    def test_time_range(self):
        """
        TestTrace: time_range is the duration of the trace
        """
        expected_duration = 6.676497

        trace = Trace(self.platform, self.trace_path,
                      self.events, normalize_time=False
        )

        self.assertAlmostEqual(trace.time_range, expected_duration, places=6)

    def test_time_range_window(self):
        """
        TestTrace: time_range is the duration of the trace in the given window
        """
        expected_duration = 4.0

        trace = Trace(self.platform, self.trace_path,
                      self.events, normalize_time=False,
                      window=(76.402065, 80.402065)
        )

        self.assertAlmostEqual(trace.time_range, expected_duration, places=6)

    def test_overutilized_time(self):
        """
        TestTrace: overutilized_time is the total time spent while system was overutilized
        """
        events = [
            76.402065,
            80.402065,
            82.001337
        ]

        trace_end = self.trace.ftrace.basetime + self.trace.ftrace.get_duration()
        # Last event should be extended to the trace's end
        expected_time = (events[1] - events[0]) + (trace_end - events[2])

        self.assertAlmostEqual(self.trace.overutilized_time, expected_time, places=6)

    def test_plotCPUIdleStateResidency(self):
        """
        Test that plotCPUIdleStateResidency doesn't crash
        """
        in_data = """
            foo-1  [000] 0.01: cpu_idle: state=0 cpu_id=0
            foo-1  [000] 0.02: cpu_idle: state=-1 cpu_id=0
            bar-2  [000] 0.03: cpu_idle: state=0 cpu_id=1
            bar-2  [000] 0.04: cpu_idle: state=-1 cpu_id=1
            baz-3  [000] 0.05: cpu_idle: state=0 cpu_id=2
            baz-3  [000] 0.06: cpu_idle: state=-1 cpu_id=2
            bam-4  [000] 0.07: cpu_idle: state=0 cpu_id=3
            bam-4  [000] 0.08: cpu_idle: state=-1 cpu_id=3
            child-5678  [002] 18765.018235: sched_switch: prev_comm=child prev_pid=5678 prev_prio=120 prev_state=1 next_comm=father next_pid=5678 next_prio=120
        """
        trace = self.make_trace(in_data)

        trace.analysis.idle.plotCPUIdleStateResidency()

    def test_deriving_cpus_count(self):
        """Test that Trace derives cpus_count if it isn't provided"""
        if self.platform:
            del self.platform['cpus_count']

        in_data = """
            father-1234  [000] 18765.018235: sched_switch: prev_comm=father prev_pid=1234 prev_prio=120 prev_state=0 next_comm=father next_pid=5678 next_prio=120
             child-5678  [002] 18765.018235: sched_switch: prev_comm=child prev_pid=5678 prev_prio=120 prev_state=1 next_comm=father next_pid=5678 next_prio=120
        """

        trace = self.make_trace(in_data)

        self.assertEqual(trace.platform['cpus_count'], 3)

    def test_dfg_cpu_wakeups(self):
        """
        Test the cpu_wakeups DataFrame getter
        """
        trace = self.make_trace("""
          <idle>-0     [004]   519.021928: cpu_idle:             state=4294967295 cpu_id=4
          <idle>-0     [004]   519.022147: cpu_idle:             state=0 cpu_id=4
          <idle>-0     [004]   519.022641: cpu_idle:             state=4294967295 cpu_id=4
          <idle>-0     [001]   519.022642: cpu_idle:             state=4294967295 cpu_id=1
          <idle>-0     [002]   519.022643: cpu_idle:             state=4294967295 cpu_id=2
          <idle>-0     [001]   519.022788: cpu_idle:             state=0 cpu_id=1
          <idle>-0     [002]   519.022831: cpu_idle:             state=2 cpu_id=2
          <idle>-0     [003]   519.022867: cpu_idle:             state=4294967295 cpu_id=3
          <idle>-0     [003]   519.023045: cpu_idle:             state=2 cpu_id=3
          <idle>-0     [004]   519.023080: cpu_idle:             state=1 cpu_id=4
        """)

        df = trace.data_frame.cpu_wakeups()

        exp_index=[519.021928, 519.022641, 519.022642, 519.022643, 519.022867]
        exp_cpus= [         4,          4,          1,          2,          3]
        self.assertListEqual(df.index.tolist(), exp_index)
        self.assertListEqual(df.cpu.tolist(), exp_cpus)

        df = trace.data_frame.cpu_wakeups([2])

        self.assertListEqual(df.index.tolist(), [519.022643])
        self.assertListEqual(df.cpu.tolist(), [2])

class TestTraceNoClusterData(TestTrace):
    """
    Test Trace without cluster data

    Inherits from TestTrace, so all the tests are run again but with
    no cluster info the platform dict.
    """
    def _get_platform(self):
        platform = super(TestTraceNoClusterData, self)._get_platform()
        del platform['clusters']
        return platform

class TestTraceNoPlatform(TestTrace):
    """
    Test Trace with platform=none

    Inherits from TestTrace, so all the tests are run again but with
    platform=None
    """
    def _get_platform(self):
        return None
