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


class TraceBase(TestCase):
    """Base class for tests for Trace class"""

    events = ['sched_switch', 'sched_load_se', 'sched_load_avg_task',
              'sched_load_cfs_rq', 'sched_load_avg_cpu']

    def __init__(self, *args, **kwargs):
        super(TraceBase, self).__init__(*args, **kwargs)

        self.test_trace = os.path.join(self.traces_dir, 'test_trace.txt')

        with open(os.path.join(self.traces_dir, 'platform.json')) as f:
            self.platform = json.load(f)

        self.trace = Trace(self.platform, self.traces_dir, self.events)

class TestTrace(TraceBase):
    """Smoke tests for LISA's Trace class"""

    traces_dir = os.path.join(os.path.dirname(__file__), 'traces')

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

        with open(self.test_trace, "w") as fout:
            fout.write(in_data)
        trace = Trace(self.platform, self.test_trace, self.events)

        self.assertEqual(trace.getTaskByPid(1234), 'father')
        self.assertEqual(trace.getTaskByPid(5678), 'child')
        self.assertEqual(trace.getTaskByName('father'), [1234])

        os.remove(self.test_trace)


class TaskSignalsBase(TraceBase):
    """Test getting scheduler task signals from traces"""

    def _test_tasks_dfs(self):
        """Helper for smoke testing _dfg methods in tasks_analysis"""
        df = self.trace.data_frame.task_load_events()
        for column in ['comm', 'pid', 'load_avg', 'util_avg', 'cluster', 'cpu']:
            msg = 'Task signals parsed from {} missing {} column'.format(
                self.trace.data_dir, column)
            self.assertIn(column, df, msg=msg)

        df = self.trace.data_frame.top_big_tasks(min_samples=1)
        for column in ['samples', 'comm']:
            msg = 'Big tasks parsed from {} missing {} column'.format(
                self.trace.data_dir, column)
            self.assertIn(column, df, msg=msg)

        # Pick an arbitrary PID to try plotting signals for.
        # Call plotTasks - although we won't check the results we can just check
        # that things aren't totally borken.
        pid = self.trace.getTasks().keys()[-1]
        self.trace.analysis.tasks.plotTasks(tasks=[pid])

    def _test_cpus_dfs(self):
        """Helper for smoke testing _dfg methods in cpus_analysis"""
        df = self.trace.data_frame.cpu_load_events()
        for column in ['cpu', 'load_avg', 'util_avg']:
            msg = 'CPU signals parsed from {} missing {} column'.format(
                self.trace.data_dir, column)
            self.assertIn(column, df, msg=msg)

        # Call plotCPU - although we won't check the results we can just check
        # that things aren't totally borken.
        self.trace.analysis.cpus.plotCPU()

class TestTraceSchedLoad(TaskSignalsBase):
    """Test parsing sched_load_* events"""

    traces_dir = os.path.join(os.path.dirname(__file__), 'traces', 'sched_load')

    def test_sched_load_signals(self):
        """Test parsing sched_load_se events from EAS upstream integration"""
        self._test_tasks_dfs()

    def test_sched_load_signals(self):
        """Test parsing sched_load_cfs_rq events from EAS upstream integration"""
        self._test_cpus_dfs()

class TestTraceSchedLoadAvg(TaskSignalsBase):
    """Test parsing sched_load_avg_* events"""

    traces_dir = os.path.join(os.path.dirname(__file__),
                              'traces', 'sched_load_avg')

    def test_sched_load_avg_signals(self):
        """Test parsing sched_load_avg_task events from EAS1.2"""
        self._test_tasks_dfs()

    def test_sched_load_avg_signals(self):
        """Test parsing sched_load_avg_cpu events from EAS1.2"""
        self._test_cpus_dfs()
