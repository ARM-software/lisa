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
from numpy.testing import assert_array_equal
import os
import pandas as pd
from unittest import TestCase

from trace import Trace

#
# Helpers for generating example text-format trace events.
# Doesn't have the exact same whitespace as real trace output, but that
# shouldn't matter.
#

def _event_common(_comm, _pid, _cpu, timestamp):
    return "{_comm}-{_pid}  [{_cpu:0>3}] {timestamp}: ".format(**locals())

def sched_switch(timestamp, cpu,
                 prev_comm, prev_pid, prev_state, next_comm, next_pid):
    return (_event_common(prev_comm, prev_pid, cpu, timestamp) + "sched_switch: "
            "prev_comm={prev_comm} prev_pid={prev_pid} prev_prio=120 prev_state={prev_state} "
            "next_comm={next_comm} next_pid={next_pid} next_prio=120").format(**locals())

def _sched_wakeup(event_name, timestamp, _cpu, comm, pid, target_cpu):
    return (_event_common('<idle>', 0, _cpu, timestamp) + "{event_name}: "
            "comm={comm} pid={pid} prio=100 success=1 target_cpu={target_cpu}"
            .format(**locals()))

# sched_wakeup_new is exactly the same as sched_wakeup, but with a different
# name.
def sched_wakeup(*args, **kwargs):
    return _sched_wakeup('sched_wakeup', *args, **kwargs)
def sched_wakeup_new(*args, **kwargs):
    return _sched_wakeup('sched_wakeup_new', *args, **kwargs)


def sched_migrate_task(timestamp, comm, pid, orig_cpu, dest_cpu):
    return (_event_common("<idle>", 0, 0, timestamp) + "sched_migrate_task: "
            "comm={comm} pid={pid} prio=100 orig_cpu={orig_cpu} dest_cpu={dest_cpu}"
            .format(**locals()))

class TraceBase(TestCase):
    traces_dir = os.path.join(os.path.dirname(__file__), 'traces')
    events = [
        'sched_switch', 'sched_wakeup', 'sched_wakeup_new', 'sched_migrate_task'
    ]

    def __init__(self, *args, **kwargs):
        super(TraceBase, self).__init__(*args, **kwargs)

        self.test_trace = os.path.join(self.traces_dir, 'test_trace.txt')

        with open(os.path.join(self.traces_dir, 'platform.json')) as f:
            self.platform = json.load(f)

        trace_path = os.path.join(self.traces_dir, 'trace.txt')
        self.trace = Trace(self.platform, trace_path, self.events)

    def make_trace(self, lines):
        """Helper to create a trace from lines of a trace.txt"""
        trace_txt = '\n'.join(lines) + '\n' # Final newline is required!

        with open(self.test_trace, "w") as fout:
            fout.write(trace_txt)

        return Trace(self.platform, self.test_trace, self.events,
                     normalize_time=False)

class TestTrace(TraceBase):
    """Smoke tests for LISA's Trace class"""

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

        trace = self.make_trace([
            sched_switch('18765.018235', 2, 'father', 1234, 0, 'father', 5678),
            sched_switch('18765.018235', 2, 'child',  5678, 1, 'father', 5678)
        ])

        self.assertEqual(trace.getTaskByPid(1234), 'father')
        self.assertEqual(trace.getTaskByPid(5678), 'child')
        self.assertEqual(trace.getTaskByName('father'), [1234])

        os.remove(self.test_trace)


class TestTasksAnalysis(TraceBase):
    def __init__(self, *args, **kwargs):
        super(TestTasksAnalysis, self).__init__(*args, **kwargs)
        self.platform = {
            'clusters': {'x': [0, 1, 2, 3, 4, 5]},
            'cpus_count': 6
        }

    def test_dfg_task_cpu_single_task(self):
        """Test the task_cpu DataFrame getter for one task"""

        comm = 'mytask'
        pid = 100

        trace = self.make_trace([
            sched_wakeup_new('0.1', 0, comm, pid, 0),
            sched_migrate_task('0.2', comm, pid, 0, 1),
            sched_migrate_task('0.3', comm, pid, 1, 2),
            sched_wakeup('0.4', 0, comm, pid, 2),
            sched_migrate_task('0.5', comm, pid, 2, 0),
        ])

        df = trace.data_frame.task_cpu()[pid].astype(int)
        assert_array_equal(df.values, [0, 1, 2,  2, 0])
        assert_array_equal(df.index,  [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_dfg_task_cpu_no_wakeup(self):
        """Test the task_cpu DataFrame getter for task without wake event"""

        comm = 'mytask'
        pid = 100

        trace = self.make_trace([
            # Task gets migrated to CPU 1
            sched_migrate_task('0.1', comm, pid, 0, 1),
            # Then to CPU 2
            sched_migrate_task('0.2', comm, pid, 1, 2),
            # Include an unrelated sched_wakeup event just so
            # hasEvents('sched_wakeup') is True
            sched_wakeup('0.5', 0, 'other', pid + 1, 0)
        ])

        df = trace.data_frame.task_cpu()[pid].astype(int)
        df = df[df.shift() != df] # drop consecutive duplicates

        assert_array_equal(df.values, [1, 2])
        assert_array_equal(df.index,  [0.1, 0.2])

    def _test_dfg_task_cpu_no_migration(self, wake_event):

        comm = 'mytask'
        pid = 100

        trace = self.make_trace([
            # Task wakes up on CPU 0
            wake_event('0.1', 0, comm, pid, 0),
            # Include unrelated events so the analyser doesn't bail out due to
            # missing events
            sched_wakeup('0.1', 0, 'other', pid + 1, 0),
            sched_migrate_task('0.5', 'other', pid + 1, 0, 1),
        ])

        df = trace.data_frame.task_cpu()[pid].astype(int)
        df = df[df.shift() != df] # drop consecutive duplicates

        assert_array_equal(df.values, [0])
        assert_array_equal(df.index,  [0.1])

    def test_dfg_task_cpu_no_migration(self):
        """Test the task_cpu DataFrame getter for task without migrate event"""
        self._test_dfg_task_cpu_no_migration(sched_wakeup)

    def test_dfg_task_cpu_no_migration(self):
        """Test the task_cpu DataFrame getter for task only sched_wakeup_new"""
        self._test_dfg_task_cpu_no_migration(sched_wakeup_new)

    def test_dfg_task_runnable(self):

        comm1, comm2 = 'task1', 'task2'
        pid1, pid2 = 101, 102

        trace = self.make_trace([
            # Task1 wakes up on CPU 0
            sched_wakeup('0.1', 0, comm1, pid1, 0),
            # Task2 goes to sleep on CPU1 at the same exact time!
            sched_switch('0.1', 1, comm2, pid2, 1, 'other2', 202),
            # Task1 gets preempted (prev_state = 0)
            sched_switch('0.2', 0, comm1, pid1, 0, 'next', pid1 + 10),
            # Task1 gets switched back in
            sched_switch('0.3', 0, 'other', pid1 + 10, 0, comm1, pid1),
            # Task2 wakes up on CPU 0
            sched_wakeup('0.4', 0, comm2, pid2, 0),
            # Task1 goes to sleep
            sched_switch('0.5', 0, comm1, pid1, 1, 'next', pid1 + 10),
            # Task1 wakes up again
            sched_wakeup('0.6', 0, comm1, pid1, 0),
        ])

        df = trace.data_frame.task_runnable()

        runnable1 = df[pid1]
        runnable1 = runnable1[runnable1.shift() != runnable1] # drop consecutive duplicates
        assert_array_equal(runnable1.values, [True, False, True])
        assert_array_equal(runnable1.index,  [0.1,  0.5,  0.6])

        runnable2 = df[pid2]
        runnable2 = runnable2[runnable2.shift() != runnable2] # drop consecutive duplicates
        assert_array_equal(runnable2.values, [False, True])
        assert_array_equal(runnable2.index,  [0.1,    0.4])

    def test_dfg_task_runnable_no_switch(self):

        comm = 'mytask'
        pid = 100

        trace = self.make_trace([
            # Task wakes up on CPU 0
            sched_wakeup('0.1', 0, comm, pid, 0),
            # Include a switch event so the hasEvents tests pass
            sched_switch('0.2', 0, 'other', pid + 1, 0, 'other2', pid + 2),
        ])

        df = trace.data_frame.task_runnable()[pid].astype(bool)

        assert_array_equal(df.values, [True])
        assert_array_equal(df.index,  [0.1])

    def test_dfg_task_runnable_no_wakeup(self):

        comm = 'mytask'
        pid = 100

        trace = self.make_trace([
            # Task goes to sleep at t=0.1
            sched_switch('0.1', 0, comm, pid, 1, 'other2', pid + 2),
            # Include a wakeup event so the hasEvents tests pass
            sched_wakeup('0.2', 0, 'other', pid + 1, 0),
        ])

        df = trace.data_frame.task_runnable()[pid].astype(bool)
        df = df[df.shift() != df] # drop consecutive duplicates

        assert_array_equal(df.values, [False])
        assert_array_equal(df.index,  [0.1])

    def test_dfg_nr_running_single_cpu(self):
        trace = self.make_trace([
            # nr_running unknown
            sched_wakeup('0.01', 0, 't1', 101, 0),
            sched_wakeup('0.02', 4, 't4', 104, 4),
            sched_switch('0.03', 0, 't2', 102, 1, 't3', 103),
            sched_switch('0.05', 0, 't3', 103, 1, 't1', 101),
            # last task goes to sleep, nr_running = 0
            sched_switch('0.07', 0, 't1', 101, 1, 'swapper', 0),
            # t1 wakes up, nr_running = 1
            sched_wakeup('0.09', 0, 't1', 101, 0),
            sched_switch('0.09', 0, 'swapper', 0, 0, 't1', 101),
            # t2 wakes up, nr_running = 2
            sched_wakeup('0.11', 0, 't2', 102, 0),
            # t3 wakesup, nr_running = 3
            sched_wakeup('0.13', 0, 't3', 103, 0),
            # t2 preempts t1, nr_running unchanged
            sched_switch('0.15', 0, 't1', 101, 0, 't2', 102),
            # t4 migrates in, nr_running = 4
            sched_migrate_task('0.16', 't4', 104, 1, 0),
            # t2 goes to sleep (t3 takes over), nr_running = 3
            sched_switch('0.17', 0, 't2', 102, 1, 't3', 103),
            # t1 migrates out, nr_running = 2
            sched_migrate_task('0.18', 't1', 101, 0, 1),
            # t3 goes to sleep (t4 takes over), nr_running = 1
            sched_switch('0.19', 0, 't3', 103, 1, 't4', 104),
            # t4 goes to sleep, nr_running = 0
            sched_switch('0.21', 0, 't4', 104, 1, 'swapper', 0),
        ])

        nr = trace.data_frame.nr_running()[0].dropna().astype(int)
        nr = nr[nr.shift() != nr]

        assert_array_equal(
            nr.values, [   0,    1,    2,    3,    4,    3,    2,    1,    0])
        assert_array_equal(
            nr.index,  [0.07, 0.09, 0.11, 0.13, 0.16, 0.17, 0.18, 0.19, 0.21])

    def test_dfg_underutilization(self):
        trace = self.make_trace([
            # last task goes to sleep, nr_running[0] = 0
            sched_switch('0.07', 0, 't1', 101, 1, 'swapper', 0),
            # last task goes to sleep, nr_running[1] = 0
            sched_switch('0.07', 1, 't5', 105, 1, 'swapper', 0),
            # last task goes to sleep, nr_running[2] = 0
            sched_switch('0.07', 2, 't7', 107, 1, 'swapper', 0),
            # last task goes to sleep, nr_running[3] = 0
            sched_switch('0.07', 3, 't8', 108, 1, 'swapper', 0),
            # last task goes to sleep, nr_running[4] = 0
            sched_switch('0.07', 4, 't9', 109, 1, 'swapper', 0),
            # last task goes to sleep, nr_running[5] = 0
            sched_switch('0.07', 5, 't10', 110, 1, 'swapper', 0),
            # t4 wakes up, nr_running[1] = 1
            sched_wakeup('0.08', 1, 't4', 104, 1),
            # t1 wakes up on, nr_running[0] = 1
            sched_wakeup('0.09', 0, 't1', 101, 0),
            sched_switch('0.09', 0, 'swapper', 0, 0, 't1', 101),

            # t2 wakes up, nr_running[0] = 2
            sched_wakeup('0.11', 0, 't2', 102, 0),

            # t3 wakesup, nr_running[0] = 3
            sched_wakeup('0.13', 0, 't3', 103, 0),

            # t4 migrates in, nr_running[0] = 4, nr_running[1] = 0
            sched_migrate_task('0.16', 't4', 104, 1, 0),

            # t5 wakes up, nr_running[1] = 1
            sched_wakeup('0.17', 1, 't5', 105, 1),

            # t2 goes to sleep (t3 takes over), nr_running[0] = 3
            sched_switch('0.18', 0, 't2', 102, 1, 't3', 103),

            # t1 migrates out, nr_running[0] = 2, nr_running[2] = 1
            sched_migrate_task('0.19', 't1', 101, 0, 2),

            # t3 goes to sleep (t4 takes over), nr_running[0] = 1
            sched_switch('0.20', 0, 't3', 103, 1, 't4', 104),
            # t4 goes to sleep, nr_running[0] = 1
            sched_switch('0.21', 0, 't4', 104, 1, 'swapper', 0),
            # t5 goes to sleep, nr_running[1] = 0
            sched_switch('0.22', 0, 't5', 105, 1, 'swapper', 0),
        ])

        uu = trace.data_frame.underutilization().astype(int)
        uu = uu[uu.shift() != uu]

        assert_array_equal(
            uu.values, [   0,    1,    2,    3,    2,    1,    0])
        assert_array_equal(
            uu.index,  [0.07, 0.11, 0.13, 0.16, 0.18, 0.19, 0.20])
