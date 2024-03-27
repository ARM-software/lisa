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
import copy
import math

import pytest
import numpy as np
import pandas as pd

from devlib.target import KernelVersion

from lisa.trace import Trace, TxtTraceParser, MockTraceParser
from lisa.analysis.tasks import TaskID
from lisa.datautils import df_squash
from lisa.platforms.platinfo import PlatformInfo
from .utils import StorageTestCase, ASSET_DIR


class TraceTestCase(StorageTestCase):
    traces_dir = ASSET_DIR
    events = [
        'sched_switch',
        'sched_wakeup',
        'sched_overutilized',
        'cpu_idle',
        'sched_load_avg_task',
        'sched_load_se'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_trace = os.path.join(self.traces_dir, 'test_trace.txt')
        self.plat_info = self._get_plat_info()

        self.trace_path = os.path.join(self.traces_dir, 'trace.txt')
        self.trace = Trace(
            self.trace_path,
            plat_info=self.plat_info,
            events=self.events,
            parser=TxtTraceParser.from_txt_file,
        )

    def make_trace(self, in_data, plat_info=None, events=None):
        """
        Get a trace from an embedded string of textual trace data
        """
        trace_path = os.path.join(self.res_dir, "test_trace.txt")
        with open(trace_path, "w") as fout:
            fout.write(in_data)

        return Trace(
            trace_path,
            plat_info=self.plat_info if plat_info is None else plat_info,
            events=self.events if events is None else events,
            normalize_time=False,
            parser=TxtTraceParser.from_txt_file,
        )

    def get_trace(self, trace_name):
        """
        Get a trace from a separate provided trace file
        """
        trace_path = os.path.join(self.traces_dir, trace_name, 'trace.dat')
        return Trace(
            trace_path,
            plat_info=self._get_plat_info(trace_name),
            events=self.events,
        )

    def _get_plat_info(self, trace_name=None):
        trace_dir = self.traces_dir
        if trace_name:
            trace_dir = os.path.join(trace_dir, trace_name)

        path = os.path.join(trace_dir, 'plat_info.yml')
        return PlatformInfo.from_yaml_map(path)


class TestTrace(TraceTestCase):
    """Smoke tests for LISA's Trace class"""

    def test_get_task_id(self):
        for name, pid in [
            ('watchdog/0', 12),
            ('jbd2/sda2-8', 1383)
        ]:
            task_id = TaskID(pid=pid, comm=name)
            task_id2 = TaskID(pid=pid, comm=None)
            task_id3 = TaskID(pid=None, comm=name)

            task_id_tuple = TaskID(pid=None, comm=name)

            for x in (pid, name, task_id, task_id2, task_id3, task_id_tuple):
                assert self.trace.ana.tasks.get_task_id(x) == task_id

        with pytest.raises(ValueError):
            for x in ('sh', 'sshd', 1639, 1642, 1702, 1717, 1718):
                self.trace.ana.tasks.get_task_id(x)

    def test_get_task_name_pids(self):
        for name, pids in [
            ('watchdog/0', [12]),
            ('sh', [1642, 1702, 1717, 1718]),
        ]:
            assert self.trace.ana.tasks.get_task_name_pids(name) == pids

        with pytest.raises(KeyError):
            self.trace.ana.tasks.get_task_name_pids('NOT_A_TASK')

    def test_get_task_pid_names(self):
        for pid, names in [
            (15, ['watchdog/1']),
            (1639, ['sshd']),
        ]:
            assert self.trace.ana.tasks.get_task_pid_names(pid) == names

        with pytest.raises(KeyError):
            self.trace.ana.tasks.get_task_pid_names(987654321)

    def test_get_tasks(self):
        """TestTrace: get_tasks() returns a dictionary mapping PIDs to a single task name"""
        tasks_dict = self.trace.ana.tasks.get_tasks()
        for pid, name in [(1, ['init']),
                          (9, ['rcu_sched']),
                          (1383, ['jbd2/sda2-8'])]:
            assert tasks_dict[pid] == name

    def test_setTaskName(self):
        """TestTrace: getTaskBy{Pid,Name}() properly track tasks renaming"""
        in_data = """
          father-1234  [002] 18765.018235: sched_switch:          prev_comm=father prev_pid=1234 prev_prio=120 prev_state=0 next_comm=father next_pid=5678 next_prio=120
           child-5678  [002] 18766.018236: sched_switch:          prev_comm=child prev_pid=5678 prev_prio=120 prev_state=1 next_comm=sh next_pid=3367 next_prio=120
        """
        trace = self.make_trace(in_data)
        ana = trace.ana.tasks

        assert ana.get_task_pid_names(1234) == ['father']
        assert ana.get_task_pid_names(5678) == ['father', 'child']
        assert ana.get_task_name_pids('father') == [1234]
        assert ana.get_task_name_pids('father', ignore_fork=False) == [1234, 5678]

    def test_time_range(self):
        """
        TestTrace: time_range is the duration of the trace
        """
        expected_duration = 6.676497

        trace = Trace(
            self.trace_path,
            plat_info=self.plat_info,
            events=self.events,
            normalize_time=False,
            parser=TxtTraceParser.from_txt_file,
        )

        assert trace.time_range == pytest.approx(expected_duration)

    def test_squash_df(self):
        """
        TestTrace: df_squash() behaves as expected
        """
        index = [float(i) for i in range(15, 20)]
        data = [(1, i % 2) for i in range(15, 20)]
        df = pd.DataFrame(index=index, data=data, columns=['delta', 'state'])

        # Test "standard" slice:

        # The df here should be:
        # Time delta state
        # 16.5  .5   0
        # 17    .5   1
        df1 = df_squash(df, 16.5, 17.5,)
        head = df1.head(1)
        tail = df1.tail(1)
        assert len(df1.index) == 2
        assert df1.index.tolist() == [16.5, 17]
        assert head['delta'].values[0] == pytest.approx(0.5)
        assert tail['delta'].values[0] == pytest.approx(0.5)
        assert head['state'].values[0] == 0
        assert tail['state'].values[0] == 1

        # Test slice where no event exists in the interval

        # The df here should be:
        # Time delta state
        # 16.2  .6   0
        df2 = df_squash(df, 16.2, 16.8)
        assert len(df2.index) == 1
        assert df2.index[0] == 16.2

        assert df2['delta'].values[0] == pytest.approx(0.6)
        assert df2['state'].values[0] == 0

        # Test slice that matches an event's index

        # The df here should be:
        # Time delta state
        # 16   1   0
        df3 = df_squash(df, 16, 17)
        assert len(df3.index) == 1
        assert df3.index[0] == 16
        assert df3['delta'].values[0] == pytest.approx(1)
        assert df3['state'].values[0] == 0

        # Test slice past last event
        # The df here should be:
        # Time delta state
        # 19.5  .5  1
        df4 = df_squash(df, 19.5, 22)
        assert len(df4.index) == 1
        assert df4.index[0] == 19.5
        assert df4['delta'].values[0] == pytest.approx(0.5)
        assert df4['state'].values[0] == 1

        # Test slice where there's no past event
        df5 = df_squash(df, 10, 30)
        assert len(df5.index) == 5

        # Test slice where that should contain nothing
        df6 = df_squash(df, 8, 9)
        assert len(df6.index) == 0

    def test_overutilized_time(self):
        """
        TestTrace: overutilized_time is the total time spent while system was overutilized
        """
        trace = self.trace

        overutilized_time = trace.ana.status.get_overutilized_time()
        expected_pct = overutilized_time / trace.time_range * 100
        overutilized_pct = trace.ana.status.get_overutilized_pct()

        assert overutilized_pct == pytest.approx(expected_pct)

    def test_plot_cpu_idle_state_residency(self):
        """
        Test that plot_cpu_idle_state_residency doesn't crash
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

        trace.ana.idle.plot_cpu_idle_state_residency(0)

    def test_deriving_cpus_count(self):
        """Test that Trace derives cpus_count if it isn't provided"""
        in_data = """
            father-1234  [000] 18765.018235: sched_switch: prev_comm=father prev_pid=1234 prev_prio=120 prev_state=0 next_comm=father next_pid=5678 next_prio=120
             child-5678  [002] 18765.018235: sched_switch: prev_comm=child prev_pid=5678 prev_prio=120 prev_state=1 next_comm=father next_pid=5678 next_prio=120
        """

        plat_info = copy.copy(
            self.make_trace(in_data).plat_info
        )
        plat_info.force_src('cpus-count', ['SOURCE THAT DOES NOT EXISTS'])
        trace = self.make_trace(in_data, plat_info=plat_info)

        assert trace.cpus_count == 3

    def test_df_cpus_wakeups(self):
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

        df = trace.ana.idle.df_cpus_wakeups()

        exp_index = [519.021928, 519.022641, 519.022642, 519.022643, 519.022867]
        exp_cpus = [4, 4, 1, 2, 3]
        assert df.index.tolist() == exp_index
        assert df.cpu.tolist() == exp_cpus

        df = df[df.cpu == 2]

        assert df.index.tolist() == [519.022643]
        assert df.cpu.tolist() == [2]

    def _test_tasks_dfs(self, trace_name):
        """Helper for smoke testing _dfg methods in tasks_analysis"""
        trace = self.get_trace(trace_name)

        lt_df = trace.ana.load_tracking.df_tasks_signal('util')
        columns = ['comm', 'pid', 'util', 'cpu']
        for column in columns:
            msg = 'Task signals parsed from {} missing {} column'.format(
                trace.trace_path, column)
            assert column in lt_df, msg

        # Pick an arbitrary PID to try plotting signals for.
        pid = lt_df['pid'].unique()[0]
        # Call plot - although we won't check the results we can just check
        # that things aren't totally borken.
        trace.ana.load_tracking.plot_task_signals(pid)

    def test_sched_load_signals(self):
        """Test parsing sched_load_se events from EAS upstream integration"""
        self._test_tasks_dfs('sched_load')

    def test_sched_load_avg_signals(self):
        """Test parsing sched_load_avg_task events from EAS1.2"""
        self._test_tasks_dfs('sched_load_avg')

    def df_peripheral_clock_effective_rate(self):
        """
        TestTrace: getPeripheralClockInfo() returns proper effective rate info.
        """
        in_data = """
          <idle>-0 [002] 380330000000: clock_enable: bus_clk state=1 cpu_id=2
          <idle>-0 [002] 380331000000: clock_set_rate: bus_clk state=750000000 cpu_id=2
          <idle>-0 [000] 380332000000: clock_disable: bus_clk state=0 cpu_id=0
          <idle>-0 [000] 380333000000: clock_enable: bus_clk state=1 cpu_id=0
          <idle>-0 [002] 380334000000: clock_set_rate: bus_clk state=100000000 cpu_id=2
          <idle>-0 [000] 380335000000: clock_disable: bus_clk state=0 cpu_id=0
          <idle>-0 [004] 380339000000: cpu_idle:             state=1 cpu_id=4
        """
        trace = self.make_trace(
            in_data,
            events=[
                'clock_set_rate',
                'clock_disable',
                'clock_enable'
            ]
        )
        df = trace.ana.frequency.df_peripheral_clock_effective_rate(clk_name='bus_clk')
        exp_effective_rate = [float('NaN'), 750000000, 0.0, 750000000, 100000000, 0.0]
        effective_rate = df['effective_rate'].tolist()
        assert len(exp_effective_rate) == len(effective_rate)

        for e, r in zip(exp_effective_rate, effective_rate):
            if (np.isnan(e)):
                assert np.isnan(r)
                continue
            assert e == r

    def test_df_tasks_states(self):
        df = self.trace.ana.tasks.df_tasks_states()

        assert len(df) == 4780
        # Proxy check for detecting delta computation changes
        assert df.delta.sum() == pytest.approx(134.568219)


class TestTraceView(TraceTestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We don't want normalized time
        self.trace = Trace(
            self.trace_path,
            plat_info=self.plat_info,
            events=self.events,
            normalize_time=False,
            parser=TxtTraceParser.from_txt_file,
        )

    def test_lower_slice(self):
        view = self.trace[81:]
        assert len(view.ana.status.df_overutilized()) == 2

    def test_upper_slice(self):
        view = self.trace[:80.402065]
        df = view.ana.status.df_overutilized()
        assert len(view.ana.status.df_overutilized()) == 1

    def test_full_slice(self):
        view = self.trace[80:81]
        assert len(view.ana.status.df_overutilized()) == 2

    def test_time_range(self):
        expected_duration = np.nextafter(4.0, math.inf)

        trace = Trace(
            self.trace_path,
            plat_info=self.plat_info,
            events=self.events,
            normalize_time=False,
            parser=TxtTraceParser.from_txt_file,
        ).get_view(window=(76.402065, 80.402065))

        assert trace.time_range == pytest.approx(expected_duration)

    def test_time_range_subscript(self):
        expected_duration = 4.0

        trace = Trace(
            self.trace_path,
            plat_info=self.plat_info,
            events=self.events,
            normalize_time=False,
            parser=TxtTraceParser.from_txt_file,
        )[76.402065:80.402065]

        assert trace.time_range == pytest.approx(expected_duration)


class TestNestedTraceView(TestTraceView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        trace = Trace(
            self.trace_path,
            plat_info=self.plat_info,
            events=self.events,
            normalize_time=False,
            parser=TxtTraceParser.from_txt_file,
        )

        self.trace = trace[trace.start:trace.end]


class TestTraceNoClusterData(TestTrace):
    """
    Test Trace without cluster data

    Inherits from TestTrace, so all the tests are run again but with
    no cluster info the platform dict.
    """

    def _get_plat_info(self, trace_name=None):
        plat_info = super()._get_plat_info(trace_name)
        plat_info = copy.copy(plat_info)
        plat_info.force_src('freq-domains', ['SOURCE THAT DOES NOT EXISTS'])
        return plat_info


class TestTraceNoPlatform(TestTrace):
    """
    Test Trace with platform=none

    Inherits from TestTrace, so all the tests are run again but with
    platform=None
    """

    def _get_plat_info(self, trace_name=None):
        return None

class TestMockTraceParser(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dfs = {
            'sched_wakeup': pd.DataFrame.from_records(
                [
                    (0, 1, 1, 'task1', 'task1', 1, 1, 1),
                    (1, 2, 1, 'task1', 'task1', 1, 1, 2),
                    (2, 4, 2, 'task2', 'task2', 2, 1, 4),
                ],
                columns=('Time', '__cpu', '__pid', '__comm', 'comm', 'pid', 'prio', 'target_cpu'),
                index='Time',
            ),
        }
        self.trace = Trace(parser=MockTraceParser(dfs, time_range=(0, 42)))

    def test_df_event(self):
        df = self.trace.df_event('sched_wakeup')
        assert not df.empty
        assert 'target_cpu' in df.columns

    def test_time_range(self):
        assert self.trace.start.as_nanoseconds == 0
        assert self.trace.end.as_nanoseconds == 42000000000

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
