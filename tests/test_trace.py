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
from pathlib import Path
import functools
import tempfile

import pytest
import numpy as np
import pandas as pd
import polars as pl

from devlib.target import KernelVersion

from lisa.trace import Trace, TraceBase, TxtTraceParser, MockTraceParser, _TraceProxy, MissingTraceEventError
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
        self.plat_info = self._get_plat_info()

    def _wrap_trace(self, trace):
        return trace

    @property
    def trace(self):
        return self._wrap_trace(
            Trace(
                os.path.join(self.traces_dir, 'trace.txt'),
                plat_info=self.plat_info,
                events=self.events,
                normalize_time=False,
                parser=TxtTraceParser.from_txt_file,
            )
        )

    def make_trace(self, in_data, plat_info=None, events=None):
        """
        Get a trace from an embedded string of textual trace data
        """
        return self._wrap_trace(
            Trace(
                None,
                plat_info=self.plat_info if plat_info is None else plat_info,
                events=self.events if events is None else events,
                normalize_time=False,
                parser=TxtTraceParser.from_string(in_data),
            )
        )

    def get_trace(self, trace_name):
        """
        Get a trace from a separate provided trace file
        """
        return self._wrap_trace(
            Trace(
                Path(self.traces_dir, trace_name, 'trace.dat'),
                plat_info=self._get_plat_info(trace_name),
                events=self.events,
            )
        )

    def _get_plat_info(self, trace_name=None):
        trace_dir = self.traces_dir
        if trace_name:
            trace_dir = os.path.join(trace_dir, trace_name)

        path = os.path.join(trace_dir, 'plat_info.yml')
        return PlatformInfo.from_yaml_map(path)

    def test_context_manager(self):
        trace = self.get_trace('doc')
        with trace:
            trace.df_event('sched_switch')

    def test_meta_event(self):
        trace = self.get_trace('doc')
        df = trace.df_event('userspace@rtapp_stats')
        assert 'userspace@rtapp_stats' in trace.available_events
        assert len(df) == 465

    def test_meta_event_2(self):
        in_data = """
          rt-app-5732  [001]   471.410977940: print:                tracing_mark_write: rtapp_main: event=start
         big_0-0-5733  [003]   471.412970020: print:                tracing_mark_write: rtapp_main: event=clock_ref data=471324860
          rt-app-5732  [002]   472.920141960: print:                tracing_mark_write: rtapp_main: event=end
        """
        trace = self.make_trace(in_data)
        df = trace.df_event('userspace@rtapp_main')
        assert 'userspace@rtapp_main' in trace.available_events
        assert len(df) == 3

    def test_meta_event_3(self):
        trace = self.get_trace('doc')

        # This window is somewhere at the beginning of the trace, before the
        # first "print" event.
        trace = trace[470.783594680:470.784129640]

        # Ensure the _MetaEventTraceView that will process our request is above
        # the _WindowTraceView in the stack.
        trace = trace.get_view()

        assert len(trace.df_event('sched_switch')) == 3

        # But it does not contain any rtapp_stats meta events.
        df = trace.df_event('userspace@rtapp_stats')
        assert list(df.columns) == [
            '__cpu',
            '__pid',
            '__comm',
            'c_period',
            'c_run',
            'period',
            'run',
           'slack',
           'wu_lat',
        ]
        assert len(df) == 0

    def test_window(self):
        trace = self.get_trace('doc')
        trace = trace[470.783594680:470.784129640]
        assert len(trace.df_event('sched_switch')) == 3

    def test_window_2(self):
        trace = self.get_trace('doc')
        trace = trace[470.783594680:470.784129640]
        trace = trace.get_view()
        assert len(trace.df_event('sched_switch')) == 3

    def test_meta_event_missing(self):
        trace = self.get_trace('doc')
        with pytest.raises(MissingTraceEventError):
            trace.df_event('userspace@foo')
        assert 'userspace@foo' not in trace.available_events

    def test_meta_event_available(self):
        trace = self.get_trace('doc')
        assert 'userspace@rtapp_stats' in trace.available_events

    def test_event_available(self):
        trace = self.get_trace('doc')
        event = 'sched_switch'
        assert event in trace.available_events
        trace.df_event(event)

    def test_event_not_available(self):
        trace = self.get_trace('doc')
        event = 'foobar_inexistent_event'
        assert event not in trace.available_events
        with pytest.raises(MissingTraceEventError):
            trace.df_event(event)

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

        for x in ('sh', 'sshd'):
            with pytest.raises(ValueError):
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
        assert self.trace.time_range == pytest.approx(expected_duration)

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

    def test_metadata_symbols_address(self):
        trace = self.get_trace('doc')
        syms = trace.get_metadata('symbols-address')
        assert len(syms) == 138329
        assert all(
            isinstance(key, int) and isinstance(value, str)
            for key, value in syms.items()
        )
        assert syms[18446603336539106032] == 'sunrpc_net_id'

    def test_metadata_time_range(self):
        trace = self.get_trace('doc')
        time_range = trace.get_metadata('time-range')
        assert time_range[0].as_nanoseconds == 470783168860
        assert time_range[1].as_nanoseconds == 473280164600

    def test_metadata_cpus_count(self):
        trace = self.get_trace('doc')
        count = trace.get_metadata('cpus-count')
        assert count == 6

    def test_metadata_available_events(self):
        trace = self.get_trace('doc')
        events = trace.get_metadata('available-events')
        assert set(events) == {
            'cpu_frequency',
            'cpu_idle',
            'print',
            'sched_cpu_capacity',
            'sched_migrate_task',
            'sched_overutilized',
            'sched_pelt_cfs',
            'sched_pelt_dl',
            'sched_pelt_irq',
            'sched_pelt_rt',
            'sched_pelt_se',
            'sched_process_exec',
            'sched_process_exit',
            'sched_process_fork',
            'sched_process_free',
            'sched_process_wait',
            'sched_stat_runtime',
            'sched_switch',
            'sched_util_est_cfs',
            'sched_util_est_se',
            'sched_wakeup',
            'sched_wakeup_new',
            'sched_waking',
            'task_newtask',
            'task_rename',
        }

    def test_metadata_trace_id(self):
        trace = self.get_trace('doc')
        trace_id = trace.get_metadata('trace-id')
        assert trace_id == 'trace.dat-8785260356321690258'

    def test_isinstance_base(self):
        assert isinstance(self.trace, TraceBase)

    def test_df_fmt(self):
        import polars as pl
        trace = self.get_trace('doc')

        df = trace.df_event('sched_switch', df_fmt='polars-lazyframe')
        assert isinstance(df, pl.LazyFrame)

        df = trace.get_view(df_fmt='polars-lazyframe').df_event('sched_switch')
        assert isinstance(df, pl.LazyFrame)

    def test_lazyframe_scan_path_rewrite(self):
        trace = self.trace
        trace = trace.get_view(df_fmt='polars-lazyframe')
        df = trace.df_event('sched_switch')

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'df.parquet'
            df.sink_parquet(path)
            df = pl.scan_parquet(path)

            paths = []
            def update_path(path):
                paths.append(path)
                return path

            from lisa.trace import _lazyframe_rewrite, _logical_plan_update_paths
            df = _lazyframe_rewrite(
                df=df,
                update_plan=functools.partial(
                    _logical_plan_update_paths,
                    update_path=update_path,
                )
            )
            assert paths == [str(path)]


class TestTrace(TraceTestCase):
    """Smoke tests for LISA's Trace class"""
    pass


class TestTraceProxy(TraceTestCase):
    """Smoke tests for LISA's TraceProxy class"""
    def _wrap_trace(self, trace):
        proxy = _TraceProxy(None)
        proxy._set_trace(trace)
        return proxy


class TestTraceView(TraceTestCase):
    def _wrap_trace(self, trace):
        return trace.get_view()

    def test_lower_slice(self):
        view = self.trace[81:]
        df = view.ana.status.df_overutilized()
        assert len(df) == 3

    def test_upper_slice(self):
        view = self.trace[:80.402065]
        df = view.ana.status.df_overutilized()
        assert len(df) == 2

    def test_full_slice(self):
        view = self.trace[80:81]
        df = view.ana.status.df_overutilized()
        assert len(df) == 3

    def test_time_range(self):
        expected_duration = np.nextafter(4.0, math.inf)
        trace = self.trace.get_view(window=(76.402065, 80.402065))

        assert trace.time_range == pytest.approx(expected_duration)

    def test_time_range_subscript(self):
        expected_duration = 4.0
        trace = self.trace[76.402065:80.402065]

        assert trace.time_range == pytest.approx(expected_duration)


class TestTraceViewNested(TestTraceView):
    def _wrap_trace(self, trace):
        for _ in range(100):
            trace = trace.get_view()
        return trace


class TestNestedTraceView(TestTraceView):
    def _wrap_trace(self, trace):
        trace = super()._wrap_trace(trace)
        view = trace[trace.start:trace.end]
        assert view is not trace
        return view


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
