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
import contextlib
import itertools
import abc
import datetime

import pytest
import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from devlib.target import KernelVersion

from lisa.trace import Trace, TraceBase, TxtTraceParser, MockTraceParser, _TraceProxy, MissingTraceEventError, TraceParserBase, MissingMetadataError, _json_checksum, _ScanPushdowns
from lisa.analysis.tasks import TaskID
from lisa.datautils import df_squash, Timestamp, _polars_parse_predicate
from lisa.platforms.platinfo import PlatformInfo
from .utils import StorageTestCase, ASSET_DIR


class TraceTestCaseBase(StorageTestCase):
    traces_dir = Path(ASSET_DIR)

    def _wrap_trace(self, trace):
        return trace

    @property
    def trace(self):
        return self._wrap_trace(
            Trace(
                self.traces_dir / 'trace_txt' / 'trace.txt',
                plat_info=self._get_plat_info('trace_txt'),
                normalize_time=False,
                # We need to specify the parser explicitly, as we don't have a
                # human-readable trace (the result of rendering each event
                # according to its format string). Instead, we have a raw text
                # trace.
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
                plat_info=plat_info,
                events=events,
                normalize_time=False,
                parser=TxtTraceParser.from_string(in_data),
            )
        )

    def get_trace(self, trace_name, **kwargs):
        """
        Get a trace from a separate provided trace file
        """
        path = self.traces_dir / trace_name
        path, = path.glob('trace.*')
        return self._wrap_trace(
            Trace(
                path,
                plat_info=self._get_plat_info(trace_name),
                **kwargs,
            )
        )

    @contextlib.contextmanager
    def get_trace_fresh_swap(self, trace_name):
        with tempfile.TemporaryDirectory() as folder:
            def f(**kwargs):
                return self.get_trace(
                    trace_name,
                    swap_dir=folder,
                    **kwargs,
                )
            yield f

    def _get_plat_info(self, trace_name):
        path = self.traces_dir / trace_name / 'plat_info.yml'
        try:
            return PlatformInfo.from_yaml_map(path)
        except FileNotFoundError:
            return PlatformInfo()


class TraceTestCase(TraceTestCaseBase):

    def test_parse_all(self):
        trace = self.trace
        trace.get_view(events='all')

    def test_preload(self):
        trace = self.trace
        trace.get_view(
            events={'switch'},
            events_namespaces=('sched_',),
            strict_events=True,
        )

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
                path = str(path)
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


    def test_iter_rows(self):
        trace = self.get_trace('doc')
        trace = trace[:470.783675]
        rows = list(trace.iter_rows())
        events = {event for event, fields in rows}

        assert len(rows) == 15
        assert events == {
            'sched_process_exit',
            'sched_waking',
            'sched_pelt_se',
            'sched_pelt_irq',
            'sched_util_est_cfs',
            'sched_pelt_cfs',
            'cpu_idle',
            'sched_wakeup',
            'cpu_idle',
        }

        assert rows[0] == (
            'sched_process_exit',
            {
                'Time': datetime.timedelta(
                    seconds=470,
                    microseconds=783168,
                ),
                '__cpu': 5,
                '__pid': 5715,
                '__comm': 'trace-cmd',
                'comm': 'trace-cmd',
                'pid': 5715,
                'prio': 120,
            }
        )



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

    def _get_plat_info(self, trace_name):
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

    def _get_plat_info(self, trace_name):
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


class TraceLogCursor:
    def __init__(self, pos, log):
        self.pos = pos
        self.log = log

    @classmethod
    def from_log(cls, log):
        return cls(
            log=iter(log),
            pos=0,
        )

    def consume(self):
        for i, entry in enumerate(self.log):
            cursor = self.__class__(
                pos=self.pos + i + 1,
                log=self.log,
            )
            yield (cursor, entry)

    def tee(self, n=2):
        pos = self.pos
        logs = itertools.tee(self.log, n)
        return (
            self.__class__(
                log=log,
                pos=pos,
            )
            for log in logs
        )

    def max_entries(self, n):
        return self.__class__(
            log=itertools.islice(self.log, n),
            pos=self.pos,
        )

class TraceLogMatch:
    def __init__(self, cursor, width):
        self.width = width
        self.cursor = cursor

    @property
    def span(self):
        width = self.width
        pos = self.cursor.pos
        return (pos - width, pos)


class NotMatchedError(Exception):
    def __init__(self, cursor, msg):
        self.msg = msg
        self.cursor = cursor

    def __str__(self):
        return self.msg


class TraceLogPattern(abc.ABC):
    def assert_log(self, log):
        cursor = TraceLogCursor.from_log(log)
        self._match(cursor)

    def __invert__(self):
        return TraceLogNotPattern(self)

    @abc.abstractmethod
    def _match(self, cursor):
        return TraceLogMatch(
            cursor=cursor,
            width=0,
        )


class TraceLogNotPattern(TraceLogPattern):
    def __init__(self, pattern):
        self._pattern = pattern

    def _match(self, cursor):
        pattern = self._pattern
        cursor, cursor_bak = cursor.tee()
        try:
            pattern._match(cursor)
        except NotMatchedError as e:
            return TraceLogMatch(
                cursor=cursor_bak,
                width=0,
            )
        else:
            raise NotMatchedError(
                cursor_bak,
                f'Could not match pattern: {self}',
            )

    def __str__(self):
        return f'~({self._pattern})'


class TraceLogPredicatePattern(TraceLogPattern):
    def __init__(self, f):
        self._f = f

    def _match(self, cursor):
        for cursor, entry in cursor.consume():
            if self._f(entry):
                return TraceLogMatch(
                    cursor=cursor,
                    width=1,
                )

        raise NotMatchedError(
            cursor,
            f'Could not match: {self}',
        )

    def __str__(self):
        return f'{self._f.__code__}'


class TraceLogLookbehindPattern(TraceLogPattern):
    def __init__(self, behind, pattern):
        self._behind = behind
        self._pattern = pattern

    def _match(self, cursor):
        saved_pos = cursor.pos
        cursor1, cursor2 = cursor.tee()

        match1 = self._pattern._match(cursor1)
        consumed = match1.cursor.pos - saved_pos

        cursor2 = cursor2.max_entries(max(0, consumed - match1.width))
        self._behind._match(cursor2)

        return match1

    def __str__(self):
        return f'(?<{self._behind})({self._pattern})'


class TraceLogLookaheadPattern(TraceLogPattern):
    def __init__(self, pattern, ahead):
        self._ahead = ahead
        self._pattern = pattern

    def _match(self, cursor):
        match1 = self._pattern._match(cursor)
        cursor1, cursor2 = match1.cursor.tee()

        try:
            self._ahead._match(cursor2)
        except NotMatchedError as e:
            raise NotMatchedError(
                cursor1,
                e.msg,
            )
        else:
            return TraceLogMatch(
                cursor=cursor1,
                width=match1.width,
            )

    def __str__(self):
        return f'({self._pattern})(?={self._ahead})'


class TraceLogLookaroundPattern(TraceLogPattern):
    def __init__(self, behind, pattern, ahead):
        self._pattern = TraceLogLookaheadPattern(
            pattern=TraceLogLookbehindPattern(
                behind=behind,
                pattern=pattern,
            ),
            ahead=ahead,
        )

    def _match(self, cursor):
        return self._pattern._match(cursor)

    def __str__(self):
        return str(self._pattern)


class TraceLogSeqPattern(TraceLogPattern):
    def __init__(self, patterns):
        self._patterns = list(patterns)

    def _match(self, cursor):
        start = cursor.pos
        for i, pattern in enumerate(self._patterns):
            match = pattern._match(cursor)
            cursor = match.cursor
            if i == 0:
                start, _ = match.span

        return TraceLogMatch(
            cursor=cursor,
            width=cursor.pos - start,
        )

    def __str__(self):
        return ''.join(
            f'({pattern})'
            for pattern in self._patterns
        )


class TestTraceCache(TraceTestCase):
    """Test the trace cache"""

    def _assert_event(self, log, event, seq):
        return self._assert_data(
            log=log,
            key=event,
            typ='event',
            seq=seq,
        )

    def _assert_metadata(self, log, key, seq):
        return self._assert_data(
            log=log,
            key=key,
            typ='metadata',
            seq=seq,
        )

    def _assert_data(self, log, key, typ, seq):

        def event_parsed(event):
            return TraceLogPredicatePattern(lambda entry: (
                entry['type'] == 'spinup-parser'
                and event in entry['data']['events']
            ))

        def event_cold(event):
            return TraceLogSeqPattern([
                TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'df-event'
                    and entry['data']['event'] == event
                )),
                TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'spinup-parser'
                    and event in entry['data']['events']
                )),
            ])

        def event_hot(event):
            return TraceLogLookaroundPattern(
                # We should not spin up the parser for a df_event() call before
                # the df-event log, if it is expected to be available in the
                # cache.
                behind=~event_parsed(event),
                pattern=TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'df-event'
                    and entry['data']['event'] == event
                )),
                # We also do not expect it _after_ the call, which could happen
                # with cold delayed events.
                ahead=~event_parsed(event),
            )

        def metadata_cold(key):
            return TraceLogSeqPattern([
                TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'get-metadata'
                    and entry['data']['key'] == key
                )),
                TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'spinup-parser'
                    and key in entry['data']['metadata']
                )),
            ])

        def metadata_hot(key):
            return TraceLogLookbehindPattern(
                # We should not spin up the parser for the 2nd get_metadata()
                # call, as we expect it to be available in the cache.
                behind=~TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'spinup-parser'
                    and key in entry['data']['metadata']
                )),
                pattern=TraceLogPredicatePattern(lambda entry: (
                    entry['type'] == 'get-metadata'
                    and entry['data']['key'] == key
                )),
            )

        def pick(pattern):
            match (typ, pattern):
                case ('event', 'hot'):
                    return event_hot(key)
                case ('event', 'cold'):
                    return event_cold(key)
                case ('event', 'parsed'):
                    return event_parsed(key)
                case ('event', '~hot'):
                    return ~event_hot(key)
                case ('event', '~cold'):
                    return ~event_cold(key)
                case ('event', '~parsed'):
                    return ~event_parsed(key)
                case ('metadata', 'hot'):
                    return metadata_hot(key)
                case ('metadata', 'cold'):
                    return metadata_cold(key)
                case ('metadata', '~hot'):
                    return ~metadata_hot(key)
                case ('metadata', '~cold'):
                    return ~metadata_cold(key)
                case _:
                    raise ValueError(f'Unknown pattern: {pattern}')

        pattern = TraceLogSeqPattern(map(pick, seq))
        pattern.assert_log(log)

    def test_basic_event(self):
        event1 = 'sched_switch'

        with self.get_trace_fresh_swap('doc') as make_trace:
            trace1 = make_trace(df_fmt='polars-lazyframe')
            with trace1._with_activity_log() as log1:
                df = trace1.df_event(event1)
                df1 = df.collect()

                self._assert_event(log1, event1, ['~hot'])
                self._assert_event(log1, event1, ['cold'])

                df = trace1.df_event(event1)
                df2 = df.collect()

                self._assert_event(log1, event1, ['cold', 'hot'])
                assert_frame_equal(df1, df2)

            del trace1

            trace1 = make_trace(df_fmt='polars-lazyframe')
            with trace1._with_activity_log() as log1:
                df = trace1.df_event(event1)
                df.collect()

                self._assert_event(log1, event1, ['~cold'])
                self._assert_event(log1, event1, ['hot'])

    def test_pushdowns(self):
        def make_predicate_pushdowns(predicate):
            return _ScanPushdowns(
                schema={
                    'prev_pid': pl.Int32,
                },
                predicate=predicate,
            )

        def make_column_pushdowns(columns):
            return _ScanPushdowns(
                schema={
                    'prev_pid': pl.Int32,
                    'prev_comm': pl.Categorical,
                },
                columns=columns,
            )

        @contextlib.contextmanager
        def make_trace():
            with self.get_trace_fresh_swap('doc') as make_trace:
                yield make_trace(df_fmt='polars-lazyframe')

        def check(f):
            i = 0
            def _f(trace):
                nonlocal i
                try:
                    return f(i, trace)
                finally:
                    i += 1

            with contextlib.ExitStack() as trace_stack:
                traces = [
                    trace_stack.enter_context(
                        make_trace()
                    )
                    for _ in range(3)
                ]

                dfs = Trace.map_traces(_f, traces)
                pl.collect_all(dfs)

        def f_predicate(i, trace, *, predicate1, predicate2):
            assert make_predicate_pushdowns(predicate1).subsumes(
                make_predicate_pushdowns(predicate2)
            )

            df = trace.df_event('sched_switch')

            if i == 0:
                df = df.filter(predicate1)
            else:
                # We collect the dataframe, with the predicate that we
                # expected from the processing of the first trace. This
                # should spinup the parser with that predicate pushdown and
                # parse it.
                df.filter(predicate1).collect()

                # We then get a new dataframe and filter it with a
                # different predicate. Since predicate1 subsumes predicate2
                # (all the rows selected by predicate2 are also selected by
                # predicate1, plus more), we should be able to re-use the
                # cache entry.
                with trace._with_activity_log() as log:
                    df = trace.df_event('sched_switch')
                    df = df.filter(predicate2)
                    df = df.collect()
                    # Check that the event we have a cache hit straight away,
                    # coming from the re-use of the previously parsed dataframe
                    # with a different predicate that subsumes this one.
                    self._assert_event(log, 'sched_switch', ['hot'])
                    df = df.lazy()

            return df

        def f_columns(i, trace, *, cols1, cols2):
            assert make_column_pushdowns(cols1).subsumes(
                make_column_pushdowns(cols2)
            )

            df = trace.df_event('sched_switch')

            if i == 0:
                df = df.select(cols1)
            else:
                df.select(cols1).collect()

                with trace._with_activity_log() as log:
                    df = trace.df_event('sched_switch')
                    df = df.select(cols2)
                    df = df.collect()
                    # Check that the event we have a cache hit straight away,
                    # coming from the re-use of the previously parsed dataframe
                    # with a different predicate that subsumes this one.
                    self._assert_event(log, 'sched_switch', ['hot'])
                    df = df.lazy()

            return df


        def f_columns_non_subsume(i, trace, *, cols1, cols2):
            assert not make_column_pushdowns(cols1).subsumes(
                make_column_pushdowns(cols2)
            )

            if i == 0:
                return trace.df_event('sched_switch')
            else:
                with trace._with_activity_log() as log:
                    df = trace.df_event('sched_switch')
                    df = df.select(cols2)
                    df = df.collect()
                    # Since cols2 is not a subset of cols1, we will trigger the
                    # path that re-parses the trace because we asked for some
                    # pushdowns that are incompatible with how the first trace
                    # was processed.
                    self._assert_event(log, 'sched_switch', ['cold'])
                    return df.lazy()

        check(
            functools.partial(
                f_predicate,
                predicate1=pl.col('prev_pid') <= pl.lit(1, pl.Int32),
                predicate2=pl.col('prev_pid') == pl.lit(1, pl.Int32),
            )
        )

        check(
            functools.partial(
                f_predicate,
                predicate1=pl.col('prev_pid') >= pl.lit(1, pl.Int32),
                predicate2=pl.col('prev_pid') == pl.lit(1, pl.Int32),
            )
        )

        check(
            functools.partial(
                f_columns,
                cols1=['prev_pid', 'prev_comm'],
                cols2=['prev_pid'],
            )
        )

        check(
            functools.partial(
                f_columns_non_subsume,
                cols1=['prev_pid', 'prev_comm'],
                cols2=['next_pid'],
            )
        )

    def test_delayed_event(self):
        event1 = 'sched_switch'
        event2 = 'sched_wakeup'

        for i in range(2):
            with self.get_trace_fresh_swap('doc') as make_trace:
                trace1 = make_trace(df_fmt='polars-lazyframe')
                with trace1._with_activity_log() as log1:

                    df1 = trace1.df_event(event1)
                    df2 = trace1.df_event(event2)

                    # With delayed events parsing, the LazyFrame is returned
                    # immediately without spinning up the parser, so it appears
                    # to be hot at first.
                    for event in (event1, event2):
                        self._assert_event(log1, event, ['hot'])

                    match i:
                        case 0:
                            df1.collect()
                        case 1:
                            df2.collect()

                    # Then after collecting one of the LazyFrame, the parsing
                    # got triggered and the load finally looks like a cold
                    # load.
                    for event in (event1, event2):
                        self._assert_event(log1, event, ['cold'])

    def test_shared_cache(self):
        event1 = 'sched_switch'
        event2 = 'sched_wakeup'

        with self.get_trace_fresh_swap('doc') as make_trace:
            trace1 = make_trace(df_fmt='polars-lazyframe')
            with trace1._with_activity_log() as log1:
                trace1.df_event(event1).collect()

                self._assert_event(log1, event1, ['~hot'])
                self._assert_event(log1, event1, ['cold'])

                # Same Trace object, query again. This should be serviced by the
                # cache.
                t1e1 = trace1.df_event(event1).collect()

                self._assert_event(log1, event1, ['cold', 'hot'])

                trace2 = make_trace(df_fmt='polars-lazyframe')
                with trace2._with_activity_log() as log2:
                    t2e2 = trace2.df_event(event2).collect()

                    # event2 should not be hot in trace2 swap
                    # But it should be cold
                    self._assert_event(log2, event2, ['~hot'])
                    self._assert_event(log2, event2, ['cold'])

                    # And now it has been parsed, so it is hot
                    trace2.df_event(event2).collect()
                    self._assert_event(log2, event2, ['cold', 'hot'])

                    # And it is hot in trace1 as well, since they share the
                    # swap
                    t1e2 = trace1.df_event(event2).collect()
                    self._assert_event(log1, event2, ['hot'])

                    # trace2 shares the swap with trace1, so it has event1 hot already
                    t2e1 = trace2.df_event(event1).collect()
                    self._assert_event(log2, event1, ['hot'])

                    assert_frame_equal(t1e1, t2e1)
                    assert_frame_equal(t1e2, t2e2)


    def test_metadata(self):
        with self.get_trace_fresh_swap('doc') as make_trace:
            trace1 = make_trace(df_fmt='polars-lazyframe')
            with trace1._with_activity_log() as log1:
                keys = sorted(TraceParserBase.METADATA_KEYS)

                def get_meta(trace, key):
                    try:
                        return trace.get_metadata(key)
                    except MissingMetadataError:
                        return None

                def get_all_meta(trace):
                    return {
                        key: value
                        for key in keys
                        if (value := get_meta(trace, key)) is not None
                    }

                # Before we parse anything in the trace, the keys should not
                # have been loaded.
                for key in keys:
                    self._assert_metadata(log1, key, ['~cold'])
                    self._assert_metadata(log1, key, ['~hot'])

                m1 = get_all_meta(trace1)

                # Now that the metadata is loaded, it should be either hot or
                # cold, as loading the first metadata may lead to load them
                # all, so some of them could appear as hot already.
                for key in m1.keys():
                    try:
                        self._assert_metadata(log1, key, ['cold'])
                    except NotMatchedError:
                        self._assert_metadata(log1, key, ['hot'])

                assert set(m1.keys()) == {
                    'available-events',
                    'cpus-count',
                    'symbols-address',
                    'time-range',
                    'trace-id',
                }

                assert m1['available-events'] == {
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

                assert m1['cpus-count'] == 6

                assert len(m1['symbols-address']) == 138329
                assert _json_checksum(
                    m1['symbols-address'],
                    method='sha256',
                ) == '756c9f59f61f272442385ee6fdc55921bf85f0773c157433c0811cbe98eea765'

                assert m1['trace-id'] == 'trace.dat-8785260356321690258'

                assert m1['time-range'] == (
                    Timestamp(470783168860, unit='ns'),
                    Timestamp(473280164600, unit='ns'),
                )

                m2 = get_all_meta(trace1)
                assert m1 == m2

                trace2 = make_trace(df_fmt='polars-lazyframe')
                with trace2._with_activity_log() as log2:

                    m3 = get_all_meta(trace2)
                    assert m1 == m3

                    # All the keys are hot now, as they were loaded by the
                    # previous Trace instance
                    for key in m1.keys():
                        self._assert_metadata(log2, key, ['hot'])

    def _test_scrub(self, count_entries, clear_kwargs, shared_cache):
        def collect_df(trace):
            df = trace.df_event('sched_switch').collect()
            # Sanity check on the dataframe, just to make sure we don't get
            # something completely broken
            assert len(df) == 1501
            return df

        with self.get_trace_fresh_swap('doc') as make_trace:
            _make_trace = functools.partial(
                make_trace,
                df_fmt='polars-lazyframe'
            )

            trace1 = _make_trace()
            collect_df(trace1)
            count1 = count_entries(trace1)

            trace2 = _make_trace(**clear_kwargs)
            count2 = count_entries(trace2)
            count1_2 = count_entries(trace1)

            # There should be something in the cache
            assert count1 > 0

            # The cache should have been purged by **clear_kwargs, so less
            # cache entries.
            assert count2 <= count1

            if shared_cache:
                # The cache is shared in real time, so the new count on the old
                # trace should be exactly the same
                assert count1_2 == count2

            # The cache should be now empty.
            assert count2 == 0

            # Make sure everything still works after purging caches
            assert_frame_equal(
                collect_df(trace1),
                collect_df(trace2)
            )

    def test_scrub_swap(self):
        def count_swap_entries(trace):
            with trace._cache._swap_db_transaction('r') as trans:
                return len(list(trans.load_all_swap_entries()))

        return self._test_scrub(
            count_entries=count_swap_entries,
            clear_kwargs=dict(max_swap_size=0),
            # The on-disk swap is shared dynamically between trace instances.
            shared_cache=True,
        )

    def test_scrub_mem(self):
        def count_cache_entries(trace):
            return len(trace._cache._cache)

        return self._test_scrub(
            count_entries=count_cache_entries,
            clear_kwargs=dict(max_mem_size=0),
            # The in-memory cache is not shared dynamically between trace
            # instances.
            shared_cache=False,
        )

class TestMapTraces(TraceTestCaseBase):
    def test_map_traces(self):
        def check(f, expected_pushdowns):
            with contextlib.ExitStack() as trace_stack:

                @contextlib.contextmanager
                def make_trace():
                    with self.get_trace_fresh_swap('doc') as make_trace:
                        yield make_trace(df_fmt='polars-lazyframe')

                traces = [
                    trace_stack.enter_context(
                        make_trace()
                    )
                    for _ in range(5)
                ]

                with contextlib.ExitStack() as log_stack:
                    logs = [
                        log_stack.enter_context(
                            trace._with_activity_log()
                        )
                        for trace in traces
                    ]

                    collected = pl.collect_all(
                        df.lazy()
                        for df in Trace.map_traces(f, traces)
                    )

                    items = [
                        {
                            'trace': trace,
                            'df': df,
                            'log': list(log),
                        }
                        for trace, log, df in zip(
                            traces,
                            logs,
                            collected,
                        )
                    ]

                collect_pushdowns = lambda df: TraceLogPredicatePattern(
                    lambda entry: (
                        entry['type'] == 'df-collect-with-pushdowns'
                        and entry['data']['parser-subsumes-collect']
                        and entry['data']['collect-pushdowns'].subsumes(
                            expected_pushdowns(df)
                        )
                    ),
                )

                parser_pushdowns = lambda df: TraceLogPredicatePattern(
                    lambda entry: (
                        entry['type'] == 'df-collect-with-pushdowns'
                        and entry['data']['parser-subsumes-collect']
                        and entry['data']['parser-pushdowns'].subsumes(
                            expected_pushdowns(df)
                        )
                    ),
                )

                # The first LazyFrame was not parsed with pushdowns as it was
                # the instrumented one.
                for item in items[1:]:
                    df = item['df']
                    log = item['log']
                    parser_pushdowns(df).assert_log(log)

                for item in items:
                    df = item['df']
                    log = item['log']
                    trace = item['trace']

                    collect_pushdowns(df).assert_log(log)

                    # Pushdowns should have been applied at the parser level
                    # and in the higher layers of the Trace infrastructure, so
                    # re-applying the same filtering should be a no-op.
                    assert_frame_equal(
                        df.lazy(),
                        expected_pushdowns(df).apply(df),
                    )

                    # Check that we get the same result regardless of whether
                    # we applied scan pushdowns or not.
                    assert_frame_equal(
                        df.lazy(),
                        f(trace).lazy(),
                    )

        f1_pushdowns = lambda df: _ScanPushdowns(
            schema=df.collect_schema(),
            n_rows=2,
            columns=('prev_comm', 'prev_pid'),
            predicate=(
                pl.col('prev_pid') == pl.lit(0, pl.Int32)
            ),
        )

        def f1(trace):
            df = trace.df_event('sched_switch')
            df = f1_pushdowns(df).apply(df)
            return df

        check(
            f=f1,
            expected_pushdowns=f1_pushdowns,
        )


        def f2(trace):
            df = f1(trace)
            return df.collect()

        check(
            f=f2,
            expected_pushdowns=f1_pushdowns,
        )


        def f3(trace):
            return f2(trace).lazy()

        check(
            f=f3,
            expected_pushdowns=f1_pushdowns,
        )


class TestScanPushdowns(TestCase):
    def test_predicate_ir(self):
        def to_ir(expr, schema):
            def binop(left, op, right):
                return {
                    'type': 'binop',
                    'data': {
                        'left': left,
                        'op': op,
                        'right': right,
                    }
                }

            def unaryop(op, expr):
                return {
                    'type': 'unaryop',
                    'data': {
                        'op': op,
                        'expr': expr,
                    }
                }

            def column(col):
                return {
                    'type': 'column',
                    'data': col,
                }

            def literal(lit):
                return {
                    'type': 'literal',
                    'data': lit,
                }

            return _polars_parse_predicate(
                expr=expr,
                schema=schema,
                binop=binop,
                unaryop=unaryop,
                column=column,
                literal=literal
            )

        def check(predicate, ir, schema):
            _ir = to_ir(
                expr=predicate,
                schema=schema,
            )
            assert ir == _ir

        check(
            pl.lit(True),
            {'type': 'literal', 'data': True},
            schema={},
        )

        check(
            pl.lit(False),
            {'type': 'literal', 'data': False},
            schema={},
        )

        check(
            pl.lit(None),
            {'type': 'literal', 'data': None},
            schema={},
        )

        check(
            pl.lit(None, pl.UInt8),
            {'type': 'literal', 'data': None},
            schema={},
        )

        check(
            pl.lit(0),
            {'type': 'literal', 'data': 0},
            schema={},
        )

        check(
            pl.lit(1),
            {'type': 'literal', 'data': 1},
            schema={},
        )

        check(
            pl.lit(1.2),
            {'type': 'literal', 'data': 1.2},
            schema={},
        )

        with pytest.raises(ValueError):
            check(
                pl.lit(float('nan')),
                {'type': 'literal', 'data': 42},
                schema={},
            )

        with pytest.raises(ValueError):
            check(
                pl.lit(float('inf')),
                {'type': 'literal', 'data': 42},
                schema={},
            )

        check(
            pl.lit("hello"),
            {'type': 'literal', 'data': 'hello'},
            schema={},
        )

        check(
            pl.lit(b"hello"),
            {'type': 'literal', 'data': b'hello'},
            schema={},
        )

        check(
            pl.col('a'),
            {'type': 'column', 'data': 'a'},
            schema={'a': pl.Int32},
        )

        check(
            ~pl.col('a'),
            {
                'type': 'unaryop',
                'data': {
                    'op': '~',
                    'expr': {'type': 'column', 'data': 'a'},
                },
            },
            schema={'a': pl.Int32},
        )

        check(
            ~pl.col('a'),
            {
                'type': 'unaryop',
                'data': {
                    'op': '!',
                    'expr': {'type': 'column', 'data': 'a'},
                },
            },
            schema={'a': pl.Boolean},
        )

        binops = [
            (pl.col('a') == 1, '=='),
            (pl.col('a') < 1, '<'),
            (pl.col('a') | 1, '|'),
        ]

        for expr, op in binops:
            check(
                expr,
                {
                    'type': 'binop',
                    'data': {
                        'op': op,
                        'left': {'type': 'column', 'data': 'a'},
                        'right': {'type': 'literal', 'data': 1},
                    }
                },
                schema={'a': pl.Int32},
            )

        check(
            pl.col('a') <= 1,
            {
                'type': 'binop',
                'data': {
                    'op': '||',
                    'left': {
                        'type': 'binop',
                        'data': {
                            'left': {'type': 'column', 'data': 'a'},
                            'right': {'type': 'literal', 'data': 1},
                            'op': '==',
                        }
                    },
                    'right': {
                        'type': 'binop',
                        'data': {
                            'left': {'type': 'column', 'data': 'a'},
                            'right': {'type': 'literal', 'data': 1},
                            'op': '<',
                        }
                    },
                }
            },
            schema={'a': pl.Int32},
        )

        check(
            pl.col('a') > 1,
            {
                'type': 'unaryop',
                'data': {
                    'op': '!',
                    'expr': {
                        'type': 'binop',
                        'data': {
                            'op': '||',
                            'left': {
                                'type': 'binop',
                                'data': {
                                    'op': '==',
                                    'left': {'type': 'column', 'data': 'a'},
                                    'right': {'type': 'literal', 'data': 1},
                                }
                            },
                            'right': {
                                'type': 'binop',
                                'data': {
                                    'op': '<',
                                    'left': {'type': 'column', 'data': 'a'},
                                    'right': {'type': 'literal', 'data': 1},
                                }
                            },
                        }
                    },
                }
            },
            schema={'a': pl.Int32},
        )


        check(
            (pl.col('a') == 1) | (pl.col('b') == 1),
            {
                'type': 'binop',
                'data': {
                    'op': '||',
                    'left': {
                        'type': 'binop',
                        'data': {
                            'left': {'type': 'column', 'data': 'a'},
                            'right': {'type': 'literal', 'data': 1},
                            'op': '==',
                        }
                    },
                    'right': {
                        'type': 'binop',
                        'data': {
                            'left': {'type': 'column', 'data': 'b'},
                            'right': {'type': 'literal', 'data': 1},
                            'op': '==',
                        }
                    },
                }
            },
            schema={
                'a': pl.Int32,
                'b': pl.Int32,
            },
        )

        check(
            pl.col('a') | pl.col('b'),
            {
                'type': 'binop',
                'data': {
                    'op': '|',
                    'left': {'type': 'column', 'data': 'a'},
                    'right': {'type': 'column', 'data': 'b'},
                }
            },
            schema={
                'a': pl.Int32,
                'b': pl.Int32,
            },
        )

    def test_predicate_subsumption(self):

        def check(predicate1, predicate2, schema, check):
            pushdowns1 = _ScanPushdowns(
                predicate=predicate1,
                schema=schema,
            )

            pushdowns2 = _ScanPushdowns(
                predicate=predicate2,
                schema=schema,
            )

            check(pushdowns1, pushdowns2)


        def strict_subsumes(pushdowns1, pushdowns2):
            assert pushdowns1.subsumes(pushdowns2)
            assert not pushdowns2.subsumes(pushdowns1)

        def neither_subsumes(pushdowns1, pushdowns2):
            assert not pushdowns1.subsumes(pushdowns2)
            assert not pushdowns2.subsumes(pushdowns1)

        def both_subsumes(pushdowns1, pushdowns2):
            assert pushdowns1.subsumes(pushdowns2)
            assert pushdowns2.subsumes(pushdowns1)


        check(
            pl.lit(True),
            pl.lit(True),
            schema={},
            check=both_subsumes,
        )

        check(
            pl.lit(True),
            pl.col('a'),
            schema={'a': pl.Int32},
            check=strict_subsumes,
        )

        check(
            pl.lit(True),
            pl.lit(False),
            schema={},
            check=strict_subsumes,
        )

        check(
            pl.col('a') <= 1,
            pl.col('a') == 1,
            schema={'a': pl.Int32},
            check=strict_subsumes,
        )

        check(
            pl.col('a') >= 1,
            pl.col('a') == 1,
            schema={'a': pl.Int32},
            check=strict_subsumes,
        )

        check(
            (pl.col('a') == 1) | (pl.col('b') == 1),
            pl.col('b') == 1,
            schema={'a': pl.Int32, 'b': pl.Int32},
            check=strict_subsumes,
        )

        check(
            pl.col('a') >= 1,
            pl.col('a') >= 1,
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            pl.col('a') > 1,
            pl.col('a') > 1,
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            pl.col('a') >= 1,
            pl.col('a') < 1,
            schema={'a': pl.Int32},
            check=neither_subsumes,
        )

        check(
            pl.col('a') > 1,
            pl.col('a') <= 1,
            schema={'a': pl.Int32},
            check=neither_subsumes,
        )

        check(
            pl.col('a'),
            pl.col('b'),
            schema={'a': pl.Int32, 'b': pl.Int32},
            check=neither_subsumes,
        )

        # De Morgan's laws
        check(
            ((pl.col('a') != 1) | (pl.col('b') != 1)),
            ~((pl.col('a') == 1) & (pl.col('b') == 1)),
            schema={'a': pl.Int32, 'b': pl.Int32},
            check=both_subsumes,
        )

        # De Morgan's laws and "|" overload as boolean OR for Boolean columns
        check(
            (~pl.col('a') | ~pl.col('b')),
            ~(pl.col('a') & pl.col('b')),
            schema={'a': pl.Boolean, 'b': pl.Boolean},
            check=both_subsumes,
        )

        check(
            ~(pl.col('a') == 1),
            pl.col('a') != 1,
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            ~pl.col('a').is_null(),
            pl.col('a').is_not_null(),
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            ~~~~(pl.col('a') == 1),
            pl.col('a') == 1,
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            ~~~(pl.col('a') == 1),
            ~(pl.col('a') == 1),
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            ~~~(pl.col('a') == 1),
            (pl.col('a') != 1),
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            (pl.col('a') == 1),
            (1 == pl.col('a')),
            schema={'a': pl.Int32},
            check=both_subsumes,
        )

        check(
            (pl.col('a') ^ pl.col('b')),
            (pl.col('a') ^ pl.col('b')),
            schema={'a': pl.Boolean, 'b': pl.Boolean},
            check=both_subsumes,
        )

        check(
            (pl.col('a') & pl.col('a')),
            pl.col('a'),
            schema={'a': pl.Boolean},
            check=both_subsumes,
        )

        check(
            pl.lit(True) | pl.lit(False),
            pl.lit(True),
            schema={},
            check=both_subsumes,
        )
        check(
            pl.lit(False) | pl.lit(True),
            pl.lit(True),
            schema={},
            check=both_subsumes,
        )

        check(
            pl.lit(True) | pl.col('a'),
            pl.lit(True),
            schema={'a': pl.Boolean},
            check=both_subsumes,
        )
        check(
            pl.col('a') | pl.lit(True),
            pl.lit(True),
            schema={'a': pl.Boolean},
            check=both_subsumes,
        )

        check(
            pl.lit(True) & pl.lit(False),
            pl.lit(False),
            schema={},
            check=both_subsumes,
        )
        check(
            pl.lit(False) & pl.lit(True),
            pl.lit(False),
            schema={},
            check=both_subsumes,
        )

        check(
            pl.lit(False) & pl.col('a'),
            pl.lit(False),
            schema={'a': pl.Boolean},
            check=both_subsumes,
        )
        check(
            pl.col('a') & pl.lit(False),
            pl.lit(False),
            schema={'a': pl.Boolean},
            check=both_subsumes,
        )

        check(
            ~pl.lit(None),
            pl.lit(None),
            schema={},
            check=both_subsumes,
        )

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
