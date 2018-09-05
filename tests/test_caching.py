#    Copyright 2015-2017 ARM Limited, Google and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import json
import shutil
import sys
import unittest
import utils_tests
import trappy
from trappy.ftrace import GenericFTrace
from trappy.systrace import SysTrace

class TestCaching(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestCaching, self).__init__(
            [("trace_sched.txt", "trace.txt"),
             ("trace_sched.txt", "trace.raw.txt"),
             ("trace_systrace.html", "trace.html")],
            *args,
            **kwargs)

    def test_cache_created(self):
        """Test cache creation when enabled"""
        GenericFTrace.disable_cache = False
        traces = (trappy.FTrace(), trappy.SysTrace(path='./trace.html'))

        for trace in traces:
            trace_path = os.path.abspath(trace.trace_path)
            trace_dir = os.path.dirname(trace_path)
            trace_file = os.path.basename(trace_path)
            cache_dir = '.' + trace_file + '.cache'

            self.assertTrue(cache_dir in os.listdir(trace_dir))

    def test_cache_not_created(self):
        """Test that cache should not be created when disabled """
        GenericFTrace.disable_cache = True
        traces = (trappy.FTrace(), trappy.SysTrace(path='./trace.html'))

        for trace in traces:
            trace_path = os.path.abspath(trace.trace_path)
            trace_dir = os.path.dirname(trace_path)
            trace_file = os.path.basename(trace_path)
            cache_dir = '.' + trace_file + '.cache'

            self.assertFalse(cache_dir in os.listdir(trace_dir))

    def test_compare_cached_vs_uncached(self):
        """ Test that the cached and uncached traces are same """
        # Build the cache, but the actual trace will be parsed
        # fresh since this is a first time parse
        GenericFTrace.disable_cache = False
        uncached_trace = trappy.FTrace()
        uncached_dfr = uncached_trace.sched_wakeup.data_frame

        # Now read from previously parsed cache by reusing the path
        cached_trace = trappy.FTrace(uncached_trace.trace_path)
        cached_dfr = cached_trace.sched_wakeup.data_frame

        # By default, the str to float conversion done when reading from csv is
        # different from the one used when reading from the trace.txt file.
        #
        # Here's an example:
        # - trace.txt string timestamps:
        #   [76.402065, 80.402065, 80.001337]
        # - parsed dataframe timestamps:
        #   [76.402065000000007, 80.402065000000007, 82.001337000000007]
        #
        # - csv string timestamps:
        #   [76.402065, 80.402065, 80.001337]
        # - cached dataframe timestamps:
        #   [76.402064999999993, 80.402064999999993, 82.001337000000007]
        #
        # To fix this, the timestamps read from the cache are converted using
        # the same conversion method as the trace.txt parser, which results in
        # cache-read timestamps being identical to trace-read timestamps.
        #
        # This test ensures that this stays true.

        cached_times = [r[0] for r in cached_dfr.iterrows()]
        uncached_times = [r[0] for r in uncached_dfr.iterrows()]

        self.assertTrue(cached_times == uncached_times)

        # compare other columns as well
        self.assertTrue([r[1].pid for r in cached_dfr.iterrows()] ==
                        [r[1].pid for r in uncached_dfr.iterrows()])

        self.assertTrue([r[1].comm for r in cached_dfr.iterrows()] ==
                        [r[1].comm for r in uncached_dfr.iterrows()])

        self.assertTrue([r[1].prio for r in cached_dfr.iterrows()] ==
                        [r[1].prio for r in uncached_dfr.iterrows()])

    def test_invalid_cache_overwritten(self):
        """Test a cache with a bad checksum is overwritten"""
        # This is a directory so we can't use the files_to_copy arg of
        # SetUpDirectory, just do it ourselves.
        cache_path = ".trace.txt.cache"
        src = os.path.join(utils_tests.TESTS_DIRECTORY, "trace_sched.txt.cache")
        shutil.copytree(src, cache_path)

        metadata_path = os.path.join(cache_path, "metadata.json")

        def read_metadata():
            with open(metadata_path, "r") as f:
                return json.load(f)

        def write_md5(md5):
            metadata = read_metadata()
            metadata["md5sum"] = md5
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)


        # Change 1 character of the stored checksum
        md5sum = read_metadata()["md5sum"]
        md5sum_inc = md5sum[:-1] + chr(ord(md5sum[-1]) + 1)
        write_md5(md5sum_inc)

        # Parse a trace, this should delete and overwrite the invalidated cache
        GenericFTrace.disable_cache = False
        trace = trappy.FTrace()

        # Check that the modified md5sum was overwritten
        self.assertNotEqual(read_metadata()["md5sum"], md5sum_inc,
                            "The invalid ftrace cache wasn't overwritten")

    def test_cache_dynamic_events(self):
        """Test that caching works if new event parsers have been registered"""

        # Parse the trace to create a cache
        GenericFTrace.disable_cache = False
        trace1 = trappy.FTrace()

        # Check we're actually testing what we think we are
        if hasattr(trace1, 'dynamic_event'):
            raise RuntimeError('Test bug: found unexpected event in trace')

        # Now register a new event type, call the constructor again, and check
        # that the newly added event (which is not present in the cache) is
        # parsed.

        parse_class = trappy.register_dynamic_ftrace("DynamicEvent", "dynamic_test_key")

        trace2 = trappy.FTrace()
        self.assertTrue(len(trace2.dynamic_event.data_frame) == 1)

        trappy.unregister_dynamic_ftrace(parse_class)

    def test_cache_normalize_time(self):
        """Test that caching doesn't break normalize_time"""
        GenericFTrace.disable_cache = False

        # Times in trace_sched.txt
        start_time = 6550.018511
        first_freq_event_time = 6550.056870

        # Parse without normalizing time
        trace1 = trappy.FTrace(events=['cpu_frequency', 'sched_wakeup'],
                               normalize_time=False)

        self.assertEqual(trace1.cpu_frequency.data_frame.index[0],
                         first_freq_event_time)

        # Parse with normalized time
        trace2 = trappy.FTrace(events=['cpu_frequency', 'sched_wakeup'],
                               normalize_time=True)

        self.assertEqual(trace2.cpu_frequency.data_frame.index[0],
                         first_freq_event_time - start_time)

    def test_cache_window_broad(self):
        """Test that caching doesn't break the 'window' parameter"""
        GenericFTrace.disable_cache = False

        trace1 = trappy.FTrace(
            events=['sched_wakeup'],
            window=(0, 1))

        # Check that we're testing what we think we're testing The trace
        # contains 2 sched_wakeup events; this window should get rid of one of
        # them.
        if len(trace1.sched_wakeup.data_frame) != 1:
            raise RuntimeError('Test bug: bad sched_wakeup event count')

        # Parse again without the window
        trace1 = trappy.FTrace(
            events=['sched_wakeup'],
            window=(0, None))

        self.assertEqual(len(trace1.sched_wakeup.data_frame), 2)

    def test_cache_window_narrow(self):
        """
        Test that applying a window to a cached trace returns EXACTLY what is expected
        """
        # As described in test_compare_cache_vs_uncached, reading from cache
        # results in slightly different timestamps
        #
        # This test verifies that applying windows results in identical
        # dataframes whether cache is used or not.
        GenericFTrace.disable_cache = False

        uncached_trace = trappy.FTrace()

        trace = trappy.FTrace(uncached_trace.trace_path,
                              normalize_time=False,
                              abs_window=(6550.100000, 6552.000002))

        self.assertAlmostEquals(trace.get_duration(), 1.900002)

        self.assertEquals(len(trace.sched_wakeup.data_frame), 2)
        self.assertEquals(len(trace.sched_wakeup_new.data_frame), 1)

    def test_ftrace_metadata(self):
        """Test that caching keeps trace metadata"""
        GenericFTrace.disable_cache = False

        self.test_cache_created()

        trace = trappy.FTrace()

        version = int(trace._version)
        cpus = int(trace._cpus)

        self.assertEquals(version, 6)
        self.assertEquals(cpus, 6)

    def test_cache_delete_single(self):
        GenericFTrace.disable_cache = False
        trace = trappy.FTrace()

        trace_path = os.path.abspath(trace.trace_path)
        trace_dir = os.path.dirname(trace_path)
        trace_file = os.path.basename(trace_path)
        cache_dir = '.' + trace_file + '.cache'
        number_of_trace_categories = 31
        self.assertEquals(len(os.listdir(cache_dir)), number_of_trace_categories)

        os.remove(os.path.join(cache_dir, 'SchedWakeup.csv'))
        self.assertEquals(len(os.listdir(cache_dir)), number_of_trace_categories - 1)

        # Generate trace again, should regenerate only the missing item
        trace = trappy.FTrace()
        self.assertEquals(len(os.listdir(cache_dir)), number_of_trace_categories)
        for c in trace.trace_classes:
            if isinstance(c, trace.class_definitions['sched_wakeup']):
                self.assertEquals(c.cached, False)
                continue
            self.assertEquals(c.cached, True)
