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
import shutil
import sys
import unittest
import utils_tests
import trappy
from trappy.ftrace import GenericFTrace

class TestCaching(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestCaching, self).__init__(
            [("trace_sched.txt", "trace.txt"),
             ("trace_sched.txt", "trace.raw.txt")],
            *args,
            **kwargs)

    def test_cache_created(self):
        """Test cache creation when enabled"""
        GenericFTrace.disable_cache = False
        trace = trappy.FTrace()

        trace_path = os.path.abspath(trace.trace_path)
        trace_dir = os.path.dirname(trace_path)
        trace_file = os.path.basename(trace_path)
        cache_dir = '.' + trace_file + '.cache'

        self.assertTrue(cache_dir in os.listdir(trace_dir))

    def test_cache_not_created(self):
        """Test that cache should not be created when disabled """
        GenericFTrace.disable_cache = True
        trace = trappy.FTrace()

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

        # Test whether timestamps are the same:
        # The cached/uncached versions of the timestamps are slightly
        # different due to floating point precision errors due to converting
        # back and forth CSV and DataFrame. For all purposes this is not relevant
        # since such rounding doesn't effect the end result.
        # Here's an example of the error, the actual normalized time when
        # calculated by hand is 0.081489, however following is what's stored
        # in the CSV for sched_wakeup events in this trace.
        # When converting the index to strings (and also what's in the CSV)
        # cached: ['0.0814890000001', '1.981491']
        # uncached: ['0.0814890000001', '1.981491']
        #
        # Keeping index as numpy.float64
        # cached: [0.081489000000100009, 1.9814909999999999]
        # uncached: [0.081489000000146916, 1.9814909999995507]
        #
        # To make it possible to test, lets just convert the timestamps to strings
        # and compare them below.

        cached_times = [str(r[0]) for r in cached_dfr.iterrows()]
        uncached_times = [str(r[0]) for r in uncached_dfr.iterrows()]

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

        md5_path = os.path.join(cache_path, "md5sum")
        def read_md5sum():
            with open(md5_path) as f:
                return f.read()

        # Change 1 character of the stored checksum
        md5sum = read_md5sum()
        # Sorry, I guess modifying strings in Python is kind of awkward?
        md5sum_inc = "".join(list(md5sum[:-1]) + [chr(ord(md5sum[-1]) + 1)])
        with open(md5_path, "w") as f:
            f.write(md5sum_inc)

        # Parse a trace, this should delete and overwrite the invalidated cache
        GenericFTrace.disable_cache = False
        trace = trappy.FTrace()

        # Check that the modified md5sum was overwritten
        self.assertNotEqual(read_md5sum(), md5sum_inc,
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

    def test_cache_window(self):
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
