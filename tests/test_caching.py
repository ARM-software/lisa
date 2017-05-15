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
