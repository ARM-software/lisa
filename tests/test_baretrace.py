#    Copyright 2015-2017 ARM Limited
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

import pandas as pd
import trappy
import unittest

class TestBareTrace(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBareTrace, self).__init__(*args, **kwargs)
        dfr0 = pd.DataFrame({"l1_misses": [24, 535,  41],
                             "l2_misses": [155, 11, 200],
                             "cpu":       [ 0,   1,   0]},
                            index=pd.Series([1.020, 1.342, 1.451], name="Time"))

        dfr1 = pd.DataFrame({"load": [ 35,  16,  21,  28],
                             "util": [279, 831, 554, 843]},
                            index=pd.Series([1.279, 1.718, 2.243, 2.465], name="Time"))

        self.dfr = [dfr0, dfr1]

    def test_bare_trace_accepts_name(self):
        """The BareTrace() accepts a name parameter"""

        trace = trappy.BareTrace(name="foo")

        self.assertEquals(trace.name, "foo")

    def test_bare_trace_can_add_parsed_event(self):
        """The BareTrace() class can add parsed events to its collection of trace events"""
        trace = trappy.BareTrace()
        trace.add_parsed_event("pmu_counters", self.dfr[0])

        self.assertEquals(len(trace.pmu_counters.data_frame), 3)
        self.assertEquals(trace.pmu_counters.data_frame["l1_misses"].iloc[0], 24)

        trace.add_parsed_event("pivoted_counters", self.dfr[0], pivot="cpu")
        self.assertEquals(trace.pivoted_counters.pivot, "cpu")

    def test_bare_trace_get_duration(self):
        """BareTrace.get_duration() works for a simple case"""

        trace = trappy.BareTrace()
        trace.add_parsed_event("pmu_counter", self.dfr[0])
        trace.add_parsed_event("load_event", self.dfr[1])

        self.assertEquals(trace.get_duration(), self.dfr[1].index[-1])

    def test_bare_trace_get_duration_normalized(self):
        """BareTrace.get_duration() works if the trace has been normalized"""

        trace = trappy.BareTrace()
        trace.add_parsed_event("pmu_counter", self.dfr[0].copy())
        trace.add_parsed_event("load_event", self.dfr[1].copy())

        basetime = self.dfr[0].index[0]
        trace.normalize_time(basetime)

        expected_duration = self.dfr[1].index[-1] - basetime
        self.assertEquals(trace.get_duration(), expected_duration)

    def test_bare_trace_normalize_time_accepts_basetime(self):
        """BareTrace().normalize_time() accepts an arbitrary basetime"""

        trace = trappy.BareTrace()
        trace.add_parsed_event("pmu_counter", self.dfr[0].copy())

        prev_first_time = trace.pmu_counter.data_frame.index[0]
        basetime = 3

        trace.normalize_time(basetime)

        self.assertEquals(trace.basetime, basetime)

        exp_first_time = prev_first_time - basetime
        self.assertEquals(round(trace.pmu_counter.data_frame.index[0] - exp_first_time, 7), 0)
