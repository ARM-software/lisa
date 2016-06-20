#    Copyright 2016 ARM Limited
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

import utils_tests

import trappy

class TestSystrace(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestSystrace, self).__init__(
             [("trace_systrace.html", "trace.html")],
             *args,
             **kwargs)

    def test_systrace_html(self):
        """Tests parsing of a systrace embedded textual trace """

        events = ["sched_switch", "sched_wakeup", "trace_event_clock_sync"]
        trace = trappy.SysTrace("trace.html", events=events)

        self.assertTrue(hasattr(trace, "sched_switch"))
        self.assertEquals(len(trace.sched_switch.data_frame), 4)
        self.assertTrue("prev_comm" in trace.sched_switch.data_frame.columns)

        self.assertTrue(hasattr(trace, "sched_wakeup"))
        self.assertEquals(len(trace.sched_wakeup.data_frame), 4)
        self.assertTrue("target_cpu" in trace.sched_wakeup.data_frame.columns)

        self.assertTrue(hasattr(trace, "trace_event_clock_sync"))
        self.assertEquals(len(trace.trace_event_clock_sync.data_frame), 1)
        self.assertTrue("realtime_ts" in trace.trace_event_clock_sync.data_frame.columns)

    def test_cpu_counting(self):
        """SysTrace traces know the number of cpus"""

        trace = trappy.SysTrace("trace.html")

        self.assertTrue(hasattr(trace, "_cpus"))
        self.assertEquals(trace._cpus, 3)


class TestLegacySystrace(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestLegacySystrace, self).__init__(
             [("trace_legacy_systrace.html", "trace.html")],
             *args,
             **kwargs)

    def test_systrace_html(self):
        """Tests parsing of a legacy systrace embedded textual trace """

        events = ["sched_switch", "sched_wakeup", "sched_contrib_scale_f"]
        trace = trappy.SysTrace("trace.html", events=events)

        self.assertTrue(hasattr(trace, "sched_switch"))
        self.assertEquals(len(trace.sched_switch.data_frame), 3)
        self.assertTrue("prev_comm" in trace.sched_switch.data_frame.columns)

        self.assertTrue(hasattr(trace, "sched_wakeup"))
        self.assertEquals(len(trace.sched_wakeup.data_frame), 2)
        self.assertTrue("target_cpu" in trace.sched_wakeup.data_frame.columns)

        self.assertTrue(hasattr(trace, "sched_contrib_scale_f"))
        self.assertEquals(len(trace.sched_contrib_scale_f.data_frame), 2)
        self.assertTrue("freq_scale_factor" in trace.sched_contrib_scale_f.data_frame.columns)

    def test_cpu_counting(self):
        """In a legacy SysTrace trace, trappy gets the number of cpus"""

        trace = trappy.SysTrace("trace.html")

        self.assertTrue(hasattr(trace, "_cpus"))
        self.assertEquals(trace._cpus, 8)
