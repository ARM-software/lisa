#    Copyright 2016-2016 ARM Limited
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

import trappy

import utils_tests

class TestSchedFunctions(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestSchedFunctions, self).__init__([], *args, **kwargs)

    def test_get_pids_for_processes_no_sched_switch(self):
        """get_pids_for_processes() raises an exception if the trace doesn't have a sched_switch event"""
        from bart.sched.functions import get_pids_for_process

        trace_file = "trace.txt"
        raw_trace_file = "trace.raw.txt"

        with open(trace_file, "w") as fout:
            fout.write("")

        with open(raw_trace_file, "w") as fout:
            fout.write("")

        trace = trappy.FTrace(trace_file)
        with self.assertRaises(ValueError):
            get_pids_for_process(trace, "foo")
