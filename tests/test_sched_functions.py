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

    def test_get_pids_for_process_funny_process_names(self):
        """get_pids_for_process() works when a process name is a substring of another"""
        from bart.sched.functions import get_pids_for_process

        trace_file = "trace.txt"
        raw_trace_file = "trace.raw.txt"
        in_data = """          <idle>-0     [001] 10826.894644: sched_switch:          prev_comm=swapper/1 prev_pid=0 prev_prio=120 prev_state=0 next_comm=rt-app next_pid=3268 next_prio=120
            wmig-3268  [001] 10826.894778: sched_switch:          prev_comm=wmig prev_pid=3268 prev_prio=120 prev_state=1 next_comm=rt-app next_pid=3269 next_prio=120
           wmig1-3269  [001] 10826.905152: sched_switch:          prev_comm=wmig1 prev_pid=3269 prev_prio=120 prev_state=1 next_comm=wmig next_pid=3268 next_prio=120
            wmig-3268  [001] 10826.915384: sched_switch:          prev_comm=wmig prev_pid=3268 prev_prio=120 prev_state=1 next_comm=swapper/1 next_pid=0 next_prio=120
          <idle>-0     [005] 10826.995169: sched_switch:          prev_comm=swapper/5 prev_pid=0 prev_prio=120 prev_state=0 next_comm=wmig1 next_pid=3269 next_prio=120
           wmig1-3269  [005] 10827.007064: sched_switch:          prev_comm=wmig1 prev_pid=3269 prev_prio=120 prev_state=0 next_comm=wmig next_pid=3268 next_prio=120
            wmig-3268  [005] 10827.019061: sched_switch:          prev_comm=wmig prev_pid=3268 prev_prio=120 prev_state=0 next_comm=wmig1 next_pid=3269 next_prio=120
           wmig1-3269  [005] 10827.031061: sched_switch:          prev_comm=wmig1 prev_pid=3269 prev_prio=120 prev_state=0 next_comm=wmig next_pid=3268 next_prio=120
            wmig-3268  [005] 10827.050645: sched_switch:          prev_comm=wmig prev_pid=3268 prev_prio=120 prev_state=1 next_comm=swapper/5 next_pid=0 next_prio=120
"""

        # We create an empty trace.txt to please trappy ...
        with open(trace_file, "w") as fout:
            fout.write("")

        # ... but we only put the sched_switch events in the raw trace
        # file because that's where trappy is going to look for
        with open(raw_trace_file, "w") as fout:
            fout.write(in_data)

        trace = trappy.FTrace(trace_file)

        self.assertEquals(get_pids_for_process(trace, "wmig"), [3268])
