
#    Copyright 2018 ARM Limited, Google and contributors
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
from utils_tests import SetupDirectory

TEST_DATA = """
    bash-7842  [002]   354.414778: print:                tracing_mark_write: TRACE_MARKER_START
 shutils-7843  [001]   354.590131: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=0
 shutils-7843  [001]   354.590253: print:                tracing_mark_write: cpu_frequency_devlib:        state=1200000 cpu_id=1
 shutils-7843  [001]   354.590369: print:                tracing_mark_write: cpu_frequency_devlib:        state=1200000 cpu_id=2
 shutils-7843  [001]   354.590477: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=3
 shutils-7843  [001]   354.590584: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=4
 shutils-7843  [001]   354.590691: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=5
 shutils-7868  [001]   357.732485: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=0
 shutils-7868  [001]   357.732607: print:                tracing_mark_write: cpu_frequency_devlib:        state=1200000 cpu_id=1
 shutils-7868  [001]   357.732726: print:                tracing_mark_write: cpu_frequency_devlib:        state=1200000 cpu_id=2
 shutils-7868  [001]   357.732833: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=3
 shutils-7868  [001]   357.732939: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=4
 shutils-7868  [001]   357.733077: print:                tracing_mark_write: cpu_frequency_devlib:        state=950000 cpu_id=5
    bash-7870  [002]   357.892659: print:                tracing_mark_write: TRACE_MARKER_STOP
"""

class TestFallback(SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestFallback, self).__init__([], *args, **kwargs)

    def test_tracing_mark_write(self):
        with open("trace.txt", "w") as fout:
            fout.write(TEST_DATA)

        trace = trappy.FTrace(events=['cpu_frequency_devlib', 'tracing_mark_write'])

        self.assertEqual(len(trace.cpu_frequency_devlib.data_frame), 12)
        self.assertEqual(len(trace.tracing_mark_write.data_frame), 2)

