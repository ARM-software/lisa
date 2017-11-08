
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
shutils-15864 [000] 27361.777314: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=999000 cpu_id=0
shutils-15864 [000] 27361.777428: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=999000 cpu_id=1
shutils-15864 [000] 27361.777523: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=999000 cpu_id=2
shutils-15864 [000] 27361.777616: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=999000 cpu_id=3
shutils-15864 [000] 27361.777709: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=2362000 cpu_id=4
shutils-15864 [000] 27361.777803: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=2362000 cpu_id=5
shutils-15864 [000] 27361.777896: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=2362000 cpu_id=6
shutils-15864 [000] 27361.777989: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=2362000 cpu_id=7
    sh-15888 [007] 27361.904411: tracing_mark_write: 0xffffff800819fac8s: TRACE_MARKER_START
    sh-15895 [004] 27362.971605: tracing_mark_write: 0xffffff800819fac8s: TRACE_MARKER_STOP
shutils-15900 [007] 27363.018426: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1844000 cpu_id=0
shutils-15900 [007] 27363.018474: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1844000 cpu_id=1
shutils-15900 [007] 27363.018506: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1844000 cpu_id=2
shutils-15900 [007] 27363.018536: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1844000 cpu_id=3
shutils-15900 [007] 27363.018566: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1805000 cpu_id=4
shutils-15900 [007] 27363.018594: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1805000 cpu_id=5
shutils-15900 [007] 27363.018623: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1805000 cpu_id=6
shutils-15900 [007] 27363.018651: tracing_mark_write: 0xffffff800819fac8s: cpu_frequency_devlib:        state=1805000 cpu_id=7
"""

class TestFallback(SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestFallback, self).__init__([], *args, **kwargs)

    def test_tracing_mark_write(self):
        with open("trace.txt", "w") as fout:
            fout.write(TEST_DATA)

        trace = trappy.FTrace(events=['cpu_frequency_devlib', 'tracing_mark_write'])

        self.assertEqual(len(trace.cpu_frequency_devlib.data_frame), 16)
        self.assertEqual(len(trace.tracing_mark_write.data_frame), 2)

    def test_print(self):
        with open("trace.txt", "w") as fout:
            data = TEST_DATA.replace('tracing_mark_write', 'print')
            fout.write(data)

        trace = trappy.FTrace(events=['cpu_frequency_devlib', 'print'])

        self.assertEqual(len(trace.cpu_frequency_devlib.data_frame), 16)
        self.assertEqual(len(trace.print_.data_frame), 2)
