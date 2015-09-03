#    Copyright 2013-2015 ARM Limited
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
import re
import time

from wlauto import AndroidBenchmark, Parameter


class Nenamark(AndroidBenchmark):

    name = 'nenamark'
    description = """
    NenaMark is an OpenGL-ES 2.0 graphics performance benchmark for Android
    devices.

    http://nena.se/nenamark_story

    From the website:

    The NenaMark2 benchmark scene averages about 45k triangles, with a span
    between 26k and 68k triangles. It averages 96 batches per frame and contains
    about 15 Mb of texture data (non-packed).
    """
    package = 'se.nena.nenamark2'
    activity = 'se.nena.nenamark2.NenaMark2'

    parameters = [
        Parameter('duration', kind=int, default=120,
                  description="""
                  Number of seconds to wait before considering the benchmark
                  finished
                  """),
    ]

    regex = re.compile('.*NenaMark2.*Score.*?([0-9\.]*)fps')

    def run(self, context):
        time.sleep(5)  # wait for nenamark menu to show up
        self.device.execute('input keyevent 23')
        time.sleep(self.duration)

    def update_result(self, context):
        super(Nenamark, self).update_result(context)
        with open(self.logcat_log) as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    score = match.group(1)
                    context.result.add_metric('nenamark score', score)
                    break

