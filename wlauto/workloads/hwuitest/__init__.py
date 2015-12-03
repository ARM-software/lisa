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

#pylint: disable=E1101,W0201

import os
import re
from collections import defaultdict

from wlauto import Workload, Parameter, Executable
from wlauto.utils.types import caseless_string


BINARY = "hwuitest"
IGNORED_METRICS = ["Stats since", "Total frames rendered"]


class HWUITest(Workload):

    name = 'hwuitest'
    description = """
    Tests UI rendering latency on android devices
    """
    supported_platforms = ['android']

    parameters = [
        Parameter('test', kind=caseless_string, default="shadowgrid",
                  allowed_values=["shadowgrid", "rectgrid", "oval"],
                  description="""
                  The test to run:
                     - ``'shadowgrid'``: creates a grid of rounded rects that
                       cast shadows, high CPU & GPU load
                     - ``'rectgrid'``: creates a grid of 1x1 rects
                     - ``'oval'``: draws 1 oval
                  """),
        Parameter('loops', kind=int, default=3,
                  description="The number of test iterations."),
        Parameter('frames', kind=int, default=150,
                  description="The number of frames to run the test over."),
    ]

    def setup(self, context):
        host_exe = context.resolver.get(Executable(self,
                                                   self.device.abi,
                                                   BINARY))
        self.device.install(host_exe)

    def run(self, context):
        self.output = self.device.execute("{} {} {} {}".format(BINARY,
                                                               self.test.lower(),
                                                               self.loops,
                                                               self.frames))

    def update_result(self, context):
        outfile = os.path.join(context.output_directory, 'hwuitest.output')
        with open(outfile, 'w') as wfh:
            wfh.write(self.output)
        context.add_artifact('hwuitest', outfile, kind='raw')

        normal = re.compile(r'(?P<value>\d*)(?P<unit>\w*)')
        with_pct = re.compile(r'(?P<value>\d*) \((?P<percent>.*)%\)')
        count = 0
        for line in self.output.splitlines():
            #Filters out "Success!" and blank lines
            try:
                metric, value_string = [p.strip() for p in line.split(':', 1)]
            except ValueError:
                continue

            # Filters out unwanted lines
            if metric in IGNORED_METRICS:
                continue

            if metric == "Janky frames":
                count += 1
                match = with_pct.match(value_string).groupdict()
                context.result.add_metric(metric,
                                          match['value'],
                                          None,
                                          classifiers={"loop": count,
                                                       "frames": self.frames})
                context.result.add_metric(metric + "_pct",
                                          match['value'],
                                          "%",
                                          classifiers={"loop": count,
                                                       "frames": self.frames})
            else:
                match = normal.match(value_string).groupdict()
                context.result.add_metric(metric,
                                          match['value'],
                                          match['unit'],
                                          classifiers={"loop": count,
                                                       "frames": self.frames})

    def teardown(self, context):
        self.device.uninstall_executable(BINARY)
