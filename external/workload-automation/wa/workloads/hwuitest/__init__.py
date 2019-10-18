#    Copyright 2013-2018 ARM Limited
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

# pylint: disable=E1101,W0201

import os
import re
from collections import defaultdict

from wa import Workload, Parameter, Executable
from wa.utils.exec_control import once
from wa.utils.types import caseless_string


BINARY = "hwuitest"
IGNORED_METRICS = ["Stats since", "Total frames rendered"]


class HWUITest(Workload):

    name = 'hwuitest'
    description = """
    Tests UI rendering latency on Android devices.

    The binary for this workload is built as part of AOSP's
    frameworks/base/libs/hwui component.
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

    def __init__(self, target, *args, **kwargs):
        super(HWUITest, self).__init__(target, *args, **kwargs)
        HWUITest.target_exe = None

    @once
    def initialize(self, context):
        host_exe = context.get_resource(Executable(self,
                                                   self.target.abi,
                                                   BINARY))
        HWUITest.target_exe = self.target.install(host_exe)

    def run(self, context):
        self.output = None
        self.output = self.target.execute("{} {} {} {}".format(self.target_exe,
                                                               self.test.lower(),
                                                               self.loops,
                                                               self.frames))

    def extract_results(self, context):
        if not self.output:
            return
        outfile = os.path.join(context.output_directory, 'hwuitest.output')
        with open(outfile, 'w') as wfh:
            wfh.write(self.output)
        context.add_artifact('hwuitest', outfile, kind='raw')

    def update_output(self, context):
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
                context.add_metric(metric,
                                   match['value'],
                                   None,
                                   classifiers={"loop": count,
                                                "frames": self.frames})
                context.add_metric(metric + "_pct",
                                   match['percent'],
                                   "%",
                                   classifiers={"loop": count,
                                                "frames": self.frames})
            else:
                match = normal.match(value_string).groupdict()
                context.add_metric(metric,
                                   match['value'],
                                   match['unit'],
                                   classifiers={"loop": count,
                                                "frames": self.frames})

    @once
    def finalize(self, context):
        if self.target_exe and self.uninstall:
            self.target.uninstall(self.target_exe)
