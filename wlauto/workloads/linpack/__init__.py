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

# pylint: disable=E1101,E0203

import os
import re

from wlauto import AndroidUiAutoBenchmark, Parameter


class Linpack(AndroidUiAutoBenchmark):

    name = 'linpack'
    description = """
    The LINPACK Benchmarks are a measure of a system's floating point computing
    power.

    http://en.wikipedia.org/wiki/LINPACK_benchmarks

    From the article:

    Introduced by Jack Dongarra, they measure how fast a computer solves
    a dense n by n system of linear equations Ax = b, which is a common task in
    engineering.

    """
    package = 'com.greenecomputing.linpackpro'
    activity = '.Linpack'
    summary_metrics = ['Linpack ST', 'Linpack MT']
    regex = re.compile(r'LINPACK RESULT: (?P<type>\w+) (?P<value>\S+)')

    parameters = [
        Parameter('output_file', default=None,
                  description='On-device output file path.'),
    ]

    def __init__(self, device, **kwargs):
        super(Linpack, self).__init__(device, **kwargs)
        if self.output_file is None:
            self.output_file = os.path.join(self.device.working_directory, 'linpack.txt')
        self.uiauto_params['output_file'] = self.output_file

    def update_result(self, context):
        super(Linpack, self).update_result(context)
        with open(self.logcat_log) as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    metric = 'Linpack ' + match.group('type')
                    value = float(match.group('value'))
                    context.result.add_metric(metric, value, 'MFLOPS')
