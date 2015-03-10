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

# pylint: disable=E1101,W0201

import os
import re

from wlauto import Workload, Parameter


THIS_DIR = os.path.dirname(__file__)


RESULT_REGEX = re.compile('Total time: ([\d.]+) s.*Bandwidth: ([\d.]+) MB/s', re.S)


class MemcpyTest(Workload):

    name = 'memcpy'
    description = """
    Runs memcpy in a loop.

    This will run memcpy in a loop for a specified number of times on a buffer
    of a specified size. Additionally, the affinity of the test can be set to one
    or more specific cores.

    This workload is single-threaded. It genrates no scores or metrics by itself.

    """

    parameters = [
        Parameter('buffer_size', kind=int, default=1024 * 1024 * 5,
                  description='Specifies the size, in bytes, of the buffer to be copied.'),
        Parameter('iterations', kind=int, default=1000,
                  description='Specfies the number of iterations that will be performed.'),
        Parameter('cpus', kind=list, default=[],
                  description="""A list of integers specifying ordinals of cores to which the affinity
                                 of the test process should be set. If not specified, all avaiable cores
                                 will be used.
                  """),
    ]

    def setup(self, context):
        self.host_binary = os.path.join(THIS_DIR, 'memcpy')
        if not self.device.is_installed('memcpy'):
            self.device_binary = self.device.install(self.host_binary)
        else:
            self.device_binary = 'memcpy'
        self.command = '{} -i {} -s {}'.format(self.device_binary, self.iterations, self.buffer_size)
        if self.cpus:
            for c in self.cpus:
                self.command += ' -c {}'.format(c)

    def run(self, context):
        self.result = self.device.execute(self.command, timeout=300)

    def update_result(self, context):
        match = RESULT_REGEX.search(self.result)
        context.result.add_metric('time', float(match.group(1)), 'seconds', lower_is_better=True)
        context.result.add_metric('bandwidth', float(match.group(2)), 'MB/s')

    def teardown(self, context):
        pass
