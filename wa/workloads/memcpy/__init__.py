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

from wa import Workload, Parameter, Executable
from wa.utils.exec_control import once
from wa.utils.types import cpu_mask


THIS_DIR = os.path.dirname(__file__)


RESULT_REGEX = re.compile('Total time: ([\d.]+) s.*Bandwidth: ([\d.]+) MB/s', re.S)


class Memcpy(Workload):

    name = 'memcpy'
    description = """
    Runs memcpy in a loop.

    This will run memcpy in a loop for a specified number of times on a buffer
    of a specified size. Additionally, the affinity of the test can be set to
    one or more specific cores.

    This workload is single-threaded. It generates no scores or metrics by
    itself.

    """

    parameters = [
        Parameter('buffer_size', kind=int, default=1024 * 1024 * 5,
                  description='''
                  Specifies the size, in bytes, of the buffer to be copied.
                  '''),
        Parameter('loops', kind=int, default=1000, aliases=['iterations'],
                  description='''
                  Specfies the number of iterations that will be performed.
                  '''),
        Parameter('cpus', kind=cpu_mask, default=0,
                  description='''
                  The cpus for which the affinity of the test
                  process should be set, specified as a mask, as a list of
                  cpus or a sysfs-style string. If not specified, all available
                  cores will be used.
                  '''),
    ]

    @once
    def initialize(self, context):
        self.binary_name = 'memcpy'
        resource = Executable(self, self.target.abi, self.binary_name)
        host_binary = context.get_resource(resource)
        Memcpy.target_exe = self.target.install_if_needed(host_binary)

    def setup(self, context):
        self.command = '{} -i {} -s {}'.format(Memcpy.target_exe, self.loops, self.buffer_size)
        for c in self.cpus.list():
            self.command += ' -c {}'.format(c)
        self.result = None

    def run(self, context):
        self.result = self.target.execute(self.command, timeout=300)

    def extract_results(self, context):
        if self.result:
            match = RESULT_REGEX.search(self.result)
            context.add_metric('time', float(match.group(1)), 'seconds', lower_is_better=True)
            context.add_metric('bandwidth', float(match.group(2)), 'MB/s')

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall('memcpy')
