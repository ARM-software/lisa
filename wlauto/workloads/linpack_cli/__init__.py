#    Copyright 2015 ARM Limited
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
# pylint: disable=attribute-defined-outside-init
import os

from wlauto import Workload, Parameter, Executable


class LinpackCliWorkload(Workload):

    name = 'linpack-cli'
    description = """
    linpack benchmark with a command line interface

    Benchmarks FLOPS (floating point operations per second).

    This is the oldschool version of the bencmark. Source may be viewed here:

        http://www.netlib.org/benchmark/linpackc.new

    """

    parameters = [
        Parameter('array_size', kind=int, default=200,
                  description='size of arrays to be used by the benchmark.'),
    ]

    binary = None  # set during initialization

    def initialize(self, context):
        host_exe = context.resolver.get(Executable(self, self.device.abi, 'linpack'))
        LinpackCliWorkload.binary = self.device.install(host_exe)

    def setup(self, context):
        self.command = '(echo {}; echo q) | {}'.format(self.array_size, self.binary)

    def run(self, context):
        self.raw_output = self.device.execute(self.command,
                                              timeout=(self.array_size / 10) ** 2,
                                              check_exit_code=False)

    def update_result(self, context):
        raw_outfile = os.path.join(context.output_directory, 'linpack-raw.txt')
        with open(raw_outfile, 'w') as wfh:
            wfh.write(self.raw_output)
        context.add_artifact('linpack-raw', raw_outfile, kind='raw')

        marker = '--------------------'
        lines = iter(self.raw_output.split('\n'))
        for line in lines:
            if marker in line:
                break

        for line in lines:
            line = line.strip()
            if not line:
                break
            parts = line.split()
            classifiers = {'reps': int(parts[0])}
            context.add_metric('time', float(parts[1]), 'seconds',
                               lower_is_better=True, classifiers=classifiers)
            context.add_metric('KFLOPS', float(parts[5]), 'KFLOPS',
                               lower_is_better=True, classifiers=classifiers)

    def finalize(self, context):
        self.device.uninstall(self.binary)
