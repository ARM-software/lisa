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
#

#pylint: disable=E1101,W0201

import os

from wlauto import Workload, Parameter, Executable
from wlauto.utils.types import list_or_integer, list_or_string


class lmbench(Workload):

    name = 'lmbench'

    # Define supported tests. Each requires a _setup_{name} routine below
    test_names = ['lat_mem_rd', 'bw_mem']

    description = """
                  Run a subtest from lmbench, a suite of portable ANSI/C microbenchmarks for UNIX/POSIX.

                  In general, lmbench measures two key features: latency and bandwidth. This workload supports a subset
                  of lmbench tests. lat_mem_rd can be used to measure latencies to memory (including caches). bw_mem
                  can be used to measure bandwidth to/from memory over a range of operations.

                  Further details, and source code are available from:
                  http://sourceforge.net/projects/lmbench/.
                  See lmbench/bin/README for license details.
                  """

    parameters = [
        Parameter('test', default='lat_mem_rd', allowed_values=test_names,
                  description="Specifies an lmbench test to run."),
        Parameter('stride', kind=list_or_integer, default=[128],
                  description='Stride for lat_mem_rd test. Workload will iterate over one or more integer values.'),
        Parameter('thrash', kind=bool, default=True, description='Sets -t flag for lat_mem_rd_test'),
        Parameter('size', kind=list_or_string, default="4m",
                  description='Data set size for lat_mem_rd bw_mem tests.'),
        Parameter('mem_category', kind=list_or_string,
                  default=('rd', 'wr', 'cp', 'frd', 'fwr', 'fcp', 'bzero', 'bcopy'),
                  description='List of memory catetories for bw_mem test.'),
        Parameter('parallelism', kind=int, default=None, description='Parallelism flag for tests that accept it.'),
        Parameter('warmup', kind=int, default=None, description='Warmup flag for tests that accept it.'),
        Parameter('repetitions', kind=int, default=None, description='Repetitions flag for tests that accept it.'),
        Parameter('force_abi', kind=str, default=None,
                  description='Override device abi with this value. Can be used to force arm32 on 64-bit devices.'),
        Parameter('run_timeout', kind=int, default=900,
                  description="""
                  Timeout for execution of the test.
                  """),
        Parameter('times', kind=int, default=1, constraint=lambda x: x > 0,
                  description="""
                  Specifies the number of times the benchmark will be run in a "tight loop",
                  i.e. without performaing setup/teardown inbetween. This parameter is distinct from
                  "repetitions", as the latter takes place within the benchmark and produces a single result.
                  """),
    ]

    def setup(self, context):

        abi = self.device.abi
        if self.force_abi:
            abi = self.force_abi

        # self.test has been pre-validated, so this _should_ only fail if there's an abi mismatch
        host_exe = context.resolver.get(Executable(self, abi, self.test))
        self.device_exe = self.device.install(host_exe)
        self.commands = []

        setup_test = getattr(self, '_setup_{}'.format(self.test))
        setup_test()

    def run(self, context):
        self.output = []

        for time in xrange(self.times):
            for command in self.commands:
                self.output.append("Output for time #{}, {}: ".format(time + 1, command))
                self.output.append(self.device.execute(command, timeout=self.run_timeout, check_exit_code=False))

    def update_result(self, context):
        for output in self.output:
            self.logger.debug(output)
        outfile = os.path.join(context.output_directory, 'lmbench.output')
        with open(outfile, 'w') as wfh:
            for output in self.output:
                wfh.write(output)

    def teardown(self, context):
        self.device.uninstall_executable(self.test)

    #
    # Test setup routines
    #
    def _setup_lat_mem_rd(self):
        command_stub = self._setup_common()
        if self.thrash:
            command_stub = command_stub + '-t '

        for size in self.size:
            command = command_stub + size + ' '
            for stride in self.stride:
                self.commands.append(command + str(stride))

    def _setup_bw_mem(self):
        command_stub = self._setup_common()

        for size in self.size:
            command = command_stub + size + ' '
            for what in self.mem_category:
                self.commands.append(command + what)

    def _setup_common(self):
        command = self.device_exe + " "
        if self.parallelism is not None:
            command = command + '-P {} '.format(self.parallelism)
        if self.warmup is not None:
            command = command + '-W {} '.format(self.warmup)
        if self.repetitions is not None:
            command = command + '-N {} '.format(self.repetitions)
        return command
