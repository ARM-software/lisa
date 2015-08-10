#    Copyright 2012-2015 ARM Limited
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

from wlauto import Workload, Parameter, Executable
from wlauto.exceptions import ConfigError
from wlauto.utils.types import list_of_strs
import os
import re

iozone_results_txt = 'iozone_results.txt'

time_res_regex = re.compile(r'Time Resolution = (\d+.\d+) (\w+)')
cache_size_regex = re.compile(r'Processor cache size set to (\d+) (\w+)')
cache_line_regex = re.compile(r'Processor cache line size set to (\d+) (\w+)')


class Iozone(Workload):

    name = 'iozone'
    description = """
    Iozone runs a series of disk I/O performance tests.

    To run specific tests, they must be written in the form of:

    ['0', '1', '5', '6']

    By default, iozone will run all tests in auto mode.

    The official website for iozone is at www.iozone.org.
    """

    parameters = [
        Parameter('tests', kind=list_of_strs, default=[],
                  description='List of performance tests to run.'),
        Parameter('auto_mode', kind=bool, default=True,
                  description='Run tests in auto mode.'),
        Parameter('timeout', kind=int, default=14400,
                  description='Timeout for the workload.'),
        Parameter('file_size', kind=int, default=0,
                  description='Fixed file size to run tests with.'),
        Parameter('record_length', kind=int, default=0,
                  description='Fixed record length.'),
        Parameter('threads', kind=int, default=0,
                  description='Number of threads'),
        Parameter('other_params', kind=str, default='',
                  description='Other parameter. Run iozone -h to see'
                              ' list of options.')
    ]

    def initialize(self, context):
        Iozone.host_binary = context.resolver.get(Executable(self,
                                                             self.device.abi,
                                                             'iozone'))
        Iozone.device_binary = self.device.install(Iozone.host_binary)

    def setup(self, context):
        self.results = os.path.join(self.device.working_directory,
                                    iozone_results_txt)
        self.command = self._build_command()

        if self.threads and self.auto_mode:
            raise ConfigError("You cannot set the number of threads and enable"
                              " auto mode at the same time.")

    def _build_command(self):
        iozone_command = '{}'.format(self.device_binary)

        if self.auto_mode:
            self.auto_option = ' -a'
            iozone_command += self.auto_option

        if self.tests:
            self.test_string = ''.join([' -i {}'.format(t) for t in self.tests])
            iozone_command += self.test_string

        if self.record_length > 0:
            self.record_option = ' -r {}'.format(self.record_length)
            iozone_command += self.record_option

        if self.threads > 0:
            self.threads_option = ' -t {}'.format(self.threads)
            iozone_command += self.threads_option

        if self.file_size > 0:
            self.file_size_option = ' -s {}'.format(self.file_size)
            iozone_command += self.file_size_option

        if self.other_params:
            other_params_string = ' ' + self.other_params
            iozone_command += other_params_string

        self.log_string = ' > {}'.format(self.results)
        iozone_command += self.log_string

        return iozone_command

    def run(self, context):
        self.device.execute(self.command, timeout=self.timeout)

    def update_result(self, context):
        self.device.pull_file(self.results, context.output_directory)
        outfile = os.path.join(context.output_directory, iozone_results_txt)

        with open(outfile, 'r') as iozone_file:
            for line in iozone_file:
                time_res_match = time_res_regex.search(line)
                if time_res_match:
                    context.result.add_metric("time_resolution",
                                              float(time_res_match.group(1)),
                                              time_res_match.group(2))

                cache_size_match = cache_size_regex.search(line)
                if cache_size_match:
                    context.result.add_metric("processor_cache_size",
                                              float(cache_size_match.group(1)),
                                              cache_size_match.group(2))

                cache_line_match = cache_line_regex.search(line)
                if cache_line_match:
                    context.result.add_metric("processor_cache_line_size",
                                              float(cache_line_match.group(1)),
                                              cache_line_match.group(2))

    def finalize(self, context):
        self.device.uninstall_executable(self.device_binary)
