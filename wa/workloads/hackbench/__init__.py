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

# pylint: disable=W0201, C0103

import os
import re

from wa import Workload, Parameter, Executable
from wa.utils.exec_control import once


timeout_buffer = 10

hackbench_results_txt = 'hackbench_results.txt'

regex_map = {"total_groups": (re.compile(r'(\d+) groups'), "groups"),
             "total_fd": (re.compile(r'(\d+) file descriptors'), "file_descriptors"),
             "total_messages": (re.compile(r'(\d+) messages'), "messages"),
             "total_bytes": (re.compile(r'(\d+) bytes'), "bytes"),
             "test_time": (re.compile(r'Time: (\d+.*)'), "seconds")
             }


class Hackbench(Workload):

    name = 'hackbench'
    description = """
    Hackbench runs a series of tests for the Linux scheduler.

    For details, go to:
    https://github.com/linux-test-project/ltp/

    """

    parameters = [
        Parameter('timeout', kind=int, default=30, aliases=['duration'],
                  description='Expected test duration in seconds.'),
        Parameter('datasize', kind=int, default=100, description='Message size in bytes.'),
        Parameter('groups', kind=int, default=10, description='Number of groups.'),
        Parameter('loops', kind=int, default=100, description='Number of loops.'),
        Parameter('fds', kind=int, default=40, description='Number of file descriptors.'),
        Parameter('extra_params', kind=str, default='',
                  description='''
                  Extra parameters to pass in. See the hackbench man page
                  or type `hackbench --help` for list of options.
                  '''),
    ]

    binary_name = 'hackbench'

    @once
    def initialize(self, context):
        host_binary = context.get_resource(Executable(self, self.target.abi, self.binary_name))
        Hackbench.target_binary = self.target.install(host_binary)

    def setup(self, context):
        self.target_output_file = self.target.get_workpath(hackbench_results_txt)
        self.run_timeout = self.timeout + timeout_buffer
        command_format = '{} -s {} -g {} -l {} {} > {}'
        self.command = command_format.format(self.target_binary, self.datasize, self.groups,
                                             self.loops, self.extra_params, self.target_output_file)

    def run(self, context):
        self.target.execute(self.command, timeout=self.run_timeout)

    def extract_results(self, context):
        host_output_file = os.path.join(context.output_directory, hackbench_results_txt)
        self.target.pull(self.target_output_file, host_output_file)
        context.add_artifact('hackbench-results', host_output_file, kind='raw')

    def update_output(self, context):
        results_file = context.get_artifact_path('hackbench-results')
        with open(results_file) as fh:
            for line in fh:
                for label, (regex, units) in regex_map.items():
                    match = regex.search(line)
                    if match:
                        context.add_metric(label, float(match.group(1)), units)

    def teardown(self, context):
        if self.cleanup_assets:
            self.target.execute('rm -f {}'.format(self.target_output_file))

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall(self.binary_name)
