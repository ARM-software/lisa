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

# pylint: disable=W0201, C0103

from wlauto import Workload, Parameter, Executable

import os
import re

hackbench_results_txt = 'hackbench_results.txt'

grp_regex = re.compile(r'(?P<group>(\d+) groups)')
fd_regex = re.compile(r'(?P<fd>(\d+) file descriptors)')
msg_regex = re.compile(r'(?P<message>(\d+) messages)')
bytes_regex = re.compile(r'(?P<bytes>(\d+) bytes)')
time_regex = re.compile(r'(Time: (\d+.*))')


class Hackbench(Workload):

    name = 'hackbench'
    description = """
    Hackbench runs a series of tests for the Linux scheduler.

    For details, go to:
    https://github.com/linux-test-project/ltp/

    """

    parameters = [
        # Workload parameters go here e.g.
        Parameter('datasize', kind=int, default=100, override=True, mandatory=False,
                  description='Message size in bytes.'),
        Parameter('groups', kind=int, default=10, override=True, mandatory=False,
                  description='Number of groups.'),
        Parameter('loops', kind=int, default=100, override=True, mandatory=False,
                  description='Number of loops.'),
        Parameter('fds', kind=int, default=40, override=True, mandatory=False,
                  description='Number of file descriptors.'),
        Parameter('extra_params', kind=str, default='', override=True, mandatory=False,
                  description='Extra parameters to pass in. See the hackbench man page'
                              ' or type `hackbench --help` for list of options.')
    ]

    def setup(self, context):
        self.command = '{} -s {} -g {} -l {} {} > {}'
        self.device_binary = None
        self.hackbench_result = os.path.join(self.device.working_directory, hackbench_results_txt)

        self.binary_name = 'hackbench'
        if not self.device.is_installed(self.binary_name):
            host_binary = context.resolver.get(Executable(self, self.device.abi, self.binary_name))
            self.device_binary = self.device.install(host_binary)
        else:
            self.device_binary = self.binary_name

        self.command = self.command.format(self.device_binary, self.datasize, self.groups,
                                           self.loops, self.extra_params, self.hackbench_result)

    def run(self, context):
        self.device.execute(self.command)

    def update_result(self, context):
        group_label = 'groups'
        fd_label = 'file_descriptors'
        msg_label = 'messages'
        bytes_label = 'bytes'
        time_label = 'time'

        self.device.pull_file(self.hackbench_result, context.output_directory)

        with open(os.path.join(context.output_directory, hackbench_results_txt)) as hackbench_file:
            for line in hackbench_file:
                group_match = grp_regex.search(line)
                if group_match:
                    context.result.add_metric(group_label, int(group_match.group(2)), group_label)

                fd_match = fd_regex.search(line)
                if fd_match:
                    context.result.add_metric(fd_label, int(fd_match.group(2)), fd_label)

                msg_match = msg_regex.search(line)
                if msg_match:
                    context.result.add_metric(msg_label, int(msg_match.group(2)), msg_label)

                bytes_match = bytes_regex.search(line)
                if bytes_match:
                    context.result.add_metric(bytes_label, int(bytes_match.group(2)), bytes_label)

                time_match = time_regex.search(line)
                if time_match:
                    context.result.add_metric(time_label, float(time_match.group(2)), 'seconds')

    def teardown(self, context):
        self.device.uninstall_executable(self.binary_name)
        self.device.execute('rm -f {}'.format(self.hackbench_result))

    def validate(self):
        pass

