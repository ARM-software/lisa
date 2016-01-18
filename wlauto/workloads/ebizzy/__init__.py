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

# pylint: disable=W0201, C0103

import os
import re

from wlauto import Workload, Parameter, Executable

results_txt = 'ebizzy_results.txt'
record_regex = re.compile(r'(?P<record>\d+) records/s')
result_regex = re.compile(r'(?P<metric>\D+)(?P<value>\d+.*\b)(?P<unit>\S+)')


class Ebizzy(Workload):

    name = 'ebizzy'
    description = """
    ebizzy is designed to generate a workload resembling common web
    application server workloads.  It is highly threaded, has a large in-memory
    working set with low locality, and allocates and deallocates memory frequently.
    When running most efficiently, it will max out the CPU.

    ebizzy description taken from the source code at
    https://github.com/linux-test-project/ltp/tree/master/utils/benchmark/ebizzy-0.3

    """

    parameters = [
        # Workload parameters go here e.g.
        Parameter('threads', kind=int, default=2, description='Number of threads to execute.'),
        Parameter('seconds', kind=int, default=10, description='Number of seconds.'),
        Parameter('chunks', kind=int, default=10,
                  description='Number of memory chunks to allocate.'),
        Parameter('extra_params', kind=str, default='',
                  description='Extra parameters to pass in (e.g. -M to disable mmap).'
                              ' See ebizzy -? for full list of options.')
    ]

    def setup(self, context):
        timeout_buf = 10
        self.command = '{} -t {} -S {} -n {} {} > {}'
        self.ebizzy_results = self.device.path.join(self.device.working_directory, results_txt)
        self.device_binary = None
        self.run_timeout = self.seconds + timeout_buf

        self.binary_name = 'ebizzy'
        host_binary = context.resolver.get(Executable(self, self.device.abi, self.binary_name))
        self.device_binary = self.device.install_if_needed(host_binary)

        self.command = self.command.format(self.device_binary, self.threads, self.seconds,
                                           self.chunks, self.extra_params, self.ebizzy_results)

    def run(self, context):
        self.device.execute(self.command, timeout=self.run_timeout)

    def update_result(self, context):
        self.device.pull_file(self.ebizzy_results, context.output_directory)

        with open(os.path.join(context.output_directory, results_txt)) as ebizzy_file:
            for line in ebizzy_file:
                record_match = record_regex.search(line)
                if record_match:
                    context.result.add_metric('total_recs', record_match.group('record'),
                                              'records/s')

                results_match = result_regex.search(line)
                if results_match:
                    context.result.add_metric(results_match.group('metric'),
                                              results_match.group('value'),
                                              results_match.group('unit'))

    def teardown(self, context):
        self.device.uninstall_executable(self.device_binary)

    def validate(self):
        pass
