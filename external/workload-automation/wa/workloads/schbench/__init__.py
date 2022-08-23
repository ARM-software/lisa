#    Copyright 2022 ARM Limited
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

import os
import re

from wa import Workload, Parameter, Executable
from wa.utils.exec_control import once

timeout_buffer = 10

regex_map = {"50.0th": re.compile(r'50.0th: (\d+)'),
             "75.0th": re.compile(r'75.0th: (\d+)'),
             "90.0th": re.compile(r'90.0th: (\d+)'),
             "95.0th": re.compile(r'95.0th: (\d+)'),
             "*99.0th": re.compile(r'\*99.0th: (\d+)'),
             "99.5th": re.compile(r'99.5th: (\d+)'),
             "99.9th": re.compile(r'99.9th: (\d+)'),
             "min": re.compile(r'min=(\d+)'),
             "max": re.compile(r'max=(\d+)')
             }


class Schbench(Workload):
    name = 'schbench'
    description = """
    Benchmark providing detailed latency distribution statistics for scheduler
    wakeups.
    """

    parameters = [
        Parameter('runtime', kind=int, default=30, aliases=['timeout'],
                  description='How long to run before exiting (seconds)'),
        Parameter('message_threads', kind=int, default=2,
                  description='number of message threads'),
        Parameter('threads', kind=int, default=16,
                  description='worker threads per message thread'),
        Parameter('sleeptime', kind=int, default=30000,
                  description='Message thread latency (usec)'),
        Parameter('cputime', kind=int, default=30000,
                  description='How long to think during loop'),
        Parameter('auto', kind=bool, default=False,
                  description='grow thread count until latencies hurt'),
        Parameter('pipe', kind=int, default=0,
                  description='transfer size bytes to simulate a pipe test'),
        Parameter('rps', kind=int, default=0,
                  description='requests per second mode (count)'),
        Parameter('intervaltime', kind=int,
                  description='interval for printing latencies (seconds)'),
    ]

    binary_name = 'schbench'
    schbench_results_txt = 'schbench_results.txt'
    output_unit = 'usec'

    @once
    def initialize(self, context):
        host_binary = context.get_resource(
            Executable(self, self.target.abi, self.binary_name))
        Schbench.target_binary = self.target.install(host_binary)

    def setup(self, context):
        self.target_output_file = self.target.get_workpath(
            self.schbench_results_txt)
        self.run_timeout = self.runtime + timeout_buffer
        self.command = "{} -m {} -t {} -r {} -s {} -c {} -p {} -R {} -i {} {}"
        self.command = self.command.format(
            self.target_binary, self.message_threads, self.threads,
            self.runtime, self.sleeptime, self.cputime, self.pipe, self.rps,
            self.runtime if not self.intervaltime else self.intervaltime,
            '-a' if self.auto else '',
        )

    def run(self, context):
        self.output = self.target.execute(
            self.command, timeout=self.run_timeout)

    def extract_results(self, context):
        host_output_file = os.path.join(
            context.output_directory, self.schbench_results_txt)
        with open(host_output_file, "w") as f:
            f.write(self.output)
        context.add_artifact('schbench-results', host_output_file, kind='raw')

    def update_output(self, context):
        results_file = context.get_artifact_path('schbench-results')
        with open(results_file) as fh:
            for line in fh:
                for label, regex in regex_map.items():
                    match = regex.search(line)
                    if match:
                        context.add_metric(label, float(match.group(1)),
                                           self.output_unit)

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall(self.binary_name)
