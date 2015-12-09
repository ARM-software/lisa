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
# pylint: disable=attribute-defined-outside-init
import os
import re

from wlauto import Workload, Parameter, Executable


stream_results_txt = 'stream_results.txt'
system_array_regex = re.compile(r'^This system uses (\d)')

regex_map = {
    "array_size": (re.compile(r'^Array size = (\d+)'), "elements"),
    "total_threads_requested": (re.compile(r'^Number of Threads requested = (\d+)'), "threads"),
    "total_thread_count": (re.compile(r'^Number of Threads counted = (\d+)'), "threads")
}

regex_map2 = {
    "memory_per_array": re.compile(r'^Memory per array = (\d*.\d*) (\w+)'),
    "total_memory_required": re.compile(r'^Total memory required = (\d*.\d*) (\w+)')
}


class Stream(Workload):

    name = 'stream'
    description = """
    Measures memory bandwidth.

    The original source code be found on:
    https://www.cs.virginia.edu/stream/FTP/Code/
    """

    parameters = [
        # Workload parameters go here e.g.
        Parameter('threads', kind=int, default=0,
                  description='The number of threads to execute if OpenMP is enabled')
    ]

    def initialize(self, context):
        Stream.stream_noomp_binary = context.resolver.get(Executable(self, self.device.abi, 'stream_noomp'))
        Stream.stream_omp_binary = context.resolver.get(Executable(self, self.device.abi, 'stream_omp'))

        Stream.stream_default = self.device.install(Stream.stream_noomp_binary)
        Stream.stream_optional = self.device.install(Stream.stream_omp_binary)

    def setup(self, context):
        self.results = os.path.join(self.device.working_directory, stream_results_txt)
        self.timeout = 50

        if self.threads:
            self.command = 'OMP_NUM_THREADS={} {} > {}'.format(self.threads, self.stream_optional, self.results)
        else:
            self.command = '{} > {}'.format(self.stream_default, self.results)

    def run(self, context):
        self.output = self.device.execute(self.command, timeout=self.timeout)

    def update_result(self, context):
        self.device.pull_file(self.results, context.output_directory)
        outfile = os.path.join(context.output_directory, stream_results_txt)

        with open(outfile, 'r') as stream_file:
            for line in stream_file:
                match = system_array_regex.search(line)
                if match:
                    context.result.add_metric('bytes_per_array_element', int(match.group(1)), 'bytes')

                for label, (regex, units) in regex_map.iteritems():
                    match = regex.search(line)
                    if match:
                        context.result.add_metric(label, float(match.group(1)), units)

                for label, regex in regex_map2.iteritems():
                    match = regex.search(line)
                    if match:
                        context.result.add_metric(label, float(match.group(1)), match.group(2))

    def finalize(self, context):
        self.device.uninstall_executable(self.stream_default)
        self.device.uninstall_executable(self.stream_optional)
