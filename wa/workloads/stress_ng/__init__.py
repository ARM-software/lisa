#    Copyright 2015, 2018 ARM Limited
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

from wa import Workload, Parameter, ConfigError, Executable
from wa.framework.exception import WorkloadError
from wa.utils.exec_control import once
from wa.utils.serializer import yaml


class StressNg(Workload):

    name = 'stress-ng'
    description = """
    Run the stress-ng benchmark.

    stress-ng will stress test a computer system in various selectable ways. It
    was designed to exercise various physical subsystems of a computer as well
    as the various operating system kernel interfaces.

    stress-ng can also measure test throughput rates; this can be useful to
    observe performance changes across different operating system releases or
    types of hardware. However, it has never been intended to be used as a
    precise benchmark test suite, so do NOT use it in this manner.

    The official website for stress-ng is at:
        http://kernel.ubuntu.com/~cking/stress-ng/

    Source code are available from:
        http://kernel.ubuntu.com/git/cking/stress-ng.git/
    """

    parameters = [
        Parameter('stressor', kind=str, default='cpu',
                  allowed_values=['cpu', 'io', 'fork', 'switch', 'vm', 'pipe',
                                  'yield', 'hdd', 'cache', 'sock', 'fallocate',
                                  'flock', 'affinity', 'timer', 'dentry',
                                  'urandom', 'sem', 'open', 'sigq', 'poll'],
                  description='''
                  Stress test case name. The cases listed in
                  allowed values come from the stable release
                  version 0.01.32. The binary included here
                  compiled from dev version 0.06.01. Refer to
                  man page for the definition of each stressor.
                  '''),
        Parameter('extra_args', kind=str, default="",
                  description='''
                  Extra arguments to pass to the workload.

                  Please note that these are not checked for validity.
                  '''),
        Parameter('threads', kind=int, default=0,
                  description='''
                  The number of workers to run. Specifying a negative
                  or zero value will select the number of online
                  processors.
                  '''),
        Parameter('duration', kind=int, default=60,
                  description='''
                  Timeout for test execution in seconds
                  ''')
    ]

    @once
    def initialize(self, context):
        if not self.target.is_rooted:
            raise WorkloadError('stress-ng requires root premissions to run')

        resource = Executable(self, self.target.abi, 'stress-ng')
        host_exe = context.get_resource(resource)
        StressNg.binary = self.target.install(host_exe)

    def setup(self, context):
        self.log = self.target.path.join(self.target.working_directory,
                                         'stress_ng_output.txt')
        self.results = self.target.path.join(self.target.working_directory,
                                             'stress_ng_results.yaml')
        self.command = ('{} --{} {} {} --timeout {}s --log-file {} --yaml {} '
                        '--metrics-brief --verbose'
                        .format(self.binary, self.stressor, self.threads,
                                self.extra_args, self.duration, self.log,
                                self.results))
        self.timeout = self.duration + 10

    def run(self, context):
        self.output = self.target.execute(self.command, timeout=self.timeout,
                                          as_root=True)

    def extract_results(self, context):
        self.host_file_log = os.path.join(context.output_directory,
                                          'stress_ng_output.txt')
        self.host_file_results = os.path.join(context.output_directory,
                                              'stress_ng_results.yaml')
        self.target.pull(self.log, self.host_file_log)
        self.target.pull(self.results, self.host_file_results)

        context.add_artifact('stress_ng_log', self.host_file_log, 'log', "stress-ng's logfile")
        context.add_artifact('stress_ng_results', self.host_file_results, 'raw', "stress-ng's results")

    def update_output(self, context):
        with open(self.host_file_results, 'r') as stress_ng_results:
            results = yaml.load(stress_ng_results)

        try:
            metric = results['metrics'][0]['stressor']
            throughput = results['metrics'][0]['bogo-ops']
            context.add_metric(metric, throughput, 'ops')
        # For some stressors like vm, if test duration is too short, stress_ng
        # may not able to produce test throughput rate.
        except TypeError:
            msg = '{} test throughput rate not found. Please increase test duration and retry.'
            self.logger.warning(msg.format(self.stressor))

    def validate(self):
        if self.stressor == 'vm' and self.duration < 60:
            raise ConfigError('vm test duration needs to be >= 60s.')

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall('stress-ng')
