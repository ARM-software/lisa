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
#

# pylint: disable=E1101,W0201,E0203

import os

from wlauto import Workload, Parameter, Executable
from wlauto.exceptions import WorkloadError, ConfigError
from wlauto.utils.misc import parse_value
from wlauto.utils.types import boolean, numeric


class Sysbench(Workload):

    name = 'sysbench'
    description = """
    SysBench is a modular, cross-platform and multi-threaded benchmark tool
    for evaluating OS parameters that are important for a system running a
    database under intensive load.

    The idea of this benchmark suite is to quickly get an impression about
    system performance without setting up complex database benchmarks or
    even without installing a database at all.

    **Features of SysBench**

       * file I/O performance
       * scheduler performance
       * memory allocation and transfer speed
       * POSIX threads implementation performance
       * database server performance


    See: https://github.com/akopytov/sysbench

    """

    parameters = [
        Parameter('timeout', kind=int, default=300,
                  description='timeout for workload exectution (adjust from default '
                              'if running on a slow device and/or specifying a large value for '
                              '``max_requests``'),
        Parameter('test', kind=str, default='cpu',
                  allowed_values=['fileio', 'cpu', 'memory', 'threads', 'mutex'],
                  description='sysbench test to run'),
        Parameter('num_threads', kind=int, default=8,
                  description='The number of threads sysbench will launch'),
        Parameter('max_requests', kind=int, default=2000,
                  description='The limit for the total number of requests'),
        Parameter('file_test_mode', default=None,
                  allowed_values=['seqwr', 'seqrewr', 'seqrd', 'rndrd', 'rndwr', 'rndrw'],
                  description='File test mode to use. this should only be specfied if ``test`` is '
                              '``"fileio"``; if that is the case and ``file_test_mode`` is not specified, '
                              'it will default to ``"seqwr"`` (please see sysbench documentation for '
                              'explanation of various modes).'),
        Parameter('cmd_params', kind=str, default='',
                  description='Additional parameters to be passed to sysbench as a single stiring'),
        Parameter('force_install', kind=boolean, default=True,
                  description='Always install binary found on the host, even if already installed on device'),
    ]

    def validate(self):
        if self.test == 'fileio' and not self.file_test_mode:
            self.logger.debug('Test is "fielio" and no file_test_mode specified -- using default.')
            self.file_test_mode = 'seqwr'
        elif self.test != 'fileio' and self.file_test_mode:
            raise ConfigError('file_test_mode must not be specified unless test is "fileio"')

    def init_resources(self, context):
        self.on_host_binary = context.resolver.get(Executable(self, 'armeabi', 'sysbench'), strict=False)

    def setup(self, context):
        self.command = self._build_command(test=self.test,
                                           num_threads=self.num_threads,
                                           max_requests=self.max_requests)
        self.results_file = self.device.path.join(self.device.working_directory, 'sysbench_result.txt')
        self._check_executable()

    def run(self, context):
        self.device.execute(self.command, timeout=self.timeout)

    def update_result(self, context):
        host_results_file = os.path.join(context.output_directory, 'sysbench_result.txt')
        self.device.pull_file(self.results_file, host_results_file)
        context.add_iteration_artifact('sysbench_output', kind='raw', path=host_results_file)

        with open(host_results_file) as fh:
            find_line_with('response time:', fh)
            extract_response_time_metric('min', fh.next(), context.result)
            extract_response_time_metric('avg', fh.next(), context.result)
            extract_response_time_metric('max', fh.next(), context.result)
            extract_response_time_metric('approx.  95 percentile', fh.next(), context.result)
            find_line_with('Threads fairness:', fh)
            extract_threads_fairness_metric('events', fh.next(), context.result)
            extract_threads_fairness_metric('execution time', fh.next(), context.result)

    def teardown(self, context):
        self.device.delete_file(self.results_file)

    def _check_executable(self):
        if self.device.is_installed('sysbench') and not self.force_install:
            self.logger.debug('sysbench found on device')
            return
        if not self.on_host_binary:
            raise WorkloadError('sysbench binary is not installed on the device, and it does not found in dependencies on the host.')
        self.device.install(self.on_host_binary)

    def _build_command(self, **parameters):
        param_strings = ['--{}={}'.format(k.replace('_', '-'), v)
                         for k, v in parameters.iteritems()]
        if self.file_test_mode:
            param_strings.append('--file-test-mode={}'.format(self.file_test_mode))
        sysbench_command = 'sysbench {} {} run'.format(' '.join(param_strings), self.cmd_params)
        return 'cd {} && {} > sysbench_result.txt'.format(self.device.working_directory, sysbench_command)


# Utility functions

def find_line_with(text, fh):
    for line in fh:
        if text in line:
            return
    message = 'Could not extract sysbench results from {}; did not see "{}"'
    raise WorkloadError(message.format(fh.name, text))


def extract_response_time_metric(metric, line, result):
    try:
        name, value_part = [part.strip() for part in line.split(':')]
        if name != metric:
            message = 'Name mismatch: expected "{}", got "{}"'
            raise WorkloadError(message.format(metric, name.strip()))
        idx = -1
        while value_part[idx - 1].isalpha() and idx:  # assumes at least one char of units
            idx -= 1
        if not idx:
            raise WorkloadError('Could not parse value "{}"'.format(value_part))
        value = numeric(value_part[:idx])
        units = value_part[idx:]
        result.add_metric('response time ' + metric,
                          value, units, lower_is_better=True)
    except Exception as e:
        message = 'Could not extract sysbench metric "{}"; got "{}"'
        raise WorkloadError(message.format(metric, e))


def extract_threads_fairness_metric(metric, line, result):
    try:
        name_part, value_part = [part.strip() for part in line.split(':')]
        name = name_part.split('(')[0].strip()
        if name != metric:
            message = 'Name mismatch: expected "{}", got "{}"'
            raise WorkloadError(message.format(metric, name))
        avg, stddev = [numeric(v) for v in value_part.split('/')]
        result.add_metric('thread fairness {} avg'.format(metric), avg)
        result.add_metric('thread fairness {} stddev'.format(metric),
                          stddev, lower_is_better=True)
    except Exception as e:
        message = 'Could not extract sysbench metric "{}"; got "{}"'
        raise WorkloadError(message.format(metric, e))
