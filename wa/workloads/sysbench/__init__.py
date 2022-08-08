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
#

# pylint: disable=E1101,W0201,E0203

import os

from wa import Workload, Parameter, Executable, WorkloadError, ConfigError
from wa.utils.exec_control import once
from wa.utils.misc import parse_value
from wa.utils.types import numeric, cpu_mask


class Sysbench(Workload):

    name = 'sysbench'
    description = """
    A modular, cross-platform and multi-threaded benchmark tool for evaluating
    OS parameters that are important for a system running a database under
    intensive load.

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
                  description='''
                  timeout for workload execution (adjust from default if
                  running on a slow target and/or specifying a large value for
                  ``max_requests``
                  '''),
        Parameter('test', kind=str, default='cpu',
                  allowed_values=['fileio', 'cpu', 'memory', 'threads', 'mutex'],
                  description='sysbench test to run'),
        Parameter('threads', kind=int, default=8, aliases=['num_threads'],
                  description='''
                  The number of threads sysbench will launch.
                  '''),
        Parameter('max_requests', kind=int, default=None,
                  description='The limit for the total number of requests.'),
        Parameter('max_time', kind=int, default=None,
                  description='''
                  The limit for the total execution time. If neither this nor
                  ``max_requests`` is specified, this will default to 30
                  seconds.
                  '''),
        Parameter('file_test_mode', default=None,
                  allowed_values=['seqwr', 'seqrewr', 'seqrd', 'rndrd', 'rndwr', 'rndrw'],
                  description='''
                  File test mode to use. This should only be specified if
                  ``test`` is ``"fileio"``; if that is the case and
                  ``file_test_mode`` is not specified, it will default to
                  ``"seqwr"`` (please see sysbench documentation for
                  explanation of various modes).
                  '''),
        Parameter('cmd_params', kind=str, default='',
                  description='''
                  Additional parameters to be passed to sysbench as a single
                  string.
                  '''),
        Parameter('cpus', kind=cpu_mask, default=0, aliases=['taskset_mask'],
                  description='''
                  The processes spawned by sysbench will be
                  pinned to cores as specified by this parameter. Can be
                  provided as a mask, a list of cpus or a sysfs-style string.
                  '''),
    ]

    def validate(self):
        if (self.max_requests is None) and (self.max_time is None):
            self.max_time = 30
        if self.max_time and (self.max_time + 10) > self.timeout:
            self.timeout = self.max_time + 10
        if self.test == 'fileio' and not self.file_test_mode:
            self.logger.debug('Test is "fileio" and no file_test_mode specified -- using default.')
            self.file_test_mode = 'seqwr'
        elif self.test != 'fileio' and self.file_test_mode:
            raise ConfigError('file_test_mode must not be specified unless test is "fileio"')

    @once
    def initialize(self, context):
        exe = Executable(self, self.target.abi, 'sysbench')
        host_binary = context.get_resource(exe)
        Sysbench.target_binary = self.target.install(host_binary)

    def setup(self, context):
        self.host_results_file = None
        params = dict(test=self.test,
                      num_threads=self.threads)
        if self.max_requests:
            params['max_requests'] = self.max_requests
        if self.max_time:
            params['max_time'] = self.max_time
        self.target_results_file = self.target.get_workpath('sysbench_result.txt')
        self.command = self._build_command(**params)

    def run(self, context):
        self.target.execute(self.command, timeout=self.timeout)

    def extract_results(self, context):
        self.host_results_file = os.path.join(context.output_directory, 'sysbench_result.txt')
        self.target.pull(self.target_results_file, self.host_results_file)
        context.add_artifact('sysbench_output', self.host_results_file, kind='raw')

    def update_output(self, context):
        if not os.path.exists(self.host_results_file):
            self.logger.warning('No results file found.')
            return

        with open(self.host_results_file) as fh:
            find_line_with('General statistics:', fh)
            extract_metric('total time', next(fh), context.output)
            extract_metric('total number of events', next(fh), context.output, lower_is_better=False)
            find_line_with('response time:', fh)
            extract_metric('min', next(fh), context.output, 'response time ')
            extract_metric('avg', next(fh), context.output, 'response time ')
            extract_metric('max', next(fh), context.output, 'response time ')
            extract_metric('approx.  95 percentile', next(fh), context.output)
            find_line_with('Threads fairness:', fh)
            extract_threads_fairness_metric('events', next(fh), context.output)
            extract_threads_fairness_metric('execution time', next(fh), context.output)

    def teardown(self, context):
        if self.cleanup_assets:
            self.target.remove(self.target_results_file)

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall('sysbench')

    def _build_command(self, **parameters):
        param_strings = ['--{}={}'.format(k.replace('_', '-'), v)
                         for k, v in parameters.items()]
        if self.file_test_mode:
            param_strings.append('--file-test-mode={}'.format(self.file_test_mode))
        sysbench_command = '{} {} {} run'.format(self.target_binary, ' '.join(param_strings), self.cmd_params)
        if self.cpus:
            taskset_string = '{} taskset {} '.format(self.target.busybox, self.cpus.mask())
        else:
            taskset_string = ''
        return 'cd {} && {} {} > sysbench_result.txt'.format(self.target.working_directory, taskset_string, sysbench_command)


# Utility functions

def find_line_with(text, fh):
    for line in fh:
        if text in line:
            return
    message = 'Could not extract sysbench results from {}; did not see "{}"'
    raise WorkloadError(message.format(fh.name, text))


def extract_metric(metric, line, output, prefix='', lower_is_better=True):
    try:
        name, value_part = [part.strip() for part in line.split(':')]
        if name != metric:
            message = 'Name mismatch: expected "{}", got "{}"'
            raise WorkloadError(message.format(metric, name.strip()))
        if not value_part or not value_part[0].isdigit():
            raise ValueError('value part does not start with a digit: {}'.format(value_part))
        idx = -1
        if not value_part[idx].isdigit():  # units detected at the end of the line
            while not value_part[idx - 1].isdigit():
                idx -= 1
            value = numeric(value_part[:idx])
            units = value_part[idx:]
        else:
            value = numeric(value_part)
            units = None
        output.add_metric(prefix + metric,
                          value, units, lower_is_better=lower_is_better)
    except Exception as e:
        message = 'Could not extract sysbench metric "{}"; got "{}"'
        raise WorkloadError(message.format(prefix + metric, e))


def extract_threads_fairness_metric(metric, line, output):
    try:
        name_part, value_part = [part.strip() for part in line.split(':')]
        name = name_part.split('(')[0].strip()
        if name != metric:
            message = 'Name mismatch: expected "{}", got "{}"'
            raise WorkloadError(message.format(metric, name))
        avg, stddev = [numeric(v) for v in value_part.split('/')]
        output.add_metric('thread fairness {} avg'.format(metric), avg)
        output.add_metric('thread fairness {} stddev'.format(metric),
                          stddev, lower_is_better=True)
    except Exception as e:
        message = 'Could not extract sysbench metric "{}"; got "{}"'
        raise WorkloadError(message.format(metric, e))
