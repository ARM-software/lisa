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

from wlauto import Workload, Parameter, File
from wlauto.exceptions import WorkloadError
from wlauto.utils.misc import parse_value


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


    See: http://sysbench.sourceforge.net/docs/

    """

    parameters = [
        Parameter('timeout', kind=int, default=300),
        Parameter('test', kind=str, default='cpu'),
        Parameter('num_threads', kind=int, default=8),
        Parameter('max_requests', kind=int, default=2000),
    ]

    def __init__(self, device, **kwargs):
        super(Sysbench, self).__init__(device)
        self.command = self._build_command(test=self.test,
                                           num_threads=self.num_threads,
                                           max_requests=self.max_requests)
        self.results_file = self.device.path.join(self.device.working_directory, 'sysbench_result.txt')

    def setup(self, context):
        self._check_executable(context)
        self.device.execute('am start -n com.android.browser/.BrowserActivity about:blank ')

    def run(self, context):
        self.device.execute(self.command, timeout=self.timeout)

    def update_result(self, context):
        host_results_file = os.path.join(context.output_directory, 'sysbench_result.txt')
        self.device.pull_file(self.results_file, host_results_file)

        with open(host_results_file) as fh:
            in_summary = False
            metric_prefix = ''
            for line in fh:
                if line.startswith('Test execution summary:'):
                    in_summary = True
                elif in_summary:
                    if not line.strip():
                        break  # end of summary section
                    parts = [p.strip() for p in line.split(':') if p.strip()]
                    if len(parts) == 2:
                        metric = metric_prefix + parts[0]
                        value, units = parse_value(parts[1])
                        context.result.add_metric(metric, value, units)
                    elif len(parts) == 1:
                        metric_prefix = line.strip() + ' '
                    else:
                        self.logger.warn('Could not parse line: "{}"'.format(line.rstrip('\n')))
        context.add_iteration_artifact('sysbench_output', kind='raw', path='sysbench_result.txt')

    def teardown(self, context):
        self.device.execute('am force-stop com.android.browser')
        self.device.delete_file(self.results_file)

    def _check_executable(self, context):
        if self.device.is_installed('sysbench'):
            return
        path = context.resolver.get(File(owner=self, path='sysbench'))
        if not path:
            raise WorkloadError('sysbench binary is not installed on the device, and it does not found in dependencies on the host.')
        self.device.install(path)

    def _build_command(self, **parameters):
        param_strings = ['--{}={}'.format(k.replace('_', '-'), v)
                         for k, v in parameters.iteritems()]
        sysbench_command = 'sysbench {} run'.format(' '.join(param_strings))
        return 'cd {} && {} > sysbench_result.txt'.format(self.device.working_directory, sysbench_command)
