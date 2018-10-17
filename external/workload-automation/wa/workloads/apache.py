#    Copyright 2018 ARM Limited
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

from __future__ import division
import os
# pylint: disable=wrong-import-order,wrong-import-position
from future.standard_library import install_aliases
install_aliases()

from urllib.request import urlopen  # pylint: disable=import-error

from wa import Workload, Parameter, Alias, WorkloadError
from wa.utils.exec_control import once
from wa.utils.misc import which, check_output


class ApacheBenchmark(Workload):

    name = 'apache'
    description = '''
    Load-test an apache installation by issueing parallel requests with ab.

    Run ab, the Apache benchmark on the host, directed at the target as the
    server.

    .. note:: It is assumed that Apache is already running on target.

    .. note:: Current implmentation only supports a very basic use of the
              benchmark.

    '''

    parameters = [
        Parameter('port', kind=int, default=80,
                  description='''
                  Port on which Apache is running.
                  '''),
        Parameter('path', default='/',
                  description='''
                  Path to request.
                  '''),
        Parameter('parallel_requests', kind=int, default=350,
                  description='''
                  The number of parallel requests at a time.
                  '''),
        Parameter('total_requests', kind=int, default=100000,
                  description='''
                  The total number of parallel requests.
                  '''),
    ]

    aliases = [
        Alias('ab'),
    ]

    supported_targets = ['linux']

    @once
    def initialize(self, context):
        ab = which('ab')
        if not ab:
            msg = 'ab not found on host; make sure apache2-utils (or you distro equivalent) package is installed.'
            raise WorkloadError(msg)

        response = urlopen('http://{}:{}{}'.format(self.target.conn.host, self.port, self.path))
        code = response.getcode()
        if code != 200:
            msg = 'HTTP request failed with status {}; is Apache running on target?'
            raise WorkloadError(msg.format(code))

    def setup(self, context):
        template = 'ab -k -c {} -n {} {}:{}{}'
        self.command = template.format(self.parallel_requests,
                                       self.total_requests,
                                       self.target.conn.host,
                                       self.port,
                                       self.path)
        self.output = None

    def run(self, context):
        self.logger.debug(self.command)
        self.output, _ = check_output(self.command, timeout=300, shell=True)

    def extract_results(self, context):
        outfile = os.path.join(context.output_directory, 'ab.output')
        with open(outfile, 'w') as wfh:
            wfh.write(self.output)
            context.add_artifact('ab-output', outfile, kind='raw')

    def update_output(self, context):  # pylint: disable=too-many-locals
        with open(context.get_artifact_path('ab-output')) as fh:
            server_software = get_line(fh, 'Server Software').split(':')[1].strip()
            context.add_metadata('server-software', server_software)

            doc_len_str = get_line(fh, 'Document Length').split(':')[1].strip()
            doc_len = int(doc_len_str.split()[0])
            context.add_metadata('document-length', doc_len)

            completed = int(get_line(fh, 'Complete requests').split(':')[1].strip())
            failed = int(get_line(fh, 'Failed requests').split(':')[1].strip())
            fail_rate = failed / completed * 100
            context.add_metric('failed_request', fail_rate, units='percent',
                               lower_is_better=True)

            rps_str = get_line(fh, 'Requests per second').split(':')[1].strip()
            rps = float(rps_str.split('[')[0])
            rps_units = rps_str.split('[')[1].split(']')[0]
            context.add_metric('requests_per_second', rps, units=rps_units)

            tpr_str = get_line(fh, 'Time per request').split(':')[1].strip()
            tpr = float(tpr_str.split('[')[0])
            tpr_units = tpr_str.split('[')[1].split(']')[0]
            context.add_metric('time_per_request', tpr, units=tpr_units)

            trate_str = get_line(fh, 'Transfer rate').split(':')[1].strip()
            trate = float(trate_str.split('[')[0])
            trate_units = trate_str.split('[')[1].split(']')[0]
            context.add_metric('transfer_rate', trate, units=trate_units)

            pc99 = int(get_line(fh, '99%').split()[1])
            context.add_metric('request_99percentile', pc99, 'ms')

            pc100 = int(get_line(fh, '100%').split()[1])
            context.add_metric('longest_request', pc100, 'ms')


def get_line(fh, text):
    for line in fh:
        if text in line:
            return line
