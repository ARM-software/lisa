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

import re

from wlauto import AndroidUiAutoBenchmark, Parameter, Alias
from wlauto.exceptions import ConfigError


class Andebench(AndroidUiAutoBenchmark):

    name = 'andebench'
    description = """
    AndEBench is an industry standard Android benchmark provided by The
    Embedded Microprocessor Benchmark Consortium (EEMBC).

    http://www.eembc.org/andebench/about.php

    From the website:

       - Initial focus on CPU and Dalvik interpreter performance
       - Internal algorithms concentrate on integer operations
       - Compares the difference between native and Java performance
       - Implements flexible multicore performance analysis
       - Results displayed in Iterations per second
       - Detailed log file for comprehensive engineering analysis

    """
    package = 'com.eembc.coremark'
    activity = 'com.eembc.coremark.splash'
    summary_metrics = ['AndEMark Java', 'AndEMark Native']

    parameters = [
        Parameter('number_of_threads', kind=int,
                  description='Number of threads that will be spawned by AndEBench.'),
        Parameter('single_threaded', kind=bool,
                  description="""
                  If ``true``, AndEBench will run with a single thread. Note: this must
                  not be specified if ``number_of_threads`` has been specified.
                  """),
    ]

    aliases = [
        Alias('andebenchst', number_of_threads=1),
    ]

    regex = re.compile('\s*(?P<key>(AndEMark Native|AndEMark Java))\s*:'
                       '\s*(?P<value>\d+)')

    def validate(self):
        if (self.number_of_threads is not None) and (self.single_threaded is not None):  # pylint: disable=E1101
            raise ConfigError('Can\'t specify both number_of_threads and single_threaded parameters.')

    def setup(self, context):
        if self.number_of_threads is None:  # pylint: disable=access-member-before-definition
            if self.single_threaded:  # pylint: disable=E1101
                self.number_of_threads = 1  # pylint: disable=attribute-defined-outside-init
            else:
                self.number_of_threads = self.device.number_of_cores  # pylint: disable=W0201
        self.logger.debug('Using {} threads'.format(self.number_of_threads))
        self.uiauto_params['number_of_threads'] = self.number_of_threads
        # Called after this setup as modifying uiauto_params
        super(Andebench, self).setup(context)

    def update_result(self, context):
        super(Andebench, self).update_result(context)
        results = {}
        with open(self.logcat_log) as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    data = match.groupdict()
                    results[data['key']] = data['value']
        for key, value in results.iteritems():
            context.result.add_metric(key, value)

