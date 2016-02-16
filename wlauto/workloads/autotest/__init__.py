#    Copyright 2015 ARM Limited
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
# pylint: disable=attribute-defined-outside-init

import os
import re

from wlauto import Workload, Parameter
from wlauto.exceptions import WorkloadError
from wlauto.utils.misc import which, check_output
from wlauto.utils.types import arguments, numeric


# Location of the power_LoadTest under the chroot
#POWER_LOADTEST_DIR = '/mnt/host/source/src/third_party/autotest/files/client/site_tests/power_LoadTest'
MARKER = '---------------------------'
STATUS_REGEX = re.compile(r'^\S+\s+\[\s*(\S+)\s*\]')
METRIC_REGEX = re.compile(r'^\S+\s+(\S+)\s*(\S+)')


class ChromeAutotest(Workload):

    name = 'autotest'
    description = '''
    Executes tests from ChromeOS autotest suite

    .. note:: This workload *must* be run inside a CromeOS SDK chroot.

    See: https://www.chromium.org/chromium-os/testing/power-testing

    '''
    supported_platforms = ['chromeos']

    parameters = [
        Parameter('test', mandatory=True,
                  description='''
                  The test to be run
                  '''),
        Parameter('test_that_args', kind=arguments, default='',
                  description='''
                  Extra arguments to be passed to test_that_invocation.
                  '''),
        Parameter('run_timeout', kind=int, default=30 * 60,
                  description='''
                  Timeout, in seconds, for the test execution.
                  '''),
    ]

    def setup(self, context):
        if self.device.os != 'chromeos':
            raise WorkloadError('{} only supports ChromeOS devices'.format(self.name))
        self.test_that = which('test_that')
        if not self.test_that:
            message = ('Could not find "test_that"; {} must be running in a ChromeOS SDK chroot '
                       '(did you execute "cros_sdk"?)')
            raise WorkloadError(message.format(self.name))
        self.command = self._build_command()
        self.raw_output = None
        # make sure no other test is running
        self.device.execute('killall -9 autotest', check_exit_code=False)

    def run(self, context):
        self.logger.debug(self.command)
        self.raw_output, _ = check_output(self.command, timeout=self.run_timeout, shell=True)

    def update_result(self, context):
        if not self.raw_output:
            self.logger.warning('No power_LoadTest output detected; run failed?')
            return
        raw_outfile = os.path.join(context.output_directory, 'autotest-output.raw')
        with open(raw_outfile, 'w') as wfh:
            wfh.write(self.raw_output)
        context.add_artifact('autotest_raw', raw_outfile, kind='raw')
        lines = iter(self.raw_output.split('\n'))
        # Results are delimitted from the rest of the output by MARKER
        for line in lines:
            if MARKER in line:
                break
        for line in lines:
            match = STATUS_REGEX.search(line)
            if match:
                status = match.group(1)
                if status != 'PASSED':
                    self.logger.warning(line)
            match = METRIC_REGEX.search(line)
            if match:
                try:
                    context.result.add_metric(match.group(1), numeric(match.group(2)), lower_is_better=True)
                except ValueError:
                    pass  # non-numeric metrics aren't supported

    def _build_command(self):
        parts = [self.test_that, self.device.host, self.test]
        parts.append(str(self.test_that_args))
        return ' '.join(parts)

