#    Copyright 2014-2015 ARM Limited
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

from wlauto import Workload, Parameter
from wlauto.exceptions import ConfigError


class ShellScript(Workload):

    name = 'shellscript'
    description = """
    Runs an arbitrary shellscript on the device.

    """

    parameters = [
        Parameter('script_file', mandatory=True,
                  description=('The path (on the host) to the shell script file. This must be '
                               'an absolute path (though it may contain ~).')),
        Parameter('argstring', default='',
                  description='A string that should contain arguments passed to the script.'),
        Parameter('timeout', kind=int, default=60,
                  description='Timeout, in seconds, for the script run time.'),
    ]

    def __init__(self, device, **kwargs):
        super(ShellScript, self).__init__(device, **kwargs)
        self.script_file = os.path.expanduser(self.script_file)
        if not os.path.isfile(self.script_file):
            raise ConfigError('Can\'t access file (is the path correct?): {}'.format(self.script_file))
        self.output = None
        self.command = None
        self.on_device_script_file = None

    def setup(self, context):
        self.on_device_script_file = self.device.path.join(self.device.working_directory,
                                                           os.path.basename(self.script_file))
        self.device.push_file(self.script_file, self.on_device_script_file)
        self.command = 'sh {} {}'.format(self.on_device_script_file, self.argstring)

    def run(self, context):
        self.output = self.device.execute(self.command, timeout=self.timeout)

    def update_result(self, context):
        with open(os.path.join(context.output_directory, 'output.txt'), 'w') as wfh:
            wfh.write(self.output)

    def teardown(self, context):
        self.device.delete_file(self.on_device_script_file)
