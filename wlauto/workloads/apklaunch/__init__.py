#    Copyright 2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=attribute-defined-outside-init
import os

from time import sleep

from wlauto import Workload, Parameter
from wlauto import File
from wlauto.exceptions import ConfigError
from wlauto.utils.android import ApkInfo


class ApkLaunchWorkload(Workload):
    name = 'apklaunch'
    description = '''
    Installs and runs a .apk file, waits wait_time_seconds, and tests if the app
    has started successfully.
    '''
    supported_platforms = ['android']

    parameters = [
        Parameter('apk_file', description='Name to the .apk to run', mandatory=True),
        Parameter('uninstall_required', kind=bool, default=False,
                  description='Set to true if the package should be uninstalled'),
        Parameter('wait_time_seconds', kind=int, default=0,
                  description='Seconds to wait before testing if the app is still alive')
    ]

    def setup(self, context):
        apk_file = context.resolver.get(File(self, self.apk_file))
        self.package = ApkInfo(apk_file).package  # pylint: disable=attribute-defined-outside-init

        self.logger.info('Installing {}'.format(apk_file))
        return self.device.install(apk_file)

    def run(self, context):
        self.logger.info('Starting {}'.format(self.package))
        self.device.execute('am start -W {}'.format(self.package))

        self.logger.info('Waiting {} seconds'.format(self.wait_time_seconds))
        sleep(self.wait_time_seconds)

    def update_result(self, context):
        app_is_running = bool([p for p in self.device.ps() if p.name == self.package])
        context.result.add_metric('ran_successfully', app_is_running)

    def teardown(self, context):
        if self.uninstall_required:
            self.logger.info('Uninstalling {}'.format(self.package))
            self.device.execute('pm uninstall {}'.format(self.package))
