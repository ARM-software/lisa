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

# pylint: disable=E1101

import time

from wlauto import Workload, Parameter
from wlauto.exceptions import WorkloadError, ConfigError


class IdleWorkload(Workload):

    name = 'idle'
    description = """
    Do nothing for the specified duration.

    On android devices, this may optionally stop the Android run time, if
    ``stop_android`` is set to ``True``.

    .. note:: This workload requires the device to be rooted.

    """

    parameters = [
        Parameter('duration', kind=int, default=20,
                  description='Specifies the duration, in seconds, of this workload.'),
        Parameter('stop_android', kind=bool, default=False,
                  description='Specifies whether the Android run time should be stopped. '
                              '(Can be set only for Android devices).'),
    ]

    def setup(self, context):
        if self.stop_android:
            if self.device.platform != 'android':
                raise ConfigError('stop_android can only be set for Android devices')
            if not self.device.is_rooted:
                raise WorkloadError('Idle workload requires the device to be rooted in order to stop Android.')

    def run(self, context):
        self.logger.debug('idling...')
        if self.stop_android:
            timeout = self.duration + 10
            self.device.execute('stop && sleep {} && start'.format(self.duration),
                                timeout=timeout, as_root=True)
        else:
            time.sleep(self.duration)

    def teardown(self, context):
        if self.stop_android:
            self.logger.debug('Waiting for Android restart to complete...')
            # Wait for the boot animation to start and then to finish.
            while self.device.execute('getprop init.svc.bootanim').strip() == 'stopped':
                time.sleep(0.2)
            while self.device.execute('getprop init.svc.bootanim').strip() == 'running':
                time.sleep(1)
