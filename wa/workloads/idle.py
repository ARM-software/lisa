#    Copyright 2014-2017 ARM Limited
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

from wa import Workload, Parameter, WorkloadError, ConfigError


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
                  description='''
                  Specifies the duration, in seconds, of this workload.
                  '''),
        Parameter('screen_off', kind=bool, default=False,
                  description='''
                  Ensure that the screen is off before idling.

                  .. note:: Make sure screen lock is disabled on the target!

                  '''),
        Parameter('stop_android', kind=bool, default=False,
                  description='''
                  Specifies whether the Android run time should be stopped.
                  (Can be set only for Android devices).
                  '''),
    ]

    def initialize(self, context):
        self.old_screen_state = None
        if self.target.os == 'android':
            if self.stop_android and not self.target.is_rooted:
                msg = 'Idle workload requires the device to be rooted in order to stop Android.'
                raise WorkloadError(msg)
        else:
            if self.stop_android or self.screen_off:
                msg = 'stop_android/screen_off can only be set for Android devices'
                raise ConfigError(msg)

    def setup(self, context):
        if self.target.os == 'android':
            self.old_screen_state = self.target.is_screen_on()
            self.target.ensure_screen_is_on()
            self.target.homescreen()
            if self.screen_off:
                self.target.ensure_screen_is_off()

    def run(self, context):
        self.logger.debug('idling...')
        if self.stop_android:
            timeout = self.duration + 10
            self.target.execute('stop && sleep {} && start'.format(self.duration),
                                timeout=timeout, as_root=True)
        else:
            self.target.sleep(self.duration)

    def teardown(self, context):
        if self.stop_android:
            self.logger.debug('Waiting for Android restart to complete...')
            # Wait for the boot animation to start and then to finish.
            while self.target.getprop('init.svc.bootanim') == 'stopped':
                self.target.sleep(0.2)
            while self.target.getprop('init.svc.bootanim') == 'running':
                self.target.sleep(1)
        if self.screen_off and self.old_screen_state:
            self.target.ensure_screen_is_on()
        elif (self.target.os == 'android'
                and not self.screen_off and not self.old_screen_state):
            self.target.ensure_screen_is_off()
