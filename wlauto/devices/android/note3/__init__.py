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


import time

from wlauto import AndroidDevice, Parameter
from wlauto.exceptions import TimeoutError
from wlauto.utils.android import adb_shell


class Note3Device(AndroidDevice):

    name = 'Note3'
    description = """
    Adapter for Galaxy Note 3.

    To be able to use Note3 in WA, the following must be true:

        - USB Debugging Mode is enabled.
        - Generate USB debugging authorisation for the host machine

    """

    parameters = [
        Parameter('core_names', default=['A15', 'A15', 'A15', 'A15'], override=True),
        Parameter('core_clusters', default=[0, 0, 0, 0], override=True),
        Parameter('working_directory', default='/storage/sdcard0/wa-working', override=True),
    ]

    def __init__(self, **kwargs):
        super(Note3Device, self).__init__(**kwargs)
        self._just_rebooted = False

    def initialize(self, context):
        self.execute('svc power stayon true', check_exit_code=False)

    def reset(self):
        super(Note3Device, self).reset()
        self._just_rebooted = True

    def hard_reset(self):
        super(Note3Device, self).hard_reset()
        self._just_rebooted = True

    def connect(self):  # NOQA pylint: disable=R0912
        super(Note3Device, self).connect()
        if self._just_rebooted:
            self.logger.debug('Waiting for boot to complete...')
            # On the Note 3, adb connection gets reset some time after booting.
            # This  causes errors during execution. To prevent this, open a shell
            # session and wait for it to be killed. Once its killed, give adb
            # enough time to restart, and then the device should be ready.
            try:
                adb_shell(self.adb_name, '', timeout=20)  # pylint: disable=no-member
                time.sleep(5)  # give adb time to re-initialize
            except TimeoutError:
                pass  # timed out waiting for the session to be killed -- assume not going to be.

            self.logger.debug('Boot completed.')
            self._just_rebooted = False
        # Swipe upwards to unlock the screen.
        time.sleep(self.long_delay)
        self.execute('input touchscreen swipe 540 1600 560 800 ')
