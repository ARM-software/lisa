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

import re

from wlauto import LinuxDevice, Parameter
from wlauto.exceptions import DeviceError
from wlauto.core.device import RuntimeParameter
from wlauto.utils.misc import convert_new_lines
from wlauto.utils.types import boolean


class ChromeOsDevice(LinuxDevice):

    name = "chromeos_test_image"
    description = """
    Chrome OS test image device. Use this if you are working on a Chrome OS device with a test
    image. An off the shelf device will not work with this device interface.

    More information on how to build a Chrome OS test image can be found here:

        https://www.chromium.org/chromium-os/developer-guide#TOC-Build-a-disk-image-for-your-board

    """

    platform = 'chromeos'
    abi = 'armeabi'
    has_gpu = True
    default_timeout = 100

    parameters = [
        Parameter('core_names', default=[], override=True),
        Parameter('core_clusters', default=[], override=True),
        Parameter('username', default='root', override=True),
        Parameter('password_prompt', default='Password:', override=True),
        Parameter('binaries_directory', default='/usr/local/bin', override=True),
        Parameter('working_directory', default='/home/root/wa-working', override=True),
    ]

    runtime_parameters = [
        RuntimeParameter('ui', 'get_ui_status', 'set_ui_status', value_name='status'),
    ]

    def __init__(self, **kwargs):
        super(ChromeOsDevice, self).__init__(**kwargs)
        self.ui_status = None

    def validate(self):
        # pylint: disable=access-member-before-definition,attribute-defined-outside-init
        if self.password is None and not self.keyfile:
            self.password = 'test0000'

    def initialize(self, context, *args, **kwargs):
        if self.busybox == 'busybox':
            self.logger.debug('Busybox already installed on the device: replacing with wa version')
            self.uninstall('busybox')
            self.busybox = self.deploy_busybox(context)

    def get_ui_status(self):
        return self.ui_status

    def set_ui_status(self, status):
        self.ui_status = boolean(status)
        if self.ui_status is None:
            pass
        elif self.ui_status:
            try:
                self.execute('start ui')
            except DeviceError:
                pass
        else:
            try:
                self.execute('stop ui')
            except DeviceError:
                pass

    def stop(self):
        if self.ui_status is None:
            pass
        elif not self.ui_status:
            try:
                self.execute('start ui')
            except DeviceError:
                pass
        else:
            pass
        self.ui_status = None

