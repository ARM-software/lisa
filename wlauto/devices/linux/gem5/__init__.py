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

# Original implementation by Rene de Jong. Updated by Sascha Bischoff.

import logging

from wlauto import LinuxDevice, Parameter
from wlauto.common.gem5.device import BaseGem5Device
from wlauto.utils import types


class Gem5LinuxDevice(BaseGem5Device, LinuxDevice):
    """
    Implements gem5 Linux device.

    This class allows a user to connect WA to a simulation using gem5. The
    connection to the device is made using the telnet connection of the
    simulator, and is used for all commands. The simulator does not have ADB
    support, and therefore we need to fall back to using standard shell
    commands.

    Files are copied into the simulation using a VirtIO 9P device in gem5. Files
    are copied out of the simulated environment using the m5 writefile command
    within the simulated system.

    When starting the workload run, the simulator is automatically started by
    Workload Automation, and a connection to the simulator is established. WA
    will then wait for Android to boot on the simulated system (which can take
    hours), prior to executing any other commands on the device. It is also
    possible to resume from a checkpoint when starting the simulation. To do
    this, please append the relevant checkpoint commands from the gem5
    simulation script to the gem5_discription argument in the agenda.

    Host system requirements:
        * VirtIO support. We rely on diod on the host system. This can be
          installed on ubuntu using the following command:

                sudo apt-get install diod

    Guest requirements:
        * VirtIO support. We rely on VirtIO to move files into the simulation.
          Please make sure that the following are set in the kernel
          configuration:

                CONFIG_NET_9P=y

                CONFIG_NET_9P_VIRTIO=y

                CONFIG_9P_FS=y

                CONFIG_9P_FS_POSIX_ACL=y

                CONFIG_9P_FS_SECURITY=y

                CONFIG_VIRTIO_BLK=y

        * m5 binary. Please make sure that the m5 binary is on the device and
          can by found in the path.
    """

    name = 'gem5_linux'
    platform = 'linux'

    parameters = [
        Parameter('core_names', default=[], override=True),
        Parameter('core_clusters', default=[], override=True),
        Parameter('host', default='localhost', override=True,
                  description='Host name or IP address for the device.'),
        Parameter('login_prompt', kind=types.list_of_strs,
                  default=['login:', 'AEL login:', 'username:'],
                  mandatory=False),
        Parameter('login_password_prompt', kind=types.list_of_strs,
                  default=['password:'], mandatory=False),
    ]

    # Overwritten from Device. For documentation, see corresponding method in
    # Device.

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('Gem5LinuxDevice')
        LinuxDevice.__init__(self, **kwargs)
        BaseGem5Device.__init__(self)

    def login_to_device(self):
        # Wait for the login prompt
        prompt = self.login_prompt + [self.sckt.UNIQUE_PROMPT]
        i = self.sckt.expect(prompt, timeout=10)
        # Check if we are already at a prompt, or if we need to log in.
        if i < len(prompt) - 1:
            self.sckt.sendline("{}".format(self.username))
            password_prompt = self.login_password_prompt + [r'# ', self.sckt.UNIQUE_PROMPT]
            j = self.sckt.expect(password_prompt, timeout=self.delay)
            if j < len(password_prompt) - 2:
                self.sckt.sendline("{}".format(self.password))
                self.sckt.expect([r'# ', self.sckt.UNIQUE_PROMPT], timeout=self.delay)

    def capture_screen(self, filepath):
        if BaseGem5Device.capture_screen(self, filepath):
            return

        # If we didn't manage to do the above, call the parent class.
        self.logger.warning("capture_screen: falling back to parent class implementation")
        LinuxDevice.capture_screen(self, filepath)

    def initialize(self, context):
        self.resize_shell()
        self.deploy_m5(context, force=False)
