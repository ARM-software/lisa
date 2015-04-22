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

import time

from wlauto import Module, Parameter
from wlauto.exceptions import DeviceError
from wlauto.utils.netio import KshellConnection


class NetioSwitchReset(Module):

    #pylint: disable=E1101
    name = 'netio_switch'
    description = """
    Enables hard reset of devices connected to a Netio ethernet power switch
    """
    capabilities = ['reset_power']

    parameters = [
        Parameter('host', default='ippowerbar',
                  description='IP address or DNS name of the Netio power switch.'),
        Parameter('port', kind=int, default=1234,
                  description='Port on which KSHELL is listening.'),
        Parameter('username', default='admin',
                  description='User name for the administrator on the Netio.'),
        Parameter('password', default='admin',
                  description='User name for the administrator on the Netio.'),
        Parameter('psu', kind=int, default=1,
                  description='The device port number on the Netio, i.e. which '
                              'PSU port the device is connected to.'),
    ]

    def hard_reset(self):
        try:
            conn = KshellConnection(host=self.host, port=self.port)
            conn.login(self.username, self.password)
            conn.disable_port(self.psu)
            time.sleep(2)
            conn.enable_port(self.psu)
            conn.close()
        except Exception as e:
            raise DeviceError('Could not reset power: {}'.format(e))
