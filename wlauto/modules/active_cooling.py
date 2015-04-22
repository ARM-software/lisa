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


from wlauto import Module, Parameter
from wlauto.utils.serial_port import open_serial_connection


class MbedFanActiveCooling(Module):

    name = 'mbed-fan'
    description = 'Controls a cooling fan via an mbed connected to a serial port.'

    capabilities = ['active_cooling']

    parameters = [
        Parameter('port', default='/dev/ttyACM0',
                  description="""The serial port for the active cooling solution (see above)."""),
        Parameter('buad', kind=int, default=115200,
                  description="""Baud for the serial port (see above)."""),
        Parameter('fan_pin', kind=int, default=0,
                  description="""Which controller pin on the mbed the fan for the active cooling solution is
                  connected to (controller pin 0 is physical pin 22 on the mbed)."""),
    ]

    timeout = 30

    def start_active_cooling(self):
        with open_serial_connection(timeout=self.timeout,
                                    port=self.port,
                                    baudrate=self.buad) as target:
            target.sendline('motor_{}_1'.format(self.fan_pin))

    def stop_active_cooling(self):
        with open_serial_connection(timeout=self.timeout,
                                    port=self.port,
                                    baudrate=self.buad) as target:
            target.sendline('motor_{}_0'.format(self.fan_pin))


class OdroidXU3ctiveCooling(Module):

    name = 'odroidxu3-fan'
    description = """
    Enabled active cooling by controling the fan an Odroid XU3

    .. note:: depending on the kernel used, it may not be possible to turn the fan
              off completely; in such situations, the fan will be set to its minimum
              speed.

    """

    capabilities = ['active_cooling']

    def start_active_cooling(self):
        self.owner.set_sysfile_value('/sys/devices/odroid_fan.15/fan_mode', 0, verify=False)
        self.owner.set_sysfile_value('/sys/devices/odroid_fan.15/pwm_duty', 255, verify=False)

    def stop_active_cooling(self):
        self.owner.set_sysfile_value('/sys/devices/odroid_fan.15/fan_mode', 0, verify=False)
        self.owner.set_sysfile_value('/sys/devices/odroid_fan.15/pwm_duty', 1, verify=False)
