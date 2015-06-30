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
import os
import re
import time

import pexpect

from wlauto import BigLittleDevice, Parameter
from wlauto.exceptions import DeviceError
from wlauto.utils.serial_port import open_serial_connection, pulse_dtr
from wlauto.utils.android import adb_connect, adb_disconnect, adb_list_devices
from wlauto.utils.uefi import UefiMenu, UefiConfig
from wlauto.utils.uboot import UbootMenu


AUTOSTART_MESSAGE = 'Press Enter to stop auto boot...'


class Juno(BigLittleDevice):

    name = 'juno'
    description = """
    ARM Juno next generation big.LITTLE development platform.
    """

    capabilities = ['reset_power']

    has_gpu = True

    core_modules = [
        'vexpress',
    ]

    parameters = [
        Parameter('retries', kind=int, default=2,
                  description="""Specifies the number of times the device will attempt to recover
                                 (normally, with a hard reset) if it detects that something went wrong."""),

        Parameter('microsd_mount_point', default='/media/JUNO',
                  description='Location at which the device\'s MicroSD card will be mounted.'),
        Parameter('port', default='/dev/ttyS0', description='Serial port on which the device is connected.'),
        Parameter('baudrate', kind=int, default=115200, description='Serial connection baud.'),
        Parameter('timeout', kind=int, default=300, description='Serial connection timeout.'),
        Parameter('core_names', default=['a53', 'a53', 'a53', 'a53', 'a57', 'a57'], override=True),
        Parameter('core_clusters', default=[0, 0, 0, 0, 1, 1], override=True),

        Parameter('bootloader', default='uefi', allowed_values=['uefi', 'u-boot'],
                  description="""Bootloader used on the device."""),
        Parameter('actually_disconnect', kind=bool, default=False,
                  description="""
                  Actually perfom "adb disconnect" on closing the connection to the device.
                  """),

        # VExpress flasher expects a device to have these:
        Parameter('uefi_entry', default='WA',
                  description='The name of the entry to use (will be created if does not exist).'),
        Parameter('uefi_config', kind=UefiConfig,
                  description='''Specifies the configuration for the UEFI entry for his device. In an
                                 entry specified by ``uefi_entry`` parameter doesn't exist in UEFI menu,
                                 it will be created using this config. This configuration will also be
                                 used, when flashing new images.''',
                  default={
                      'image_name': 'Image',
                      'image_args': 'console=ttyAMA0,115200 '
                                    'earlyprintk=pl011,0x7ff80000 '
                                    'verbose debug init=/init '
                                    'root=/dev/sda1 rw ip=dhcp rootwait',
                      'fdt_support': True,
                  }
                  ),
    ]

    short_delay = 1
    firmware_prompt = 'Cmd>'

    def boot(self, **kwargs):
        self.logger.debug('Resetting the device.')
        self.reset()
        if self.bootloader == 'uefi':
            self._boot_via_uefi()
        else:
            self._boot_via_uboot(**kwargs)

    def _boot_via_uboot(self, **kwargs):
        if not kwargs:
            # Standard linaro configuration will proceed directly to the kernel
            return
        with open_serial_connection(port=self.port,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0) as target:
            menu = UbootMenu(target)
            self.logger.debug('Waiting for U-Boot prompt...')
            menu.open(timeout=120)
            for var, value in kwargs.iteritems():
                menu.setenv(var, value)
            menu.boot()

    def _boot_via_uefi(self):
        with open_serial_connection(port=self.port,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0) as target:
            menu = UefiMenu(target)
            self.logger.debug('Waiting for UEFI menu...')
            menu.open(timeout=120)
            try:
                menu.select(self.uefi_entry)
            except LookupError:
                self.logger.debug('{} UEFI entry not found.'.format(self.uefi_entry))
                self.logger.debug('Attempting to create one using default flasher configuration.')
                menu.create_entry(self.uefi_entry, self.uefi_config)
                menu.select(self.uefi_entry)
            self.logger.debug('Waiting for the Android prompt.')
            target.expect(self.android_prompt, timeout=self.timeout)

    def connect(self):
        if not self._is_ready:
            if not self.adb_name:  # pylint: disable=E0203
                with open_serial_connection(timeout=self.timeout,
                                            port=self.port,
                                            baudrate=self.baudrate,
                                            init_dtr=0) as target:
                    target.sendline('')
                    self.logger.debug('Waiting for the Android prompt.')
                    target.expect(self.android_prompt)

                    self.logger.debug('Waiting for IP address...')
                    wait_start_time = time.time()
                    while True:
                        target.sendline('ip addr list eth0')
                        time.sleep(1)
                        try:
                            target.expect('inet ([1-9]\d*.\d+.\d+.\d+)', timeout=10)
                            self.adb_name = target.match.group(1) + ':5555'  # pylint: disable=W0201
                            break
                        except pexpect.TIMEOUT:
                            pass  # We have our own timeout -- see below.
                        if (time.time() - wait_start_time) > self.ready_timeout:
                            raise DeviceError('Could not acquire IP address.')

            if self.adb_name in adb_list_devices():
                adb_disconnect(self.adb_name)
            adb_connect(self.adb_name, timeout=self.timeout)
            super(Juno, self).connect()  # wait for boot to complete etc.
            self._is_ready = True

    def disconnect(self):
        if self._is_ready:
            super(Juno, self).disconnect()
            if self.actually_disconnect:
                adb_disconnect(self.adb_name)
            self._is_ready = False

    def reset(self):
        # Currently, reboot is not working in Android on Juno, so
        # perfrom a ahard reset instead
        self.hard_reset()

    def hard_reset(self):
        self.disconnect()
        self.adb_name = None  # Force re-acquire IP address on reboot. pylint: disable=attribute-defined-outside-init
        with open_serial_connection(port=self.port,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0,
                                    get_conn=True) as (target, conn):
            pulse_dtr(conn, state=True, duration=0.1)  # TRM specifies a pulse of >=100ms

            i = target.expect([AUTOSTART_MESSAGE, self.firmware_prompt])
            if i:
                self.logger.debug('Saw firmware prompt.')
                time.sleep(self.short_delay)
                target.sendline('reboot')
            else:
                self.logger.debug('Saw auto boot message.')

    def wait_for_microsd_mount_point(self, target, timeout=100):
        attempts = 1 + self.retries
        if os.path.exists(os.path.join(self.microsd_mount_point, 'config.txt')):
            return

        self.logger.debug('Waiting for VExpress MicroSD to mount...')
        for i in xrange(attempts):
            if i:  # Do not reboot on the first attempt.
                target.sendline('reboot')
            for _ in xrange(timeout):
                time.sleep(self.short_delay)
                if os.path.exists(os.path.join(self.microsd_mount_point, 'config.txt')):
                    return
        raise DeviceError('Did not detect MicroSD mount on {}'.format(self.microsd_mount_point))

    def get_android_id(self):
        # Android ID currenlty not set properly in Juno Android builds.
        return 'abad1deadeadbeef'

