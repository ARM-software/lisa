#
#    Copyright 2015-2018 ARM Limited
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
import os
import time
import tarfile
import shutil
from subprocess import CalledProcessError

from devlib.module import HardRestModule, BootModule, FlashModule
from devlib.exception import TargetError, TargetStableError, HostError
from devlib.utils.serial_port import open_serial_connection, pulse_dtr, write_characters
from devlib.utils.uefi import UefiMenu, UefiConfig
from devlib.utils.uboot import UbootMenu


OLD_AUTOSTART_MESSAGE = 'Press Enter to stop auto boot...'
AUTOSTART_MESSAGE = 'Hit any key to stop autoboot:'
POWERUP_MESSAGE = 'Powering up system...'
DEFAULT_MCC_PROMPT = 'Cmd>'


class VexpressDtrHardReset(HardRestModule):

    name = 'vexpress-dtr'
    stage = 'early'

    @staticmethod
    def probe(target):
        return True

    def __init__(self, target, port='/dev/ttyS0', baudrate=115200,
                 mcc_prompt=DEFAULT_MCC_PROMPT, timeout=300):
        super(VexpressDtrHardReset, self).__init__(target)
        self.port = port
        self.baudrate = baudrate
        self.mcc_prompt = mcc_prompt
        self.timeout = timeout

    def __call__(self):
        try:
            if self.target.is_connected:
                self.target.execute('sync')
        except (TargetError, CalledProcessError):
            pass
        with open_serial_connection(port=self.port,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0,
                                    get_conn=True) as (_, conn):
            pulse_dtr(conn, state=True, duration=0.1)  # TRM specifies a pulse of >=100ms


class VexpressReboottxtHardReset(HardRestModule):

    name = 'vexpress-reboottxt'
    stage = 'early'

    @staticmethod
    def probe(target):
        return True

    def __init__(self, target,
                 port='/dev/ttyS0', baudrate=115200,
                 path='/media/VEMSD',
                 mcc_prompt=DEFAULT_MCC_PROMPT, timeout=30, short_delay=1):
        super(VexpressReboottxtHardReset, self).__init__(target)
        self.port = port
        self.baudrate = baudrate
        self.path = path
        self.mcc_prompt = mcc_prompt
        self.timeout = timeout
        self.short_delay = short_delay
        self.filepath = os.path.join(path, 'reboot.txt')

    def __call__(self):
        try:
            if self.target.is_connected:
                self.target.execute('sync')
        except (TargetError, CalledProcessError):
            pass

        if not os.path.exists(self.path):
            self.logger.debug('{} does not exisit; attempting to mount...'.format(self.path))
            with open_serial_connection(port=self.port,
                                        baudrate=self.baudrate,
                                        timeout=self.timeout,
                                        init_dtr=0) as tty:
                wait_for_vemsd(self.path, tty, self.mcc_prompt, self.short_delay)
        with open(self.filepath, 'w'):
            pass


class VexpressBootModule(BootModule):

    stage = 'early'

    @staticmethod
    def probe(target):
        return True

    def __init__(self, target, uefi_entry=None,
                 port='/dev/ttyS0', baudrate=115200,
                 mcc_prompt=DEFAULT_MCC_PROMPT,
                 timeout=120, short_delay=1):
        super(VexpressBootModule, self).__init__(target)
        self.port = port
        self.baudrate = baudrate
        self.uefi_entry = uefi_entry
        self.mcc_prompt = mcc_prompt
        self.timeout = timeout
        self.short_delay = short_delay

    def __call__(self):
        with open_serial_connection(port=self.port,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0) as tty:
            self.get_through_early_boot(tty)
            self.perform_boot_sequence(tty)
            self.wait_for_android_prompt(tty)

    def perform_boot_sequence(self, tty):
        raise NotImplementedError()

    def get_through_early_boot(self, tty):
        self.logger.debug('Establishing initial state...')
        tty.sendline('')
        i = tty.expect([AUTOSTART_MESSAGE, OLD_AUTOSTART_MESSAGE, POWERUP_MESSAGE, self.mcc_prompt])
        if i == 3:
            self.logger.debug('Saw MCC prompt.')
            time.sleep(self.short_delay)
            tty.sendline('reboot')
        elif i == 2:
            self.logger.debug('Saw powering up message (assuming soft reboot).')
        else:
            self.logger.debug('Saw auto boot message.')
            tty.sendline('')
            time.sleep(self.short_delay)
            # could be either depending on where in the boot we are
            tty.sendline('reboot')
            tty.sendline('reset')

    def get_uefi_menu(self, tty):
        menu = UefiMenu(tty)
        self.logger.debug('Waiting for UEFI menu...')
        menu.wait(timeout=self.timeout)
        return menu

    def wait_for_android_prompt(self, tty):
        self.logger.debug('Waiting for the Android prompt.')
        tty.expect(self.target.shell_prompt, timeout=self.timeout)
        # This delay is needed to allow the platform some time to finish
        # initilizing; querying the ip address too early from connect() may
        # result in a bogus address being assigned to eth0.
        time.sleep(5)


class VexpressUefiBoot(VexpressBootModule):

    name = 'vexpress-uefi'

    def __init__(self, target, uefi_entry,
                 image, fdt, bootargs, initrd,
                 *args, **kwargs):
        super(VexpressUefiBoot, self).__init__(target, uefi_entry=uefi_entry,
                                               *args, **kwargs)
        self.uefi_config = self._create_config(image, fdt, bootargs, initrd)

    def perform_boot_sequence(self, tty):
        menu = self.get_uefi_menu(tty)
        try:
            menu.select(self.uefi_entry)
        except LookupError:
            self.logger.debug('{} UEFI entry not found.'.format(self.uefi_entry))
            self.logger.debug('Attempting to create one using default flasher configuration.')
            menu.create_entry(self.uefi_entry, self.uefi_config)
            menu.select(self.uefi_entry)

    def _create_config(self, image, fdt, bootargs, initrd):  # pylint: disable=R0201
        config_dict = {
            'image_name': image,
            'image_args': bootargs,
            'initrd': initrd,
        }

        if fdt:
            config_dict['fdt_support'] = True
            config_dict['fdt_path'] = fdt
        else:
            config_dict['fdt_support'] = False

        return UefiConfig(config_dict)


class VexpressUefiShellBoot(VexpressBootModule):

    name = 'vexpress-uefi-shell'

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, target, uefi_entry='^Shell$',
                 efi_shell_prompt='Shell>',
                 image='kernel', bootargs=None,
                 *args, **kwargs):
        super(VexpressUefiShellBoot, self).__init__(target, uefi_entry=uefi_entry,
                                                    *args, **kwargs)
        self.efi_shell_prompt = efi_shell_prompt
        self.image = image
        self.bootargs = bootargs

    def perform_boot_sequence(self, tty):
        menu = self.get_uefi_menu(tty)
        try:
            menu.select(self.uefi_entry)
        except LookupError:
            raise TargetStableError('Did not see "{}" UEFI entry.'.format(self.uefi_entry))
        tty.expect(self.efi_shell_prompt, timeout=self.timeout)
        if self.bootargs:
            tty.sendline('')  # stop default boot
            time.sleep(self.short_delay)
            efi_shell_command = '{} {}'.format(self.image, self.bootargs)
            self.logger.debug(efi_shell_command)
            write_characters(tty, efi_shell_command)
            tty.sendline('\r\n')


class VexpressUBoot(VexpressBootModule):

    name = 'vexpress-u-boot'

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, target, env=None,
                 *args, **kwargs):
        super(VexpressUBoot, self).__init__(target, *args, **kwargs)
        self.env = env

    def perform_boot_sequence(self, tty):
        if self.env is None:
            return  # Will boot automatically

        menu = UbootMenu(tty)
        self.logger.debug('Waiting for U-Boot prompt...')
        menu.open(timeout=120)
        for var, value in self.env.items():
            menu.setenv(var, value)
        menu.boot()


class VexpressBootmon(VexpressBootModule):

    name = 'vexpress-bootmon'

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, target,
                 image, fdt, initrd, bootargs,
                 uses_bootscript=False,
                 bootmon_prompt='>',
                 *args, **kwargs):
        super(VexpressBootmon, self).__init__(target, *args, **kwargs)
        self.image = image
        self.fdt = fdt
        self.initrd = initrd
        self.bootargs = bootargs
        self.uses_bootscript = uses_bootscript
        self.bootmon_prompt = bootmon_prompt

    def perform_boot_sequence(self, tty):
        if self.uses_bootscript:
            return  # Will boot automatically

        time.sleep(self.short_delay)
        tty.expect(self.bootmon_prompt, timeout=self.timeout)
        with open_serial_connection(port=self.port,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0) as tty_conn:
            write_characters(tty_conn, 'fl linux fdt {}'.format(self.fdt))
            write_characters(tty_conn, 'fl linux initrd {}'.format(self.initrd))
            write_characters(tty_conn, 'fl linux boot {} {}'.format(self.image,
                                                                    self.bootargs))


class VersatileExpressFlashModule(FlashModule):

    name = 'vexpress-vemsd'
    description = """
    Enables flashing of kernels and firmware to ARM Versatile Express devices.

    This modules enables flashing of image bundles or individual images to ARM
    Versatile Express-based devices (e.g. JUNO) via host-mounted MicroSD on the
    board.

    The bundle, if specified, must reflect the directory structure of the MicroSD
    and will be extracted directly into the location it is mounted on the host. The
    images, if  specified, must be a dict mapping the absolute path of the image on
    the host to the destination path within the board's MicroSD; the destination path
    may be either absolute, or relative to the MicroSD mount location.

    """

    stage = 'early'

    @staticmethod
    def probe(target):
        if not target.has('hard_reset'):
            return False
        return True

    def __init__(self, target, vemsd_mount, mcc_prompt=DEFAULT_MCC_PROMPT, timeout=30, short_delay=1):
        super(VersatileExpressFlashModule, self).__init__(target)
        self.vemsd_mount = vemsd_mount
        self.mcc_prompt = mcc_prompt
        self.timeout = timeout
        self.short_delay = short_delay

    def __call__(self, image_bundle=None, images=None, bootargs=None):
        self.target.hard_reset()
        with open_serial_connection(port=self.target.platform.serial_port,
                                    baudrate=self.target.platform.baudrate,
                                    timeout=self.timeout,
                                    init_dtr=0) as tty:
            # pylint: disable=no-member
            i = tty.expect([self.mcc_prompt, AUTOSTART_MESSAGE, OLD_AUTOSTART_MESSAGE])
            if i:
                tty.sendline('')  # pylint: disable=no-member
            wait_for_vemsd(self.vemsd_mount, tty, self.mcc_prompt, self.short_delay)
        try:
            if image_bundle:
                self._deploy_image_bundle(image_bundle)
            if images:
                self._overlay_images(images)
            os.system('sync')
        except (IOError, OSError) as e:
            msg = 'Could not deploy images to {}; got: {}'
            raise TargetStableError(msg.format(self.vemsd_mount, e))
        self.target.boot()
        self.target.connect(timeout=30)

    def _deploy_image_bundle(self, bundle):
        self.logger.debug('Validating {}'.format(bundle))
        validate_image_bundle(bundle)
        self.logger.debug('Extracting {} into {}...'.format(bundle, self.vemsd_mount))
        with tarfile.open(bundle) as tar:
            tar.extractall(self.vemsd_mount)

    def _overlay_images(self, images):
        for dest, src in images.items():
            dest = os.path.join(self.vemsd_mount, dest)
            self.logger.debug('Copying {} to {}'.format(src, dest))
            shutil.copy(src, dest)


# utility functions

def validate_image_bundle(bundle):
    if not tarfile.is_tarfile(bundle):
        raise HostError('Image bundle {} does not appear to be a valid TAR file.'.format(bundle))
    with tarfile.open(bundle) as tar:
        try:
            tar.getmember('config.txt')
        except KeyError:
            try:
                tar.getmember('./config.txt')
            except KeyError:
                msg = 'Tarball {} does not appear to be a valid image bundle (did not see config.txt).'
                raise HostError(msg.format(bundle))


def wait_for_vemsd(vemsd_mount, tty, mcc_prompt=DEFAULT_MCC_PROMPT, short_delay=1, retries=3):
    attempts = 1 + retries
    path = os.path.join(vemsd_mount, 'config.txt')
    if os.path.exists(path):
        return
    for _ in range(attempts):
        tty.sendline('')  # clear any garbage
        tty.expect(mcc_prompt, timeout=short_delay)
        tty.sendline('usb_on')
        time.sleep(short_delay * 3)
        if os.path.exists(path):
            return
    raise TargetStableError('Could not mount {}'.format(vemsd_mount))
