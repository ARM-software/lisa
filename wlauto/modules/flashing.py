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

# pylint: disable=attribute-defined-outside-init
import os
import time
import tarfile
import tempfile
import shutil

from wlauto import Module
from wlauto.exceptions import ConfigError, DeviceError
from wlauto.utils.android import fastboot_flash_partition, fastboot_command
from wlauto.utils.serial_port import open_serial_connection
from wlauto.utils.uefi import UefiMenu
from wlauto.utils.misc import merge_dicts


class Flasher(Module):
    """
    Implements a mechanism for flashing a device. The images to be flashed can be
    specified either as a tarball "image bundle" (in which case instructions for
    flashing are provided as flasher-specific metadata also in the bundle), or as
    individual image files, in which case instructions for flashing as specified
    as part of  flashing config.

    .. note:: It is important that when resolving configuration, concrete flasher
              implementations prioritise settings specified in the config over those
              in the bundle (if they happen to clash).

    """

    capabilities = ['flash']

    def flash(self, image_bundle=None, images=None):
        """
        Flashes the specified device using the specified config. As a post condition,
        the device must be ready to run workloads upon returning from this method (e.g.
        it must be fully-booted into the OS).

        """
        raise NotImplementedError()


class FastbootFlasher(Flasher):

    name = 'fastboot'
    description = """
    Enables automated flashing of images using the fastboot utility.

    To use this flasher, a set of image files to be flused are required.
    In addition a mapping between partitions and image file is required. There are two ways
    to specify those requirements:

    - Image mapping: In this mode, a mapping between partitions and images is given in the agenda.
    - Image Bundle: In This mode a tarball is specified, which must contain all image files as well
      as well as a partition file, named ``partitions.txt`` which contains the mapping between
      partitions and images.

    The format of ``partitions.txt`` defines one mapping per line as such: ::

        kernel zImage-dtb
        ramdisk ramdisk_image

    """

    delay = 0.5
    serial_timeout = 30
    partitions_file_name = 'partitions.txt'

    def flash(self, image_bundle=None, images=None):
        self.prelude_done = False
        to_flash = {}
        if image_bundle:  # pylint: disable=access-member-before-definition
            image_bundle = expand_path(image_bundle)
            to_flash = self._bundle_to_images(image_bundle)
        to_flash = merge_dicts(to_flash, images or {}, should_normalize=False)
        for partition, image_path in to_flash.iteritems():
            self.logger.debug('flashing {}'.format(partition))
            self._flash_image(self.owner, partition, expand_path(image_path))
        fastboot_command('reboot')

    def _validate_image_bundle(self, image_bundle):
        if not tarfile.is_tarfile(image_bundle):
            raise ConfigError('File {} is not a tarfile'.format(image_bundle))
        with tarfile.open(image_bundle) as tar:
            files = [tf.name for tf in tar.getmembers()]
            if not any(pf in files for pf in (self.partitions_file_name, '{}/{}'.format(files[0], self.partitions_file_name))):
                ConfigError('Image bundle does not contain the required partition file (see documentation)')

    def _bundle_to_images(self, image_bundle):
        """
        Extracts the bundle to a temporary location and creates a mapping between the contents of the bundle
        and images to be flushed.
        """
        self._validate_image_bundle(image_bundle)
        extract_dir = tempfile.mkdtemp()
        with tarfile.open(image_bundle) as tar:
            tar.extractall(path=extract_dir)
            files = [tf.name for tf in tar.getmembers()]
            if self.partitions_file_name not in files:
                extract_dir = os.path.join(extract_dir, files[0])
        partition_file = os.path.join(extract_dir, self.partitions_file_name)
        return get_mapping(extract_dir, partition_file)

    def _flash_image(self, device, partition, image_path):
        if not self.prelude_done:
            self._fastboot_prelude(device)
        fastboot_flash_partition(partition, image_path)
        time.sleep(self.delay)

    def _fastboot_prelude(self, device):
        with open_serial_connection(port=device.port,
                                    baudrate=device.baudrate,
                                    timeout=self.serial_timeout,
                                    init_dtr=0,
                                    get_conn=False) as target:
            device.reset()
            time.sleep(self.delay)
            target.sendline(' ')
            time.sleep(self.delay)
            target.sendline('fast')
            time.sleep(self.delay)
        self.prelude_done = True


class VersatileExpressFlasher(Flasher):

    name = 'vexpress'
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

    def flash(self, image_bundle=None, images=None, recreate_uefi_entry=True):  # pylint: disable=arguments-differ
        device = self.owner
        if not hasattr(device, 'port') or not hasattr(device, 'microsd_mount_point'):
            msg = 'Device {} does not appear to support VExpress flashing.'
            raise ConfigError(msg.format(device.name))
        with open_serial_connection(port=device.port,
                                    baudrate=device.baudrate,
                                    timeout=device.timeout,
                                    init_dtr=0) as target:
            target.sendline('usb_on')  # this will cause the MicroSD to be mounted on the host
            device.wait_for_microsd_mount_point(target)
            self.deploy_images(device, image_bundle, images)

        self.logger.debug('Resetting the device.')
        device.hard_reset()

        with open_serial_connection(port=device.port,
                                    baudrate=device.baudrate,
                                    timeout=device.timeout,
                                    init_dtr=0) as target:
            menu = UefiMenu(target)
            menu.open(timeout=300)
            if recreate_uefi_entry and menu.has_option(device.uefi_entry):
                self.logger.debug('Deleting existing device entry.')
                menu.delete_entry(device.uefi_entry)
                menu.create_entry(device.uefi_entry, device.uefi_config)
            elif not menu.has_option(device.uefi_entry):
                menu.create_entry(device.uefi_entry, device.uefi_config)
            menu.select(device.uefi_entry)
            target.expect(device.android_prompt, timeout=device.timeout)

    def deploy_images(self, device, image_bundle=None, images=None):
        try:
            if image_bundle:
                self.deploy_image_bundle(device, image_bundle)
            if images:
                self.overlay_images(device, images)
            os.system('sync')
        except (IOError, OSError), e:
            msg = 'Could not deploy images to {}; got: {}'
            raise DeviceError(msg.format(device.microsd_mount_point, e))

    def deploy_image_bundle(self, device, bundle):
        self.logger.debug('Validating {}'.format(bundle))
        validate_image_bundle(bundle)
        self.logger.debug('Extracting {} into {}...'.format(bundle, device.microsd_mount_point))
        with tarfile.open(bundle) as tar:
            tar.extractall(device.microsd_mount_point)

    def overlay_images(self, device, images):
        for dest, src in images.iteritems():
            dest = os.path.join(device.microsd_mount_point, dest)
            self.logger.debug('Copying {} to {}'.format(src, dest))
            shutil.copy(src, dest)


# utility functions

def get_mapping(base_dir, partition_file):
    mapping = {}
    with open(partition_file) as pf:
        for line in pf:
            pair = line.split()
            if len(pair) != 2:
                ConfigError('partitions.txt is not properly formated')
            image_path = os.path.join(base_dir, pair[1])
            if not os.path.isfile(expand_path(image_path)):
                ConfigError('file {} was not found in the bundle or was misplaced'.format(pair[1]))
            mapping[pair[0]] = image_path
    return mapping


def expand_path(original_path):
    path = os.path.abspath(os.path.expanduser(original_path))
    if not os.path.exists(path):
        raise ConfigError('{} does not exist.'.format(path))
    return path


def validate_image_bundle(bundle):
    if not tarfile.is_tarfile(bundle):
        raise ConfigError('Image bundle {} does not appear to be a valid TAR file.'.format(bundle))
    with tarfile.open(bundle) as tar:
        try:
            tar.getmember('config.txt')
        except KeyError:
            try:
                tar.getmember('./config.txt')
            except KeyError:
                msg = 'Tarball {} does not appear to be a valid image bundle (did not see config.txt).'
                raise ConfigError(msg.format(bundle))

