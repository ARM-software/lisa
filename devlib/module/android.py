#    Copyright 2014-2018 ARM Limited
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

from devlib.module import FlashModule
from devlib.exception import HostError
from devlib.utils.android import fastboot_flash_partition, fastboot_command
from devlib.utils.misc import merge_dicts


class FastbootFlashModule(FlashModule):

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
    partitions_file_name = 'partitions.txt'

    @staticmethod
    def probe(target):
        return target.os == 'android'

    def __call__(self, image_bundle=None, images=None, bootargs=None):
        if bootargs:
            raise ValueError('{} does not support boot configuration'.format(self.name))
        self.prelude_done = False
        to_flash = {}
        if image_bundle:  # pylint: disable=access-member-before-definition
            image_bundle = expand_path(image_bundle)
            to_flash = self._bundle_to_images(image_bundle)
        to_flash = merge_dicts(to_flash, images or {}, should_normalize=False)
        for partition, image_path in to_flash.items():
            self.logger.debug('flashing {}'.format(partition))
            self._flash_image(self.target, partition, expand_path(image_path))
        fastboot_command('reboot')
        self.target.connect(timeout=180)

    def _validate_image_bundle(self, image_bundle):
        if not tarfile.is_tarfile(image_bundle):
            raise HostError('File {} is not a tarfile'.format(image_bundle))
        with tarfile.open(image_bundle) as tar:
            files = [tf.name for tf in tar.getmembers()]
            if not any(pf in files for pf in (self.partitions_file_name, '{}/{}'.format(files[0], self.partitions_file_name))):
                HostError('Image bundle does not contain the required partition file (see documentation)')

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

    def _flash_image(self, target, partition, image_path):
        if not self.prelude_done:
            self._fastboot_prelude(target)
        fastboot_flash_partition(partition, image_path)
        time.sleep(self.delay)

    def _fastboot_prelude(self, target):
        target.reset(fastboot=True)
        time.sleep(self.delay)
        self.prelude_done = True


# utility functions

def expand_path(original_path):
    path = os.path.abspath(os.path.expanduser(original_path))
    if not os.path.exists(path):
        raise HostError('{} does not exist.'.format(path))
    return path


def get_mapping(base_dir, partition_file):
    mapping = {}
    with open(partition_file) as pf:
        for line in pf:
            pair = line.split()
            if len(pair) != 2:
                HostError('partitions.txt is not properly formated')
            image_path = os.path.join(base_dir, pair[1])
            if not os.path.isfile(expand_path(image_path)):
                HostError('file {} was not found in the bundle or was misplaced'.format(pair[1]))
            mapping[pair[0]] = image_path
    return mapping
