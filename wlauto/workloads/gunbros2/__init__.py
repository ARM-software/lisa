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

# pylint: disable=R0801
import os
import time
import tarfile
import shutil

from wlauto import settings
from wlauto.common.android.workload import GameWorkload
from wlauto.exceptions import WorkloadError, DeviceError
from wlauto.utils.misc import check_output
from wlauto.common.resources import PluginAsset


class GunBros(GameWorkload):

    name = 'gunbros2'
    description = """
    Gun Bros. 2 game.

    """
    package = 'com.glu.gunbros2'
    activity = 'com.google.android.vending.expansion.downloader_impl.DownloaderActivity'
    asset_file = 'com.glu.gunbros2.tar.gz'
    ondevice_asset_root = '/data'
    loading_time = 20
    install_timeout = 500

