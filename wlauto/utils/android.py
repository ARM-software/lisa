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


"""
Utility functions for working with Android devices through adb.

"""
# pylint: disable=E1103
import os
import time
import subprocess
import logging
import re

from wlauto.exceptions import DeviceError, ConfigError, HostError
from wlauto.utils.misc import check_output, escape_single_quotes, escape_double_quotes, get_null

from devlib.utils.android import ANDROID_VERSION_MAP, adb_command, ApkInfo

# See:
# http://developer.android.com/guide/topics/security/normal-permissions.html
ANDROID_NORMAL_PERMISSIONS = [
    'ACCESS_LOCATION_EXTRA_COMMANDS',
    'ACCESS_NETWORK_STATE',
    'ACCESS_NOTIFICATION_POLICY',
    'ACCESS_WIFI_STATE',
    'BLUETOOTH',
    'BLUETOOTH_ADMIN',
    'BROADCAST_STICKY',
    'CHANGE_NETWORK_STATE',
    'CHANGE_WIFI_MULTICAST_STATE',
    'CHANGE_WIFI_STATE',
    'DISABLE_KEYGUARD',
    'EXPAND_STATUS_BAR',
    'GET_PACKAGE_SIZE',
    'INTERNET',
    'KILL_BACKGROUND_PROCESSES',
    'MODIFY_AUDIO_SETTINGS',
    'NFC',
    'READ_SYNC_SETTINGS',
    'READ_SYNC_STATS',
    'RECEIVE_BOOT_COMPLETED',
    'REORDER_TASKS',
    'REQUEST_INSTALL_PACKAGES',
    'SET_TIME_ZONE',
    'SET_WALLPAPER',
    'SET_WALLPAPER_HINTS',
    'TRANSMIT_IR',
    'USE_FINGERPRINT',
    'VIBRATE',
    'WAKE_LOCK',
    'WRITE_SYNC_SETTINGS',
    'SET_ALARM',
    'INSTALL_SHORTCUT',
    'UNINSTALL_SHORTCUT',
]
