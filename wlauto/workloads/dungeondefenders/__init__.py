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

from wlauto import GameWorkload
from wlauto.exceptions import WorkloadError, DeviceError


class DungeonDefenders(GameWorkload):

    name = 'dungeondefenders'
    description = """
    Dungeon Defenders game.

    """
    package = 'com.trendy.ddapp'
    activity = 'com.trendy.ddapp.ddapp'
    loading_time = 20
    asset_file = 'com.trendy.ddapp.tar.gz'
