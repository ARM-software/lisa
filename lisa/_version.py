# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# THIS MODULE MUST BE EXECUTABLE ON ITS OWN WITHOUT DEPENDING ON ANYTHING
# PROVIDED BY LISA.
#
# This way, setup.py can run before the lisa package becomes importable.


import os

version_tuple = (3, 1, 0)

def format_version(version):
    return '.'.join(str(part) for part in version)

def parse_version(version):
    return tuple(int(part) for part in version.split('.'))

__version__ = format_version(version_tuple)
