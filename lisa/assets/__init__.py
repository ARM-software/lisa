#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
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
import os
import stat
import platform

def _get_abi():
    machine = platform.machine()
    return dict(
        x86_64='x86_64',
        aarch64='arm64',
        arm='arm',
    )[machine]

HOST_ABI = _get_abi()
"""
ABI of the machine that imported that module.
"""
del _get_abi

ASSETS_PATH = os.path.dirname(__file__)
"""
Path in which all assets the ``lisa`` package relies on are located in.
"""

def _get_abi_bin():
    def list_binaries(path):
        return {
            entry.name: os.path.abspath(entry.path)
            for entry in os.scandir(path)
            if entry.stat().st_mode & stat.S_IXUSR
        }

    bin_path = os.path.join(ASSETS_PATH, 'binaries')
    return {
        entry.name: list_binaries(entry.path)
        for entry in os.scandir(bin_path)
        if entry.is_dir()
    }

ABI_BINARIES = _get_abi_bin()
del _get_abi_bin

HOST_BINARIES = ABI_BINARIES[HOST_ABI]


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
