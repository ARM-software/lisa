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
import shutil
from pathlib import Path

from lisa.utils import LISA_HOST_ABI

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

HOST_BINARIES = ABI_BINARIES[LISA_HOST_ABI]

def _make_path(abi=None):
    abi = abi or LISA_HOST_ABI

    compos = [
        os.path.join(ASSETS_PATH, 'binaries', abi),
        os.path.join(ASSETS_PATH, 'scripts'),
    ]

    if abi == LISA_HOST_ABI:
        path = os.environ['PATH']
        use_system = bool(int(os.environ.get('LISA_USE_SYSTEM_BIN', 0)))
        if use_system:
            compos = [path] + compos
        else:
            compos = compos + [path]

    return ':'.join(compos)

HOST_PATH = _make_path(LISA_HOST_ABI)
"""
Value to be used as the ``PATH`` env var on the host.
"""

def get_bin(name, abi=None):
    """
    Return the path to a tool bundled in LISA.

    :param abi: ABI of the binary. If ``abi`` is not the host ABI,
        ``LISA_USE_SYSTEM_BIN`` is ignored.
    :type abi: str or None

    The result is influenced by the ``LISA_USE_SYSTEM_BIN`` environment
    variable:

        * If it is set to ``0`` or unset, it will give priority to the binaries
          bundled inside the :mod:`lisa` package.
        * If it is set to ``1``, it will use the bundled binaries as a fallback
          only.
    """
    path = shutil.which(name, path=_make_path(abi))
    if path:
        return Path(path).resolve()
    else:
        raise FileNotFoundError(f'Could not locate the tool: {name}')

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
