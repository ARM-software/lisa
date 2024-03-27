# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2023, Arm Limited and contributors.
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

from unittest import TestCase
from pathlib import Path
import os
import io
import subprocess
import lzma
import logging
from concurrent.futures import ProcessPoolExecutor

from lisa._btf import parse_btf, dump_c

from .utils import ASSET_DIR

def _test_btf(btf_path):
    """
    Compile-test the dumped C header. Since this header contains
    _Static_assert() to check member offsets and type sizes, this is almost
    equivalent to a roundtrip test.
    """
    logging.info(f'Checking BTF blob: {btf_path}')

    with lzma.open(btf_path, 'rb') as f:
        buf = f.read()

    typs = parse_btf(buf)
    with io.StringIO() as header:
        dump_c(typs, fileobj=header)
        header.flush()
        header = header.getvalue()

    header = header.encode('utf-8')

    cmd = (
        'clang',
        '--target=aarch64-linux-gnu-',
        '-xc',

        '-std=gnu11',
        '-pedantic',
        '-fshort-wchar',

        '-Wall',
        '-Wextra',
        '-Wno-attributes',
        '-Wno-zero-length-array',
        '-Wno-pedantic',

        '-c',
        '-o', '/dev/null',
        '/dev/stdin',
    )

    subprocess.check_output(cmd, input=header)


class BTFDump(TestCase):
    def test_c_dump(self):
        btf_folder = Path(ASSET_DIR) / 'btf'

        paths = sorted(
            Path(root) / filename
            for (root, dirs, files) in os.walk(btf_folder)
            for filename in files
        )
        with ProcessPoolExecutor() as executor:
            executor.map(
                _test_btf,
                paths
            )


