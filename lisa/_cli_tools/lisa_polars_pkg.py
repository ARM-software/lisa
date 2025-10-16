#! /usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, Arm Limited and contributors.
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
import sys
import subprocess
import importlib.metadata

def check_polars():
    # Import in a separate process to resist coredumps due to illegal
    # instructions (e.g. SSE on old CPUs).
    completed = subprocess.run(
        [sys.executable, '-c', 'import polars'],
        # Resist coredump
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            # Ensure we have warnings enabled.
            'PYTHONWARNINGS': 'always',
        }
    )
    out = completed.stdout.decode().lower().replace('\n', ' ')
    old_cpu = 'cpu' in out and 'feature' in out
    return old_cpu


def main():
    def var_is_lts(var):
        return 'lts' in os.environ[var]

    # Respect user's choice
    try:
        old_cpu = var_is_lts('POLARS_FORCE_PKG')
    except KeyError:
        try:
            old_cpu = var_is_lts('POLARS_PREFER_PKG')
        except KeyError:
            old_cpu = check_polars()

    if old_cpu:
        pkg = 'polars[rtcompat]'
    else:
        pkg = 'polars'

    # Taking the version into account is important, as dependency resolution
    # might have avoided some versions according to the constraints used in
    # setup.py
    version = importlib.metadata.version('polars')
    print(f'{pkg}=={version}')


if __name__ == '__main__':
    main()
