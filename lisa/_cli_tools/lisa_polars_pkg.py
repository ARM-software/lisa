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
    out = completed.stdout

    # Get the version without importing the package. This avoids coredumps.
    version = importlib.metadata.version('polars')

    out = out.decode().lower().replace('\n', ' ')
    old_cpu = 'cpu' in out and 'feature' in out
    return (old_cpu, version)


def main():
    try:
        importlib.metadata.version('polars-lts-cpu')
    except importlib.metadata.PackageNotFoundError:
        old_cpu, version = check_polars()
    # If we have polars-lts-cpu installed, we assume it is for good reasons
    # because we have an old cpu.
    else:
        old_cpu = True
        version = importlib.metadata.version('polars-lts-cpu')

    if old_cpu:
        pkg = 'polars-lts-cpu'
    else:
        pkg = 'polars'

    # Taking the version into account is important, as dependency resolution
    # might have avoided some versions according to the constraints used in
    # setup.py
    print(f'{pkg}=={version}')


if __name__ == '__main__':
    main()
