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

from warnings import catch_warnings

def main():
    with catch_warnings(record=True) as ws:
        import polars

    version = polars.__version__

    assert ws is not None
    w = ' '.join(
        str(w.message)
        for w in ws
    )
    w = w.lower()

    old_cpu = 'cpu' in w and 'feature' in w

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
