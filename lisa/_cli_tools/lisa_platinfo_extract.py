#! /usr/bin/env python3
#
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

# Basic tool to extract a lisa.platform.plat_info.PlatformInfo from a live
# target

from lisa.target import Target

def main():
    params = {
        'plat-info': dict(
            help='Path to the PlatformInfo file to generate',
        )
    }
    args, target = Target.from_custom_cli(params=params)

    with target.closing() as target:
        return _main(args, target)

def _main(args, target):
    plat_info = target.plat_info

    # Make sure we get all the information we can, even if it means running for
    # a bit longer. RTapp calibration will be computed as it is a DeferredValue
    plat_info.eval_deferred(error='log')

    print(plat_info)

    path = args.plat_info or f'{target.name}.plat_info.yml'
    plat_info.to_yaml_map(path)
    print(f'\nPlatform info written to: {path}')

if __name__ == "__main__":
    main()
