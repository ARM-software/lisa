#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

from lisa.target import Target
from devlib.module.sched import SchedDomain

target = Target.from_cli()

sd_info = target.sched.get_sd_info()

for cpuid, cpu in sd_info.cpus.items():
    print("== CPU{} ==".format(cpuid))
    for domain in cpu.domains.values():
        print("\t{} level".format(domain.name))
        for flag in domain.flags:
            print("\t\t{} - {}".format(flag.name, flag.__doc__))
