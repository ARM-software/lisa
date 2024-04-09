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

def parse_kallsyms(kallsyms):
    """
    Parse the content of ``/proc/kallsyms``.
    """
    def parse_line(line):
        splitted = line.split(maxsplit=3)

        addr = int(splitted[0], base=16)
        symtype = splitted[1]
        name = splitted[2]

        try:
            module = splitted[3]
        except IndexError:
            module = None
        else:
            module = module.strip('[]')

        return (addr, name, symtype, module)

    return sorted(map(parse_line, kallsyms.splitlines()))
