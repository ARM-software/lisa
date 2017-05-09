# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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
# Parser for dumpsys gfxinfo output

import ast, re

def get_value(token):
    try:
        v = ast.literal_eval(token)
    except:
        return token
    return v

class GfxInfo(object):
    """
    Class for parsing and accessing GfxInfo output
    """
    __properties = {}

    def __init__(self, path):
        """
        Initialize gfxinfo parser object

        :param path: Path to file containing output of gfxinfo
        """
        self.path = path
        self.parse_gfxinfo()

    def parse_gfxinfo(self):
        """
        Parser for gfxinfo output, creates a name/value pair
        """
        with open(self.path) as f:
            content = f.readlines()

        # gfxinfo has several statistics of the format
        # <string>: <value>
        for line in content:
            line = line.rstrip()
            # Ignore any lines that aren't of this format
            if not ':' in line or line.endswith(':'):
                continue

            tokens = line.split(':')
            # convert <string> into a key by replacing spaces with '_'
            tokens = [t.strip() for t in tokens]
            tokens[0] = tokens[0].replace(' ', '_').lower()

            # Parse janky_frames. Ex: "Janky frames: 44 (26.99%)"
            if tokens[0] == 'janky_frames':
                (frames, pc) = tokens[1].split(' ')
                self.__properties["janky_frames"] = get_value(frames)
                pc = re.sub('[\(\)\%]', '', pc)
                self.__properties["janky_frames_pc"] = get_value(pc)
                continue
            # Parse 'nth_percentile: <int>ms' into nth_percentile_ms=<int>
            if tokens[1].endswith('ms'):
                tokens[0] = tokens[0] + '_ms'
                tokens[1] = tokens[1][:-2]
            # Regular parsing
            self.__properties[tokens[0]] = get_value(tokens[1])

    def __dir__(self):
        """
        List all available attributes including ones parsed from
        gfxinfo output
        """
        return self.__properties.keys()

    def __getattr__(self, name):
        """
        Get the gfxinfo property using the period operator
        Ex: obj.number_missed_vsync
        """
        return self.__properties[name]

    def __getitem__(self, name):
        """
        Get the gfxinfo property using the [] opertator
        Useful for attributes like "50th_percentile" that can't
        be fetched with the period operator
        """
        return self.__properties[name]
