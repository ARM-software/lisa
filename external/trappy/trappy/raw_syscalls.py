#    Copyright 2016-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import unicode_literals

import re
from trappy.base import Base
from trappy.dynamic import register_ftrace_parser

class SysEnter(Base):
    """Parse sys enter"""

    unique_word = "sys_enter"
    pivot = "pid"
    parse_raw = False
    pat = re.compile(r"NR (\d+) \(([-+]?[\da-fA-F]+), ([-+]?[\da-fA-F]+), ([-+]?[\da-fA-F]+), ([-+]?[\da-fA-F]+), ([-+]?[\da-fA-F]+), ([-+]?[\da-fA-F]+)\)")

    def __init__(self):
        super(SysEnter, self).__init__(parse_raw=self.parse_raw)

    def explode(self, line):
        """trace-cmd-29016 [000] 99937.172691: sys_enter:            NR 64 (1, 7764ee9000, 103, 0, 0, 4)"""

        match = None
        try:
            match = self.pat.search(line)
            num, *args = match.group(1,2,3,4,5,6,7)
            num = int(num)
            args = [int(x, 16) for x in args]
            ret = "nr={} arg0={} arg1={} arg2={} arg3={} arg4={} arg5={}".format(num, args[0], args[1], args[2], args[3], args[4], args[5])
            return ret
        except Exception as e:
            raise ValueError("failed to parse line {}: {}".format(line, e))

    def create_dataframe(self):
        self.data_array = [self.explode(line) for line in self.data_array]
        super(SysEnter, self).create_dataframe()

register_ftrace_parser(SysEnter)



class SysExit(Base):
    """Parse sys exit"""

    unique_word = "sys_exit"
    pivot = "pid"
    parse_raw = False
    pat = re.compile("NR ([-+]?\d+) = ([-+]?\d+)")

    def __init__(self):
        super(SysExit, self).__init__(parse_raw=self.parse_raw)

    def explode(self, line):
        """     trace-cmd-29016 [000] 99937.172659: sys_exit:             NR 64 = 1 """
        match = None
        try:
            match = self.pat.search(line)
            num, ret = match.group(1,2)
            num = int(num)
            ret = int(ret)
            foo = "nr={} ret={}".format(num, ret)
            return foo
        except Exception as e:
            raise ValueError("failed to parse line {}: {}".format(line, e))

    def create_dataframe(self):
        self.data_array = [self.explode(line) for line in self.data_array]
        super(SysExit, self).create_dataframe()

register_ftrace_parser(SysExit)
