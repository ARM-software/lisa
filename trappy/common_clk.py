#    Copyright 2017 Google, ARM Limited
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


"""
Definitions of common_clk (CONFIG_COMMON_CLK) trace parsers
registered by the FTrace class
"""
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from trappy.base import Base
from trappy.dynamic import register_ftrace_parser, register_dynamic_ftrace

class CommonClkBase(Base):
    #clock traces are of the form "clk_name field0=x field1=y ..."
    def generate_data_dict(self, data_str):
        clk_name, fields = data_str.split(' ', 1)
        ret = super(CommonClkBase, self).generate_data_dict(fields)
        ret['clk_name'] = clk_name
        return ret

class CommonClkEnable(CommonClkBase):
    """Corresponds to Linux kernel trace event clock_enable"""

    unique_word = "clock_enable:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(CommonClkEnable)

class CommonClkDisable(CommonClkBase):
    """Corresponds to Linux kernel trace event clock_disable"""

    unique_word = "clock_disable:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(CommonClkDisable)

class CommonClkSetRate(CommonClkBase):
    """Corresponds to Linux kernel trace event clock_set_rate"""

    unique_word = "clock_set_rate:"
    """The unique word that will be matched in a trace line"""

    def finalize_object(self):
        self.data_frame.rename(columns={'state':'rate'}, inplace=True)

register_ftrace_parser(CommonClkSetRate)
