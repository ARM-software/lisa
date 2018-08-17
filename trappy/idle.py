from __future__ import unicode_literals
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

from trappy.base import Base
from trappy.dynamic import register_ftrace_parser

class CpuIdle(Base):
    """Parse cpu_idle"""

    unique_word = "cpu_idle"
    pivot = "cpu_id"

    def finalize_object(self):
        # The trace contains "4294967295" instead of "-1" when exiting an idle
        # state.
        uint32_max = (2 ** 32) - 1
        self.data_frame.replace(uint32_max, -1, inplace=True)
        super(CpuIdle, self).finalize_object()

register_ftrace_parser(CpuIdle)
