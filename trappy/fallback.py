#    Copyright 2017 ARM Limited, Google and contributors
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
This module contains the class for representing fallback events used for ftrace
events injected from userspace, which are free-form and could contain any
string.
"""

from trappy.base import Base
from trappy.dynamic import register_ftrace_parser

class FallbackEvent(Base):
    """
    Parse free-form events that couldn't be matched with more specific unique
    words. This class is always used as a fallback if nothing more specific
    could match the particular event.
    """

    def generate_data_dict(self, data_str):
        if self.tracer:
            data_dict = self.tracer.generate_data_dict(data_str)
            if data_dict:
                return data_dict

        return { 'string': data_str }


    def __init__(self):
        super(FallbackEvent, self).__init__(fallback=True)

class TracingMarkWrite(FallbackEvent):
    unique_word = "tracing_mark_write:"

register_ftrace_parser(TracingMarkWrite)

class Print(FallbackEvent):
    unique_word = "print:"
    name = 'print_' # To avoid keyword collision

register_ftrace_parser(Print)
