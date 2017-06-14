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

"""This module contains the class for representing a tracing_mark_write
trace_event used for ftrace events injected from userspace.
"""

from trappy.base import Base
from trappy.dynamic import register_ftrace_parser

class TracingMarkWrite(Base):
    """Parse tracing_mark_write events that couldn't be matched with more specific unique words
       This class is always used as a fallback if nothing more specific could match the particular
       tracing_mark_write event.
    """

    unique_word = "tracing_mark_write"

    def generate_data_dict(self, data_str):
        if self.tracer:
            data_dict = self.tracer.generate_data_dict(data_str)
            if data_dict:
                return data_dict

        data_dict = { 'string': data_str }
        return data_dict

    def __init__(self):
        super(TracingMarkWrite, self).__init__(fallback=True)

register_ftrace_parser(TracingMarkWrite)
