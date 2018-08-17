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

from builtins import object
from trappy.ftrace import GenericFTrace
import re

SYSTRACE_EVENT = re.compile(
    r'^(?P<event>[A-Z])(\|(?P<pid>\d+)\|(?P<func>[^|]*)(\|(?P<data>.*))?)?')

class drop_before_trace(object):
    """Object that, when called, returns True if the line is not part of
the trace

    We have to first look for the "<!-- BEGIN TRACE -->" and then skip
    the headers that start with #

    """
    def __init__(self, tracer):
        self.before_begin_trace = True
        self.before_actual_trace = True
        self.tracer = tracer

    def __call__(self, line):
        if self.before_begin_trace:
            if line.startswith("<!-- BEGIN TRACE -->") or \
               line.startswith("<title>Android System Trace</title>"):
                self.before_begin_trace = False
        elif self.before_actual_trace:
            if line.startswith('  <script class="trace-data"') or \
               line.startswith("  var linuxPerfData"):
                self.before_actual_trace = False

        if not self.before_actual_trace:
            base_call = super(SysTrace, self.tracer).trace_hasnt_started()
            return base_call(line)
        else:
            return True

class SysTrace(GenericFTrace):
    """A wrapper that parses all events of a SysTrace run

    It receives the same parameters as :mod:`trappy.ftrace.FTrace`.

    """

    def __init__(self, path=".", name="", normalize_time=True, scope="all",
                 events=[], window=(0, None), abs_window=(0, None)):

        self.trace_path = path

        super(SysTrace, self).__init__(name, normalize_time, scope, events,
                                       window, abs_window)

        try:
            self._cpus = 1 + self.sched_switch.data_frame["__cpu"].max()
        except AttributeError:
            pass

    def trace_hasnt_started(self):
        return drop_before_trace(self)

    def trace_hasnt_finished(self):
        """Return a function that returns True while the current line is still part of the trace

        In Systrace, the first line that is not part of the trace is
        </script>.  There's a further "<!-- END TRACE -->" but there's
        not point scanning for it, we should stop parsing as soon as
        we see the </script>

        """
        return lambda x: not x.endswith("</script>\n")

    def generate_data_dict(self, data_str):
        """ Custom parsing for systrace's userspace events """
        data_dict = None

        match = SYSTRACE_EVENT.match(data_str)
        if match:
            data_dict = {
                          'event': match.group('event'),
                          'pid'  : int(match.group('pid')) if match.group('pid') else None,
                          'func' : match.group('func' ),
                          'data' : match.group('data' )
                        }

        return data_dict
