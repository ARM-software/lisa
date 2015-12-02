#    Copyright 2016 ARM Limited
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

from trappy.ftrace import GenericFTrace

class drop_before_trace(object):
    """Object that, when called, returns True if the line is not part of
the trace

    We have to first look for the "<!-- BEGIN TRACE -->" and then skip
    the headers that start with #

    """
    def __init__(self):
        self.before_begin_trace = True
        self.before_script_trace_data = True
        self.before_actual_trace = True

    def __call__(self, line):
        if self.before_begin_trace:
            if line.startswith("<!-- BEGIN TRACE -->"):
                self.before_begin_trace = False
        elif self.before_script_trace_data:
            if line.startswith('  <script class="trace-data"'):
                self.before_script_trace_data = False
        elif not line.startswith("#"):
            self.before_actual_trace = False

        return self.before_actual_trace

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
        return drop_before_trace()

    def trace_hasnt_finished(self):
        """Return a function that returns True while the current line is still part of the trace

        In Systrace, the first line that is not part of the trace is
        </script>.  There's a further "<!-- END TRACE -->" but there's
        not point scanning for it, we should stop parsing as soon as
        we see the </script>

        """
        return lambda x: not x.endswith("</script>\n")
