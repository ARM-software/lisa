#    Copyright 2015-2015 ARM Limited
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

"""A library for asserting scheduler scenarios based on the
statistics aggregation framework"""

import re
import inspect
import trappy
from trappy.stats import SchedConf as sconf
from trappy.plotter.Utils import listify
from bart.sched.SchedAssert import SchedAssert
from bart import Utils

class SchedMultiAssert(object):

    """The primary focus of this class is to assert and verify
    predefined scheduler scenarios. This does not compare parameters
    across runs"""

    def __init__(self, run, topology, execnames):
        """Args:
                run (trappy.Run): A single trappy.Run object
                    or a path that can be passed to trappy.Run
                topology(trappy.stats.Topology): The CPU topology
                execname(str, list): List of execnames or single task
        """

        self._execnames = listify(execnames)
        self._run = Utils.init_run(run)
        self._pids = self._populate_pids()
        self._topology = topology
        self._asserts = self._populate_asserts()
        self._populate_methods()

    def _populate_asserts(self):
        """Populate SchedAsserts for the PIDs"""

        asserts = {}

        for pid in self._pids:
            asserts[pid] = SchedAssert(self._run, self._topology, pid=pid)

        return asserts

    def _populate_pids(self):
        """Map the input execnames to PIDs"""

        if len(self._execnames) == 1:
            return sconf.get_pids_for_process(self._run, self._execnames[0])

        pids = []

        for proc in self._execnames:
            pids += sconf.get_pids_for_process(self._run, proc)

        return list(set(pids))

    def _create_method(self, attr_name):
        """A wrapper function to create a dispatch function"""

        return lambda *args, **kwargs: self._dispatch(attr_name, *args, **kwargs)

    def _populate_methods(self):
        """Populate Methods from SchedAssert"""

        for attr_name in dir(SchedAssert):
            attr = getattr(SchedAssert, attr_name)

            valid_method = attr_name.startswith("get") or \
                           attr_name.startswith("assert")
            if inspect.ismethod(attr) and valid_method:
                func = self._create_method(attr_name)
                setattr(self, attr_name, func)

    def get_task_name(self, pid):
        """Get task name for the PID"""
        return self._asserts[pid].execname


    def _dispatch(self, func_name, *args, **kwargs):
        """The dispatch function to call into the SchedAssert
           Method
        """

        assert_func = func_name.startswith("assert")
        num_true = 0

        rank = kwargs.pop("rank", None)
        result = kwargs.pop("result", {})
        param = kwargs.pop("param", re.sub(r"assert|get", "", func_name, count=1).lower())

        for pid in self._pids:

            if pid not in result:
                result[pid] = {}
                result[pid]["task_name"] = self.get_task_name(pid)

            attr = getattr(self._asserts[pid], func_name)
            result[pid][param] = attr(*args, **kwargs)

            if assert_func and result[pid][param]:
                num_true += 1

        if assert_func and rank:
            return num_true == rank
        else:
            return result

    def generate_events(self, level, window=None):
        """Generate Events for the trace plot"""

        events = {}
        for s_assert in self._asserts.values():
            events[s_assert.name] = s_assert.generate_events(level, window=window)

        return events

    def plot(self, level="cpu", window=None, xlim=None):
        """
        Returns:
            trappy.plotter.AbstractDataPlotter. Call .view() for
                displaying the plot
        """

        if not xlim:
            if not window:
                xlim = [0, self._run.get_duration()]
            else:
                xlim = list(window)

        events = self.generate_events(level, window)
        names = [s.name for s in self._asserts.values()]
        num_lanes = self._topology.level_span(level)
        lane_prefix = level.upper() + ": "
        return trappy.EventPlot(events, names, lane_prefix, num_lanes, xlim)
