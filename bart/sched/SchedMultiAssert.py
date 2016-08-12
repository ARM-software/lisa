#    Copyright 2015-2016 ARM Limited
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
from bart.sched import functions as sched_funcs
from bart.sched.SchedAssert import SchedAssert
from bart.common import Utils

class SchedMultiAssert(object):
    """This is vector assertion class built on top of
    :mod:`bart.sched.SchedAssert.SchedAssert`

    :param ftrace: A single trappy.FTrace object
        or a path that can be passed to trappy.FTrace
    :type ftrace: :mod:`trappy.ftrace.FTrace`

    :param topology: A topology that describes the arrangement of
        CPU's on a system. This is useful for multi-cluster systems
        where data needs to be aggregated at different topological
        levels
    :type topology: :mod:`trappy.stats.Topology.Topology`

    :param execnames: The execnames of the task to be analysed

        A single execname or a list of execnames can be passed.
        There can be multiple processes associated with a single
        execname parameter. The execnames are searched using a prefix
        match.
    :type execname: list, str

    :param pids: The process IDs of the tasks to be analysed
    :type pids: list, int

    Consider the following processes which need to be analysed

        ===== ==============
         PID    execname
        ===== ==============
         11     task_1
         22     task_2
         33     task_3
        ===== ==============

    A :mod:`bart.sched.SchedMultiAssert.SchedMultiAssert` instance be created
    following different ways:

        - Using execname prefix match
          ::

            SchedMultiAssert(ftrace, topology, execnames="task_")

        - Individual Task names
          ::

            SchedMultiAssert(ftrace, topology, execnames=["task_1", "task_2", "task_3"])

        - Using Process IDs
          ::

            SchedMultiAssert(ftrace, topology, pids=[11, 22, 33])


    All the functionality provided in :mod:`bart.sched.SchedAssert.SchedAssert` is available
    in this class with the addition of handling vector assertions.

    For example consider the use of :func:`getDutyCycle`
    ::

        >>> s = SchedMultiAssert(ftrace, topology, execnames="task_")
        >>> s.getDutyCycle(window=(start, end))
        {
            "11": {
                "task_name": "task_1",
                "dutycycle": 10.0
            },
            "22": {
                "task_name": "task_2",
                "dutycycle": 20.0
            },
            "33": {
                "task_name": "task_3",
                "dutycycle": 30.0
            },
        }

    The assertions can be used in a similar way
    ::

        >>> import operator as op
        >>> s = SchedMultiAssert(ftrace, topology, execnames="task_")
        >>> s.assertDutyCycle(15, op.ge, window=(start, end))
        {
            "11": {
                "task_name": "task_1",
                "dutycycle": False
            },
            "22": {
                "task_name": "task_2",
                "dutycycle": True
            },
            "33": {
                "task_name": "task_3",
                "dutycycle": True
            },
        }

    The above result can be coalesced using a :code:`rank` parameter
    As we know that only 2 processes have duty cycles greater than 15%
    we can do the following:
    ::

        >>> import operator as op
        >>> s = SchedMultiAssert(ftrace, topology, execnames="task_")
        >>> s.assertDutyCycle(15, op.ge, window=(start, end), rank=2)
        True

    See :mod:`bart.sched.SchedAssert.SchedAssert` for the available
    functionality
    """

    def __init__(self, ftrace, topology, execnames=None, pids=None):

        self._ftrace = Utils.init_ftrace(ftrace)
        self._topology = topology

        if execnames and pids:
            raise ValueError('Either pids or execnames must be specified')
        if execnames:
            self._execnames = Utils.listify(execnames)
            self._pids = self._populate_pids()
        elif pids:
            self._pids = pids
        else:
            raise ValueError('One of PIDs or execnames must be specified')

        self._asserts = self._populate_asserts()
        self._populate_methods()

    def _populate_asserts(self):
        """Populate SchedAsserts for the PIDs"""

        asserts = {}

        for pid in self._pids:
            asserts[pid] = SchedAssert(self._ftrace, self._topology, pid=pid)

        return asserts

    def _populate_pids(self):
        """Map the input execnames to PIDs"""

        if len(self._execnames) == 1:
            return sched_funcs.get_pids_for_process(self._ftrace, self._execnames[0])

        pids = []

        for proc in self._execnames:
            pids += sched_funcs.get_pids_for_process(self._ftrace, proc)

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

    def getCPUBusyTime(self, level, node, window=None, percent=False):
        """Get the amount of time the cpus in the system were busy executing the
        tasks

        :param level: The topological level to which the group belongs
        :type level: string

        :param node: The group of CPUs for which to calculate busy time
        :type node: list

        :param window: A (start, end) tuple to limit the scope of the
        calculation.
        :type window: tuple

        :param percent: If True the result is normalized to the total
        time of the period, either the window or the full lenght of
        the trace.
        :type percent: bool

        .. math::

            R = \\frac{T_{busy} \\times 100}{T_{total}}

        """
        residencies = self.getResidency(level, node, window=window)

        busy_time = sum(v["residency"] for v in residencies.itervalues())

        if percent:
            if window:
                total_time = window[1] - window[0]
            else:
                total_time = self._ftrace.get_duration()
            num_cpus = len(node)
            return busy_time / (total_time * num_cpus) * 100
        else:
            return busy_time

    def generate_events(self, level, window=None):
        """Generate Events for the trace plot

        .. note::
            This is an internal function for plotting data
        """

        events = {}
        for s_assert in self._asserts.values():
            events[s_assert.name] = s_assert.generate_events(level, window=window)

        return events

    def plot(self, level="cpu", window=None, xlim=None):
        """
        :return: :mod:`trappy.plotter.AbstractDataPlotter` instance
            Call :func:`view` to draw the graph
        """

        if not xlim:
            if not window:
                xlim = [0, self._ftrace.get_duration()]
            else:
                xlim = list(window)

        events = self.generate_events(level, window)
        names = [s.name for s in self._asserts.values()]
        num_lanes = self._topology.level_span(level)
        lane_prefix = level.upper() + ": "
        return trappy.EventPlot(events, names, xlim,
                                lane_prefix=lane_prefix,
                                num_lanes=num_lanes)
