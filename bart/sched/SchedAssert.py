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

"""
:mod:`bart.sched.SchedAssert` provides ability to assert scheduler behaviour.
The analysis is based on TRAPpy's statistics framework and is potent enough
to aggregate statistics over processor hierarchies.
"""

import trappy
import itertools
import math
from trappy.stats.Aggregator import MultiTriggerAggregator
from bart.sched import functions as sched_funcs
from bart.common import Utils
import numpy as np

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
class SchedAssert(object):

    """The primary focus of this class is to assert and verify
    predefined scheduler scenarios. This does not compare parameters
    across runs

    :param ftrace: A single trappy.FTrace object
        or a path that can be passed to trappy.FTrace
    :type ftrace: :mod:`trappy.ftrace.FTrace`

    :param topology: A topology that describes the arrangement of
        CPU's on a system. This is useful for multi-cluster systems
        where data needs to be aggregated at different topological
        levels
    :type topology: :mod:`trappy.stats.Topology.Topology`

    :param execname: The execname of the task to be analysed

        .. note::

                There should be only one PID that maps to the specified
                execname. If there are multiple PIDs :mod:`bart.sched.SchedMultiAssert`
                should be used

    :type execname: str

    :param pid: The process ID of the task to be analysed
    :type pid: int

    .. note:

        One of pid or execname is mandatory. If only execname
        is specified, The current implementation will fail if
        there are more than one processes with the same execname
    """

    def __init__(self, ftrace, topology, execname=None, pid=None):

        ftrace = Utils.init_ftrace(ftrace)

        if not execname and not pid:
            raise ValueError("Need to specify at least one of pid or execname")

        self.execname = execname
        self._ftrace = ftrace
        self._pid = self._validate_pid(pid)
        self._aggs = {}
        self._topology = topology
        self._triggers = sched_funcs.sched_triggers(self._ftrace, self._pid,
                                              trappy.sched.SchedSwitch)
        self.name = "{}-{}".format(self.execname, self._pid)

    def _validate_pid(self, pid):
        """Validate the passed pid argument"""

        if not pid:
            pids = sched_funcs.get_pids_for_process(self._ftrace,
                                              self.execname)

            if len(pids) != 1:
                raise RuntimeError(
                    "There should be exactly one PID {0} for {1}".format(
                        pids,
                        self.execname))

            return pids[0]

        elif self.execname:

            pids = sched_funcs.get_pids_for_process(self._ftrace,
                                              self.execname)
            if pid not in pids:
                raise RuntimeError(
                    "PID {0} not mapped to {1}".format(
                        pid,
                        self.execname))
        else:
            self.execname = sched_funcs.get_task_name(self._ftrace, pid)

        return pid

    def _aggregator(self, aggfunc):
        """
        Return an aggregator corresponding to the
        aggfunc, the aggregators are memoized for performance

        :param aggfunc: Function parameter that
            accepts a :mod:`pandas.Series` object and
            returns a vector/scalar

        :type: function(:mod:`pandas.Series`)
        """

        if aggfunc not in self._aggs.keys():
            self._aggs[aggfunc] = MultiTriggerAggregator(self._triggers,
                                                         self._topology,
                                                         aggfunc)
        return self._aggs[aggfunc]

    def getResidency(self, level, node, window=None, percent=False):
        """
        Residency of the task is the amount of time it spends executing
        a particular group of a topological level. For example:
        ::

            from trappy.stats.Topology import Topology

            big = [1, 2]
            little = [0, 3, 4, 5]

            topology = Topology(clusters=[little, big])

            s = SchedAssert(trace, topology, pid=123)
            s.getResidency("cluster", big)

        This will return the residency of the task on the big cluster. If
        percent is specified it will be normalized to the total runtime
        of the task

        :param level: The topological level to which the group belongs
        :type level: str

        :param node: The group of CPUs for which residency
            needs to calculated
        :type node: list

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param percent: If true the result is normalized to the total runtime
            of the task and returned as a percentage
        :type percent: bool

        .. math::

            R = \\frac{T_{group} \\times 100}{T_{total}}

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.assertResidency`
        """

        # Get the index of the node in the level
        node_index = self._topology.get_index(level, node)

        agg = self._aggregator(sched_funcs.residency_sum)
        level_result = agg.aggregate(level=level, window=window)

        node_value = level_result[node_index]

        if percent:
            total = agg.aggregate(level="all", window=window)[0]
            node_value = node_value * 100
            node_value = node_value / total

        return node_value

    def assertResidency(
            self,
            level,
            node,
            expected_value,
            operator,
            window=None,
            percent=False):
        """
        :param level: The topological level to which the group belongs
        :type level: str

        :param node: The group of CPUs for which residency
            needs to calculated
        :type node: list

        :param expected_value: The expected value of the residency
        :type expected_value: double

        :param operator: A binary operator function that returns
            a boolean. For example:
            ::

                import operator
                op = operator.ge
                assertResidency(level, node, expected_value, op)

            Will do the following check:
            ::

                getResidency(level, node) >= expected_value

            A custom function can also be passed:
            ::

                THRESHOLD=5
                def between_threshold(a, expected):
                    return abs(a - expected) <= THRESHOLD

        :type operator: function

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param percent: If true the result is normalized to the total runtime
            of the task and returned as a percentage
        :type percent: bool

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.getResidency`
        """
        node_value = self.getResidency(level, node, window, percent)
        return operator(node_value, expected_value)

    def getStartTime(self):
        """
        :return: The first time the task ran across all the CPUs
        """

        agg = self._aggregator(sched_funcs.first_time)
        result = agg.aggregate(level="all", value=sched_funcs.TASK_RUNNING)
        return min(result[0])

    def getEndTime(self):
        """
        :return: The first last time the task ran across
            all the CPUs
        """

        agg = self._aggregator(sched_funcs.first_time)
        agg = self._aggregator(sched_funcs.last_time)
        result = agg.aggregate(level="all", value=sched_funcs.TASK_RUNNING)
        return max(result[0])

    def _relax_switch_window(self, series, direction, window):
        """
            direction == "left"
                return the last time the task was running
                if no such time exists in the window,
                extend the window's left extent to
                getStartTime

            direction == "right"
                return the first time the task was running
                in the window. If no such time exists in the
                window, extend the window's right extent to
                getEndTime()

            The function returns a None if
            len(series[series == TASK_RUNNING]) == 0
            even in the extended window
        """

        series = series[series == sched_funcs.TASK_RUNNING]
        w_series = sched_funcs.select_window(series, window)
        start, stop = window

        if direction == "left":
            if len(w_series):
                return w_series.index.values[-1]
            else:
                start_time = self.getStartTime()
                w_series = sched_funcs.select_window(
                    series,
                    window=(
                        start_time,
                        start))

                if not len(w_series):
                    return None
                else:
                    return w_series.index.values[-1]

        elif direction == "right":
            if len(w_series):
                return w_series.index.values[0]
            else:
                end_time = self.getEndTime()
                w_series = sched_funcs.select_window(series, window=(stop, end_time))

                if not len(w_series):
                    return None
                else:
                    return w_series.index.values[0]
        else:
            raise ValueError("direction should be either left or right")

    def assertSwitch(
            self,
            level,
            from_node,
            to_node,
            window,
            ignore_multiple=True):
        """
        This function asserts that there is context switch from the
           :code:`from_node` to the :code:`to_node`:

        :param level: The topological level to which the group belongs
        :type level: str

        :param from_node: The node from which the task switches out
        :type from_node: list

        :param to_node: The node to which the task switches
        :type to_node: list

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param ignore_multiple: If true, the function will ignore multiple
           switches in the window, If false the assert will be true if and
           only if there is a single switch within the specified window
        :type ignore_multiple: bool
        """

        from_node_index = self._topology.get_index(level, from_node)
        to_node_index = self._topology.get_index(level, to_node)

        agg = self._aggregator(sched_funcs.csum)
        level_result = agg.aggregate(level=level)

        from_node_result = level_result[from_node_index]
        to_node_result = level_result[to_node_index]

        from_time = self._relax_switch_window(from_node_result, "left", window)
        if ignore_multiple:
            to_time = self._relax_switch_window(to_node_result, "left", window)
        else:
            to_time = self._relax_switch_window(
                to_node_result,
                "right", window)

        if from_time and to_time:
            if from_time < to_time:
                return True

        return False

    def getRuntime(self, window=None, percent=False):
        """Return the Total Runtime of a task

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param percent: If True, the result is returned
            as a percentage of the total execution time
            of the run.
        :type percent: bool

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.assertRuntime`
        """

        agg = self._aggregator(sched_funcs.residency_sum)
        run_time = agg.aggregate(level="all", window=window)[0]

        if percent:

            if window:
                begin, end = window
                total_time = end - begin
            else:
                total_time = self._ftrace.get_duration()

            run_time = run_time * 100
            run_time = run_time / total_time

        return run_time

    def assertRuntime(
            self,
            expected_value,
            operator,
            window=None,
            percent=False):
        """Assert on the total runtime of the task

        :param expected_value: The expected value of the runtime
        :type expected_value: double

        :param operator: A binary operator function that returns
            a boolean. For example:
            ::

                import operator
                op = operator.ge
                assertRuntime(expected_value, op)

            Will do the following check:
            ::

                getRuntime() >= expected_value

            A custom function can also be passed:
            ::

                THRESHOLD=5
                def between_threshold(a, expected):
                    return abs(a - expected) <= THRESHOLD

        :type operator: function

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param percent: If True, the result is returned
            as a percentage of the total execution time
            of the run.
        :type percent: bool

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.getRuntime`
        """

        run_time = self.getRuntime(window, percent)
        return operator(run_time, expected_value)

    def getPeriod(self, window=None, align="start"):
        """Return the period of the task in (ms)

        Let's say a task started execution at the following times:

            .. math::

                T_1, T_2, ...T_n

        The period is defined as:

            .. math::

                Median((T_2 - T_1), (T_4 - T_3), ....(T_n - T_{n-1}))

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param align:
            :code:`"start"` aligns period calculation to switch-in events
            :code:`"end"` aligns the calculation to switch-out events
        :type param: str

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.assertPeriod`
        """

        agg = self._aggregator(sched_funcs.period)
        deltas = agg.aggregate(level="all", window=window)[0]

        if not len(deltas):
            return float("NaN")
        else:
            return np.median(deltas) * 1000

    def assertPeriod(
            self,
            expected_value,
            operator,
            window=None,
            align="start"):
        """Assert on the period of the task

        :param expected_value: The expected value of the runtime
        :type expected_value: double

        :param operator: A binary operator function that returns
            a boolean. For example:
            ::

                import operator
                op = operator.ge
                assertPeriod(expected_value, op)

            Will do the following check:
            ::

                getPeriod() >= expected_value

            A custom function can also be passed:
            ::

                THRESHOLD=5
                def between_threshold(a, expected):
                    return abs(a - expected) <= THRESHOLD

        :param window: A (start, end) tuple to limit the scope of the
            calculation.
        :type window: tuple

        :param align:
            :code:`"start"` aligns period calculation to switch-in events
            :code:`"end"` aligns the calculation to switch-out events
        :type param: str

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.getPeriod`
        """

        period = self.getPeriod(window, align)
        return operator(period, expected_value)

    def getDutyCycle(self, window):
        """Return the duty cycle of the task

        :param window: A (start, end) tuple to limit the scope of the
            calculation.
        :type window: tuple

        Duty Cycle:
            The percentage of time the task spends executing
            in the given window of time

            .. math::

                \delta_{cycle} = \\frac{T_{exec} \\times 100}{T_{window}}

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.assertDutyCycle`
        """

        return self.getRuntime(window, percent=True)

    def assertDutyCycle(self, expected_value, operator, window):
        """
        :param operator: A binary operator function that returns
            a boolean. For example:
            ::

                import operator
                op = operator.ge
                assertPeriod(expected_value, op)

            Will do the following check:
            ::

                getPeriod() >= expected_value

            A custom function can also be passed:
            ::

                THRESHOLD=5
                def between_threshold(a, expected):
                    return abs(a - expected) <= THRESHOLD

        :param window: A (start, end) tuple to limit the scope of the
            calculation.
        :type window: tuple

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.getDutyCycle`

        """
        return self.assertRuntime(
            expected_value,
            operator,
            window,
            percent=True)

    def getFirstCpu(self, window=None):
        """
        :return: The first CPU the task ran on

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.assertFirstCPU`
        """

        agg = self._aggregator(sched_funcs.first_cpu)
        result = agg.aggregate(level="cpu", window=window)
        result = list(itertools.chain.from_iterable(result))

        min_time = min(result)
        if math.isinf(min_time):
            return -1
        index = result.index(min_time)
        return self._topology.get_node("cpu", index)[0]

    def assertFirstCpu(self, cpus, window=None):
        """
        Check if the Task started (first ran on in the duration
        of the trace) on a particular CPU(s)

        :param cpus: A list of acceptable CPUs
        :type cpus: int, list

        .. seealso:: :mod:`bart.sched.SchedAssert.SchedAssert.getFirstCPU`
        """

        first_cpu = self.getFirstCpu(window=window)
        cpus = Utils.listify(cpus)
        return first_cpu in cpus

    def getLastCpu(self, window=None):
        """Return the last CPU the task ran on"""

        agg = self._aggregator(sched_funcs.last_cpu)
        result = agg.aggregate(level="cpu", window=window)
        result = list(itertools.chain.from_iterable(result))

        end_time = max(result)
        if not end_time:
            return -1

        return result.index(end_time)

    def generate_events(self, level, start_id=0, window=None):
        """Generate events for the trace plot

        .. note::
            This is an internal function accessed by the
            :mod:`bart.sched.SchedMultiAssert` class for plotting data
        """

        agg = self._aggregator(sched_funcs.trace_event)
        result = agg.aggregate(level=level, window=window)
        events = []

        for idx, level_events in enumerate(result):
            if not len(level_events):
                continue
            events += np.column_stack((level_events, np.full(len(level_events), idx))).tolist()

        return sorted(events, key = lambda x : x[0])

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

        events = {}
        events[self.name] = self.generate_events(level, window)
        names = [self.name]
        num_lanes = self._topology.level_span(level)
        lane_prefix = level.upper() + ": "
        return trappy.EventPlot(events, names, xlim,
                                lane_prefix=lane_prefix,
                                num_lanes=num_lanes)
