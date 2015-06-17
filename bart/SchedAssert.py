# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        SchedAssert.py
# ----------------------------------------------------------------
# $
#
"""A library for asserting scheduler scenarios based on the
statistics aggregation framework"""

import cr2
import itertools
import math
from cr2.plotter.Utils import listify
from cr2.stats.Aggregator import MultiTriggerAggregator
from cr2.stats import SchedConf as sconf
from sheye import Utils

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
class SchedAssert(object):

    """The primary focus of this class is to assert and verify
    predefined scheduler scenarios. This does not compare parameters
    across runs"""

    def __init__(self, run, topology, execname=None, pid=None):
        """Args:
                run (cr2.Run): A single cr2.Run object
                    or a path that can be passed to cr2.Run
                topology(cr2.stats.Topology): The CPU topology
                execname(str, optional): Optional execname of the task
                     under consideration.
                PID(int): The PID of the task to be checked

            One of pid or execname is mandatory. If only execname
            is specified, The current implementation will fail if
            there are more than one processes with the same execname
        """

        run = Utils.init_run(run)

        if not execname and not pid:
            raise ValueError("Need to specify at least one of pid or execname")

        self.execname = execname
        self._run = run
        self._pid = self._validate_pid(pid)
        self._aggs = {}
        self._topology = topology
        self._triggers = sconf.sched_triggers(self._run, self._pid,
                                              cr2.sched.SchedSwitch)
        self.name = "{}-{}".format(self.execname, self._pid)

    def _validate_pid(self, pid):
        """Validate the passed pid argument"""

        if not pid:
            pids = sconf.get_pids_for_process(self._run,
                                              self.execname)

            if len(pids) != 1:
                raise RuntimeError(
                    "There should be exactly one PID {0} for {1}".format(
                        pids,
                        self.execname))

            return pids[0]

        elif self.execname:

            pids = sconf.get_pids_for_process(self._run,
                                              self.execname)
            if pid not in pids:
                raise RuntimeError(
                    "PID {0} not mapped to {1}".format(
                        pid,
                        self.execname))
        else:
            self.execname = sconf.get_task_name(self._run, pid)

        return pid

    def _aggregator(self, aggfunc):
        """
        Returns an aggregator corresponding to the
        aggfunc, the aggregators are memoized for performance

        Args:
            aggfunc (function(pandas.Series)): Function parameter that
            accepts a pandas.Series object and returns a vector/scalar result
        """

        if aggfunc not in self._aggs.keys():
            self._aggs[aggfunc] = MultiTriggerAggregator(self._triggers,
                                                         self._topology,
                                                         aggfunc)
        return self._aggs[aggfunc]

    def getResidency(self, level, node, window=None, percent=False):
        """
        Residency of the task is the amount of time it spends executing
        a particular node of a topological level. For example:

        clusters=[]
        big = [1,2]
        little = [0,3,4,5]

        topology = Topology(clusters=clusters)

        level="cluster"
        node = [1,2]

        Will return the residency of the task on the big cluster. If
        percent is specified it will be normalized to the total RUNTIME
        of the TASK

        Args:
            level (hashable): The level to which the node belongs
            node (list): The node for which residency needs to calculated
            window (tuple): A (start, end) tuple to limit the scope of the
                residency calculation.
            percent: If true the result is normalized to the total runtime
                of the task and returned as a percentage
        """

        # Get the index of the node in the level
        node_index = self._topology.get_index(level, node)

        agg = self._aggregator(sconf.residency_sum)
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
        Args:
            level (hashable): The level to which the node belongs
            node (list): The node for which residency needs to assert
            expected_value (double): The expected value of the residency
            operator (function): A binary operator function that returns
                a boolean
            window (tuple): A (start, end) tuple to limit the scope of the
                residency calculation.
            percent: If true the result is normalized to the total runtime
                of the task and returned as a percentage
        """
        node_value = self.getResidency(level, node, window, percent)
        return operator(node_value, expected_value)

    def getStartTime(self):
        """
        Returns the first time the task ran
        (across all CPUs)
        """

        agg = self._aggregator(sconf.first_time)
        result = agg.aggregate(level="all", value=sconf.TASK_RUNNING)
        return min(result[0])

    def getEndTime(self):
        """
        Returns the last time the task ran
        (across all CPUs)
        """

        agg = self._aggregator(sconf.first_time)
        agg = self._aggregator(sconf.last_time)
        result = agg.aggregate(level="all", value=sconf.TASK_RUNNING)
        return max(result[0])
