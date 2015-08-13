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

"""
The SchedMatrix provides an ability to compare two executions
of benchmarks with multiple processes.

For example, consider a benchmark that spawns 4 identical threads
and any two threads should exhibit a certain behaviours and the
remaining another identical but different behaviour.

SchedMatrix creates a Matrix of Scheduler Waveform Correlations

A = Reference Execution
B = Execution to be Evaluated

           +---+  +---+
           |   |  |   |
A1, B3 +---+   +--+   +--------------+
                      +---+  +---+
                      |   |  |   |
A2, B4 +--------------+   +--+   +---+
           +---+  +---+
           |   |  |   |
A3, B1 +---+   +--+   +--------------+
                      +---+  +---+
                      |   |  |   |
A4, B2 +--------------+   +--+   +---+


Correlation Matrix

    B1   B2   B3   B4
A1  1    0    1    0

A2  0    1    0    1

A3  1    0    1    0

A4  0    1    0    1


Thus a success criteria can be defined as

A1 has two similar threads in the
evaluated execution

assertSiblings(A1, 2, operator.eq)
assertSiblings(A2, 2, operator.eq)
assertSiblings(A3, 2, operator.eq)
assertSiblings(A4, 2, operator.eq)
"""


import sys
import trappy
import numpy as np
from trappy.stats.Aggregator import MultiTriggerAggregator
from trappy.stats.Correlator import Correlator
from trappy.plotter.Utils import listify
from trappy.stats import SchedConf as sconf
from bart import Utils

POSITIVE_TOLERANCE = 0.80

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments


class SchedMatrix(object):

    """Valid cases are:

        * Single execname, multiple PIDs
        * PID List
        * Multiple execname, one-to-one PID
            association
    """

    def __init__(
            self,
            reference_trace,
            trace,
            topology,
            execnames,
            aggfunc=sconf.csum):

        run = Utils.init_run(trace)
        reference_run = Utils.init_run(reference_trace)

        self._execnames = listify(execnames)
        self._reference_pids = self._populate_pids(reference_run)
        self._pids = self._populate_pids(run)
        self._dimension = len(self._pids)
        self._topology = topology
        self._matrix = self._generate_matrix(run, reference_run, aggfunc)

        if len(self._pids) != len(self._reference_pids):
            raise RuntimeError(
                "The runs do not have the same number of PIDs for {0}".format(
                    str(execnames)))

    def _populate_pids(self, run):
        """Populate the qualifying PIDs from the run"""

        if len(self._execnames) == 1:
            return sconf.get_pids_for_process(run, self._execnames[0])

        pids = []

        for proc in self._execnames:
            pids += sconf.get_pids_for_process(run, proc)

        return list(set(pids))

    def _generate_matrix(self, run, reference_run, aggfunc):
        """Generate the Correlation Matrix"""

        reference_aggs = []
        aggs = []

        for idx in range(self._dimension):

            reference_aggs.append(
                MultiTriggerAggregator(
                    sconf.sched_triggers(
                        reference_run,
                        self._reference_pids[idx],
                        trappy.sched.SchedSwitch
                        ),
                    self._topology,
                    aggfunc))

            aggs.append(
                MultiTriggerAggregator(
                    sconf.sched_triggers(
                        run,
                        self._pids[idx],
                        trappy.sched.SchedSwitch
                        ),
                    self._topology,
                    aggfunc))

        agg_pair_gen = ((r_agg, agg)
                        for r_agg in reference_aggs for agg in aggs)

        # pylint fails to recognize numpy members.
        # pylint: disable=no-member
        matrix = np.zeros((self._dimension, self._dimension))
        # pylint: enable=no-member

        for (ref_result, test_result) in agg_pair_gen:
            i = reference_aggs.index(ref_result)
            j = aggs.index(test_result)
            corr = Correlator(
                ref_result,
                test_result,
                corrfunc=sconf.binary_correlate,
                filter_gaps=True)
            _, total = corr.correlate(level="cluster")

            matrix[i][j] = total

        return matrix

    def print_matrix(self):
        """Print the correlation matrix"""

        # pylint fails to recognize numpy members.
        # pylint: disable=no-member
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=False)
        np.savetxt(sys.stdout, self._matrix, "%5.5f")
        # pylint: enable=no-member

    def getSiblings(self, pid, tolerance=POSITIVE_TOLERANCE):
        """Return the number of processes in the
           reference trace that have a correlation
           greater than tolerance
        """

        ref_pid_idx = self._reference_pids.index(pid)
        pid_result = self._matrix[ref_pid_idx]
        return len(pid_result[pid_result > tolerance])

    def assertSiblings(self, pid, expected_value, operator,
                       tolerance=POSITIVE_TOLERANCE):
        """Assert that the number of siblings in the reference
           trace match the expected value and the operator

           Args:
               pid: The PID in the reference trace
               expected_value: the second argument to the operator
               operator: a function of the type f(a, b) that returns
                    a boolean
         """
        num_siblings = self.getSiblings(pid, tolerance)
        return operator(num_siblings, expected_value)
