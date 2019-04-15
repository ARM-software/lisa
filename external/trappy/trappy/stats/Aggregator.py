#    Copyright 2015-2017 ARM Limited
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

"""Aggregators are responsible for aggregating information
for further analysis. These aggregations can produce
both scalars and vectors and each aggregator implementation
is expected to handle its "aggregation" mechanism.
"""
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


from builtins import object
from trappy.utils import listify
from trappy.stats.Indexer import MultiTriggerIndexer
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass


class AbstractAggregator(with_metaclass(ABCMeta, object)):
    """Abstract class for all aggregators

    :param indexer: Indexer is passed on by the Child class
        for handling indices during correlation
    :type indexer: :mod:`trappy.stats.Indexer.Indexer`

    :param aggfunc: Function that accepts a pandas.Series and
        process it for aggregation.
    :type aggfunc: function
    """

    # The current implementation needs the index to
    # be unified across data frames to account for
    # variable sampling across data frames
    def __init__(self, indexer, aggfunc=None):

        self._result = {}
        self._aggregated = False
        self._aggfunc = aggfunc
        self.indexer = indexer

    def _add_result(self, pivot, series):
        """Add the result for the given pivot and trace

        :param pivot: The pivot for which the result is being generated
        :type pivot(hashable)

        :param series: series to be added to result
        :type series: :mod:`pandas.Series`
        """

        if pivot not in self._result:
            self._result[pivot] = self.indexer.series()

        for idx in series.index:
                self._result[pivot][idx] = series[idx]

    @abstractmethod
    def aggregate(self, trace_idx, **kwargs):
        """Abstract Method for aggregating data for various
        pivots.

        :param trace_idx: Index of the trace to be aggregated
        :type trace_idx: int

        :return: The aggregated result

        """

        raise NotImplementedError("Method Not Implemented")


class MultiTriggerAggregator(AbstractAggregator):

    """This aggregator accepts a list of triggers and each trigger has
     a value associated with it.
    """

    def __init__(self, triggers, topology, aggfunc=None):
        """
        :param triggers: trappy.stat.Trigger): A list or a singular trigger object
        :type triggers: :mod:`trappy.stat.Trigger.Trigger`

        :param topology (trappy.stat.Topology): A topology object for aggregation
                levels
        :type topology: :mod:`trappy.stat.Topology`

        :param aggfunc: A function to be applied on each series being aggregated.
                For each topology node, a series will be generated and this
                will be processed by the aggfunc
        :type aggfunc: function
        """
        self._triggers = triggers
        self.topology = topology
        super(
            MultiTriggerAggregator,
            self).__init__(MultiTriggerIndexer(triggers), aggfunc)

    def aggregate(self, **kwargs):
        """
        Aggregate implementation that aggregates
        triggers for a given topological level. All the arguments passed to
        it are forwarded to the aggregator function except level (if present)

        :return: A scalar or a vector aggregated result. Each group in the
            level produces an element in the result list with a one to one
            index correspondence
            ::

                groups["level"] = [[1,2], [3,4]]
                result = [result_1, result_2]
        """

        level = kwargs.pop("level", "all")

        # This function is a hot spot in the code. It is
        # worth considering a memoize decorator to cache
        # the function. The memoization can also be
        # maintained by the aggregator object. This will
        # help the code scale efficeintly
        level_groups = self.topology.get_level(level)
        result = []


        if not self._aggregated:
            self._aggregate_base()

        for group in level_groups:
            group = listify(group)
            if self._aggfunc is not None:
                level_res = self._aggfunc(self._result[group[0]], **kwargs)
            else:
                level_res = self._result[group[0]]

            for node in group[1:]:
                if self._aggfunc is not None:
                    node_res = self._aggfunc(self._result[node], **kwargs)
                else:
                    node_res = self._result[node]

                level_res += node_res

            result.append(level_res)

        return result

    def _aggregate_base(self):
        """A memoized function to generate the base series
        for each node in the flattened topology
        ::

            topo["level_1"] = [[1, 2], [3, 4]]

       This function will generate the fundamental
       aggregations for all nodes 1, 2, 3, 4 and
       store the result in _agg_result
       """

        for trigger in self._triggers:
            for node in self.topology.flatten():
                result_series = trigger.generate(node)
                self._add_result(node, result_series)

        self._aggregated = True
