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
# File:        Aggregator.py
# ----------------------------------------------------------------
# $
#
"""Aggregators are responsible for aggregating information
   for further analysis. These aggregations can produce
   both scalars and vectors and each aggregator implementation
   is expected to handle its "aggregation" mechanism.
"""


from cr2.plotter.Utils import listify
from cr2.stats.Indexer import MultiTriggerIndexer
from abc import ABCMeta, abstractmethod


class AbstractAggregator(object):

    """Abstract class for all aggregators"""

    __metaclass__ = ABCMeta

    # The current implementation needs the index to
    # be unified across data frames to account for
    # variable sampling across data frames
    def __init__(self, indexer, aggfunc=None):
        """Args:

            indexer (Indexer): Indexer iss passed on by the Child class
                for handling indices during correlation
            aggfunc (function): Function that accepts a pandas.Series and
                process it for aggregation.
        """

        self._result = {}
        self._aggregated = False
        self._aggfunc = aggfunc
        self.indexer = indexer

    def _add_result(self, pivot, data_frame, value):
        """Add the result for the given pivot and run

          Args:
             pivot (hashable): The pivot for which the result is being generated
             data_frame (pandas.DataFrame): pandas data frame of result values
             value (str, numeric): If value is str, the corresponding
                 column is used as a vector of resultant values. If
                 numeric, each index in data frame gets the numeric

        """

        if pivot not in self._result:
            self._result[pivot] = self.indexer.series()

        for idx in data_frame.index:
            if isinstance(value, basestring):
                self._result[pivot][idx] = data_frame[value][idx]
            else:
                self._result[pivot][idx] = value

    @abstractmethod
    def aggregate(self, run_idx, **kwargs):
        """Abstract Method for aggregating data for various
           pivots.

            Args:
                run_idx: Index of the run to be aggregated

            Returns:
                The aggregated result

        """

        raise NotImplementedError("Method Not Implemented")


class MultiTriggerAggregator(AbstractAggregator):

    """This aggregator accepts a list of triggers and each trigger has
     a value associated with it.
    """

    def __init__(self, triggers, topology, aggfunc=None):
        """
            Args:

            triggers (cr2.stat.Trigger): A list or a singular trigger object
            topology (cr2.stat.Topology): A topology object for aggregation
                levels
            aggfunc: A function to be applied on each series being aggregated.
                For each topology node, a series will be generated and this
                will be processed by the aggfunc

        """
        self._triggers = triggers
        self.topology = topology
        super(
            MultiTriggerAggregator,
            self).__init__(MultiTriggerIndexer(triggers), aggfunc)

    def aggregate(self, **kwargs):
        """
            Aggregate implementation that aggrgates
            triggers for a given topological level

            Args:
                level can be specified. If not the default level is
                taken to be all

            Returns:
                A scalar or a vector aggregated result.
                Each group in the level produces an element
                in the result list with a one to one
                index correspondence

                groups["level"] = [[1,2], [3,4]]
                result = [result_1, result_2]
        """

        level = kwargs.get("level", "all")

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
            level_res = self._aggfunc(self._result[group[0]])

            for node in group[1:]:
                if self._aggfunc is not None:
                    node_res = self._aggfunc(self._result[node])
                else:
                    node_res = self._result[node]

                level_res += node_res

            result.append(level_res)

        return result

    def _aggregate_base(self):
        """A memoized function to generate the base series
           for each node in the flattened topology.

            eg topo["level_1"] = [[1, 2], [3, 4]]
            This function will generate the fundamental
            aggregations for all nodes 1, 2, 3, 4 and
            store the result in _agg_result

        """

        for trigger in self._triggers:
            for node in self.topology.get_level("all"):
                result_df = trigger.generate(node)
                self._add_result(node, result_df, trigger.value)

        self._aggregated = True
