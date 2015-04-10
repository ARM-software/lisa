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
# File:        Indexer.py
# ----------------------------------------------------------------
# $
#
"""Indexers are responsible for providind indexes for
   aggregations and provide specific functions like
   unification and resampling.
"""

import pandas as pd
import numpy as np
from cr2.plotter.Utils import listify
from cr2.stats import StatConf

class Indexer(object):
    """Indexer base class is an encapsulation
       around the pandas Index object with some
       special functionality"""

    def __init__(self, index):
        """
            Args:
                index (pandas.Index): Pandas index. This can be
                    non-unoform and non-unique
                runs (cr2.Run): CR2 Run list/singular object
        """
        self.index = index

    def series(self):
        """Returns an empty series with the initialized index
        """
        return pd.Series(np.zeros(len(self.index)), index=self.index)

    def get_uniform(self, delta=StatConf.DELTA_DEFAULT):
        """
            Args:
                delta: Difference between two indices. This has a
                    default value specified in StatConf.DELTA_DEFAULT

            Returns:
                Returns a uniformly spaced index.
        """

        uniform_start = self.index.values[0]
        uniform_end = self.index.values[-1]
        new_index = np.arange(uniform_start, uniform_end, delta)
        return new_index

def get_unified_indexer(indexers):
    """Unify the List of Indexers

        Args:
            indexers (stats.Indexer): A list of indexers

        Returns:
            An indexer with a unfied index
    """


    new_index = indexers[0].index

    for idx in indexers[1:]:
        new_index = new_index.union(idx.index)

    return Indexer(new_index)

class MultiTriggerIndexer(Indexer):
    """"The index unifies the indices of all trigger
     events.
    """


    def __init__(self, triggers):
        """
            Args:
                triggers (stat.Trigger): A list (or single) trigger
        """

        self._triggers = listify(triggers)
        super(MultiTriggerIndexer, self).__init__(self._unify())

    def _unify(self):
        """Function to unify all the indices of each trigger
        """

        idx = pd.Index([])
        for trigger in self._triggers:
            run = trigger.run
            cr2_event = getattr(run, trigger.template.name)
            idx = idx.union(cr2_event.data_frame.index)


        return pd.Index(np.unique(idx.values))
