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

"""Indexers are responsible for providing indexes for
   aggregations and provide specific functions like
   unification and resampling.
"""

import pandas as pd
import numpy as np
from trappy.plotter.Utils import listify
from trappy.stats import StatConf

class Indexer(object):
    """Indexer base class is an encapsulation
    around the pandas Index object with some
    special functionality

    :param index: Pandas index object. This can be
                    non-unoform and non-unique
    :type index: :mod:`pandas.Index`

    :param runs: trappy Run list/singular object
    :type runs: :mod:`trappy.run.Run`
    """

    def __init__(self, index):
        self.index = index

    def series(self):
        """Returns an empty series with the initialized index
        """
        return pd.Series(np.zeros(len(self.index)), index=self.index)

    def get_uniform(self, delta=StatConf.DELTA_DEFAULT):
        """
        :param delta: Difference between two indices. This has a
            default value specified in StatConf.DELTA_DEFAULT
        :type delta: float

        :return: A uniformly spaced index.
        """

        uniform_start = self.index.values[0]
        uniform_end = self.index.values[-1]
        new_index = np.arange(uniform_start, uniform_end, delta)
        return new_index

def get_unified_indexer(indexers):
    """Unify the List of Indexers

    :param indexers: A list of indexers
    :type indexers: :mod:`trappy.stats.Indexer.Indexer`

    :return: A :mod:`pandas.Indexer.Indexer`
        with a unfied index
    """

    new_index = indexers[0].index

    for idx in indexers[1:]:
        new_index = new_index.union(idx.index)

    return Indexer(new_index)

class MultiTriggerIndexer(Indexer):
    """"The index unifies the indices of all trigger
     events.

    :param triggers: A (list or single) trigger
    :type triggers: :mod:`trappy.stats.Trigger.Trigger`
    """

    def __init__(self, triggers):

        self._triggers = listify(triggers)
        super(MultiTriggerIndexer, self).__init__(self._unify())

    def _unify(self):
        """Function to unify all the indices of each trigger
        """

        idx = pd.Index([])
        for trigger in self._triggers:
            run = trigger.run
            trappy_event = getattr(run, trigger.template.name)
            idx = idx.union(trappy_event.data_frame.index)


        return pd.Index(np.unique(idx.values))
