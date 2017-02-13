# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from collections import namedtuple, OrderedDict

import pandas as pd
import numpy as np

from devlib.utils.misc import memoized

ActiveState = namedtuple("ActiveState", ["capacity", "power"])
ActiveState.__new__.__defaults__ = (None, None)

class EnergyModelNode(namedtuple("EnergyModelNode",
                                 ["cpus", "active_states", "idle_states",
                                  "power_domain", "freq_domain"])):
    @property
    def max_capacity(self):
        return max(s.capacity for s in self.active_states.values())

    def idle_state_idx(self, state):
        return self.idle_states.keys().index(state)

EnergyModelNode.__new__.__defaults__ = (None, None, None, None, None)

class PowerDomain(object):
    def __init__(self, idle_states, parent, cpus):
        self.cpus = set()
        self.idle_states = idle_states

        self.parent = parent
        self.add_cpus(set(cpus))

    def add_cpus(self, cpus):
        self.cpus = self.cpus.union(cpus)
        if self.parent:
            self.parent.add_cpus(self.cpus)

    def __repr__(self):
        return "PowerDomain(cpus={})".format(list(self.cpus))

class EnergyModel(object):
    """
    Represents hierarchical CPU topology with power and capacity data

    Describes a CPU topology similarly to trappy's Topology class, additionally
    describing relative CPU compute capacity, frequency domains and energy costs
    in various configurations.

    The topology is stored in "levels", currently hard-coded to be "cpu" and
    "cluster". Each level is a list of EnergyModelNode objects. An EnergyModel
    node is a CPU or group of CPUs with associated power and (optionally)
    capacity characteristics.
    """

    # TODO check that this is the highest cap available
    capacity_scale = 1024

    def __init__(self, levels=None):
        self._levels = levels

        self.num_cpus = len(self._levels["cpu"])
        self.cpus = [n.cpus[0] for n in levels["cpu"]]
        if self.cpus != range(self.num_cpus):
            raise ValueError("CPUs are sparse or out of order")
        if any(len(n.cpus) != 1 for n in levels["cpu"]):
            raise ValueError("'cpu' level nodes must all have exactly 1 CPU")

    @property
    @memoized
    def biggest_cpus(self):
        max_cap = max(n.max_capacity for n in self._levels["cpu"])
        return [n.cpus[0] for n in self._levels["cpu"]
                if n.max_capacity == max_cap]

    @property
    @memoized
    def littlest_cpus(self):
        min_cap = min(n.max_capacity for n in self._levels["cpu"])
        return [n.cpus[0] for n in self._levels["cpu"]
                if n.max_capacity == min_cap]


    @property
    @memoized
    def is_heterogeneous(self):
        """
        True iff CPUs do not all have the same efficiency and OPP range
        """
        states = self._levels["cpu"][0].active_states
        return any(c.active_states != states for c in self._levels["cpu"][1:])
