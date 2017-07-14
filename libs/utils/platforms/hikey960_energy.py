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
from energy_model import (ActiveState, EnergyModelNode, EnergyModelRoot,
                          PowerDomain, EnergyModel)

from collections import OrderedDict


#
# the raw hikey960 energy model data is simply calculated based on Pixel OPP
#

#
# LITTLE CA53 Core
#

CA53_cluster_active_states = OrderedDict([
    ( 533000, ActiveState(power=4)),
    ( 999000, ActiveState(power=16)),
    ( 1402000, ActiveState(power=49)),
    ( 1709000, ActiveState(power=55)),
    ( 1844000, ActiveState(power=70))
])

CA53_cluster_idle_states = OrderedDict([
    ('WFI', 20),
    ('cpu-sleep', 20),
    ('cluster-sleep', 0)
])

CA53_cpu_active_states = OrderedDict([
    ( 533000,   ActiveState(capacity=257,   power=160)),
    ( 999000,   ActiveState(capacity=475,   power=390)),
    ( 1402000,  ActiveState(capacity=684,   power=752)),
    ( 1709000,  ActiveState(capacity=810,   power=1060)),
    ( 1844000,  ActiveState(capacity=870,   power=1210))
])

CA53_cpu_idle_states = OrderedDict([
    ('WFI', 6),
    ('cpu-sleep', 0),
    ('cluster-sleep', 0)
])


#
# Big CA73 Core
#

CA73_cluster_active_states = OrderedDict([
    ( 903000, ActiveState(power=15)),
    ( 1421000, ActiveState(power=49)),
    ( 1805000, ActiveState(power=72)),
    ( 2112000, ActiveState(power=95)),
    ( 2362000, ActiveState(power=118))
])

CA73_cluster_idle_states = OrderedDict([
    ('WFI', 47),
    ('cpu-sleep', 47),
    ('cluster-sleep', 0)
])

CA73_cpu_active_states = OrderedDict([
    ( 903000,  ActiveState(capacity=428,  power=332)),
    ( 1421000,  ActiveState(capacity=684,  power=752)),
    ( 1805000,  ActiveState(capacity=868,  power=1209)),
    ( 2112000,  ActiveState(capacity=979,  power=1520)),
    ( 2362000, ActiveState(capacity=1024, power=1715))
])

CA73_cpu_idle_states = OrderedDict([
    ('WFI', 10),
    ('cpu-sleep', 0),
    ('cluster-sleep', 0)
])

little_cores = [0, 1, 2, 3]
big_cores = [4, 5, 6, 7]

def little_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=CA53_cpu_active_states,
                           idle_states=CA53_cpu_idle_states)

def big_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=CA73_cpu_active_states,
                           idle_states=CA73_cpu_idle_states)


def cpu_pd(cpu):
    return PowerDomain(cpu=cpu, idle_states=['WFI', 'cpu-sleep'])

hikey960_energy = EnergyModel(
    root_node=EnergyModelRoot(children=[
        EnergyModelNode(name='cluster0',
                        children=[little_cpu_node(c) for c in little_cores ],
                        active_states=CA53_cluster_active_states,
                        idle_states=CA53_cluster_idle_states),
        EnergyModelNode(name='cluster1',
                        children=[big_cpu_node(c) for c in big_cores ],
                        active_states=CA73_cluster_active_states,
                        idle_states=CA73_cluster_idle_states)]),
    root_power_domain=PowerDomain(idle_states=[], children=[
        PowerDomain(idle_states=["cluster-sleep"], children=[
            cpu_pd(c) for c in little_cores ]),
        PowerDomain(idle_states=["cluster-sleep"], children=[
            cpu_pd(c) for c in big_cores ])]),
    freq_domains=[little_cores,big_cores])
