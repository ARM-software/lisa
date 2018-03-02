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
from collections import OrderedDict

from energy_model import (ActiveState, EnergyModelNode, EnergyModelRoot,
                          PowerDomain, EnergyModel)

a53_cluster_active_states = OrderedDict([
    (533000, ActiveState(power=12)),
    (999000, ActiveState(power=22)),
    (1402000, ActiveState(power=36)),
    (1709000, ActiveState(power=67)),
    (1844000, ActiveState(power=144)),
])

# TODO warn if any of the idle states aren't represented by power domains
a53_cluster_idle_states = OrderedDict([
    ("WFI",               12),
    ("cpu-sleep-0",       12),
    ("cluster-sleep-0",   0),
])

a53_cpu_active_states = OrderedDict([
    (533000, ActiveState(capacity=133, power=87)),
    (999000, ActiveState(capacity=250, power=164)),
    (1402000, ActiveState(capacity=351, power=265)),
    (1709000, ActiveState(capacity=429, power=388)),
    (1844000, ActiveState(capacity=462, power=502)),
])

a53_cpu_idle_states = OrderedDict([
    ("WFI",               5),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

a53s = [0, 1, 2, 3]

def a53_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=a53_cpu_active_states,
                           idle_states=a53_cpu_idle_states)

a72_cluster_active_states = OrderedDict([
    (903000, ActiveState(power=102)),
    (1421000, ActiveState(power=124)),
    (1805000, ActiveState(power=221)),
    (2112000, ActiveState(power=330)),
    (2362000, ActiveState(power=433)),
])

a72_cluster_idle_states = OrderedDict([
    ("WFI",               102),
    ("cpu-sleep-0",       102),
    ("cluster-sleep-0",   0),
])

a72_cpu_active_states = OrderedDict([
    (903000, ActiveState(capacity=390, power=404)),
    (1421000, ActiveState(capacity=615, power=861)),
    (1805000, ActiveState(capacity=782, power=1398)),
    (2112000, ActiveState(capacity=915, power=2200)),
    (2362000, ActiveState(capacity=1024, power=2848)),
])

a72_cpu_idle_states = OrderedDict([
    ("WFI",               18),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

a72s = [4, 5, 6, 7]

def a72_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=a72_cpu_active_states,
                           idle_states=a72_cpu_idle_states)

hikey960_energy = EnergyModel(
    root_node=EnergyModelRoot(
        children=[
            EnergyModelNode(
                name="cluster_a53",
                active_states=a53_cluster_active_states,
                idle_states=a53_cluster_idle_states,
                children=[a53_cpu_node(c) for c in a53s]),
            EnergyModelNode(
                name="cluster_a72",
                active_states=a72_cluster_active_states,
                idle_states=a72_cluster_idle_states,
                children=[a72_cpu_node(c) for c in a72s])]),
    root_power_domain=PowerDomain(idle_states=[], children=[
        PowerDomain(
            idle_states=["cluster-sleep-0"],
            children=[PowerDomain(idle_states=["WFI", "cpu-sleep-0"], cpu=c)
                      for c in a72s]),
        PowerDomain(
            idle_states=["cluster-sleep-0"],
            children=[PowerDomain(idle_states=["WFI", "cpu-sleep-0"], cpu=c)
                      for c in a53s])]),
    freq_domains=[a53s, a72s])
