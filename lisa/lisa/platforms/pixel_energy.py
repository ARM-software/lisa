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
from lisa.energy_model import (ActiveState, EnergyModelNode, EnergyModelRoot,
                          PowerDomain, EnergyModel)

from collections import OrderedDict

silver_cpu_active_states = OrderedDict([
    ( 307200, ActiveState(capacity=149, power=90)),
    ( 384000, ActiveState(capacity=188, power=111)),
    ( 460800, ActiveState(capacity=225, power=133)),
    ( 537600, ActiveState(capacity=257, power=160)),
    ( 614400, ActiveState(capacity=281, power=182)),
    ( 691200, ActiveState(capacity=315, power=210)),
    ( 768000, ActiveState(capacity=368, power=251)),
    ( 844800, ActiveState(capacity=406, power=306)),
    ( 902400, ActiveState(capacity=428, power=332)),
    ( 979200, ActiveState(capacity=469, power=379)),
    (1056000, ActiveState(capacity=502, power=438)),
    (1132800, ActiveState(capacity=538, power=494)),
    (1209600, ActiveState(capacity=581, power=550)),
    (1286400, ActiveState(capacity=611, power=613)),
    (1363200, ActiveState(capacity=648, power=670)),
    (1440000, ActiveState(capacity=684, power=752)),
    (1516800, ActiveState(capacity=729, power=848)),
    (1593600, ActiveState(capacity=763, power=925)),
])

silver_cluster_active_states = OrderedDict([
    ( 307200, ActiveState(power=4)),
    ( 384000, ActiveState(power=4)),
    ( 460800, ActiveState(power=4)),
    ( 537600, ActiveState(power=4)),
    ( 614400, ActiveState(power=4)),
    ( 691200, ActiveState(power=4)),
    ( 768000, ActiveState(power=8)),
    ( 844800, ActiveState(power=9)),
    ( 902400, ActiveState(power=15)),
    ( 979200, ActiveState(power=16)),
    (1056000, ActiveState(power=21)),
    (1132800, ActiveState(power=22)),
    (1209600, ActiveState(power=29)),
    (1286400, ActiveState(power=32)),
    (1363200, ActiveState(power=42)),
    (1440000, ActiveState(power=49)),
    # This power value is 41 (invalid!) in the released kernel. Patch it to
    # avoid errors.
    (1516800, ActiveState(power=52)),
    (1593600, ActiveState(power=52)),
])

gold_cpu_active_states = OrderedDict([
    ( 307200, ActiveState(capacity=149, power=93)),
    ( 384000, ActiveState(capacity=188, power=111)),
    ( 460800, ActiveState(capacity=225, power=133)),
    ( 537600, ActiveState(capacity=257, power=160)),
    ( 614400, ActiveState(capacity=281, power=182)),
    ( 691200, ActiveState(capacity=315, power=210)),
    ( 748800, ActiveState(capacity=348, power=252)),
    ( 825600, ActiveState(capacity=374, power=290)),
    ( 902400, ActiveState(capacity=428, power=332)),
    ( 979200, ActiveState(capacity=469, power=379)),
    (1056000, ActiveState(capacity=502, power=438)),
    (1132800, ActiveState(capacity=538, power=494)),
    (1209600, ActiveState(capacity=581, power=550)),
    (1286400, ActiveState(capacity=611, power=613)),
    (1363200, ActiveState(capacity=648, power=670)),
    (1440000, ActiveState(capacity=684, power=752)),
    (1516800, ActiveState(capacity=729, power=848)),
    (1593600, ActiveState(capacity=763, power=925)),
    (1670400, ActiveState(capacity=795, power=1018)),
    (1747200, ActiveState(capacity=832, power=1073)),
    (1824000, ActiveState(capacity=868, power=1209)),
    (1900800, ActiveState(capacity=905, power=1298)),
    (1977600, ActiveState(capacity=952, power=1428)),
    (2054400, ActiveState(capacity=979, power=1521)),
    (2150400, ActiveState(capacity=1024, power=1715)),
])

gold_cluster_active_states = OrderedDict([
    ( 307200, ActiveState(power=4)),
    ( 384000, ActiveState(power=4)),
    ( 460800, ActiveState(power=4)),
    ( 537600, ActiveState(power=4)),
    ( 614400, ActiveState(power=4)),
    ( 691200, ActiveState(power=4)),
    ( 748800, ActiveState(power=7)),
    ( 825600, ActiveState(power=10)),
    ( 902400, ActiveState(power=15)),
    ( 979200, ActiveState(power=16)),
    (1056000, ActiveState(power=21)),
    (1132800, ActiveState(power=22)),
    (1209600, ActiveState(power=29)),
    (1286400, ActiveState(power=32)),
    (1363200, ActiveState(power=42)),
    (1440000, ActiveState(power=49)),
    # This power value is 41 (invalid!) in the released kernel. Patch it to
    # avoid errors.
    (1516800, ActiveState(power=52)),
    (1593600, ActiveState(power=52)),
    (1670400, ActiveState(power=62)),
    (1747200, ActiveState(power=69)),
    (1824000, ActiveState(power=75)),
    (1900800, ActiveState(power=81)),
    (1977600, ActiveState(power=90)),
    (2054400, ActiveState(power=93)),
    (2150400, ActiveState(power=96)),
])

# TODO warn if any of the idle states aren't represented by power domains
cpu_idle_states = OrderedDict([
    ("WFI",               2),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

cluster_idle_states = OrderedDict([
    ("WFI",               0),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

silvers = [0, 1]
golds = [2, 3]

def silver_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=silver_cpu_active_states,
                           idle_states=cpu_idle_states)

def gold_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=gold_cpu_active_states,
                           idle_states=cpu_idle_states)

def cpu_pd(cpu):
    return PowerDomain(cpu=cpu, idle_states=["WFI", "cpu-sleep-0"])

pixel_energy = EnergyModel(
    root_node=EnergyModelRoot(children=[
        EnergyModelNode(name='cluster_silver',
                        children=[silver_cpu_node(c) for c in silvers],
                        active_states=silver_cluster_active_states,
                        idle_states=cluster_idle_states),
        EnergyModelNode(name='cluster_gold',
                        children=[gold_cpu_node(c) for c in golds],
                        active_states=gold_cluster_active_states,
                        idle_states=cluster_idle_states)]),
    root_power_domain=PowerDomain(idle_states=[], children=[
        PowerDomain(idle_states=['cluster-sleep-0'], children=[
            cpu_pd(c) for c in silvers]),
        PowerDomain(idle_states=['cluster-sleep-0'], children=[
            cpu_pd(c) for c in golds])]),
    freq_domains=[silvers, golds])

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
