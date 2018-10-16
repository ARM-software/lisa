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

from lisa.energy_model import (ActiveState, EnergyModelNode, EnergyModelRoot,
                          PowerDomain, EnergyModel)

a53_cluster_active_states = OrderedDict([
    (450000, ActiveState(power=26)),
    (575000, ActiveState(power=30)),
    (700000, ActiveState(power=39)),
    (775000, ActiveState(power=47)),
    (850000, ActiveState(power=57)),
])

# TODO warn if any of the idle states aren't represented by power domains
a53_cluster_idle_states = OrderedDict([
    ("WFI",               56),
    ("cpu-sleep-0",       56),
    ("cluster-sleep-0",   17),
])

a53_cpu_active_states = OrderedDict([
    (450000, ActiveState(capacity=235, power=33)),
    (575000, ActiveState(capacity=302, power=46)),
    (700000, ActiveState(capacity=368, power=61)),
    (775000, ActiveState(capacity=406, power=76)),
    (850000, ActiveState(capacity=447, power=93)),
])

a53_cpu_idle_states = OrderedDict([
    ("WFI",               6),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

a53s = [0, 3, 4, 5]

def a53_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=a53_cpu_active_states,
                           idle_states=a53_cpu_idle_states)

a57_cluster_active_states = OrderedDict([
    ( 450000, ActiveState(power=24)),
    ( 625000, ActiveState(power=32)),
    ( 800000, ActiveState(power=43)),
    ( 950000, ActiveState(power=49)),
    (1100000, ActiveState(power=64)),
])

a57_cluster_idle_states = OrderedDict([
    ("WFI",               65),
    ("cpu-sleep-0",       65),
    ("cluster-sleep-0",   24),
])

a57_cpu_active_states = OrderedDict([
    (450000,  ActiveState(capacity=417,   power=168)),
    (625000,  ActiveState(capacity=579,   power=251)),
    (800000,  ActiveState(capacity=744,   power=359)),
    (950000,  ActiveState(capacity=883,   power=479)),
    (1100000, ActiveState(capacity=1023,  power=616)),
])

a57_cpu_idle_states = OrderedDict([
    ("WFI",               15),
    ("cpu-sleep-0",       0),
    ("cluster-sleep-0",   0),
])

a57s = [1, 2]

def a57_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=a57_cpu_active_states,
                           idle_states=a57_cpu_idle_states)

nrg_model = EnergyModel(
    root_node=EnergyModelRoot(
        children=[
            EnergyModelNode(
                name="cluster_a53",
                active_states=a53_cluster_active_states,
                idle_states=a53_cluster_idle_states,
                children=[a53_cpu_node(c) for c in a53s]),
            EnergyModelNode(
                name="cluster_a57",
                active_states=a57_cluster_active_states,
                idle_states=a57_cluster_idle_states,
                children=[a57_cpu_node(c) for c in a57s])]),
    root_power_domain=PowerDomain(idle_states=[], children=[
        PowerDomain(
            idle_states=["cluster-sleep-0"],
            children=[PowerDomain(idle_states=["WFI", "cpu-sleep-0"], cpu=c)
                      for c in a57s]),
        PowerDomain(
            idle_states=["cluster-sleep-0"],
            children=[PowerDomain(idle_states=["WFI", "cpu-sleep-0"], cpu=c)
                      for c in a53s])]),
    freq_domains=[a53s, a57s])

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
