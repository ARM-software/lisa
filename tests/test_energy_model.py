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
from unittest import TestCase
import os
import shutil
import tempfile

import pytest

from devlib.target import KernelVersion

from lisa.energy_model import (EnergyModel, ActiveState, EnergyModelCapacityError,
                               EnergyModelNode, EnergyModelRoot, PowerDomain)
from lisa.platforms.platinfo import PlatformInfo
from lisa.trace import Trace, TxtTraceParser
from .utils import StorageTestCase

""" A very basic test suite for the EnergyModel class."""

# WARNING!
# Note that the tests below have hard-coded expectations about the result. If
# you change the numbers in this EM, you'll need to recalculate those hard-coded
# values (or just refactor these tests)

little_cluster_active_states = OrderedDict([
    (1000, ActiveState(power=10)),
    (1500, ActiveState(power=15)),
    (2000, ActiveState(power=20)),
])

little_cluster_idle_states = OrderedDict([
    ('WFI', 5),
    ('cpu-sleep-0', 5),
    ('cluster-sleep-0', 1),
])

little_cpu_active_states = OrderedDict([
    (1000, ActiveState(capacity=100, power=100)),
    (1500, ActiveState(capacity=150, power=150)),
    (2000, ActiveState(capacity=200, power=200)),
])

little_cpu_idle_states = OrderedDict([
    ('WFI', 5),
    ('cpu-sleep-0', 0),
    ('cluster-sleep-0', 0),
])

littles = [0, 1]


def little_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=little_cpu_active_states,
                           idle_states=little_cpu_idle_states)


big_cluster_active_states = OrderedDict([
    (3000, ActiveState(power=30)),
    (4000, ActiveState(power=40)),
])

big_cluster_idle_states = OrderedDict([
    ('WFI', 8),
    ('cpu-sleep-0', 8),
    ('cluster-sleep-0', 2),
])

big_cpu_active_states = OrderedDict([
    (3000, ActiveState(capacity=300, power=300)),
    (4000, ActiveState(capacity=400, power=400)),
])

big_cpu_idle_states = OrderedDict([
    ('WFI', 9),
    ('cpu-sleep-0', 0),
    ('cluster-sleep-0', 0),
])

bigs = [2, 3]


def big_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=big_cpu_active_states,
                           idle_states=big_cpu_idle_states)


em = EnergyModel(
    root_node=EnergyModelRoot(children=[
        EnergyModelNode(name='cluster_little',
                        active_states=little_cluster_active_states,
                        idle_states=little_cluster_idle_states,
                        children=[little_cpu_node(0),
                                  little_cpu_node(1)]),
        EnergyModelNode(name='cluster_big',
                        active_states=big_cluster_active_states,
                        idle_states=big_cluster_idle_states,
                        children=[big_cpu_node(2),
                                  big_cpu_node(3)])
    ]),
    root_power_domain=PowerDomain(idle_states=[], children=[
        PowerDomain(
            idle_states=['cluster-sleep-0'],
            children=[PowerDomain(idle_states=['WFI', 'cpu-sleep-0'], cpu=c)
                      for c in littles]),
        PowerDomain(
            idle_states=['cluster-sleep-0'],
            children=[PowerDomain(idle_states=['WFI', 'cpu-sleep-0'], cpu=c)
                      for c in bigs]),
    ]),
    freq_domains=[littles, bigs]
)


class TestInvalid(TestCase):
    """Test the sanity checks in EnerygModel setup"""

    def test_overlapping_freq_doms(self):
        """Can't build an EM with energy nodes overlapping freq domains"""

        # To make this easy we'll just use a single active state everywhere, and
        # no idle states
        active_states = {10000: ActiveState(capacity=1024, power=100)}

        def cpu_node(cpu):
            return EnergyModelNode(cpu=cpu,
                                   active_states=active_states,
                                   idle_states=[])

        root_node = EnergyModelRoot(children=[
            EnergyModelNode(
                name='cluster1',
                active_states=active_states,
                idle_states=[],
                children=[cpu_node(0), cpu_node(1)])])

        def cpu_pd(cpu):
            return PowerDomain(idle_states=[], cpu=cpu)

        with pytest.raises(ValueError):
            EnergyModel(root_node=root_node,
                        root_power_domain=PowerDomain(
                            idle_states=[], children=[cpu_pd(0), cpu_pd(1)]),
                        freq_domains=[[0], [1]]),


class TestOptimalPlacement(TestCase):
    def assert_placement_list_equal(self, l1, l2):
        """
        Assert that a pair of lists of lists contain the same lists in any order
        """
        s1 = {tuple(l) for l in l1}
        s2 = {tuple(l) for l in l2}
        assert s1 == s2

    def test_single_small(self):
        placements = em.get_optimal_placements({'task0': 1})
        self.assert_placement_list_equal(placements, [[1, 0, 0, 0],
                                                   [0, 1, 0, 0]])

    def test_single_big(self):
        placements = em.get_optimal_placements({'task0': 350})
        self.assert_placement_list_equal(placements, [[0, 0, 350, 0],
                                                   [0, 0, 0, 350]])

    def test_packing(self):
        tasks = {'task' + str(i): 10 for i in list(range(5))}
        placements = em.get_optimal_placements(tasks)
        total_util = sum(tasks.values())
        self.assert_placement_list_equal(placements, [[total_util, 0, 0, 0],
                                                   [0, total_util, 0, 0]])

    def test_overutilized_single(self):
        with pytest.raises(EnergyModelCapacityError):
            em.get_optimal_placements({'task0': 401})

    def test_capacity_margin_single(self):
        with pytest.raises(EnergyModelCapacityError):
            em.get_optimal_placements(
                {'task0': 350},
                capacity_margin_pct=20
            )

    def test_overutilized_many(self):
        total_cap = 400 * 2 + 200 * 2
        task_size = 200
        tasks = {
            'task' + str(i): task_size
            for i in range((total_cap // task_size) + 1)
        }
        with pytest.raises(EnergyModelCapacityError):
            em.get_optimal_placements(tasks)


class TestBiggestCpus(TestCase):
    def test_biggest_cpus(self):
        assert em.biggest_cpus == [2, 3]


class TestLittlestCpus(TestCase):
    def test_littlest_cpus(self):
        assert em.littlest_cpus == [0, 1]


class TestMaxCap(TestCase):
    def test_max_cap(self):
        max_caps = [n.max_capacity for n in em.cpu_nodes]
        assert max_caps == [200, 200, 400, 400]


class TestEnergyEst(TestCase):
    def test_all_overutilized(self):
        big_cpu = 400 * 2
        little_cpu = 200 * 2
        big_cluster = 40
        little_cluster = 20

        total = big_cpu + little_cpu + big_cluster + little_cluster

        power = em.estimate_from_cpu_util([10000] * 4)
        exp = {
            (0): {'active': little_cpu, 'idle': 0},
            (1): {'active': little_cpu, 'idle': 0},
            (2): {'active': big_cpu, 'idle': 0},
            (3): {'active': big_cpu, 'idle': 0},
            (0, 1): {'active': little_cluster, 'idle': 0},
            (2, 3): {'active': big_cluster, 'idle': 0}
        }
        for k, v in power.items():
            assert v == pytest.approx(power[k])

    def test_all_idle(self):
        nrg = (
            # CPU power = 0
            0 * 4
            # big cluster power
            + 2
            # LITTLE cluster power
            + 1
        )
        assert sum(em.estimate_from_cpu_util([0, 0, 0, 0]).values()) == nrg

    def test_one_little_half_lowest(self):
        cpu0_util = 100 * 0.5
        nrg = (
            # CPU0 active power
            (0.5 * 100)
            # CPU0 idle power
            + (0.5 * 5)
            # LITTLE cluster idle power
            + (0.5 * 5)
            # LITTLE cluster active power
            + (0.5 * 10)
            # big cluster power
            + 2
        )
        assert sum(em.estimate_from_cpu_util([cpu0_util, 0, 0, 0]).values()) == nrg


class TestIdleStates(TestCase):
    def test_zero_util_deepest(self):
        assert em.guess_idle_states([0] * 4) == ['cluster-sleep-0'] * 4

    def test_single_cpu_used(self):
        states = em.guess_idle_states([0, 0, 0, 1])
        assert states == [
            'cluster-sleep-0',
            'cluster-sleep-0',
            'cpu-sleep-0',
            'WFI'
        ]

        states = em.guess_idle_states([0, 1, 0, 0])
        assert states == [
            'cpu-sleep-0',
            'WFI',
            'cluster-sleep-0',
            'cluster-sleep-0'
        ]

    def test_all_cpus_used(self):
        states = em.guess_idle_states([1, 1, 1, 1])
        assert states == ['WFI'] * 4

    def test_one_cpu_per_cluster(self):
        states = em.guess_idle_states([0, 1, 0, 1])
        assert states == ['cpu-sleep-0', 'WFI'] * 2


class TestFreqs(TestCase):

    def test_zero_util_slowest(self):
        assert em.guess_freqs([0] * 4) == [1000, 1000, 3000, 3000]

    def test_high_util_fastest(self):
        assert em.guess_freqs([100000] * 4) == [2000, 2000, 4000, 4000]

    def test_freq_domains(self):
        assert em.guess_freqs([0, 0, 0, 10000]) == [1000, 1000, 4000, 4000]

        assert em.guess_freqs([0, 10000, 0, 10000]) == [2000, 2000, 4000, 4000]

        assert em.guess_freqs([0, 10000, 0, 0]) == [2000, 2000, 3000, 3000]

    def test_middle_freq(self):
        assert em.guess_freqs([0, 110, 0, 0]) == [1500, 1500, 3000, 3000]


class TestNames(TestCase):
    """Test that the default names for CPU nodes get set"""

    def test_names(self):
        names = ['cpu0', 'cpu1', 'cpu2', 'cpu3']
        assert [n.name for n in em.cpu_nodes] == names


class TestCpuGroups(TestCase):
    """Test the cpu_groups property"""

    def test_cpu_groups(self):
        assert em.cpu_groups == [[0, 1], [2, 3]]


class TestGetCpuCapacity(TestCase):
    """Test the get_cpu_capacity method"""

    def test_get_cpu_capacity(self):
        for node in em.root.iter_leaves():
            [cpu] = node.cpus
            assert em.get_cpu_capacity(cpu) == node.max_capacity
            for freq, active_state in node.active_states.items():
                assert em.get_cpu_capacity(cpu, freq) == active_state.capacity


class TestEstimateFromTrace(TestCase):
    def test_estimate_from_trace(self):
        trace_data = (
            # Set all CPUs at lowest freq
            """
            <idle>-0  [000] 0000.0001: cpu_frequency:   state=1000 cpu_id=0
            <idle>-0  [000] 0000.0001: cpu_frequency:   state=1000 cpu_id=1
            <idle>-0  [000] 0000.0001: cpu_frequency:   state=3000 cpu_id=2
            <idle>-0  [000] 0000.0001: cpu_frequency:   state=3000 cpu_id=3
            """  # Set all CPUs in deepest CPU-level idle state
            """
            <idle>-0  [000] 0000.0002: cpu_idle:        state=1 cpu_id=0
            <idle>-0  [000] 0000.0002: cpu_idle:        state=1 cpu_id=1
            <idle>-0  [000] 0000.0002: cpu_idle:        state=1 cpu_id=2
            <idle>-0  [000] 0000.0002: cpu_idle:        state=1 cpu_id=3
            """  # Wake up cpu 0
            """
            <idle>-0  [000] 0000.0005: cpu_idle:        state=4294967295 cpu_id=0
            """  # Ramp up everybody's freqs to 2nd OPP
            """
            <idle>-0  [000] 0000.0010: cpu_frequency:   state=1500 cpu_id=0
            <idle>-0  [000] 0000.0010: cpu_frequency:   state=1500 cpu_id=1
            <idle>-0  [000] 0000.0010: cpu_frequency:   state=4000 cpu_id=2
            <idle>-0  [000] 0000.0010: cpu_frequency:   state=4000 cpu_id=3
            """  # Wake up the other CPUs one by one
            """
            <idle>-0  [000] 0000.0011: cpu_idle:        state=4294967295 cpu_id=1
            <idle>-0  [000] 0000.0012: cpu_idle:        state=4294967295 cpu_id=2
            <idle>-0  [000] 0000.0013: cpu_idle:        state=4294967295 cpu_id=3
            """  # Put CPU2 into "cluster sleep" (note CPU3 is still awake)
            """
            <idle>-0  [000] 0000.0020: cpu_idle:        state=2 cpu_id=2
            """
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'trace.txt')
            with open(path, 'w') as f:
                f.write(trace_data)

            trace = Trace(path,
                events=['cpu_idle', 'cpu_frequency'],
                normalize_time=False,
                # Disable swap since the folder is going to get removed
                enable_swap=False,
                # Parse all the events eagerly since the trace file is going to
                # be removed.
                strict_events=True,
                parser=TxtTraceParser.from_txt_file,
            )

        energy_df = em.estimate_from_trace(trace)

        exp_entries = [
            # Everybody idle
            (0.0002, {
                '0': 0.0,
                '1': 0.0,
                '2': 0.0,
                '3': 0.0,
                '0-1': 5.0,
                '2-3': 8.0,
            }),
            # CPU0 wakes up
            (0.0005, {
                '0': 100.0,  # CPU 0 now active
                '1': 0.0,
                '2': 0.0,
                '3': 0.0,
                '0-1': 10.0,  # little cluster now active
                '2-3': 8.0,
            }),
            # Ramp freqs up to 2nd OPP
            (0.0010, {
                '0': 150.0,
                '1': 0.0,
                '2': 0.0,
                '3': 0.0,
                '0-1': 15.0,
                '2-3': 8.0,
            }),
            # Wake up CPU1
            (0.0011, {
                '0': 150.0,
                '1': 150.0,
                '2': 0.0,
                '3': 0.0,
                '0-1': 15.0,
                '2-3': 8.0,
            }),
            # Wake up CPU2
            (0.0012, {
                '0': 150.0,
                '1': 150.0,
                '2': 400.0,
                '3': 0.0,
                '0-1': 15.0,
                '2-3': 40.0,  # big cluster now active
            }),
            # Wake up CPU3
            (0.0013, {
                '0': 150.0,
                '1': 150.0,
                '2': 400.0,
                '3': 400.0,
                '0-1': 15.0,
                '2-3': 40.0,
            }),
        ]

        # We don't know the exact index that will come out of the parsing
        # (because of handle_duplicate_index). Furthermore the value of the
        # energy estimation will change for infinitessimal moments between each
        # cpu_frequency event, and we don't care about that - we only care about
        # the stable value. So we'll take the value of the returned signal at
        # 0.01ms after each set of events, and assert based on that.
        df = energy_df.reindex([e[0] + 0.00001 for e in exp_entries],
                               method='ffill')

        for i, (exp_index, exp_values) in enumerate(exp_entries):
            row = df.iloc[i]
            assert row.name == pytest.approx(exp_index, abs=1e-4)
            assert row.to_dict() == exp_values


class TestSerialization(StorageTestCase):
    """
    A test class that verifies the Energy Model's serialization
    """

    def test_serialization(self):
        path = os.path.join(self.res_dir, "nrg_model")
        em.to_path(path)
        stored_em = EnergyModel.from_path(path)


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
