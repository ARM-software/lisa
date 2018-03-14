# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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
import tests.utils.em as em

from unittest import TestCase
from env import TestEnv
from test import LisaTest

from libs.utils.platforms.juno_r0_energy import juno_r0_energy
from libs.utils.platforms.pixel_energy import pixel_energy
from libs.utils.platforms.hikey_energy import hikey_energy
from libs.utils.platforms.hikey960_energy import hikey960_energy

class TestStaticEnergyModel(TestCase):
    """
    This test iterates over all the static energy model to sanity check them
    and reports any anomalies/irregularities observed.
    """
    energy_model = [('juno_r0', juno_r0_energy),
                    ('hikey', hikey_energy),
                    ('hikey960', hikey960_energy),
                    ('pixel',pixel_energy)]

    def test_is_active_states_coherent(self):
        """Test coherency of the active states"""
        tests_report = ''
        tests_succeed = True
        for (name, model) in self.energy_model:
            (succeed, msg) = em.is_power_increasing(model)
            if not succeed:
                tests_succeed = False
                tests_report += 'In platform {}: {}\n\n\t\t'.format(name, msg)

            (succeed, msg) = em.is_efficiency_decreasing(model)
            if not succeed:
                tests_succeed = False
                tests_report += 'In platform {}: {}\n\n\t\t'.format(name, msg)

        self.assertTrue(tests_succeed, tests_report)

    def test_get_opp_overutilized(self):
        """Get opp that are in overutilized zone"""
        for (name, model) in self.energy_model:
            opp_overutilized = em.get_opp_overutilized(model)
            msg = "\n"
            for i, opp in enumerate(opp_overutilized):
                msg += "\tGroup {}: {}\n".format(i, opp[1])
            logging.info('\nFor platform {}: {}'.format(name, msg))

    def test_get_avg_opp_per_group(self):
        """
        Get average workload that can be run on each cpus group
        """
        for (name, model) in self.energy_model:
            avg_load = em.get_avg_cap(model)
            msg = "\n"
            for i, load in enumerate(avg_load):
                msg += "\tGroup {}: {}\n".format(i, load)
            logging.info('\nFor platform {}: {}'.format(name, msg))

    def test_compare_big_little_opp(self):
        """
        Compare the opp efficiency between little and big cpus
        """
        tests_report = ''
        tests_succeed = True
        for (name, model) in self.energy_model:
            (succeed, msg) = em.compare_big_little_opp(model)
            if not succeed:
                tests_succeed = False
                tests_report += 'In platform {}: {}\n\n\t\t'.format(name, msg)

        self.assertTrue(tests_succeed, tests_report)

    def test_ideal_placements(self):
        """Test placement of simple workloads"""
        tests_report = ''
        tests_succeed = True
        for (name, model) in self.energy_model:
            (succeed, msg) = em.ideal_placements(model)
            if not succeed:
                tests_succeed = False
                tests_report += 'In platform {}: {}\n\n\t\t'.format(name, msg)

        self.assertTrue(tests_succeed, tests_report)

class TestTargetEnergyModel(LisaTest):
    """
    The goal of this test is to control the energy model of the target and
    to report any anomalies/irregularities observed.
    """
    TEST_CONF = {
        'modules': ['cpufreq'],
        'tools': [
            'sysbench',
        ]
    }
    @classmethod
    def setUpClass(cls):
        cls.env = TestEnv(test_conf=cls.TEST_CONF)
        cls.target = cls.env.target

        if not cls.env.nrg_model:
            try:
                cls.env.nrg_model = EnergyModel.from_target(cls.env.target)
            except Exception as e:
                raise SkipTest(
                    'This test requires an EnergyModel for the platform. '
                    'Either provide one manually or ensure it can be read '
                    'from the filesystem: {}'.format(e))

    def test_is_power_coherent(self):
        """Test that the power is increasing"""
        (succeed, msg) = em.is_power_increasing(self.env.nrg_model)
        self.assertTrue(succeed, msg)

    def test_is_energy_efficiency_coherent(self):
        """Test that the energy efficiency (cap / pow) is decreasing"""
        (succeed, msg) = em.is_efficiency_decreasing(self.env.nrg_model)
        self.assertTrue(succeed, msg)

    def test_nb_active_states(self):
        """Test the number of active states for each group of cpus"""
        freqs = []
        for cluster in self.env.nrg_model.root.children:
            cpu = cluster.children[0]
            freqs.append(len(self.target.cpufreq.list_frequencies(cpu.cpus[0])))
        (succeed, msg) = em.check_active_states_nb(self.env.nrg_model, freqs)
        self.assertTrue(succeed, msg)

    def test_get_opp_in_overutilized(self):
        """Get opp that are in overutilized zone"""
        opp_overutilized = em.get_opp_overutilized(self.env.nrg_model)
        msg = "\n"
        for i, opp in enumerate(opp_overutilized):
            msg += "\tGroup {}: {}\n".format(i, opp[1])
        logging.info(msg)

    def test_get_avg_opp_per_group(self):
        """
        Get average workload that can be run on each cpus group
        """
        avg_load = em.get_avg_cap(self.env.nrg_model)
        msg = "\n"
        for i, load in enumerate(avg_load):
            msg += "\tGroup {}: {}\n".format(i, load)
        logging.info(msg)

    def test_compare_big_little_opp(self):
        """
        Compare the opp efficiency between little and big cpus
        """
        (succeed, msg) = em.compare_big_little_opp(self.env.nrg_model)
        self.assertTrue(succeed, msg)

    def test_ideal_placements(self):
        """Test placement of simple workloads"""
        (succeed, msg) = em.ideal_placements(self.env.nrg_model)
        self.assertTrue(succeed, msg)
