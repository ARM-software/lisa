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

from unittest import SkipTest

from bart.sched.SchedMultiAssert import SchedMultiAssert
from devlib.target import TargetError
from test import LisaTest

WORKLOAD_DURATION_S = 5
WORKLOAD_PERIOD_MS =  10
SWITCH_WINDOW_HALF = 0.5
SMALL_DCYCLE = 10
BIG_DCYCLE = 100
STEP_HIGH_DCYCLE = 50
STEP_LOW_DCYCLE = 10
EXPECTED_RESIDENCY_PCT = 85
OFFLOAD_EXPECTED_BUSY_TIME_PCT = 97
SET_IS_BIG_LITTLE = True
SET_INITIAL_TASK_UTIL = True
OFFLOAD_MIGRATION_MIGRATOR_DELAY = 1

energy_aware_conf = {
    'tag' : 'energy_aware',
    'flags' : 'ftrace',
    'sched_features' : 'ENERGY_AWARE',
}

class EasTest(LisaTest):
    """
    Base class for EAS tests
    """

    test_conf = {
        "ftrace" : {
            "events" : [
                "sched_overutilized",
                "sched_energy_diff",
                "sched_load_avg_task",
                "sched_load_avg_cpu",
                "sched_migrate_task",
                "sched_switch",
                "cpu_frequency"
            ],
        },
        "modules": ["cgroups"],
        "cpufreq" : {
            "governor" : "sched",
        },
    }

    # Set to true to run a test only on heterogeneous systems
    skip_on_smp = False

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(EasTest, cls)._init(*args, **kwargs)

    @classmethod
    def _getExperimentsConf(cls, test_env):
        if cls.skip_on_smp and not test_env.nrg_model.is_heterogeneous:
            raise SkipTest('Test not required on symmetric systems')
        return super(EasTest, cls)._getExperimentsConf(test_env)

    @classmethod
    def _experimentsInit(cls, *args, **kwargs):
        super(EasTest, cls)._experimentsInit(*args, **kwargs)

        if SET_IS_BIG_LITTLE:
            # This flag doesn't exist on mainline-integration kernels, so
            # don't worry if the file isn't present (hence verify=False)
            cls.target.write_value(
                "/proc/sys/kernel/sched_is_big_little", 1, verify=False)

        if SET_INITIAL_TASK_UTIL:
            # This flag doesn't exist on all kernels, so don't worry if the file
            # isn't present (hence verify=False)
            cls.target.write_value(
                "/proc/sys/kernel/sched_initial_task_util", 1024, verify=False)

    def _do_test_first_cpu(self, experiment, tasks):
        """Test that all tasks start on a big CPU"""

        sched_assert = self.get_multi_assert(experiment)

        self.assertTrue(
            sched_assert.assertFirstCpu(
                self.te.nrg_model.biggest_cpus,
                rank=len(tasks)),
            msg="Not all the new generated tasks started on a big CPU")
