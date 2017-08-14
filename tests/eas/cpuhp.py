# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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

import os.path
import logging
import random
import json
import sys

from test import LisaTest, experiment_test
from target_script import TargetScript
from devlib.module.hotplug import HotplugModule

class _CpuHotplugTest(LisaTest):
    """
    "Abstract" base class for generic EAS tests under CPU hotplug stress

    Subclasses should provide a .workloads member to populate the 'wloads' field
    of the experiments_conf for the Executor. Furthermore, subclasses may
    overwrite the .hp_stress member containing the configuration used while
    generating the random hotplug stress sequence.

    The test will be successful if the system does not halt or kill the
    workloads because of hotplug stress. It means that the test passes if the
    workload completes.
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
                "cpu_frequency",
                "cpu_idle",
                "cpu_capacity",
                "cpuhp_enter",
                "cpuhp_exit",
                "cpuhp_multi_enter",
            ],
        },
        "modules": ["cgroups", "hotplug"],
        "tools": ["rt-app", "trace-cmd"],
    }

    hp_stress = {
        'seed' : None,                  # Seed of the random number generator
        'sequence_len' : 100,           # Number of operations in the sequence
        'sleep' : {
            'min_ms' : 10,              # Min sleep duration between hotplugs
            'max_ms' : 100,             # Max sleep duration between hotplugs
        },                              #   max_ms <= 0 will encode 'no sleep'
        'max_cpus_off' : sys.maxint,    # Max number of CPUs plugged-off
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(_CpuHotplugTest, cls).runExperiments(*args, **kwargs)

    @classmethod
    def _getExperimentsConf(cls, test_env):
        conf = {
            'tag' : 'energy_aware',
            'flags' : ['ftrace', 'freeze_userspace'],
            'sched_features' : 'ENERGY_AWARE',
        }

        return {
            'wloads' : cls.workloads,
            'confs' : [conf],
        }

    @classmethod
    def _experimentsInit(cls, *args, **kwargs):
        cls._log = logging.getLogger('CpuhpTest')
        # Choose a random seed explicitly if not given
        if cls.hp_stress.get('seed') is None:
            random.seed()
            cls.hp_stress['seed'] = random.randint(0, sys.maxint)
        random.seed(cls.hp_stress['seed'])
        cls._log.info('Random sequence of CPU Hotplug generated with: ')
        cls._log.info(cls.hp_stress)
        with open(os.path.join(cls.te.res_dir, 'hotplug_cfg.json'), 'w') as f:
            json.dump(cls.hp_stress, f, sort_keys=True, indent=4)

        # Play with (online) hotpluggable CPUs only
        cls.hotpluggable_cpus = filter(
                lambda cpu: cls.target.file_exists(cls._cpuhp_path(cpu)),
                cls.target.list_online_cpus())
        if not cls.hotpluggable_cpus:
            raise RuntimeError('Cannot find any hotpluggable CPU online')
        cls._log.info('Hotpluggable CPUs found on target: ')
        cls._log.info(cls.hotpluggable_cpus)

        # Run random hotplug sequence on target
        cls.cpuhp_seq_script = cls._random_cpuhp_script()
        cls.cpuhp_seq_script.push()
        cls.cpuhp_seq_script.run(as_root=True, background=True)
        cls._log.info('Hotplug stress has now started on target')

    @classmethod
    def _experimentsFinalize(cls, *args, **kwargs):
        cls._log.info('Stopping hotplug stress on the target now')
        cls.cpuhp_seq_script.kill()
        # Resume to original state
        cls.te._log.info('Plugging back in currently offline CPUs')
        cls.target.hotplug.online(*cls.hotpluggable_cpus)

    @classmethod
    def _cpuhp_path(cls, cpu):
        cpu = 'cpu{}'.format(cpu)
        return cls.target.path.join(HotplugModule.base_path, cpu, 'online')

    @classmethod
    def _random_cpuhp_script(cls):
        '''
        Generate a script consisting of a random sequence of hotplugs operations

        Two consecutive hotplugs can be separated by a random (and configurable
        through .hp_stress) sleep in the script. Each hotplug operation is
        logged in a ftrace marker file for post processing. The return value is
        the TargetScript object.

        Example of generated script:
        > while true; do
        >     echo 0 > /sys/devices/system/cpu/cpu1/online
        >     sleep 0.4245
        >     echo 1 > /sys/devices/system/cpu/cpu1/online
        >     sleep 0.178
        > done
        '''
        shift = '    '
        marker = cls.te.ftrace.marker_file
        sleep_min = cls.hp_stress['sleep']['min_ms']
        sleep_max = cls.hp_stress['sleep']['max_ms']
        script = TargetScript(cls.te, 'random_cpuhp.sh')

        # Record configuration
        script.append('# File generated automatically')
        script.append('# Configuration:')
        script.append('# {}'.format(cls.hp_stress))
        script.append('# Hotpluggable CPUs:')
        script.append('# {}'.format(cls.hotpluggable_cpus))

        script.append('while true; do')
        for cpu, plug_way in cls._random_cpuhp_seq():
            # Write in sysfs entry
            cmd = 'echo {} > {}'.format(plug_way, cls._cpuhp_path(cpu))
            script.append(shift + cmd)
            # Sleep if necessary
            if sleep_max > 0:
                sleep_dur_sec = random.randint(sleep_min, sleep_max)/1000.0
                script.append(shift + 'sleep {}'.format(sleep_dur_sec))
        script.append('done')
        return script

    @classmethod
    def _random_cpuhp_seq(cls):
        '''
        Yield a consitent random sequence of CPU hotplug operations

        "Consistent" means that a CPU will be plugged-in only if it was
        plugged-off before (and vice versa). Moreover the state of the CPUs
        once the sequence has completed should the same as it was before. The
        length of the sequence (that is the number of plug operations) is
        defined in the hp_stress member of the class. The actual length of the
        sequence might differ from the requested one by 1 because it's easier
        to implement and it shouldn't be an issue for most test cases.
        '''
        cur_on_cpus = cls.hotpluggable_cpus[:]
        cur_off_cpus = []
        max_cpus_off = cls.hp_stress['max_cpus_off']
        i = 0
        while i < cls.hp_stress['sequence_len'] - len(cur_off_cpus):
            if len(cur_on_cpus)<=1 or len(cur_off_cpus)>=max_cpus_off:
                # Force plug IN when only 1 CPU is on or too many are off
                plug_way = 1
            elif not cur_off_cpus:
                # Force plug OFF if all CPUs are on
                plug_way = 0 # Plug OFF
            else:
                plug_way = random.randint(0,1)
            src = cur_off_cpus if plug_way else cur_on_cpus
            dst = cur_on_cpus if plug_way else cur_off_cpus
            cpu = random.choice(src)
            src.remove(cpu)
            dst.append(cpu)
            i += 1
            yield cpu, plug_way

        # Re-plug offline cpus to come back to original state
        for cpu in cur_off_cpus:
            yield cpu, 1

    @classmethod
    def _test_target_is_alive(cls):
        '''Test that the target is responsive'''
        try:
            cls.target.check_responsive()
        except Exception:
            raise AssertionError("the target is not responsive")

class ThreeSmallTasks(_CpuHotplugTest):
    """
    Test EAS for 3 20% tasks over 60 seconds
    """
    workloads = {
        'cpuhp_three_small' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 20,
                    'duration_s': 60,
                    'period_ms': 16,
                },
                'tasks' : 3,
                'prefix' : 'many',
            },
        },
    }

    @experiment_test
    def test_random_hotplugs(self, experiment, tasks):
        '''Test that the system doesn't crash while under hotplug stress'''
        self._test_target_is_alive()

