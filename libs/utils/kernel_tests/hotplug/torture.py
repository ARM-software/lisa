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

from env import TestEnv
from test import LisaTest, experiment_test
from target_script import TargetScript
from devlib.module.hotplug import HotplugModule
from devlib.exception import TimeoutError

class _CpuHotplugTest(LisaTest):
    """
    "Abstract" base class for generic EAS tests under CPU hotplug stress

    The test will be successful if the system does not crash during the hotplug
    stress, and if all CPUs come back online properly.
    """

    test_conf = {
        "modules": ["hotplug"],
        # Remove the modules that silently talk to the target
        "exclude_modules": ["hwmon", "ftrace"],
    }

    duration_sec = 10                   # Duration of the hotplug stress

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
    def runExperiments(cls):
        cls.te = TestEnv(test_conf=cls._getTestConf())
        cls.target = cls.te.target
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
        cls.target.hotplug.online_all()
        cls.hotpluggable_cpus = filter(
                lambda cpu: cls.target.file_exists(cls._cpuhp_path(cpu)),
                cls.target.list_online_cpus())
        if not cls.hotpluggable_cpus:
            raise RuntimeError('Cannot find any hotpluggable CPU online')
        cls._log.info('Hotpluggable CPUs found on target: ')
        cls._log.info(cls.hotpluggable_cpus)

        # Run random hotplug sequence on target
        cls.cpuhp_seq_script = cls._random_cpuhp_script(cls.duration_sec)
        cls.cpuhp_seq_script.push()
        msg = 'Starting hotplug stress for {} seconds'
        cls._log.info(msg.format(cls.duration_sec))
        cls.target_alive = True

        # The script should run on the target for 'cls.duration_sec' seconds.
        # If there is no life sign of the target 1 minute after that, we
        # consider it dead.
        timeout = cls.duration_sec + 60
        try:
            cls.cpuhp_seq_script.run(as_root=True, timeout=timeout)
        except TimeoutError:
            msg = 'Target not responding after {} seconds ...'
            cls._log.info(msg.format(timeout))
            cls.target_alive = False
            return

        cls._log.info('Hotplug stress completed')
        cls.target.hotplug.online_all()

    @classmethod
    def _cpuhp_path(cls, cpu):
        cpu = 'cpu{}'.format(cpu)
        return cls.target.path.join(HotplugModule.base_path, cpu, 'online')

    @classmethod
    def _random_cpuhp_script(cls, timeout_s):
        '''
        Generate a script consisting of a random sequence of hotplugs operations

        Two consecutive hotplugs can be separated by a random (and configurable
        through .hp_stress) sleep in the script. The hotplug stress must be
        stopped after some time using the timeout_s parameter (in seconds).
        '''
        shift = '    '
        sleep_min = cls.hp_stress['sleep']['min_ms']
        sleep_max = cls.hp_stress['sleep']['max_ms']
        script = TargetScript(cls.te, 'random_cpuhp.sh')

        # Record configuration
        script.append('# File generated automatically')
        script.append('# Configuration:')
        script.append('# {}'.format(cls.hp_stress))
        script.append('# Hotpluggable CPUs:')
        script.append('# {}'.format(cls.hotpluggable_cpus))


        script.append('while true')
        script.append('do')
        for cpu, plug_way in cls._random_cpuhp_seq():
            # Write in sysfs entry
            cmd = 'echo {} > {}'.format(plug_way, cls._cpuhp_path(cpu))
            script.append(shift + cmd)
            # Sleep if necessary
            if sleep_max > 0:
                sleep_dur_sec = random.randint(sleep_min, sleep_max)/1000.0
                script.append(shift + 'sleep {}'.format(sleep_dur_sec))
        script.append('done &')

        # Make sure to stop the hotplug stress after timeout_s seconds
        script.append('LOOP_PID=$!')
        script.append('sleep {}'.format(timeout_s))
        script.append('kill -9 $LOOP_PID')

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

    def test_system_state(cls):
        '''Check if system state is clean after hotplug stress'''
        if not cls.target_alive:
            raise AssertionError('Target crashed under hotplug stress')

        # This should really be a separate test but there is no point
        # in doing it if the target isn't alive, so let's leave it here until
        # we find a way to express dependencies between tests
        if cls.target.list_offline_cpus():
            raise AssertionError('Some CPUs failed to come back online')

class _Torture(_CpuHotplugTest):
    """
    Torture hotplug stress
    """

    hp_stress = {
        'seed' : None,
        'sequence_len' : 100,
        'sleep' : {
            'min_ms' : -1,
            'max_ms' : -1, # No sleep time between hotplug
        },
        'max_cpus_off' : sys.maxint,
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(_Torture, cls).runExperiments(*args, **kwargs)


class Torture10(_Torture):
    """
    Torture hotplug during 10 seconds
    """
    duration_sec = 10


class Torture300(_Torture):
    """
    Torture hotplug during 5 minutes
    """
    duration_sec = 300
