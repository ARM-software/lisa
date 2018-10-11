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

import sys
import random
import os.path

from devlib.module.hotplug import HotplugModule
from devlib.exception import TimeoutError

from lisa.tests.kernel.test_bundle import TestMetric, ResultBundle, TestBundle
from lisa.target_script import TargetScript
from lisa.env import TestEnv, ArtifactPath

class HotplugTorture(TestBundle):

    def __init__(self, target_alive, hotpluggable_cpus, live_cpus):
        self.target_alive = target_alive
        self.hotpluggable_cpus = hotpluggable_cpus
        self.live_cpus = live_cpus

    @classmethod
    def _random_cpuhp_seq(cls, nr_operations,
                          hotpluggable_cpus, max_cpus_off):
        """
        Yield a consistent random sequence of CPU hotplug operations

        :param nr_operations: Number of operations in the sequence
            <= 0 will encode 'no sleep'
        :param max_cpus_off: Max number of CPUs plugged-off

        "Consistent" means that a CPU will be plugged-in only if it was
        plugged-off before (and vice versa). Moreover the state of the CPUs
        once the sequence has completed should the same as it was before.

        The actual length of the sequence might differ from the requested one
        by 1 because it's easier to implement and it shouldn't be an issue for
        most test cases.
        """
        cur_on_cpus = hotpluggable_cpus[:]
        cur_off_cpus = []
        i = 0
        while i < nr_operations - len(cur_off_cpus):
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
    def _random_cpuhp_script(cls, te, res_dir, sequence, sleep_min_ms,
                             sleep_max_ms, timeout_s):
        shift = '    '
        script = TargetScript(te, 'random_cpuhp.sh', res_dir)

        # Record configuration
        # script.append('# File generated automatically')
        # script.append('# Configuration:')
        # script.append('# {}'.format(cls.hp_stress))
        # script.append('# Hotpluggable CPUs:')
        # script.append('# {}'.format(cls.hotpluggable_cpus))

        script.append('while true')
        script.append('do')
        for cpu, plug_way in sequence:
            # Write in sysfs entry
            cmd = 'echo {} > {}'.format(plug_way, HotplugModule._cpu_path(te.target, cpu))
            script.append(shift + cmd)
            # Sleep if necessary
            if sleep_max_ms > 0:
                sleep_dur_sec = random.randint(sleep_min_ms, sleep_max_ms)/1000.0
                script.append(shift + 'sleep {}'.format(sleep_dur_sec))
        script.append('done &')

        # Make sure to stop the hotplug stress after timeout_s seconds
        script.append('LOOP_PID=$!')
        script.append('sleep {}'.format(timeout_s))
        script.append('[ $(ps -q $LOOP_PID | wc -l) -gt 1 ] && kill -9 $LOOP_PID')

        return script

    @classmethod
    def _from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None, seed=None, nr_operations=100,
            sleep_min_ms=10, sleep_max_ms=100, duration_s=10,
            max_cpus_off=sys.maxsize) -> 'HotplugTorture':

        if not seed:
            random.seed()
            seed = random.randint(0, sys.maxsize)
        else:
            random.seed(seed)

        te.target.hotplug.online_all()
        hotpluggable_cpus = te.target.hotplug.list_hotpluggable_cpus()

        sequence = cls._random_cpuhp_seq(
            nr_operations, hotpluggable_cpus, max_cpus_off)

        script = cls._random_cpuhp_script(
            te, res_dir, sequence, sleep_min_ms, sleep_max_ms, duration_s
        )

        script.push()

        target_alive = True
        timeout = duration_s + 60

        try:
            script.run(as_root=True, timeout=timeout)
            te.target.hotplug.online_all()
        except TimeoutError:
            #msg = 'Target not responding after {} seconds ...'
            #cls._log.info(msg.format(timeout))
            target_alive = False

        live_cpus = te.target.list_online_cpus() if target_alive else []

        return cls(target_alive, hotpluggable_cpus, live_cpus)

    def test_target_alive(self) -> ResultBundle:
        """
        Test that the hotplugs didn't leave the target in an unusable state
        """
        return ResultBundle.from_bool(self.target_alive)

    def test_cpus_alive(self) -> ResultBundle:
        """
        Test that all CPUs came back online after the hotplug operations
        """
        res = ResultBundle.from_bool(self.hotpluggable_cpus == self.live_cpus)
        res.add_metric("hotpluggable CPUs", self.hotpluggable_cpus)
        res.add_metric("Online CPUs", self.live_cpus)
        return res

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
