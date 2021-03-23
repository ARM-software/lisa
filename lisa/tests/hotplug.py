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

import abc
import sys
import random
import os.path
import collections
import time
from time import sleep
from threading import Thread
from operator import itemgetter

from devlib.module.hotplug import HotplugModule
from devlib.exception import TargetNotRespondingError

from lisa.tests.base import TestMetric, ResultBundle, TestBundle, DmesgTestBundle
from lisa.target import Target
from lisa.utils import ArtifactPath


class CPUHPSequenceError(Exception):
    pass


class HotplugBase(DmesgTestBundle, TestBundle):
    def __init__(self, res_dir, plat_info, target_alive, hotpluggable_cpus, live_cpus):
        super().__init__(res_dir, plat_info)
        self.target_alive = target_alive
        self.hotpluggable_cpus = hotpluggable_cpus
        self.live_cpus = live_cpus

    @classmethod
    def _check_cpuhp_seq_consistency(cls, nr_operations, hotpluggable_cpus,
                                     max_cpus_off, sequence):
        """
        Check that a hotplug sequence given by :meth:`cpuhp_seq`
        is consistent. Parameters are the same as for :meth:`cpuhp_seq`,
        with the addition of:

        :param sequence: A hotplug sequence, consisting of a sequence of
            2-tuples (CPU and hot plug way)
        :type sequence: Sequence

        """
        if len(sequence) != nr_operations:
            raise CPUHPSequenceError(f'{nr_operations} operations requested, but got {len(sequence)}')

        # Assume als CPUs are plugged in at the beginning
        state = collections.defaultdict(lambda: 1)

        for step, (cpu, plug_way) in enumerate(sequence):
            if cpu not in hotpluggable_cpus:
                raise CPUHPSequenceError('CPU {cpu} is plugged {way} but is not part of hotpluggable CPUs: {cpu_list}'.format(
                    cpu=cpu,
                    way='in' if plug_way else 'out',
                    cpu_list=str(hotpluggable_cpus),
                ))

            # Forbid plugging OFF offlined CPUs and plugging IN online CPUs
            if plug_way == state[cpu]:
                raise CPUHPSequenceError('Cannot plug {way} a CPU that is already plugged {way}'.format(
                    way='in' if plug_way else 'out'
                ))

            state[cpu] = plug_way
            cpus_off = [cpu for cpu, state in state.items() if state == 0]
            if len(cpus_off) > max_cpus_off:
                raise CPUHPSequenceError(f'A maximum of {max_cpus_off} CPUs is allowed to be plugged out, but {len(cpus_off)} CPUs were plugged out at step {step}')

        for cpu, state in state.items():
            if state != 1:
                raise CPUHPSequenceError(f'CPU {cpu} is plugged out but not plugged in at the end of the sequence')

    @classmethod
    @abc.abstractmethod
    def cpuhp_seq(cls, nr_operations, hotpluggable_cpus, max_cpus_off, random_gen):
        """
        Yield a consistent random sequence of CPU hotplug operations

        :param nr_operations: Number of operations in the sequence
        :param max_cpus_off: Max number of CPUs plugged-off

        :param random_gen: A random generator instance
        :type random_gen: ``random.Random``

        "Consistent" means that a CPU will be plugged-in only if it was
        plugged-off before (and vice versa). Moreover the state of the CPUs
        once the sequence has completed should the same as it was before.
        """
        pass

    @classmethod
    def _cpuhp_func(cls, target, res_dir, sequence, sleep_min_ms,
                      sleep_max_ms, random_gen):
        """
        Generate a script consisting of a random sequence of hotplugs operations

        Two consecutive hotplugs can be separated by a random sleep in the script.
        """

        def make_sleep():
            if sleep_max_ms:
                return random_gen.randint(sleep_min_ms, sleep_max_ms) / 1000
            else:
                return 0

        sequence = [
            dict(
                path=HotplugModule._cpu_path(target, cpu),
                sleep=make_sleep(),
                way=plug_way,
            )
            for cpu, plug_way in sequence
        ]

        # The main contributor to the execution time are sleeps, so set a
        # timeout to 10 times the total sleep time. This should be enough to
        # take into account sysfs writes too
        timeout = 10 * sum(map(itemgetter('sleep'), sequence))

        # This function will be executed on the target directly to avoid the
        # overhead of executing the calls one by one, which could mask
        # concurrency issues in the kernel
        @target.remote_func(timeout=timeout, as_root=True)
        def do_hotplug():
            for desc in sequence:
                with open(desc['path'], 'w') as f:
                    f.write(str(desc['way']))

                sleep = desc['sleep']
                if sleep:
                    time.sleep(sleep)

        return do_hotplug

    @classmethod
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath = None, seed=None,
                     nr_operations=100, sleep_min_ms=10, sleep_max_ms=100,
                     max_cpus_off=sys.maxsize, collector=None) -> 'HotplugBase':
        """
        :param seed: Seed of the RNG used to create the hotplug sequences
        :type seed: int

        :param nr_operations: Number of operations in the sequence
        :type nr_operations: int

        :param sleep_min_ms: Minimum sleep duration between hotplug operations
        :type sleep_min_ms: int

        :param sleep_max_ms: Maximum sleep duration between hotplug operations
          (0 would lead to no sleep)
        :type sleep_max_ms: int

        :param max_cpus_off: Maximum number of CPUs hotplugged out at any given
          moment
        :type max_cpus_off: int
        """

        # Instantiate a generator so we can change the seed without any global
        # effect
        random_gen = random.Random()
        random_gen.seed(seed)

        target.hotplug.online_all()
        hotpluggable_cpus = target.hotplug.list_hotpluggable_cpus()

        sequence = list(cls.cpuhp_seq(
            nr_operations, hotpluggable_cpus, max_cpus_off, random_gen))

        cls._check_cpuhp_seq_consistency(nr_operations, hotpluggable_cpus,
            max_cpus_off, sequence)

        do_hotplug = cls._cpuhp_func(
            target, res_dir, sequence, sleep_min_ms, sleep_max_ms, random_gen)

        # We don't want a timeout but we do want to detect if/when the target
        # stops responding. So handle the hotplug remote func in a separate
        # thread and keep polling the target
        thread = Thread(target=do_hotplug, daemon=True)

        with collector:
            try:
                thread.start()
                while thread.is_alive():
                    # We might have a thread hanging off in that case, but there is
                    # not much we can do since the remote func cannot really be
                    # canceled. Since it was spawned with a timeout, it will
                    # eventually die.
                    if not target.check_responsive():
                        break
                    sleep(0.1)
            finally:
                target_alive = bool(target.check_responsive())
                target.hotplug.online_all()

        live_cpus = target.list_online_cpus() if target_alive else []
        return cls(res_dir, target.plat_info, target_alive, hotpluggable_cpus, live_cpus)

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
        dead_cpus = sorted(set(self.hotpluggable_cpus) - set(self.live_cpus))
        res.add_metric("dead CPUs", dead_cpus)
        res.add_metric("number of dead CPUs", len(dead_cpus))
        return res


class HotplugTorture(HotplugBase):

    @classmethod
    def cpuhp_seq(cls, nr_operations, hotpluggable_cpus, max_cpus_off, random_gen):
        """
        FIXME: is that actually still true ?
        The actual length of the sequence might differ from the requested one
        by 1 because it's easier to implement and it shouldn't be an issue for
        most test cases.
        """

        cur_on_cpus = hotpluggable_cpus[:]
        cur_off_cpus = []
        i = 0
        while i < nr_operations - len(cur_off_cpus):
            if not (1 < len(cur_on_cpus) < max_cpus_off):
                # Force plug IN when only 1 CPU is on or too many are off
                plug_way = 1
            elif not cur_off_cpus:
                # Force plug OFF if all CPUs are on
                plug_way = 0  # Plug OFF
            else:
                plug_way = random_gen.randint(0, 1)

            src = cur_off_cpus if plug_way else cur_on_cpus
            dst = cur_on_cpus if plug_way else cur_off_cpus
            cpu = random_gen.choice(src)
            src.remove(cpu)
            dst.append(cpu)
            i += 1
            yield cpu, plug_way

        # Re-plug offline cpus to come back to original state
        for cpu in cur_off_cpus:
            yield cpu, 1

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
