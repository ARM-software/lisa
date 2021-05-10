# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, Arm Limited and contributors.
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
import operator
import collections
import time
from time import sleep
from threading import Thread
from functools import partial
from itertools import chain

import pandas as pd
from devlib.module.hotplug import HotplugModule
from devlib.exception import TargetNotRespondingError, TargetStableError

from lisa.datautils import df_merge
from lisa.tests.base import TestMetric, ResultBundle, TestBundle, DmesgTestBundle, FtraceTestBundle, CannotCreateError
from lisa.target import Target
from lisa.trace import requires_events
from lisa.utils import ArtifactPath


class CPUHPSequenceError(Exception):
    pass


class HotplugDmesgTestBundle(DmesgTestBundle):
    DMESG_IGNORED_PATTERNS = [
        *DmesgTestBundle.DMESG_IGNORED_PATTERNS,
        DmesgTestBundle.CANNED_DMESG_IGNORED_PATTERNS['hotplug-irq-affinity'],
        DmesgTestBundle.CANNED_DMESG_IGNORED_PATTERNS['hotplug-irq-affinity-failed'],
    ]


class HotplugBase(HotplugDmesgTestBundle, TestBundle):
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
        timeout = 10 * sum(map(operator.itemgetter('sleep'), sequence))

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


class HotplugRollback(TestBundle, HotplugDmesgTestBundle, FtraceTestBundle):

    @classmethod
    def _online(cls, target, cpu, online, verify=True):
        try:
            if online:
                target.hotplug.online(cpu)
            else:
                target.hotplug.offline(cpu)
        except TargetStableError as e:
            if verify:
                raise e

    @classmethod
    def _reset_fail(cls, target, cpu):
        target.hotplug.fail(cpu, -1)

    @classmethod
    def _state_can_fail(cls, target, cpu, state, up):
        """
        There are no way of probing the kernel for a list of hotplug states
        that can fail and for which we can test the rollback. We need therefore
        to try:
        - If we can set the state in the kernel 'fail' interface.
        - If the hotplug is reset actually failing (some states can fail only
          when going up or down)
        """
        try:
            target.hotplug.fail(cpu, state)
        except TargetStableError:
            return False

        try:
            cls._online(target, cpu, up)
            cls._reset_fail(target, cpu)
            cls._online(target, cpu, not up)
            #If we can go up/down without a failure, that's because this state
            #doesn't have a up/down callback and can't fail.
            return False
        except TargetStableError:
            return True

    @classmethod
    def _prepare_hotplug(cls, target, cpu, up):
        cls._reset_fail(target, cpu)
        cls._online(target, cpu, not up)

    @classmethod
    def _get_states(cls, target, cpu, up):
        states = target.hotplug.get_states()
        cls._prepare_hotplug(target, cpu, not up)
        return [
                state
                for state in states
                if cls._state_can_fail(target, cpu, state, up)
        ]

    @classmethod
    def _mark_trace(cls, target, collector, start=True,
                    expected=False, up=False, failing_state=0):
        """
        Convert start, expected and up to int for a lighter trace
        """
        target.write_value(
                collector['ftrace'].marker_file,
                "hotplug_rollback: test={} expected={} up={} failing_state={}".format(
                    int(start), int(expected), int(up), failing_state),
                verify=False
        )

    @classmethod
    def _test_rollback(cls, target, collector, cpu, failing_state, up):
        cls._prepare_hotplug(target, cpu, up=up)
        target.hotplug.fail(cpu, failing_state)
        cls._mark_trace(target, collector, up=up,
                        failing_state=failing_state)
        cls._online(target, cpu, online=up, verify=False)
        cls._mark_trace(target, collector, start=False)

    @classmethod
    def _do_from_target(cls, target, res_dir, collector, cpu):
        # Get the list of each state that can fail
        states_down = cls._get_states(target, cpu, up=False)
        states_up = cls._get_states(target, cpu, up=True)

        cls._prepare_hotplug(target, cpu, up=False)
        with collector:
            # Get the expected list of states for a complete Hotplug
            cls._mark_trace(target, collector, expected=True, up=False)
            cls._online(target, cpu, online=False)
            cls._mark_trace(target, collector, expected=True, up=True)
            cls._online(target, cpu, online=True)
            cls._mark_trace(target, collector, start=False)

            # Test hotunplug rollback for each possible state failure
            for failing_state in states_down:
                cls._test_rollback(target, collector, cpu=cpu,
                                   failing_state=failing_state, up=False)

            # Test hotplug rollback for each possible state failure
            for failing_state in states_up:
                cls._test_rollback(target, collector, cpu=cpu,
                                   failing_state=failing_state, up=True)

            # TODO: trace-cmd is relying on _SC_NPROCESSORS_CONF to know how
            # many CPUs are present in the system and what to flush from the
            # ftrace buffer to the trace.dat file. The problem is that the Musl
            # libc that we use to build trace-cmd in LISA is returning, for
            # _SC_NPROCESSORS_CONF, the number of CPUs  _online_. We then need,
            # until this problem is fixed to set the CPU back online before
            # collecting the trace, or some data would be missing.
            cls._online(target, cpu, online=True)

        return cls(res_dir, target.plat_info)

    @classmethod
    def _from_target(cls, target, *,
                     res_dir: ArtifactPath = None, collector=None) -> 'HotplugRollback':
        cpu = min(target.hotplug.list_hotpluggable_cpus())
        cls._online(target, cpu, online=True)

        try:
            return cls._do_from_target(target, res_dir, collector, cpu)
        finally:
            cls._reset_fail(target, cpu)
            cls._online(target, cpu, online=True)

    @classmethod
    def check_from_target(cls, target):
        try:
            cls._reset_fail(target, 0)
        except TargetStableError:
            raise CannotCreateError(
                "Target can't reset the hotplug fail interface")

    @classmethod
    def _get_expected_states(cls, df, up):
        df = df[(df['expected']) & (df['up'] == up)]

        return df['idx'].dropna()

    @requires_events('userspace@hotplug_rollback', 'cpuhp_enter')
    def test_hotplug_rollback(self) -> ResultBundle:
        """
        Test that the hotplug can rollback to its previous state after a
        failure. All possible steps, up/down combinations will be tested. For
        each combination, also verify that the hotplug is going through all the
        steps it is supposed to.
        """
        df = df_merge([
            self.trace.df_event('userspace@hotplug_rollback'),
            self.trace.df_event('cpuhp_enter')
        ])

        # Keep only the states delimited by _mark_trace()
        df['test'].ffill(inplace=True)
        df = df[df['test'] == 1]
        df.drop(columns='test', inplace=True)

        df['up'].ffill(inplace=True)
        df['up'] = df['up'].astype(bool)

        # Read the expected states from full hot(un)plug
        df['expected'].ffill(inplace=True)
        df['expected'] = df['expected'].astype(bool)
        expected_down = self._get_expected_states(df, up=False)
        expected_up = self._get_expected_states(df, up=True)
        df = df[~df['expected']]
        df.drop(columns='expected', inplace=True)

        def _get_expected_rollback(up, failing_state):
            return list(
                    filter(
                        partial(
                            operator.gt if up else operator.lt,
                            failing_state,
                        ),
                        chain(expected_up, expected_down) if up else
                        chain(expected_down, expected_up)
                    )
            )

        def _verify_rollback(df):
            failing_state = df['failing_state'].iloc[0]
            up = df['up'].iloc[0]
            expected = _get_expected_rollback(up, failing_state)

            return pd.DataFrame(data={
                'failing_state': df['failing_state'],
                'up': up,
                'result': df['idx'].tolist() == expected
            })

        df['failing_state'].ffill(inplace=True)
        df.dropna(inplace=True)
        df = df.groupby(['up', 'failing_state'],
                        observed=True).apply(_verify_rollback)
        df.drop_duplicates(inplace=True)

        res = ResultBundle.from_bool(df['result'].all())
        res.add_metric('Failed rollback states',
                       df[~df['result']]['failing_state'].tolist())

        return res
