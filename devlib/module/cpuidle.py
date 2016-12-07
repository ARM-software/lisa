#    Copyright 2014-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=attribute-defined-outside-init
from devlib.module import Module
from devlib.utils.misc import memoized
from devlib.utils.types import integer, boolean


class CpuidleState(object):

    @property
    def usage(self):
        return integer(self.get('usage'))

    @property
    def time(self):
        return integer(self.get('time'))

    @property
    def is_enabled(self):
        return not boolean(self.get('disable'))

    @property
    def ordinal(self):
        i = len(self.id)
        while self.id[i - 1].isdigit():
            i -= 1
            if not i:
                raise ValueError('invalid idle state name: "{}"'.format(self.id))
        return int(self.id[i:])

    def __init__(self, target, index, path):
        self.target = target
        self.index = index
        self.path = path
        self.id = self.target.path.basename(self.path)
        self.cpu = self.target.path.basename(self.target.path.dirname(path))

    @property
    @memoized
    def desc(self):
        return self.get('desc')

    @property
    @memoized
    def name(self):
        return self.get('name')

    @property
    @memoized
    def latency(self):
        """Exit latency in uS"""
        return self.get('latency')

    @property
    @memoized
    def power(self):
        """Power usage in mW

        ..note::

            This value is not always populated by the kernel and may be garbage.
        """
        return self.get('power')

    @property
    @memoized
    def target_residency(self):
        """Target residency in uS

        This is the amount of time in the state required to 'break even' on
        power - the system should avoid entering the state for less time than
        this.
        """
        return self.get('residency')

    def enable(self):
        self.set('disable', 0)

    def disable(self):
        self.set('disable', 1)

    def get(self, prop):
        property_path = self.target.path.join(self.path, prop)
        return self.target.read_value(property_path)

    def set(self, prop, value):
        property_path = self.target.path.join(self.path, prop)
        self.target.write_value(property_path, value)

    def __eq__(self, other):
        if isinstance(other, CpuidleState):
            return (self.name == other.name) and (self.desc == other.desc)
        elif isinstance(other, basestring):
            return (self.name == other) or (self.desc == other)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return 'CpuidleState({}, {})'.format(self.name, self.desc)

    __repr__ = __str__


class Cpuidle(Module):

    name = 'cpuidle'
    root_path = '/sys/devices/system/cpu/cpuidle'

    @staticmethod
    def probe(target):
        return target.file_exists(Cpuidle.root_path)

    def get_driver(self):
        return self.target.read_value(self.target.path.join(self.root_path, 'current_driver'))

    def get_governor(self):
        return self.target.read_value(self.target.path.join(self.root_path, 'current_governor_ro'))

    @memoized
    def get_states(self, cpu=0):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        states_dir = self.target.path.join(self.target.path.dirname(self.root_path), cpu, 'cpuidle')
        idle_states = []
        for state in self.target.list_directory(states_dir):
            if state.startswith('state'):
                index = int(state[5:])
                idle_states.append(CpuidleState(self.target, index, self.target.path.join(states_dir, state)))
        return idle_states

    def get_state(self, state, cpu=0):
        if isinstance(state, int):
            try:
                return self.get_states(cpu)[state]
            except IndexError:
                raise ValueError('Cpuidle state {} does not exist'.format(state))
        else:  # assume string-like
            for s in self.get_states(cpu):
                if state in [s.id, s.name, s.desc]:
                    return s
            raise ValueError('Cpuidle state {} does not exist'.format(state))

    def enable(self, state, cpu=0):
        self.get_state(state, cpu).enable()

    def disable(self, state, cpu=0):
        self.get_state(state, cpu).disable()

    def enable_all(self, cpu=0):
        for state in self.get_states(cpu):
            state.enable()

    def disable_all(self, cpu=0):
        for state in self.get_states(cpu):
            state.disable()

    def perturb_cpus(self):
        """
        Momentarily wake each CPU. Ensures cpu_idle events in trace file.
        """
        output = self.target._execute_util('cpuidle_wake_all_cpus')
        print(output)
