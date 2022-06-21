#    Copyright 2014-2018 ARM Limited
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
from past.builtins import basestring

from operator import attrgetter
from pprint import pformat

from devlib.module import Module
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

    def __init__(self, target, index, path, name, desc, power, latency, residency):
        self.target = target
        self.index = index
        self.path = path
        self.name = name
        self.desc = desc
        self.power = power
        self.latency = latency
        self.residency = residency
        self.id = self.target.path.basename(self.path)
        self.cpu = self.target.path.basename(self.target.path.dirname(path))

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

    def __init__(self, target):
        super(Cpuidle, self).__init__(target)

        basepath = '/sys/devices/system/cpu/'
        values_tree = self.target.read_tree_values(basepath, depth=4, check_exit_code=False)

        self._states = {
            cpu_name: sorted(
                (
                    CpuidleState(
                        self.target,
                        # state_name is formatted as "state42"
                        index=int(state_name[len('state'):]),
                        path=self.target.path.join(basepath, cpu_name, 'cpuidle', state_name),
                        name=state_node['name'],
                        desc=state_node['desc'],
                        power=int(state_node['power']),
                        latency=int(state_node['latency']),
                        residency=int(state_node['residency']) if 'residency' in state_node else None,
                    )
                    for state_name, state_node in cpu_node['cpuidle'].items()
                    if state_name.startswith('state')
                ),
                key=attrgetter('index'),
            )

            for cpu_name, cpu_node in values_tree.items()
            if cpu_name.startswith('cpu') and 'cpuidle' in cpu_node
        }

        self.logger.debug('Adding cpuidle states:\n{}'.format(pformat(self._states)))

    def get_states(self, cpu=0):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        return self._states.get(cpu, [])

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
        # pylint: disable=protected-access
        self.target._execute_util('cpuidle_wake_all_cpus')

    def get_driver(self):
        return self.target.read_value(self.target.path.join(self.root_path, 'current_driver'))

    def get_governor(self):
        path = self.target.path.join(self.root_path, 'current_governor_ro')
        if not self.target.file_exists(path):
            path = self.target.path.join(self.root_path, 'current_governor')
        return self.target.read_value(path)
