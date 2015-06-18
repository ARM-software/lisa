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
import wlauto.core.signal as signal
from wlauto import Module
from wlauto.exceptions import DeviceError


class CpuidleState(object):

    @property
    def usage(self):
        return self.get('usage')

    @property
    def time(self):
        return self.get('time')

    @property
    def disable(self):
        return self.get('disable')

    @disable.setter
    def disable(self, value):
        self.set('disable', value)

    @property
    def ordinal(self):
        i = len(self.id)
        while self.id[i - 1].isdigit():
            i -= 1
            if not i:
                raise ValueError('invalid idle state name: "{}"'.format(self.id))
        return int(self.id[i:])

    def __init__(self, device, path):
        self.device = device
        self.path = path
        self.id = self.device.path.basename(self.path)
        self.cpu = self.device.path.basename(self.device.path.dirname(path))
        self.desc = self.get('desc')
        self.name = self.get('name')
        self.latency = self.get('latency')
        self.power = self.get('power')

    def get(self, prop):
        property_path = self.device.path.join(self.path, prop)
        return self.device.get_sysfile_value(property_path)

    def set(self, prop, value):
        property_path = self.device.path.join(self.path, prop)
        self.device.set_sysfile_value(property_path, value)

    def __eq__(self, other):
        if isinstance(other, CpuidleState):
            return (self.name == other.name) and (self.desc == other.desc)
        elif isinstance(other, basestring):
            return (self.name == other) or (self.desc == other)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Cpuidle(Module):

    name = 'cpuidle'
    description = """
    Adds cpuidle state query and manupution APIs to a Device interface.

    """
    capabilities = ['cpuidle']

    root_path = '/sys/devices/system/cpu/cpuidle'

    def probe(self, device):
        return device.file_exists(self.root_path)

    def initialize(self, context):
        self.device = self.root_owner
        signal.connect(self._on_device_init, signal.RUN_INIT, priority=1)

    def get_cpuidle_driver(self):
        return self.device.get_sysfile_value(self.device.path.join(self.root_path, 'current_driver')).strip()

    def get_cpuidle_governor(self):
        return self.device.get_sysfile_value(self.device.path.join(self.root_path, 'current_governor_ro')).strip()

    def get_cpuidle_states(self, cpu=0):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        states_dir = self.device.path.join(self.device.path.dirname(self.root_path), cpu, 'cpuidle')
        idle_states = []
        for state in self.device.listdir(states_dir):
            if state.startswith('state'):
                idle_states.append(CpuidleState(self.device, self.device.path.join(states_dir, state)))
        return idle_states

    def _on_device_init(self, context):  # pylint: disable=unused-argument
        if not self.device.file_exists(self.root_path):
            raise DeviceError('Device kernel does not appear to have cpuidle enabled.')

