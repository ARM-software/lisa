#    Copyright 2018 ARM Limited
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

from devlib.module import Module
from devlib.exception import TargetTransientError


class HotplugModule(Module):

    name = 'hotplug'
    base_path = '/sys/devices/system/cpu'

    @classmethod
    def probe(cls, target):  # pylint: disable=arguments-differ
        # If a system has just 1 CPU, it makes not sense to hotplug it.
        # If a system has more than 1 CPU, CPU0 could be configured to be not
        # hotpluggable. Thus, check for hotplug support by looking at CPU1
        path = cls._cpu_path(target, 1)
        return target.file_exists(path) and target.is_rooted

    @classmethod
    def _cpu_path(cls, target, cpu):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        return target.path.join(cls.base_path, cpu, 'online')

    def list_hotpluggable_cpus(self):
        return [cpu for cpu in range(self.target.number_of_cpus)
                if self.target.file_exists(self._cpu_path(self.target, cpu))]

    def online_all(self, verify=True):
        self.target._execute_util('hotplug_online_all',  # pylint: disable=protected-access
                                  as_root=self.target.is_rooted)
        if verify:
            offline = set(self.target.list_offline_cpus())
            if offline:
                raise TargetTransientError('The following CPUs failed to come back online: {}'.format(offline))

    def online(self, *args):
        for cpu in args:
            self.hotplug(cpu, online=True)

    def offline(self, *args):
        for cpu in args:
            self.hotplug(cpu, online=False)

    def hotplug(self, cpu, online):
        path = self._cpu_path(self.target, cpu)
        if not self.target.file_exists(path):
            return
        value = 1 if online else 0
        self.target.write_value(path, value)

    def _get_path(self, path):
        return self.target.path.join(self.base_path,
            path)

    def fail(self, cpu, state):
        path = self._get_path('cpu{}/hotplug/fail'.format(cpu))
        return self.target.write_value(path, state)

    def get_state(self, cpu):
        path = self._get_path('cpu{}/hotplug/state'.format(cpu))
        return self.target.read_value(path)

    def get_states(self):
        path = self._get_path('hotplug/states')
        states_string = self.target.read_value(path)
        return dict(
                map(str.strip, string.split(':', 1))
                for string in states_string.strip().splitlines()
        )
