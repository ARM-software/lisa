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

    def online_all(self):
        self.target._execute_util('hotplug_online_all',  # pylint: disable=protected-access
                                  as_root=self.target.is_rooted)

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
