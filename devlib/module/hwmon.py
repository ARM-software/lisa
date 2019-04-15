#    Copyright 2015-2018 ARM Limited
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
import re
from collections import defaultdict

from devlib import TargetStableError
from devlib.module import Module
from devlib.utils.types import integer


HWMON_ROOT = '/sys/class/hwmon'
HWMON_FILE_REGEX = re.compile(r'(?P<kind>\w+?)(?P<number>\d+)_(?P<item>\w+)')


class HwmonSensor(object):

    def __init__(self, device, path, kind, number):
        self.device = device
        self.path = path
        self.kind = kind
        self.number = number
        self.target = self.device.target
        self.name = '{}/{}{}'.format(self.device.name, self.kind, self.number)
        self.label = self.name
        self.items = set()

    def add(self, item):
        self.items.add(item)
        if item == 'label':
            self.label = self.get('label')

    def get(self, item):
        path = self.get_file(item)
        value = self.target.read_value(path)
        try:
            return  integer(value)
        except (TypeError, ValueError):
            return value

    def set(self, item, value):
        path = self.get_file(item)
        self.target.write_value(path, value)

    def get_file(self, item):
        if item not in self.items:
            raise ValueError('item "{}" does not exist for {}'.format(item, self.name))
        filename = '{}{}_{}'.format(self.kind, self.number, item)
        return self.target.path.join(self.path, filename)

    def __str__(self):
        if self.name != self.label:
            text = 'HS({}, {})'.format(self.name, self.label)
        else:
            text = 'HS({})'.format(self.name)
        return text

    __repr__ = __str__


class HwmonDevice(object):

    @property
    def sensors(self):
        all_sensors = []
        for sensors_of_kind in self._sensors.values():
            all_sensors.extend(list(sensors_of_kind.values()))
        return all_sensors

    def __init__(self, target, path, name, fields):
        self.target = target
        self.path = path
        self.name = name
        self._sensors = defaultdict(dict)
        path = self.path
        if not path.endswith(self.target.path.sep):
            path += self.target.path.sep
        for entry in fields:
            match = HWMON_FILE_REGEX.search(entry)
            if match:
                kind = match.group('kind')
                number = int(match.group('number'))
                item = match.group('item')
                if number not in self._sensors[kind]:
                    sensor = HwmonSensor(self, self.path, kind, number)
                    self._sensors[kind][number] = sensor
                self._sensors[kind][number].add(item)

    def get(self, kind, number=None):
        if number is None:
            return [s for _, s in sorted(self._sensors[kind].items(),
                                         key=lambda x: x[0])]
        else:
            return self._sensors[kind].get(number)

    def __str__(self):
        return 'HD({})'.format(self.name)

    __repr__ = __str__


class HwmonModule(Module):

    name = 'hwmon'

    @staticmethod
    def probe(target):
        try:
            target.list_directory(HWMON_ROOT, as_root=target.is_rooted)
        except TargetStableError:
            # Doesn't exist or no permissions
            return False
        return True

    @property
    def sensors(self):
        all_sensors = []
        for device in self.devices:
            all_sensors.extend(device.sensors)
        return all_sensors

    def __init__(self, target):
        super(HwmonModule, self).__init__(target)
        self.root = HWMON_ROOT
        self.devices = []
        self.scan()

    def scan(self):
        values_tree = self.target.read_tree_values(self.root, depth=3, tar=True)
        for entry_id, fields in values_tree.items():
            path = self.target.path.join(self.root, entry_id)
            name = fields.pop('name', None)
            if name is None:
                continue
            self.logger.debug('Adding device {}'.format(name))
            device = HwmonDevice(self.target, path, name, fields)
            self.devices.append(device)
