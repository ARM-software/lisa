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

import re

from devlib.module import Module

class TripPoint(object):
    def __init__(self, zone, _id):
        self._id = _id
        self.zone = zone
        self.temp_node = 'trip_point_' + _id + '_temp'
        self.type_node = 'trip_point_' + _id + '_type'

    @property
    def target(self):
        return self.zone.target

    def get_temperature(self):
        """Returns the currently configured temperature of the trip point"""
        temp_file = self.target.path.join(self.zone.path, self.temp_node)
        return self.target.read_int(temp_file)

    def set_temperature(self, temperature):
        temp_file = self.target.path.join(self.zone.path, self.temp_node)
        self.target.write_value(temp_file, temperature)

    def get_type(self):
        """Returns the type of trip point"""
        type_file = self.target.path.join(self.zone.path, self.type_node)
        return self.target.read_value(type_file)

class ThermalZone(object):
    def __init__(self, target, root, _id):
        self.target = target
        self.name = 'thermal_zone' + _id
        self.path = target.path.join(root, self.name)
        self.trip_points = {}

        for entry in self.target.list_directory(self.path, as_root=target.is_rooted):
            re_match = re.match('^trip_point_([0-9]+)_temp', entry)
            if re_match is not None:
                self.add_trip_point(re_match.group(1))

    def add_trip_point(self, _id):
        self.trip_points[int(_id)] = TripPoint(self, _id)

    def is_enabled(self):
        """Returns a boolean representing the 'mode' of the thermal zone"""
        value = self.target.read_value(self.target.path.join(self.path, 'mode'))
        return value == 'enabled'

    def set_enabled(self, enabled=True):
        value = 'enabled' if enabled else 'disabled'
        self.target.write_value(self.target.path.join(self.path, 'mode'), value)

    def get_temperature(self):
        """Returns the temperature of the thermal zone"""
        temp_file = self.target.path.join(self.path, 'temp')
        return self.target.read_int(temp_file)

class ThermalModule(Module):
    name = 'thermal'
    thermal_root = '/sys/class/thermal'

    @staticmethod
    def probe(target):

        if target.file_exists(ThermalModule.thermal_root):
            return True

    def __init__(self, target):
        super(ThermalModule, self).__init__(target)

        self.zones = {}
        self.cdevs = []

        for entry in target.list_directory(self.thermal_root):
            re_match = re.match('^(thermal_zone|cooling_device)([0-9]+)', entry)
            if not re_match:
                self.logger.warning('unknown thermal entry: %s', entry)
                continue

            if re_match.group(1) == 'thermal_zone':
                self.add_thermal_zone(re_match.group(2))
            elif re_match.group(1) == 'cooling_device':
                # TODO
                pass

    def add_thermal_zone(self, _id):
        self.zones[int(_id)] = ThermalZone(self.target, self.thermal_root, _id)

    def disable_all_zones(self):
        """Disables all the thermal zones in the target"""
        for zone in self.zones.values():
            zone.set_enabled(False)
