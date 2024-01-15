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
import logging
import devlib.utils.asyn as asyn

from devlib.module import Module
from devlib.exception import TargetStableCalledProcessError

class TripPoint(object):
    def __init__(self, zone, _id):
        self._id = _id
        self.zone = zone
        self.temp_node = 'trip_point_' + _id + '_temp'
        self.type_node = 'trip_point_' + _id + '_type'

    @property
    def target(self):
        return self.zone.target

    @asyn.asyncf
    async def get_temperature(self):
        """Returns the currently configured temperature of the trip point"""
        temp_file = self.target.path.join(self.zone.path, self.temp_node)
        return await self.target.read_int.asyn(temp_file)

    @asyn.asyncf
    async def set_temperature(self, temperature):
        temp_file = self.target.path.join(self.zone.path, self.temp_node)
        await self.target.write_value.asyn(temp_file, temperature)

    @asyn.asyncf
    async def get_type(self):
        """Returns the type of trip point"""
        type_file = self.target.path.join(self.zone.path, self.type_node)
        return await self.target.read_value.asyn(type_file)

class ThermalZone(object):
    def __init__(self, target, root, _id):
        self.target = target
        self.name = 'thermal_zone' + _id
        self.path = target.path.join(root, self.name)
        self.trip_points = {}
        self.type = self.target.read_value(self.target.path.join(self.path, 'type'))

        for entry in self.target.list_directory(self.path, as_root=target.is_rooted):
            re_match = re.match('^trip_point_([0-9]+)_temp', entry)
            if re_match is not None:
                self._add_trip_point(re_match.group(1))

    def _add_trip_point(self, _id):
        self.trip_points[int(_id)] = TripPoint(self, _id)

    @asyn.asyncf
    async def is_enabled(self):
        """Returns a boolean representing the 'mode' of the thermal zone"""
        value = await self.target.read_value.asyn(self.target.path.join(self.path, 'mode'))
        return value == 'enabled'

    @asyn.asyncf
    async def set_enabled(self, enabled=True):
        value = 'enabled' if enabled else 'disabled'
        await self.target.write_value.asyn(self.target.path.join(self.path, 'mode'), value)

    @asyn.asyncf
    async def get_temperature(self):
        """Returns the temperature of the thermal zone"""
        sysfs_temperature_file = self.target.path.join(self.path, 'temp')
        return await self.target.read_int.asyn(sysfs_temperature_file)

    @asyn.asyncf
    async def get_policy(self):
        """Returns the policy of the thermal zone"""
        temp_file = self.target.path.join(self.path, 'policy')
        return await self.target.read_value.asyn(temp_file)

    @asyn.asyncf
    async def set_policy(self, policy):
        """
        Sets the policy of the thermal zone

        :params policy: Thermal governor name
        :type policy: str
        """
        await self.target.write_value.asyn(self.target.path.join(self.path, 'policy'), policy)

    @asyn.asyncf
    async def get_offset(self):
        """Returns the temperature offset of the thermal zone"""
        offset_file = self.target.path.join(self.path, 'offset')
        return await self.target.read_value.asyn(offset_file)

    @asyn.asyncf
    async def set_offset(self, offset):
        """
        Sets the temperature offset in milli-degrees of the thermal zone

        :params offset: Temperature offset in milli-degrees
        :type policy: int
        """
        await self.target.write_value.asyn(self.target.path.join(self.path, 'offset'), policy)

    @asyn.asyncf
    async def set_emul_temp(self, offset):
        """
        Sets the emulated temperature in milli-degrees of the thermal zone

        :params offset: Emulated temperature in milli-degrees
        :type policy: int
        """
        await self.target.write_value.asyn(self.target.path.join(self.path, 'emul_temp'), policy)

    @asyn.asyncf
    async def get_available_policies(self):
        """Returns the policies available for the thermal zone"""
        temp_file = self.target.path.join(self.path, 'available_policies')
        return await self.target.read_value.asyn(temp_file)

class ThermalModule(Module):
    name = 'thermal'
    thermal_root = '/sys/class/thermal'

    @staticmethod
    def probe(target):

        if target.file_exists(ThermalModule.thermal_root):
            return True

    def __init__(self, target):
        super(ThermalModule, self).__init__(target)

        self.logger = logging.getLogger(self.name)
        self.logger.debug('Initialized [%s] module', self.name)

        self.zones = {}
        self.cdevs = []

        for entry in target.list_directory(self.thermal_root):
            re_match = re.match('^(thermal_zone|cooling_device)([0-9]+)', entry)
            if not re_match:
                self.logger.warning('unknown thermal entry: %s', entry)
                continue

            if re_match.group(1) == 'thermal_zone':
                self._add_thermal_zone(re_match.group(2))
            elif re_match.group(1) == 'cooling_device':
                # TODO
                pass

    def _add_thermal_zone(self, _id):
        self.zones[int(_id)] = ThermalZone(self.target, self.thermal_root, _id)

    def disable_all_zones(self):
        """Disables all the thermal zones in the target"""
        for zone in self.zones.values():
            zone.set_enabled(False)

    @asyn.asyncf
    async def get_all_temperatures(self, error='raise'):
        """
        Returns dictionary with current reading of all thermal zones.

        :params error: Sensor read error handling (raise or ignore)
        :type error: str

        :returns: a dictionary in the form: {tz_type:temperature}
        """

        async def get_temperature_noexcep(item):
            tzid, tz = item
            try:
                temperature = await tz.get_temperature.asyn()
            except TargetStableCalledProcessError as e:
                if error == 'raise':
                    raise e
                elif error == 'ignore':
                    self.logger.warning(f'Skipping thermal_zone_id={tzid} thermal_zone_type={tz.type} error="{e}"')
                    return None
                else:
                    raise ValueError(f'Unknown error parameter value: {error}')
            return temperature

        tz_temps = await self.target.async_manager.map_concurrently(get_temperature_noexcep, self.zones.items())

        return {tz.type: temperature for (tzid, tz), temperature in tz_temps.items() if temperature is not None}
