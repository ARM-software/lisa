#    Copyright 2013-2015 ARM Limited
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


# pylint: disable=W0613,E1101
from __future__ import division
from collections import OrderedDict

from wlauto import Parameter, Instrument
from wlauto.exceptions import InstrumentError, ConfigError
from wlauto.utils.hwmon import discover_sensors
from wlauto.utils.types import list_of_strs


# sensor_kind: (report_type, units, conversion)
HWMON_SENSORS = {
    'energy': ('diff', 'Joules', lambda x: x / 10 ** 6),
    'temp': ('before/after', 'Celsius', lambda x: x / 10 ** 3),
}

HWMON_SENSOR_PRIORITIES = ['energy', 'temp']


class HwmonInstrument(Instrument):

    name = 'hwmon'
    description = """
    Hardware Monitor (hwmon) is a generic Linux kernel subsystem,
    providing access to hardware monitoring components like temperature or
    voltage/current sensors.

    The following web page has more information:

        http://blogs.arm.com/software-enablement/925-linux-hwmon-power-management-and-arm-ds-5-streamline/

    You can specify which sensors HwmonInstrument looks for by specifying
    hwmon_sensors in your config.py, e.g. ::

        hwmon_sensors = ['energy', 'temp']

    If this setting is not specified, it will look for all sensors it knows about.
    Current valid values are::

        :energy: Collect energy measurements and report energy consumed
                 during run execution (the diff of before and after readings)
                 in Joules.
        :temp: Collect temperature measurements and report the before and
               after readings in degrees Celsius.

    """

    parameters = [
        Parameter('sensors', kind=list_of_strs, default=['energy', 'temp'],
                  global_alias='hwmon_sensors',
                  description='The kinds of sensors hwmon instrument will look for')
    ]

    def __init__(self, device, **kwargs):
        super(HwmonInstrument, self).__init__(device, **kwargs)

        if self.sensors:
            self.sensor_kinds = {}
            for kind in self.sensors:
                if kind in HWMON_SENSORS:
                    self.sensor_kinds[kind] = HWMON_SENSORS[kind]
                else:
                    message = 'Unexpected sensor type: {}; must be in {}'.format(kind, HWMON_SENSORS.keys())
                    raise ConfigError(message)
        else:
            self.sensor_kinds = HWMON_SENSORS

        self.sensors = []

    def initialize(self, context):
        self.sensors = []
        self.logger.debug('Searching for HWMON sensors.')
        discovered_sensors = discover_sensors(self.device, self.sensor_kinds.keys())
        for sensor in sorted(discovered_sensors, key=lambda s: HWMON_SENSOR_PRIORITIES.index(s.kind)):
            self.logger.debug('Adding {}'.format(sensor.filepath))
            self.sensors.append(sensor)

    def setup(self, context):
        for sensor in self.sensors:
            sensor.clear_readings()

    def fast_start(self, context):
        for sensor in reversed(self.sensors):
            sensor.take_reading()

    def fast_stop(self, context):
        for sensor in self.sensors:
            sensor.take_reading()

    def update_result(self, context):
        for sensor in self.sensors:
            try:
                report_type, units, conversion = HWMON_SENSORS[sensor.kind]
                if report_type == 'diff':
                    before, after = sensor.readings
                    diff = conversion(after - before)
                    context.result.add_metric(sensor.label, diff, units)
                elif report_type == 'before/after':
                    before, after = sensor.readings
                    mean = conversion((after + before) / 2)
                    context.result.add_metric(sensor.label, mean, units)
                    context.result.add_metric(sensor.label + ' before', conversion(before), units)
                    context.result.add_metric(sensor.label + ' after', conversion(after), units)
                else:
                    raise InstrumentError('Unexpected report_type: {}'.format(report_type))
            except ValueError, e:
                self.logger.error('Could not collect all {} readings for {}'.format(sensor.kind, sensor.label))
                self.logger.error('Got: {}'.format(e))

