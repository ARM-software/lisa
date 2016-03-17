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


from wlauto.exceptions import DeviceError


HWMON_ROOT = '/sys/class/hwmon'


class HwmonSensor(object):

    def __init__(self, device, kind, label, filepath):
        self.device = device
        self.kind = kind
        self.label = label
        self.filepath = filepath
        self.readings = []

    def take_reading(self):
        reading = self.device.get_sysfile_value(self.filepath, int)
        self.readings.append(reading)

    def clear_readings(self):
        self.readings = []


def discover_sensors(device, sensor_kinds):
    """
    Discovers HWMON sensors available on the device.

        :device: Device on which to discover HWMON sensors. Must be an instance
                 of  :class:`AndroidDevice`.
        :sensor_kinds: A list of names of sensor types to be discovered. The names
                       must be as they appear prefixed to ``*_input`` files in
                       sysfs. E.g. ``'energy'``.

        :returns: A list of :class:`HwmonSensor` instantces for each found sensor. If
                  no sensors of the specified types were discovered, an empty list
                  will be returned.

    """
    hwmon_devices = device.list_directory(HWMON_ROOT)
    path = device.path
    sensors = []
    for hwmon_device in hwmon_devices:
        try:
            device_path = path.join(HWMON_ROOT, hwmon_device, 'device')
            name = device.get_sysfile_value(path.join(device_path, 'name'))
        except DeviceError:  # probably a virtual device
            device_path = path.join(HWMON_ROOT, hwmon_device)
            name = device.get_sysfile_value(path.join(device_path, 'name'))

        for sensor_kind in sensor_kinds:
            i = 1
            input_path = path.join(device_path, '{}{}_input'.format(sensor_kind, i))
            while device.file_exists(input_path):
                label_path = path.join(device_path, '{}{}_label'.format(sensor_kind, i))
                if device.file_exists(label_path):
                    name += ' ' + device.get_sysfile_value(label_path)
                sensors.append(HwmonSensor(device, sensor_kind, name, input_path))
                i += 1
                input_path = path.join(device_path, '{}{}_input'.format(sensor_kind, i))
    return sensors

