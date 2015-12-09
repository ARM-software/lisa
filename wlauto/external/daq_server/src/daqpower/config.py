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


import argparse

from daqpower.common import Serializable


class ConfigurationError(Exception):
    """Raised when configuration passed into DaqServer is invaid."""
    pass


class DeviceConfiguration(Serializable):
    """Encapulates configuration for the DAQ, typically, passed from
    the client."""

    valid_settings = ['device_id', 'v_range', 'dv_range', 'sampling_rate', 'resistor_values', 'labels']

    default_device_id = 'Dev1'
    default_v_range = 2.5
    default_dv_range = 0.2
    default_sampling_rate = 10000
    # Channel map used in DAQ 6363 and similar.
    default_channel_map = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)

    @property
    def number_of_ports(self):
        return len(self.resistor_values)

    def __init__(self, **kwargs):  # pylint: disable=W0231
        try:
            self.device_id = kwargs.pop('device_id') or self.default_device_id
            self.v_range = float(kwargs.pop('v_range') or self.default_v_range)
            self.dv_range = float(kwargs.pop('dv_range') or self.default_dv_range)
            self.sampling_rate = int(kwargs.pop('sampling_rate') or self.default_sampling_rate)
            self.resistor_values = kwargs.pop('resistor_values') or []
            self.channel_map = kwargs.pop('channel_map') or self.default_channel_map
            self.labels = (kwargs.pop('labels') or
                           ['PORT_{}.csv'.format(i) for i in xrange(len(self.resistor_values))])
        except KeyError, e:
            raise ConfigurationError('Missing config: {}'.format(e.message))
        if kwargs:
            raise ConfigurationError('Unexpected config: {}'.format(kwargs))

    def validate(self):
        if not self.number_of_ports:
            raise ConfigurationError('No resistor values were specified.')
        if len(self.resistor_values) != len(self.labels):
            message = 'The number  of resistors ({}) does not match the number of labels ({})'
            raise ConfigurationError(message.format(len(self.resistor_values), len(self.labels)))

    def __str__(self):
        return self.serialize()

    __repr__ = __str__


class ServerConfiguration(object):
    """Client-side server configuration."""

    valid_settings = ['host', 'port']

    default_host = '127.0.0.1'
    default_port = 45677

    def __init__(self, **kwargs):
        self.host = kwargs.pop('host', None) or self.default_host
        self.port = kwargs.pop('port', None) or self.default_port
        if kwargs:
            raise ConfigurationError('Unexpected config: {}'.format(kwargs))

    def validate(self):
        if not self.host:
            raise ConfigurationError('Server host not specified.')
        if not self.port:
            raise ConfigurationError('Server port not specified.')
        elif not isinstance(self.port, int):
            raise ConfigurationError('Server port must be an integer.')


class UpdateDeviceConfig(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setting = option_string.strip('-').replace('-', '_')
        if setting not in DeviceConfiguration.valid_settings:
            raise ConfigurationError('Unkown option: {}'.format(option_string))
        setattr(namespace._device_config, setting, values)  # pylint: disable=protected-access


class UpdateServerConfig(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setting = option_string.strip('-').replace('-', '_')
        if setting not in namespace.server_config.valid_settings:
            raise ConfigurationError('Unkown option: {}'.format(option_string))
        setattr(namespace.server_config, setting, values)


class ConfigNamespace(object):

    class _N(object):
        def __init__(self):
            self.device_id = None
            self.v_range = None
            self.dv_range = None
            self.sampling_rate = None
            self.resistor_values = None
            self.labels = None
            self.channel_map = None

    @property
    def device_config(self):
        return DeviceConfiguration(**self._device_config.__dict__)

    def __init__(self):
        self._device_config = self._N()
        self.server_config = ServerConfiguration()


class ConfigArgumentParser(argparse.ArgumentParser):

    def parse_args(self, *args, **kwargs):
        kwargs['namespace'] = ConfigNamespace()
        return super(ConfigArgumentParser, self).parse_args(*args, **kwargs)


def get_config_parser(server=True, device=True):
    parser = ConfigArgumentParser()
    if device:
        parser.add_argument('--device-id', action=UpdateDeviceConfig)
        parser.add_argument('--v-range', action=UpdateDeviceConfig, type=float)
        parser.add_argument('--dv-range', action=UpdateDeviceConfig, type=float)
        parser.add_argument('--sampling-rate', action=UpdateDeviceConfig, type=int)
        parser.add_argument('--resistor-values', action=UpdateDeviceConfig, type=float, nargs='*')
        parser.add_argument('--labels', action=UpdateDeviceConfig, nargs='*')
    if server:
        parser.add_argument('--host', action=UpdateServerConfig)
        parser.add_argument('--port', action=UpdateServerConfig, type=int)
    return parser
