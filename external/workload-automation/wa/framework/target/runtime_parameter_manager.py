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

from collections import namedtuple

from wa.framework.exception import ConfigError
from wa.framework.target.runtime_config import (SysfileValuesRuntimeConfig,
                                                HotplugRuntimeConfig,
                                                CpufreqRuntimeConfig,
                                                CpuidleRuntimeConfig,
                                                AndroidRuntimeConfig)
from wa.utils.types import obj_dict, caseless_string
from wa.framework import pluginloader


class RuntimeParameterManager(object):

    runtime_config_cls = [
        # order matters
        SysfileValuesRuntimeConfig,
        HotplugRuntimeConfig,
        CpufreqRuntimeConfig,
        CpuidleRuntimeConfig,
        AndroidRuntimeConfig,
    ]

    def __init__(self, target):
        self.target = target
        self.runtime_params = {}

        try:
            for rt_cls in pluginloader.list_plugins(kind='runtime-config'):
                if rt_cls not in self.runtime_config_cls:
                    self.runtime_config_cls.append(rt_cls)
        except ValueError:
            pass
        self.runtime_configs = [cls(self.target) for cls in self.runtime_config_cls]

        runtime_parameter = namedtuple('RuntimeParameter', 'cfg_point, rt_config')
        for cfg in self.runtime_configs:
            for param in cfg.supported_parameters:
                if param.name in self.runtime_params:
                    msg = 'Duplicate runtime parameter name "{}": in both {} and {}'
                    raise RuntimeError(msg.format(param.name,
                                                  self.runtime_params[param.name].rt_config.name,
                                                  cfg.name))
                self.runtime_params[param.name] = runtime_parameter(param, cfg)

    # Uses corresponding config point to merge parameters
    def merge_runtime_parameters(self, parameters):
        merged_params = obj_dict()
        for source in parameters:
            for name, value in parameters[source].items():
                cp = self.get_cfg_point(name)
                cp.set_value(merged_params, value)
        return dict(merged_params)

    # Validates runtime_parameters against each other
    def validate_runtime_parameters(self, parameters):
        self.clear_runtime_parameters()
        self.set_runtime_parameters(parameters)
        for cfg in self.runtime_configs:
            cfg.validate_parameters()

    # Writes the given parameters to the device.
    def commit_runtime_parameters(self, parameters):
        self.clear_runtime_parameters()
        self.set_runtime_parameters(parameters)
        for cfg in self.runtime_configs:
            cfg.commit()

    # Stores a set of parameters performing isolated validation when appropriate
    def set_runtime_parameters(self, parameters):
        for name, value in parameters.items():
            cfg = self.get_config_for_name(name)
            if cfg is None:
                msg = 'Unsupported runtime parameter: "{}"'
                raise ConfigError(msg.format(name))
            cfg.set_runtime_parameter(name, value)

    def clear_runtime_parameters(self):
        for cfg in self.runtime_configs:
            cfg.clear()
            cfg.set_defaults()

    def get_config_for_name(self, name):
        name = caseless_string(name)
        for k, v in self.runtime_params.items():
            if name == k:
                return v.rt_config
        return None

    def get_cfg_point(self, name):
        name = caseless_string(name)
        for k, v in self.runtime_params.items():
            if name == k or name in v.cfg_point.aliases:
                return v.cfg_point
        raise ConfigError('Unknown runtime parameter: {}'.format(name))
