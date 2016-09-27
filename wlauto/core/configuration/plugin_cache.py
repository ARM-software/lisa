#    Copyright 2016 ARM Limited
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

from copy import copy
from collections import defaultdict

from wlauto.core import pluginloader
from wlauto.exceptions import ConfigError
from wlauto.utils.types import obj_dict
from devlib.utils.misc import memoized

GENERIC_CONFIGS = ["device_config", "workload_parameters",
                   "boot_parameters", "runtime_parameters"]


class PluginCache(object):
    """
    The plugin cache is used to store configuration that cannot be processed at
    this stage, whether thats because it is unknown if its needed
    (in the case of disabled plug-ins) or it is not know what it belongs to (in
    the case of "device-config" ect.). It also maintains where configuration came
    from, and the priority order of said sources.
    """

    def __init__(self, loader=pluginloader):
        self.loader = loader
        self.sources = []
        self.plugin_configs = defaultdict(lambda: defaultdict(dict))
        self.global_alias_values = defaultdict(dict)

        # Generate a mapping of what global aliases belong to
        self._global_alias_map = defaultdict(dict)
        self._list_of_global_aliases = set()
        for plugin in self.loader.list_plugins():
            for param in plugin.parameters:
                if param.global_alias:
                    self._global_alias_map[plugin.name][param.global_alias] = param
                    self._list_of_global_aliases.add(param.global_alias)

    def add_source(self, source):
        if source in self.sources:
            raise Exception("Source has already been added.")
        self.sources.append(source)

    def add_global_alias(self, alias, value, source):
        if source not in self.sources:
            msg = "Source '{}' has not been added to the plugin cache."
            raise RuntimeError(msg.format(source))

        if not self.is_global_alias(alias):
            msg = "'{} is not a valid global alias'"
            raise RuntimeError(msg.format(alias))

        self.global_alias_values[alias][source] = value

    def add_configs(self, plugin_name, values, source):
        if self.is_global_alias(plugin_name):
            self.add_global_alias(plugin_name, values, source)
            return
        for name, value in values.iteritems():
            self.add_config(plugin_name, name, value, source)

    def add_config(self, plugin_name, name, value, source):
        if source not in self.sources:
            msg = "Source '{}' has not been added to the plugin cache."
            raise RuntimeError(msg.format(source))

        if not self.loader.has_plugin(plugin_name) and plugin_name not in GENERIC_CONFIGS:
            msg = 'configuration provided for unknown plugin "{}"'
            raise ConfigError(msg.format(plugin_name))

        if (plugin_name not in GENERIC_CONFIGS and
                name not in self.get_plugin_parameters(plugin_name)):
            msg = "'{}' is not a valid parameter for '{}'"
            raise ConfigError(msg.format(name, plugin_name))

        self.plugin_configs[plugin_name][source][name] = value

    def is_global_alias(self, name):
        return name in self._list_of_global_aliases

    def get_plugin_config(self, plugin_name, generic_name=None):
        config = obj_dict(not_in_dict=['name'])
        config.name = plugin_name

        # Load plugin defaults
        cfg_points = self.get_plugin_parameters(plugin_name)
        for cfg_point in cfg_points.itervalues():
            cfg_point.set_value(config, check_mandatory=False)

        # Merge global aliases
        for alias, param in self._global_alias_map[plugin_name].iteritems():
            if alias in self.global_alias_values:
                for source in self.sources:
                    if source not in self.global_alias_values[alias]:
                        continue
                    param.set_value(config, value=self.global_alias_values[alias][source])

        # Merge user config
        # Perform a simple merge with the order of sources representing priority
        if generic_name is None:
            plugin_config = self.plugin_configs[plugin_name]
            for source in self.sources:
                if source not in plugin_config:
                    continue
                for name, value in plugin_config[source].iteritems():
                    cfg_points[name].set_value(config, value=value)
        # A more complicated merge that involves priority of sources and specificity
        else:
            self._merge_using_priority_specificity(plugin_name, generic_name, config)

        return config

    @memoized
    def get_plugin_parameters(self, name):
        params = self.loader.get_plugin_class(name).parameters
        return {param.name: param for param in params}

    # pylint: disable=too-many-nested-blocks, too-many-branches
    def _merge_using_priority_specificity(self, specific_name, generic_name, final_config):
        """
        WA configuration can come from various sources of increasing priority, as well
        as being specified in a generic and specific manner (e.g. ``device_config``
        and ``nexus10`` respectivly). WA has two rules for the priority of configuration:

            - Configuration from higher priority sources overrides configuration from
              lower priority sources.
            - More specific configuration overrides less specific configuration.

        There is a situation where these two rules come into conflict. When a generic
        configuration is given in config source of high priority and a specific
        configuration is given in a config source of lower priority. In this situation
        it is not possible to know the end users intention and WA will error.

        :param generic_name: The name of the generic configuration e.g ``device_config``
        :param specific_name: The name of the specific configuration used, e.g ``nexus10``
        :param cfg_point: A dict of ``ConfigurationPoint``s to be used when merging configuration.
                          keys=config point name, values=config point

        :rtype: A fully merged and validated configuration in the form of a obj_dict.
        """
        generic_config = copy(self.plugin_configs[generic_name])
        specific_config = copy(self.plugin_configs[specific_name])
        cfg_points = self.get_plugin_parameters(specific_name)
        sources = self.sources
        seen_specific_config = defaultdict(list)

        # set_value uses the 'name' attribute of the passed object in it error
        # messages, to ensure these messages make sense the name will have to be
        # changed several times during this function.
        final_config.name = specific_name

        # pylint: disable=too-many-nested-blocks
        for source in sources:
            try:
                if source in generic_config:
                    final_config.name = generic_name
                    for name, cfg_point in cfg_points.iteritems():
                        if name in generic_config[source]:
                            if name in seen_specific_config:
                                msg = ('"{generic_name}" configuration "{config_name}" has already been '
                                       'specified more specifically for {specific_name} in:\n\t\t{sources}')
                                msg = msg.format(generic_name=generic_name,
                                                 config_name=name,
                                                 specific_name=specific_name,
                                                 sources=", ".join(seen_specific_config[name]))
                                raise ConfigError(msg)
                            value = generic_config[source][name]
                            cfg_point.set_value(final_config, value, check_mandatory=False)

                if source in specific_config:
                    final_config.name = specific_name
                    for name, cfg_point in cfg_points.iteritems():
                        if name in specific_config[source]:
                            seen_specific_config[name].append(str(source))
                            value = specific_config[source][name]
                            cfg_point.set_value(final_config, value, check_mandatory=False)

            except ConfigError as e:
                raise ConfigError('Error in "{}":\n\t{}'.format(source, str(e)))

        # Validate final configuration
        final_config.name = specific_name
        for cfg_point in cfg_points.itervalues():
            cfg_point.validate(final_config)
