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
from itertools import chain

from devlib.utils.misc import memoized

from wa.framework import pluginloader
from wa.framework.configuration.core import get_config_point_map
from wa.framework.exception import ConfigError
from wa.framework.target.descriptor import get_target_descriptions
from wa.utils.types import obj_dict

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
        self.targets = {td.name: td for td in get_target_descriptions()}

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
            msg = "Source '{}' has already been added."
            raise Exception(msg.format(source))
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

        if source not in self.sources:
            msg = "Source '{}' has not been added to the plugin cache."
            raise RuntimeError(msg.format(source))

        if (not self.loader.has_plugin(plugin_name) and
                plugin_name not in GENERIC_CONFIGS):
            msg = 'configuration provided for unknown plugin "{}"'
            raise ConfigError(msg.format(plugin_name))

        if not hasattr(values, 'iteritems'):
            msg = 'Plugin configuration for "{}" not a dictionary ({} is {})'
            raise ConfigError(msg.format(plugin_name, repr(values), type(values)))

        for name, value in values.iteritems():
            if (plugin_name not in GENERIC_CONFIGS and
                    name not in self.get_plugin_parameters(plugin_name)):
                msg = "'{}' is not a valid parameter for '{}'"
                raise ConfigError(msg.format(name, plugin_name))

            self.plugin_configs[plugin_name][source][name] = value

    def is_global_alias(self, name):
        return name in self._list_of_global_aliases

    def list_plugins(self, kind=None):
        return self.loader.list_plugins(kind)

    def get_plugin_config(self, plugin_name, generic_name=None, is_final=True):
        config = obj_dict(not_in_dict=['name'])
        config.name = plugin_name

        if plugin_name not in GENERIC_CONFIGS:
            self._set_plugin_defaults(plugin_name, config)
            self._set_from_global_aliases(plugin_name, config)

        if generic_name is None:
            # Perform a simple merge with the order of sources representing
            # priority
            plugin_config = self.plugin_configs[plugin_name]
            cfg_points = self.get_plugin_parameters(plugin_name)
            for source in self.sources:
                if source not in plugin_config:
                    continue
                for name, value in plugin_config[source].iteritems():
                    cfg_points[name].set_value(config, value=value)
        else:
            # A more complicated merge that involves priority of sources and
            # specificity
            self._merge_using_priority_specificity(plugin_name, generic_name,
                                                   config, is_final)

        return config

    def get_plugin(self, name, kind=None, *args, **kwargs):
        config = self.get_plugin_config(name)
        kwargs = dict(config.items() + kwargs.items())
        return self.loader.get_plugin(name, kind=kind, *args, **kwargs)

    def get_plugin_class(self, name, kind=None):
        return self.loader.get_plugin_class(name, kind)

    @memoized
    def get_plugin_parameters(self, name):
        if name in self.targets:
            return self._get_target_params(name)
        params = self.loader.get_plugin_class(name).parameters
        return get_config_point_map(params)

    def _set_plugin_defaults(self, plugin_name, config):
        cfg_points = self.get_plugin_parameters(plugin_name)
        for cfg_point in cfg_points.itervalues():
            cfg_point.set_value(config, check_mandatory=False)

    def _set_from_global_aliases(self, plugin_name, config):
        for alias, param in self._global_alias_map[plugin_name].iteritems():
            if alias in self.global_alias_values:
                for source in self.sources:
                    if source not in self.global_alias_values[alias]:
                        continue
                    val = self.global_alias_values[alias][source]
                    param.set_value(config, value=val)

    def _get_target_params(self, name):
        td = self.targets[name]
        return get_config_point_map(chain(td.target_params, td.platform_params, td.conn_params))

    # pylint: disable=too-many-nested-blocks, too-many-branches
    def _merge_using_priority_specificity(self, specific_name,
                                          generic_name, merged_config, is_final=True):
        """
        WA configuration can come from various sources of increasing priority,
        as well as being specified in a generic and specific manner (e.g
        ``device_config`` and ``nexus10`` respectivly). WA has two rules for
        the priority of configuration:

            - Configuration from higher priority sources overrides
              configuration from lower priority sources.
            - More specific configuration overrides less specific configuration.

        There is a situation where these two rules come into conflict. When a
        generic configuration is given in config source of high priority and a
        specific configuration is given in a config source of lower priority.
        In this situation it is not possible to know the end users intention
        and WA will error.

        :param specific_name: The name of the specific configuration used
                              e.g ``nexus10``
        :param generic_name: The name of the generic configuration
                             e.g ``device_config``
        :param merge_config: A dict of ``ConfigurationPoint``s to be used when
                             merging configuration.  keys=config point name,
                             values=config point
        :param is_final: if ``True`` (the default) make sure that mandatory
                         parameters are set.

        :rtype: A fully merged and validated configuration in the form of a
                obj_dict.
        """
        ms = MergeState()
        ms.generic_name = generic_name
        ms.specific_name = specific_name
        ms.generic_config = copy(self.plugin_configs[generic_name])
        ms.specific_config = copy(self.plugin_configs[specific_name])
        ms.cfg_points = self.get_plugin_parameters(specific_name)
        sources = self.sources

        # set_value uses the 'name' attribute of the passed object in it error
        # messages, to ensure these messages make sense the name will have to be
        # changed several times during this function.
        merged_config.name = specific_name

        for source in sources:
            try:
                update_config_from_source(merged_config, source, ms)
            except ConfigError as e:
                raise ConfigError('Error in "{}":\n\t{}'.format(source, str(e)))

        # Validate final configuration
        merged_config.name = specific_name
        for cfg_point in ms.cfg_points.itervalues():
            cfg_point.validate(merged_config, check_mandatory=is_final)


class MergeState(object):

    def __init__(self):
        self.generic_name = None
        self.specific_name = None
        self.generic_config = None
        self.specific_config = None
        self.cfg_points = None
        self.seen_specific_config = defaultdict(list)


def update_config_from_source(final_config, source, state):
    if source in state.generic_config:
        final_config.name = state.generic_name
        for name, cfg_point in state.cfg_points.iteritems():
            if name in state.generic_config[source]:
                if name in state.seen_specific_config:
                    msg = ('"{generic_name}" configuration "{config_name}" has '
                            'already been specified more specifically for '
                            '{specific_name} in:\n\t\t{sources}')
                    seen_sources = state.seen_specific_config[name]
                    msg = msg.format(generic_name=generic_name,
                                        config_name=name,
                                        specific_name=specific_name,
                                        sources=", ".join(seen_sources))
                    raise ConfigError(msg)
                value = state.generic_config[source].pop(name)
                cfg_point.set_value(final_config, value, check_mandatory=False)

        if state.generic_config[source]:
            msg = 'Unexpected values for {}: {}'
            raise ConfigError(msg.format(state.generic_name,
                                         state.generic_config[source]))

    if source in state.specific_config:
        final_config.name = state.specific_name
        for name, cfg_point in state.cfg_points.iteritems():
            if name in state.specific_config[source]:
                seen_state.specific_config[name].append(str(source))
                value = state.specific_config[source].pop(name)
                cfg_point.set_value(final_config, value, check_mandatory=False)

        if state.specific_config[source]:
            msg = 'Unexpected values for {}: {}'
            raise ConfigError(msg.format(state.specific_name,
                                         state.specific_config[source]))
