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


from collections import OrderedDict

from wlauto.utils.types import obj_dict


class PluginCache(object):
    """
    The plugin cache is used to store configuration that cannot be processed at
    this stage, whether thats because it is unknown if its needed
    (in the case of disabled plug-ins) or it is not know what it belongs to (in
    the case of "device-config" ect.). It also maintains where configuration came
    from, and the priority order of said sources.
    """

    def __init__(self):
        self.plugin_configs = {}
        self.global_alias = {}
        self.sources = []
        self.finalised = False
        # TODO: Build dicts of global_alias: [list of destinations]

    def add_source(self, source):
        if source in self.sources:
            raise Exception("Source has already been added.")
        self.sources.append(source)

    def _add_config(self, destination, name, value, source):
        if source not in self.sources:
            msg = "Source '{}' has not been added to the plugin cache."
            raise Exception(msg.format(source))

        if name not in destination:
            destination[name] = OrderedDict()
        destination[name][source] = value

    def add_plugin_config(self, name, config, source):
        self._add_config(self.plugin_configs, name, config, source)

    def add_global_alias(self, name, config, source):
        self._add_config(self.global_alias, name, config, source)

    def finalise_config(self):
        pass

    def is_global_alias(self, name):
        pass

    def get_plugin_config(self, name):
        return self.plugin_configs[name]

    def get_plugin_config_points(self, name):
        pass
