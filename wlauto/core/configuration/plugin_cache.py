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


class PluginCache(object):

    def __init__(self):
        self.plugin_configs = {}
        self.device_config = OrderedDict()
        self.source_list = []
        self.finalised = False
        # TODO: Build dics of global_alias: [list of destinations]

    def add_source(self, source):
        if source in self.source_list:
            raise Exception("Source has already been added.")
        self.source_list.append(source)

    def add(self, name, config, source):
        if source not in self.source_list:
            msg = "Source '{}' has not been added to the plugin cache."
            raise Exception(msg.format(source))

        if name not in self.plugin_configs:
            self.plugin_configs[name] = OrderedDict()
        self.plugin_configs[name][source] = config

    def finalise_config(self):
        pass

    def disable_instrument(self, instrument):
        pass

    def add_device_config(self, config):
        pass

    def is_global_alias(self, name):
        pass

    def add_global_alias(self, name, value):
        pass
