#    Copyright 2013-2018 ARM Limited
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
import sys


class __LoaderWrapper(object):

    @property
    def kinds(self):
        if not self._loader:
            self.reset()
        return list(self._loader.kind_map.keys())

    @property
    def kind_map(self):
        if not self._loader:
            self.reset()
        return self._loader.kind_map

    def __init__(self):
        self._loader = None

    def reset(self):
        # These imports cannot be done at top level, because of
        # sys.modules manipulation below
        # pylint: disable=import-outside-toplevel
        from wa.framework.plugin import PluginLoader
        from wa.framework.configuration.core import settings
        self._loader = PluginLoader(settings.plugin_packages,
                                    settings.plugin_paths, [])

    def update(self, packages=None, paths=None, ignore_paths=None):
        if not self._loader:
            self.reset()
        self._loader.update(packages, paths, ignore_paths)

    def reload(self):
        if not self._loader:
            self.reset()
        self._loader.reload()

    def list_plugins(self, kind=None):
        if not self._loader:
            self.reset()
        return self._loader.list_plugins(kind)

    def has_plugin(self, name, kind=None):
        if not self._loader:
            self.reset()
        return self._loader.has_plugin(name, kind)

    def get_plugin_class(self, name, kind=None):
        if not self._loader:
            self.reset()
        return self._loader.get_plugin_class(name, kind)

    def get_plugin(self, name=None, kind=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        if not self._loader:
            self.reset()
        return self._loader.get_plugin(name=name, kind=kind, *args, **kwargs)

    def get_default_config(self, name):
        if not self._loader:
            self.reset()
        return self._loader.get_default_config(name)

    def resolve_alias(self, name):
        if not self._loader:
            self.reset()
        return self._loader.resolve_alias(name)

    def __getattr__(self, name):
        if not self._loader:
            self.reset()
        return getattr(self._loader, name)


sys.modules[__name__] = __LoaderWrapper()
