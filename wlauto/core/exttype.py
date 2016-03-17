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


# Separate module to avoid circular dependencies
from wlauto.core.config.core import settings
from wlauto.core.plugin import Plugin
from wlauto.utils.misc import load_class
from wlauto.core import pluginloader


def get_plugin_type(ext):
    """Given an instance of ``wlauto.core.Plugin``, return a string representing
    the type of the plugin (e.g. ``'workload'`` for a Workload subclass instance)."""
    if not isinstance(ext, Plugin):
        raise ValueError('{} is not an instance of Plugin'.format(ext))
    for name, cls in pluginloaderkind_map.iteritems():
        if isinstance(ext, cls):
            return name
    raise ValueError('Unknown plugin type: {}'.format(ext.__class__.__name__))
