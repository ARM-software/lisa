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


"""
Defines infrastructure for resource resolution. This is used to find
various dependencies/assets/etc that WA objects rely on in a flexible way.

"""
import logging
from collections import defaultdict

# Note: this is the modified louie library in wlauto/external.
#       prioritylist does not exist in vanilla louie.
from wlauto.utils.types import prioritylist  # pylint: disable=E0611,F0401

from wlauto.exceptions import ResourceError
from wlauto.core import pluginloader

class ResourceResolver(object):
    """
    Discovers and registers getters, and then handles requests for
    resources using registered getters.

    """

    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.getters = defaultdict(prioritylist)
        self.config = config

    def load(self):
        """
        Discover getters under the specified source. The source could
        be either a python package/module or a path.

        """

        for rescls in self.config.ext_loader.list_resource_getters():
            getter = self.config.get_plugin(name=rescls.name, kind="resource_getter", resolver=self)
            getter.register()

    def get(self, resource, strict=True, *args, **kwargs):
        """
        Uses registered getters to attempt to discover a resource of the specified
        kind and matching the specified criteria. Returns path to the resource that
        has been discovered. If a resource has not been discovered, this will raise
        a ``ResourceError`` or, if ``strict`` has been set to ``False``, will return
        ``None``.

        """
        self.logger.debug('Resolving {}'.format(resource))
        for getter in self.getters[resource.name]:
            self.logger.debug('Trying {}'.format(getter))
            result = getter.get(resource, *args, **kwargs)
            if result is not None:
                self.logger.debug('Resource {} found using {}:'.format(resource, getter))
                self.logger.debug('\t{}'.format(result))
                return result
        if strict:
            raise ResourceError('{} could not be found'.format(resource))
        self.logger.debug('Resource {} not found.'.format(resource))
        return None

    def register(self, getter, kind, priority=0):
        """
        Register the specified resource getter as being able to discover a resource
        of the specified kind with the specified priority.

        This method would typically be invoked by a getter inside its __init__.
        The idea being that getters register themselves for resources they know
        they can discover.

        *priorities*

        getters that are registered with the highest priority will be invoked first. If
        multiple getters are registered under the same priority, they will be invoked
        in the order they were registered (i.e. in the order they were discovered). This is
        essentially non-deterministic.

        Generally getters that are more likely to find a resource, or would find a
        "better" version of the resource should register with higher (positive) priorities.
        Fall-back getters that should only be invoked if a resource is not found by usual
        means should register with lower (negative) priorities.

        """
        self.logger.debug('Registering {} for {} resources'.format(getter.name, kind))
        self.getters[kind].add(getter, priority)

    def unregister(self, getter, kind):
        """
        Unregister a getter that has been registered earlier.

        """
        self.logger.debug('Unregistering {}'.format(getter.name))
        try:
            self.getters[kind].remove(getter)
        except ValueError:
            raise ValueError('Resource getter {} is not installed.'.format(getter.name))
