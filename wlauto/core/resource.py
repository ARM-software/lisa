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

from wlauto.core.bootstrap import settings
from wlauto.core.extension import Extension


class GetterPriority(object):
    """
    Enumerates standard ResourceGetter priorities. In general, getters should register
    under one of these, rather than specifying other priority values.


    :cached: The cached version of the resource. Look here first. This priority also implies
             that the resource at this location is a "cache" and is not the only version of the
             resource, so it may be cleared without losing access to the resource.
    :preferred: Take this resource in favour of the environment resource.
    :environment: Found somewhere under ~/.workload_automation/ or equivalent, or
                    from environment variables, external configuration files, etc.
                    These will override resource supplied with the package.
    :external_package: Resource provided by another package.
    :package: Resource provided with the package.
    :remote: Resource will be downloaded from a remote location (such as an HTTP server
                or a samba share). Try this only if no other getter was successful.

    """
    cached = 20
    preferred = 10
    remote = 5
    environment = 0
    external_package = -5
    package = -10


class Resource(object):
    """
    Represents a resource that needs to be resolved. This can be pretty much
    anything: a file, environment variable, a Python object, etc. The only thing
    a resource *has* to have is an owner (which would normally be the
    Workload/Instrument/Device/etc object that needs the resource). In addition,
    a resource have any number of attributes to identify, but all of them are resource
    type specific.

    """

    name = None

    def __init__(self, owner):
        self.owner = owner

    def delete(self, instance):
        """
        Delete an instance of this resource type. This must be implemented by the concrete
        subclasses based on what the resource looks like, e.g. deleting a file or a directory
        tree, or removing an entry from a database.

        :note: Implementation should *not* contain any logic for deciding whether or not
               a resource should be deleted, only the actual deletion. The assumption is
               that if this method is invoked, then the decision has already been made.

        """
        raise NotImplementedError()

    def __str__(self):
        return '<{}\'s {}>'.format(self.owner, self.name)


class ResourceGetter(Extension):
    """
    Base class for implementing resolvers. Defines resolver interface. Resolvers are
    responsible for discovering resources (such as particular kinds of files) they know
    about based on the parameters that are passed to them. Each resolver also has a dict of
    attributes that describe its operation, and may be used to determine which get invoked.
    There is no pre-defined set of attributes and resolvers may define their own.

    Class attributes:

    :name: Name that uniquely identifies this getter. Must be set by any concrete subclass.
    :resource_type: Identifies resource type(s) that this getter can handle. This must
                    be either a string (for a single type) or a list of strings for
                    multiple resource types. This must be set by any concrete subclass.
    :priority: Priority with which this getter will be invoked. This should be one of
                the standard priorities specified in ``GetterPriority`` enumeration. If not
                set, this will default to ``GetterPriority.environment``.

    """

    name = None
    resource_type = None
    priority = GetterPriority.environment

    def __init__(self, resolver, **kwargs):
        super(ResourceGetter, self).__init__(**kwargs)
        self.resolver = resolver

    def register(self):
        """
        Registers with a resource resolver. Concrete implementations must override this
        to invoke ``self.resolver.register()`` method to register ``self`` for specific
        resource types.

        """
        if self.resource_type is None:
            raise ValueError('No resource type specified for {}'.format(self.name))
        elif isinstance(self.resource_type, list):
            for rt in self.resource_type:
                self.resolver.register(self, rt, self.priority)
        else:
            self.resolver.register(self, self.resource_type, self.priority)

    def unregister(self):
        """Unregister from a resource resolver."""
        if self.resource_type is None:
            raise ValueError('No resource type specified for {}'.format(self.name))
        elif isinstance(self.resource_type, list):
            for rt in self.resource_type:
                self.resolver.unregister(self, rt)
        else:
            self.resolver.unregister(self, self.resource_type)

    def get(self, resource, **kwargs):
        """
        This will get invoked by the resolver when attempting to resolve a resource, passing
        in the resource to be resolved as the first parameter. Any additional parameters would
        be specific to a particular resource type.

        This method will only be invoked for resource types that the getter has registered for.

        :param resource: an instance of :class:`wlauto.core.resource.Resource`.

        :returns: Implementations of this method must return either the discovered resource or
                  ``None`` if the resource could not be discovered.

        """
        raise NotImplementedError()

    def delete(self, resource, *args, **kwargs):
        """
        Delete the resource if it is discovered. All arguments are passed to a call
        to``self.get()``. If that call returns a resource, it is deleted.

        :returns: ``True`` if the specified resource has been discovered and deleted,
                  and ``False`` otherwise.

        """
        discovered = self.get(resource, *args, **kwargs)
        if discovered:
            resource.delete(discovered)
            return True
        else:
            return False

    def __str__(self):
        return '<ResourceGetter {}>'.format(self.name)


class __NullOwner(object):
    """Represents an owner for a resource not owned by anyone."""

    name = 'noone'
    dependencies_directory = settings.dependencies_directory

    def __getattr__(self, name):
        return None

    def __str__(self):
        return 'no-one'

    __repr__ = __str__


NO_ONE = __NullOwner()
