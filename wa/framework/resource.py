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
import os
import sys
import glob
import shutil
import inspect
import logging
from collections import defaultdict

from wa.framework import pluginloader
from wa.framework.plugin import Plugin, Parameter
from wa.framework.exception import ResourceError
from wa.framework.configuration import settings
from wa.utils.misc import ensure_directory_exists as _d
from wa.utils.types import boolean
from wa.utils.types import prioritylist


class GetterPriority(object):
    """
    Enumerates standard ResourceGetter priorities. In general, getters should
    register under one of these, rather than specifying other priority values.


    :cached: The cached version of the resource. Look here first. This
    priority also implies
             that the resource at this location is a "cache" and is not
             the only version of the resource, so it may be cleared without
             losing access to the resource.
    :preferred: Take this resource in favour of the environment resource.
    :environment: Found somewhere under ~/.workload_automation/ or equivalent,
                  or from environment variables, external configuration
                  files, etc.  These will override resource supplied with
                  the package.
    :external_package: Resource provided by another package.  :package:
                       Resource provided with the package.  :remote:
                       Resource will be downloaded from a remote location
                       (such as an HTTP server or a samba share). Try this
                       only if no other getter was successful.

    """
    cached = 20
    preferred = 10
    environment = 0
    external_package = -5
    package = -10
    remote = -20


class Resource(object):
    """
    Represents a resource that needs to be resolved. This can be pretty much
    anything: a file, environment variable, a Python object, etc. The only
    thing a resource *has* to have is an owner (which would normally be the
    Workload/Instrument/Device/etc object that needs the resource). In
    addition, a resource have any number of attributes to identify, but all of
    them are resource type specific.

    """

    name = None

    def __init__(self, owner):
        self.owner = owner

    def delete(self, instance):
        """
        Delete an instance of this resource type. This must be implemented
        by the concrete subclasses based on what the resource looks like,
        e.g. deleting a file or a directory tree, or removing an entry from
        a database.

        :note: Implementation should *not* contain any logic for deciding
               whether or not a resource should be deleted, only the actual
               deletion. The assumption is that if this method is invoked,
               then the decision has already been made.

        """
        raise NotImplementedError()

    def __str__(self):
        return '<{}\'s {}>'.format(self.owner, self.name)


class ResourceGetter(Plugin):
    """
    Base class for implementing resolvers. Defines resolver
    interface. Resolvers are responsible for discovering resources (such as
    particular kinds of files) they know about based on the parameters that are
    passed to them. Each resolver also has a dict of attributes that describe
    it's operation, and may be used to determine which get invoked.  There is
    no pre-defined set of attributes and resolvers may define their own.

    Class attributes:

    :name: Name that uniquely identifies this getter. Must be set by any
           concrete subclass.
    :resource_type: Identifies resource type(s) that this getter can
                    handle. This must be either a string (for a single type)
                    or a list of strings for multiple resource types. This
                    must be set by any concrete subclass.
    :priority: Priority with which this getter will be invoked. This should
               be one of the standard priorities specified in
               ``GetterPriority`` enumeration. If not set, this will default
               to ``GetterPriority.environment``.

    """

    name = None
    kind = 'resource_getter'
    resource_type = None
    priority = GetterPriority.environment

    def __init__(self, resolver, **kwargs):
        super(ResourceGetter, self).__init__(**kwargs)
        self.resolver = resolver

    def register(self):
        """
        Registers with a resource resolver. Concrete implementations must
        override this to invoke ``self.resolver.register()`` method to register
        ``self`` for specific resource types.

        """
        if self.resource_type is None:
            message = 'No resource type specified for {}'
            raise ValueError(message.format(self.name))
        elif isinstance(self.resource_type, list):
            for rt in self.resource_type:
                self.resolver.register(self, rt, self.priority)
        else:
            self.resolver.register(self, self.resource_type, self.priority)

    def unregister(self):
        """Unregister from a resource resolver."""
        if self.resource_type is None:
            message = 'No resource type specified for {}'
            raise ValueError(message.format(self.name))
        elif isinstance(self.resource_type, list):
            for rt in self.resource_type:
                self.resolver.unregister(self, rt)
        else:
            self.resolver.unregister(self, self.resource_type)

    def get(self, resource, **kwargs):
        """
        This will get invoked by the resolver when attempting to resolve a
        resource, passing in the resource to be resolved as the first
        parameter. Any additional parameters would be specific to a particular
        resource type.

        This method will only be invoked for resource types that the getter has
        registered for.

        :param resource: an instance of :class:`wlauto.core.resource.Resource`.

        :returns: Implementations of this method must return either the
                  discovered resource or ``None`` if the resource could not
                  be discovered.

        """
        raise NotImplementedError()

    def delete(self, resource, *args, **kwargs):
        """
        Delete the resource if it is discovered. All arguments are passed to a
        call to``self.get()``. If that call returns a resource, it is deleted.

        :returns: ``True`` if the specified resource has been discovered
                  and deleted, and ``False`` otherwise.

        """
        discovered = self.get(resource, *args, **kwargs)
        if discovered:
            resource.delete(discovered)
            return True
        else:
            return False

    def __str__(self):
        return '<ResourceGetter {}>'.format(self.name)


class ResourceResolver(object):
    """
    Discovers and registers getters, and then handles requests for
    resources using registered getters.

    """

    def __init__(self):
        self.logger = logging.getLogger('resolver')
        self.getters = defaultdict(prioritylist)

    def load(self, loader=pluginloader):
        """
        Discover getters under the specified source. The source could
        be either a python package/module or a path.

        """
        for rescls in loader.list_resource_getters():
            getter = loader.get_resource_getter(rescls.name, resolver=self)
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
        self.logger.debug('Registering {}'.format(getter.name))
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


class FileResource(Resource):
    """
    Base class for all resources that are a regular file in the
    file system.

    """

    def delete(self, instance):
        os.remove(instance)


class File(FileResource):

    name = 'file'

    def __init__(self, owner, path, url=None):
        super(File, self).__init__(owner)
        self.path = path
        self.url = url

    def __str__(self):
        return '<{}\'s {} {}>'.format(self.owner, self.name, self.path or self.url)


class ExtensionAsset(File):

    name = 'extension_asset'

    def __init__(self, owner, path):
        super(ExtensionAsset, self).__init__(
            owner, os.path.join(owner.name, path))


class Executable(FileResource):

    name = 'executable'

    def __init__(self, owner, platform, filename):
        super(Executable, self).__init__(owner)
        self.platform = platform
        self.filename = filename

    def __str__(self):
        return '<{}\'s {} {}>'.format(self.owner, self.platform, self.filename)


class ReventFile(FileResource):

    name = 'revent'

    def __init__(self, owner, stage):
        super(ReventFile, self).__init__(owner)
        self.stage = stage


class JarFile(FileResource):

    name = 'jar'


class ApkFile(FileResource):

    name = 'apk'


class PackageFileGetter(ResourceGetter):

    name = 'package_file'
    description = """
    Looks for exactly one file with the specified extension in the owner's
    directory. If a version is specified on invocation of get, it will filter
    the discovered file based on that version.  Versions are treated as
    case-insensitive.
    """

    extension = None

    def register(self):
        self.resolver.register(self, self.extension, GetterPriority.package)

    def get(self, resource, **kwargs):
        resource_dir = os.path.dirname(
            sys.modules[resource.owner.__module__].__file__)
        version = kwargs.get('version')
        return get_from_location_by_extension(resource, resource_dir, self.extension, version)


class EnvironmentFileGetter(ResourceGetter):

    name = 'environment_file'
    description = """
    Looks for exactly one file with the specified extension in the owner's
    directory. If a version is specified on invocation of get, it will filter
    the discovered file based on that version.  Versions are treated as
    case-insensitive.
    """

    extension = None

    def register(self):
        self.resolver.register(self, self.extension,
                               GetterPriority.environment)

    def get(self, resource, **kwargs):
        resource_dir = resource.owner.dependencies_directory
        version = kwargs.get('version')
        return get_from_location_by_extension(resource, resource_dir, self.extension, version)


class ReventGetter(ResourceGetter):
    """Implements logic for identifying revent files."""

    def get_base_location(self, resource):
        raise NotImplementedError()

    def register(self):
        self.resolver.register(self, 'revent', GetterPriority.package)

    def get(self, resource, **kwargs):
        filename = '.'.join([resource.owner.device.name,
                             resource.stage, 'revent']).lower()
        location = _d(os.path.join(
            self.get_base_location(resource), 'revent_files'))
        for candidate in os.listdir(location):
            if candidate.lower() == filename.lower():
                return os.path.join(location, candidate)


class PackageApkGetter(PackageFileGetter):
    name = 'package_apk'
    extension = 'apk'


class PackageJarGetter(PackageFileGetter):
    name = 'package_jar'
    extension = 'jar'


class PackageReventGetter(ReventGetter):

    name = 'package_revent'

    def get_base_location(self, resource):
        return _get_owner_path(resource)


class EnvironmentApkGetter(EnvironmentFileGetter):
    name = 'environment_apk'
    extension = 'apk'


class EnvironmentJarGetter(EnvironmentFileGetter):
    name = 'environment_jar'
    extension = 'jar'


class EnvironmentReventGetter(ReventGetter):

    name = 'enviroment_revent'

    def get_base_location(self, resource):
        return resource.owner.dependencies_directory


class ExecutableGetter(ResourceGetter):

    name = 'exe_getter'
    resource_type = 'executable'
    priority = GetterPriority.environment

    def get(self, resource, **kwargs):
        if settings.binaries_repository:
            path = os.path.join(settings.binaries_repository,
                                resource.platform, resource.filename)
            if os.path.isfile(path):
                return path


class PackageExecutableGetter(ExecutableGetter):

    name = 'package_exe_getter'
    priority = GetterPriority.package

    def get(self, resource, **kwargs):
        path = os.path.join(_get_owner_path(resource), 'bin',
                            resource.platform, resource.filename)
        if os.path.isfile(path):
            return path


class EnvironmentExecutableGetter(ExecutableGetter):

    name = 'env_exe_getter'

    def get(self, resource, **kwargs):
        paths = [
            os.path.join(resource.owner.dependencies_directory, 'bin',
                         resource.platform, resource.filename),
            os.path.join(settings.environment_root, 'bin',
                         resource.platform, resource.filename),
        ]
        for path in paths:
            if os.path.isfile(path):
                return path


class DependencyFileGetter(ResourceGetter):

    name = 'filer'
    description = """
    Gets resources from the specified mount point. Copies them the local dependencies
    directory, and returns the path to the local copy.

    """
    resource_type = 'file'
    relative_path = ''  # May be overridden by subclasses.

    default_mount_point = '/'
    priority = GetterPriority.remote

    parameters = [
        Parameter('mount_point', default='/', global_alias='filer_mount_point',
                  description='Local mount point for the remote filer.'),
    ]

    def __init__(self, resolver, **kwargs):
        super(DependencyFileGetter, self).__init__(resolver, **kwargs)
        self.mount_point = settings.filer_mount_point or self.default_mount_point

    def get(self, resource, **kwargs):
        force = kwargs.get('force')
        remote_path = os.path.join(
            self.mount_point, self.relative_path, resource.path)
        local_path = os.path.join(
            resource.owner.dependencies_directory, os.path.basename(resource.path))

        if not os.path.isfile(local_path) or force:
            if not os.path.isfile(remote_path):
                return None
            self.logger.debug('Copying {} to {}'.format(
                remote_path, local_path))
            shutil.copy(remote_path, local_path)

        return local_path


class PackageCommonDependencyGetter(ResourceGetter):

    name = 'packaged_common_dependency'
    resource_type = 'file'
    priority = GetterPriority.package - 1  # check after owner-specific locations

    def get(self, resource, **kwargs):
        path = os.path.join(settings.package_directory,
                            'common', resource.path)
        if os.path.exists(path):
            return path


class EnvironmentCommonDependencyGetter(ResourceGetter):

    name = 'environment_common_dependency'
    resource_type = 'file'
    # check after owner-specific locations
    priority = GetterPriority.environment - 1

    def get(self, resource, **kwargs):
        path = os.path.join(settings.dependencies_directory,
                            os.path.basename(resource.path))
        if os.path.exists(path):
            return path


class PackageDependencyGetter(ResourceGetter):

    name = 'packaged_dependency'
    resource_type = 'file'
    priority = GetterPriority.package

    def get(self, resource, **kwargs):
        owner_path = inspect.getfile(resource.owner.__class__)
        path = os.path.join(os.path.dirname(owner_path), resource.path)
        if os.path.exists(path):
            return path


class EnvironmentDependencyGetter(ResourceGetter):

    name = 'environment_dependency'
    resource_type = 'file'
    priority = GetterPriority.environment

    def get(self, resource, **kwargs):
        path = os.path.join(resource.owner.dependencies_directory,
                            os.path.basename(resource.path))
        if os.path.exists(path):
            return path


class ExtensionAssetGetter(DependencyFileGetter):

    name = 'extension_asset'
    resource_type = 'extension_asset'
    relative_path = 'workload_automation/assets'


class RemoteFilerGetter(ResourceGetter):

    name = 'filer_assets'
    description = """
    Finds resources on a (locally mounted) remote filer and caches them locally.

    This assumes that the filer is mounted on the local machine (e.g. as a samba share).

    """
    priority = GetterPriority.remote
    resource_type = ['apk', 'file', 'jar', 'revent']

    parameters = [
        Parameter('remote_path', global_alias='remote_assets_path', default='',
                  description="""
                  Path, on the local system, where the assets are located.
                  """),
        Parameter('always_fetch', kind=boolean, default=False, global_alias='always_fetch_remote_assets',
                  description="""
                  If ``True``, will always attempt to fetch assets from the
                  remote, even if a local cached copy is available.
                  """),
    ]

    def get(self, resource, **kwargs):
        version = kwargs.get('version')
        if resource.owner:
            remote_path = os.path.join(self.remote_path, resource.owner.name)
            local_path = os.path.join(
                settings.environment_root, resource.owner.dependencies_directory)
            return self.try_get_resource(resource, version, remote_path, local_path)
        else:
            result = None
            for entry in os.listdir(remote_path):
                remote_path = os.path.join(self.remote_path, entry)
                local_path = os.path.join(
                    settings.environment_root, settings.dependencies_directory, entry)
                result = self.try_get_resource(
                    resource, version, remote_path, local_path)
                if result:
                    break
            return result

    def try_get_resource(self, resource, version, remote_path, local_path):
        if not self.always_fetch:
            result = self.get_from(resource, version, local_path)
            if result:
                return result
        if remote_path:
            # Didn't find it cached locally; now check the remoted
            result = self.get_from(resource, version, remote_path)
            if not result:
                return result
        else:  # remote path is not set
            return None
        # Found it remotely, cache locally, then return it
        local_full_path = os.path.join(
            _d(local_path), os.path.basename(result))
        self.logger.debug('cp {} {}'.format(result, local_full_path))
        shutil.copy(result, local_full_path)
        return local_full_path

    def get_from(self, resource, version, location):  # pylint: disable=no-self-use
        if resource.name in ['apk', 'jar']:
            return get_from_location_by_extension(resource, location, resource.name, version)
        elif resource.name == 'file':
            filepath = os.path.join(location, resource.path)
            if os.path.exists(filepath):
                return filepath
        elif resource.name == 'revent':
            filename = '.'.join(
                [resource.owner.device.name, resource.stage, 'revent']).lower()
            alternate_location = os.path.join(location, 'revent_files')
            # There tends to be some confusion as to where revent files should
            # be placed. This looks both in the extension's directory, and in
            # 'revent_files' subdirectory under it, if it exists.
            if os.path.isdir(alternate_location):
                for candidate in os.listdir(alternate_location):
                    if candidate.lower() == filename.lower():
                        return os.path.join(alternate_location, candidate)
            if os.path.isdir(location):
                for candidate in os.listdir(location):
                    if candidate.lower() == filename.lower():
                        return os.path.join(location, candidate)
        else:
            message = 'Unexpected resource type: {}'.format(resource.name)
            raise ValueError(message)


# Utility functions

def get_from_location_by_extension(resource, location, extension, version=None):
    found_files = glob.glob(os.path.join(location, '*.{}'.format(extension)))
    if version:
        found_files = [ff for ff in found_files 
                       if version.lower() in os.path.basename(ff).lower()]
    if len(found_files) == 1:
        return found_files[0]
    elif not found_files:
        return None
    else:
        raise ResourceError('More than one .{} found in {} for {}.'.format(extension,
                                                                           location,
                                                                           resource.owner.name))


def _get_owner_path(resource):
    if resource.owner is NO_ONE:
        return os.path.join(os.path.dirname(__base_filepath), 'common')
    else:
        return os.path.dirname(sys.modules[resource.owner.__module__].__file__)
