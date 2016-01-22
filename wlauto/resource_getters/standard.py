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
This module contains the standard set of resource getters used by Workload Automation.

"""
import os
import sys
import shutil
import inspect
import httplib
import logging
import json

import requests

from wlauto import ResourceGetter, GetterPriority, Parameter, NO_ONE, settings, __file__ as __base_filepath
from wlauto.exceptions import ResourceError
from wlauto.utils.misc import ensure_directory_exists as _d, ensure_file_directory_exists as _f, sha256, urljoin
from wlauto.utils.types import boolean


logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class PackageFileGetter(ResourceGetter):

    name = 'package_file'
    description = """
    Looks for exactly one file with the specified extension in the owner's directory. If a version
    is specified on invocation of get, it will filter the discovered file based on that version.
    Versions are treated as case-insensitive.
    """

    extension = None

    def register(self):
        self.resolver.register(self, self.extension, GetterPriority.package)

    def get(self, resource, **kwargs):
        resource_dir = os.path.dirname(sys.modules[resource.owner.__module__].__file__)
        version = kwargs.get('version')
        return get_from_location_by_extension(resource, resource_dir, self.extension, version)


class EnvironmentFileGetter(ResourceGetter):

    name = 'environment_file'
    description = """Looks for exactly one file with the specified extension in the owner's directory. If a version
    is specified on invocation of get, it will filter the discovered file based on that version.
    Versions are treated as case-insensitive."""

    extension = None

    def register(self):
        self.resolver.register(self, self.extension, GetterPriority.environment)

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
        filename = '.'.join([resource.owner.device.name, resource.stage, 'revent']).lower()
        location = _d(os.path.join(self.get_base_location(resource), 'revent_files'))
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
        return get_owner_path(resource)


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
            path = os.path.join(settings.binaries_repository, resource.platform, resource.filename)
            if os.path.isfile(path):
                return path


class PackageExecutableGetter(ExecutableGetter):

    name = 'package_exe_getter'
    priority = GetterPriority.package

    def get(self, resource, **kwargs):
        path = os.path.join(get_owner_path(resource), 'bin', resource.platform, resource.filename)
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

    priority = GetterPriority.remote

    parameters = [
        Parameter('mount_point', default='/', global_alias='remote_assets_path',
                  description='Local mount point for the remote filer.'),
    ]

    def __init__(self, resolver, **kwargs):
        super(DependencyFileGetter, self).__init__(resolver, **kwargs)

    def get(self, resource, **kwargs):
        force = kwargs.get('force')
        remote_path = os.path.join(self.mount_point, self.relative_path, resource.path)
        local_path = os.path.join(resource.owner.dependencies_directory, os.path.basename(resource.path))

        if not os.path.isfile(local_path) or force:
            if not os.path.isfile(remote_path):
                return None
            self.logger.debug('Copying {} to {}'.format(remote_path, local_path))
            shutil.copy(remote_path, local_path)

        return local_path


class PackageCommonDependencyGetter(ResourceGetter):

    name = 'packaged_common_dependency'
    resource_type = 'file'
    priority = GetterPriority.package - 1  # check after owner-specific locations

    def get(self, resource, **kwargs):
        path = os.path.join(settings.package_directory, 'common', resource.path)
        if os.path.exists(path):
            return path


class EnvironmentCommonDependencyGetter(ResourceGetter):

    name = 'environment_common_dependency'
    resource_type = 'file'
    priority = GetterPriority.environment - 1  # check after owner-specific locations

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
        path = os.path.join(resource.owner.dependencies_directory, os.path.basename(resource.path))
        if os.path.exists(path):
            return path


class ExtensionAssetGetter(DependencyFileGetter):

    name = 'extension_asset'
    resource_type = 'extension_asset'


class HttpGetter(ResourceGetter):

    name = 'http_assets'
    description = """
    Downloads resources from a server based on an index fetched from the specified URL.

    Given a URL, this will try to fetch ``<URL>/index.json``. The index file maps extension
    names to a list of corresponing asset descriptons. Each asset description continas a path
    (relative to the base URL) of the resource and a SHA256 hash, so that this Getter can
    verify whether the resource on the remote has changed.

    For example, let's assume we want to get the APK file for workload "foo", and that
    assets are hosted at ``http://example.com/assets``. This Getter will first try to
    donwload ``http://example.com/assests/index.json``. The index file may contian
    something like ::

        {
            "foo": [
                {
                    "path": "foo-app.apk",
                    "sha256": "b14530bb47e04ed655ac5e80e69beaa61c2020450e18638f54384332dffebe86"
                },
                {
                    "path": "subdir/some-other-asset.file",
                    "sha256": "48d9050e9802246d820625717b72f1c2ba431904b8484ca39befd68d1dbedfff"
                }
            ]
        }

    This Getter will look through the list of assets for "foo" (in this case, two) check
    the paths until it finds one matching the resource (in this case, "foo-app.apk").
    Finally, it will try to dowload that file relative to the base URL and extension name
    (in this case, "http://example.com/assets/foo/foo-app.apk"). The downloaded version
    will be cached locally, so that in the future, the getter will check the SHA256 hash
    of the local file against the one advertised inside index.json, and provided that hasn't
    changed, it won't try to download the file again.

    """
    priority = GetterPriority.remote
    resource_type = ['apk', 'file', 'jar', 'revent']

    parameters = [
        Parameter('url', global_alias='remote_assets_url',
                  description="""URL of the index file for assets on an HTTP server."""),
        Parameter('username',
                  description="""User name for authenticating with assets URL"""),
        Parameter('password',
                  description="""Password for authenticationg with assets URL"""),
        Parameter('always_fetch', kind=boolean, default=False, global_alias='always_fetch_remote_assets',
                  description="""If ``True``, will always attempt to fetch assets from the remote, even if
                                 a local cached copy is available."""),
        Parameter('chunk_size', kind=int, default=1024,
                  description="""Chunk size for streaming large assets."""),
    ]

    def __init__(self, resolver, **kwargs):
        super(HttpGetter, self).__init__(resolver, **kwargs)
        self.index = None

    def get(self, resource, **kwargs):
        if not resource.owner:
            return  # TODO: add support for unowned resources
        if not self.index:
            self.index = self.fetch_index()
        asset = self.resolve_resource(resource)
        if not asset:
            return
        return self.download_asset(asset, resource.owner.name)

    def fetch_index(self):
        if not self.url:
            return {}
        index_url = urljoin(self.url, 'index.json')
        response = self.geturl(index_url)
        if response.status_code != httplib.OK:
            message = 'Could not fetch "{}"; recieved "{} {}"'
            self.logger.error(message.format(index_url, response.status_code, response.reason))
            return {}
        return json.loads(response.content)

    def download_asset(self, asset, owner_name):
        url = urljoin(self.url, owner_name, asset['path'])
        local_path = _f(os.path.join(settings.dependencies_directory, '__remote',
                                     owner_name, asset['path'].replace('/', os.sep)))
        if os.path.isfile(local_path) and not self.always_fetch:
            local_sha = sha256(local_path)
            if local_sha == asset['sha256']:
                self.logger.debug('Local SHA256 matches; not re-downloading')
                return local_path
        self.logger.debug('Downloading {}'.format(url))
        response = self.geturl(url, stream=True)
        if response.status_code != httplib.OK:
            message = 'Could not download asset "{}"; recieved "{} {}"'
            self.logger.warning(message.format(url, response.status_code, response.reason))
            return
        with open(local_path, 'wb') as wfh:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                wfh.write(chunk)
        return local_path

    def geturl(self, url, stream=False):
        if self.username:
            auth = (self.username, self.password)
        else:
            auth = None
        return requests.get(url, auth=auth, stream=stream)

    def resolve_resource(self, resource):
        assets = self.index.get(resource.owner.name, {})
        if not assets:
            return {}
        if resource.name in ['apk', 'jar']:
            paths = [a['path'] for a in assets]
            version = getattr(resource, 'version', None)
            found = get_from_list_by_extension(resource, paths, resource.name, version)
            if found:
                for a in assets:
                    if a['path'] == found:
                        return a
        elif resource.name == 'revent':
            filename = '.'.join([resource.owner.device.name, resource.stage, 'revent']).lower()
            for asset in assets:
                pathname = os.path.basename(asset['path']).lower()
                if pathname == filename:
                    return asset
        else:  # file
            for asset in assets:
                if asset['path'].lower() == resource.path.lower():
                    return asset


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
                  description="""Path, on the local system, where the assets are located."""),
        Parameter('always_fetch', kind=boolean, default=False, global_alias='always_fetch_remote_assets',
                  description="""If ``True``, will always attempt to fetch assets from the remote, even if
                                 a local cached copy is available."""),
    ]

    def get(self, resource, **kwargs):
        version = kwargs.get('version')
        if resource.owner:
            remote_path = os.path.join(self.remote_path, resource.owner.name)
            local_path = os.path.join(settings.environment_root, '__filer', resource.owner.dependencies_directory)
            return self.try_get_resource(resource, version, remote_path, local_path)
        else:
            result = None
            for entry in os.listdir(remote_path):
                remote_path = os.path.join(self.remote_path, entry)
                local_path = os.path.join(settings.environment_root, '__filer', settings.dependencies_directory, entry)
                result = self.try_get_resource(resource, version, remote_path, local_path)
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
        local_full_path = os.path.join(_d(local_path), os.path.basename(result))
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
            filename = '.'.join([resource.owner.device.name, resource.stage, 'revent']).lower()
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
            raise ValueError('Unexpected resource type: {}'.format(resource.name))


# Utility functions

def get_from_location_by_extension(resource, location, extension, version=None):
    try:
        found_files = [os.path.join(location, f) for f in os.listdir(location)]
    except OSError:
        return None
    try:
        return get_from_list_by_extension(resource, found_files, extension, version)
    except ResourceError:
        raise ResourceError('More than one .{} found in {} for {}.'.format(extension,
                                                                           location,
                                                                           resource.owner.name))


def get_from_list_by_extension(resource, filelist, extension, version=None):
    filelist = [ff for ff in filelist
                if os.path.splitext(ff)[1].lower().endswith(extension)]
    if version:
        filelist = [ff for ff in filelist if version.lower() in os.path.basename(ff).lower()]
    if len(filelist) == 1:
        return filelist[0]
    elif not filelist:
        return None
    else:
        raise ResourceError('More than one .{} found in {} for {}.'.format(extension,
                                                                           filelist,
                                                                           resource.owner.name))


def get_owner_path(resource):
    if resource.owner is NO_ONE:
        return os.path.join(os.path.dirname(__base_filepath), 'common')
    else:
        return os.path.dirname(sys.modules[resource.owner.__module__].__file__)
