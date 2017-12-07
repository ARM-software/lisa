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
import httplib
import json
import logging
import os
import shutil
import sys

import requests


from wa import Parameter, settings, __file__ as _base_filepath
from wa.framework.resource import ResourceGetter, SourcePriority, NO_ONE
from wa.framework.exception import ResourceError
from wa.utils.misc import (ensure_directory_exists as _d,
                           ensure_file_directory_exists as _f, sha256, urljoin)
from wa.utils.types import boolean, caseless_string

# Because of use of Enum (dynamic attrs)
# pylint: disable=no-member

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger('resource')


def get_by_extension(path, ext):
    if not ext.startswith('.'):
        ext = '.' + ext
    ext = caseless_string(ext)

    found = []
    for entry in os.listdir(path):
        entry_ext = os.path.splitext(entry)[1]
        if entry_ext == ext:
            found.append(os.path.join(path, entry))
    return found


def get_generic_resource(resource, files):
    matches = []
    for f in files:
        if resource.match(f):
            matches.append(f)
    if not matches:
        return None
    if len(matches) > 1:
        msg = 'Multiple matches for {}: {}'
        raise ResourceError(msg.format(resource, matches))
    return matches[0]


def get_path_matches(resource, files):
    matches = []
    for f in files:
        if resource.match_path(f):
            matches.append(f)
    return  matches


def get_from_location(basepath, resource):
    if resource.kind == 'file':
        path = os.path.join(basepath, resource.path)
        if os.path.exists(path):
            return path
    elif resource.kind == 'executable':
        path = os.path.join(basepath, 'bin', resource.abi, resource.filename)
        if os.path.exists(path):
            return path
    elif resource.kind == 'revent':
        path = os.path.join(basepath, 'revent_files')
        if os.path.exists(path):
            files = get_by_extension(path, resource.kind)
            found_resource = get_generic_resource(resource, files)
            if found_resource:
                return found_resource
        files = get_by_extension(basepath, resource.kind)
        return get_generic_resource(resource, files)
    elif resource.kind in ['apk', 'jar']:
        files = get_by_extension(basepath, resource.kind)
        return get_generic_resource(resource, files)

    return None


class Package(ResourceGetter):

    name = 'package'

    def register(self, resolver):
        resolver.register(self.get, SourcePriority.package)

    # pylint: disable=no-self-use
    def get(self, resource):
        if resource.owner == NO_ONE:
            basepath = os.path.join(os.path.dirname(_base_filepath), 'assets')
        else:
            modname = resource.owner.__module__
            basepath = os.path.dirname(sys.modules[modname].__file__)
        return get_from_location(basepath, resource)


class UserDirectory(ResourceGetter):

    name = 'user'

    def register(self, resolver):
        resolver.register(self.get, SourcePriority.local)

    # pylint: disable=no-self-use
    def get(self, resource):
        basepath = settings.dependencies_directory
        directory = _d(os.path.join(basepath, resource.owner.name))
        return get_from_location(directory, resource)


class Http(ResourceGetter):

    name = 'http'
    description = """
    Downloads resources from a server based on an index fetched from the
    specified URL.

    Given a URL, this will try to fetch ``<URL>/index.json``. The index file
    maps extension names to a list of corresponing asset descriptons. Each
    asset description continas a path (relative to the base URL) of the
    resource and a SHA256 hash, so that this Getter can verify whether the
    resource on the remote has changed.

    For example, let's assume we want to get the APK file for workload "foo",
    and that assets are hosted at ``http://example.com/assets``. This Getter
    will first try to donwload ``http://example.com/assests/index.json``. The
    index file may contian something like ::

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

    This Getter will look through the list of assets for "foo" (in this case,
    two) check the paths until it finds one matching the resource (in this
    case, "foo-app.apk").  Finally, it will try to dowload that file relative
    to the base URL and extension name (in this case,
    "http://example.com/assets/foo/foo-app.apk"). The downloaded version will
    be cached locally, so that in the future, the getter will check the SHA256
    hash of the local file against the one advertised inside index.json, and
    provided that hasn't changed, it won't try to download the file again.

    """
    parameters = [
        Parameter('url', global_alias='remote_assets_url',
                  description="""
                  URL of the index file for assets on an HTTP server.
                  """),
        Parameter('username',
                  description="""
                  User name for authenticating with assets URL
                  """),
        Parameter('password',
                  description="""
                  Password for authenticationg with assets URL
                  """),
        Parameter('always_fetch', kind=boolean, default=False,
                  global_alias='always_fetch_remote_assets',
                  description="""
                  If ``True``, will always attempt to fetch assets from the
                  remote, even if a local cached copy is available.
                  """),
        Parameter('chunk_size', kind=int, default=1024,
                  description="""
                  Chunk size for streaming large assets.
                  """),
    ]

    def __init__(self, **kwargs):
        super(Http, self).__init__(**kwargs)
        self.logger = logger
        self.index = None

    def register(self, resolver):
        resolver.register(self.get, SourcePriority.remote)

    def get(self, resource):
        if not resource.owner:
            return  # TODO: add support for unowned resources
        if not self.index:
            self.index = self.fetch_index()
        if resource.kind == 'apk':
            # APKs must always be downloaded to run ApkInfo for version
            # information.
            return self.resolve_apk(resource)
        else:
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
            self.logger.error(message.format(index_url,
                                             response.status_code,
                                             response.reason))
            return {}
        return json.loads(response.content)

    def download_asset(self, asset, owner_name):
        url = urljoin(self.url, owner_name, asset['path'])
        local_path = _f(os.path.join(settings.dependencies_directory, '__remote',
                                     owner_name, asset['path'].replace('/', os.sep)))
        if os.path.exists(local_path) and not self.always_fetch:
            local_sha = sha256(local_path)
            if local_sha == asset['sha256']:
                self.logger.debug('Local SHA256 matches; not re-downloading')
                return local_path
        self.logger.debug('Downloading {}'.format(url))
        response = self.geturl(url, stream=True)
        if response.status_code != httplib.OK:
            message = 'Could not download asset "{}"; recieved "{} {}"'
            self.logger.warning(message.format(url,
                                               response.status_code,
                                               response.reason))
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

    def resolve_apk(self, resource):
        assets = self.index.get(resource.owner.name, {})
        if not assets:
            return None
        asset_map = {a['path']: a for a in assets}
        paths = get_path_matches(resource, asset_map.keys())
        local_paths = []
        for path in paths:
            local_paths.append(self.download_asset(asset_map[path],
                                                   resource.owner.name))
        for path in local_paths:
            if resource.match(path):
                return path

    def resolve_resource(self, resource):
        # pylint: disable=too-many-branches,too-many-locals
        assets = self.index.get(resource.owner.name, {})
        if not assets:
            return {}

        asset_map = {a['path']: a for a in assets}
        if resource.kind in ['jar', 'revent']:
            path = get_generic_resource(resource, asset_map.keys())
            if path:
                return asset_map[path]
        elif resource.kind == 'executable':
            path = '/'.join(['bin', resource.abi, resource.filename])
            for asset in assets:
                if asset['path'].lower() == path.lower():
                    return asset
        else:  # file
            for asset in assets:
                if asset['path'].lower() == resource.path.lower():
                    return asset


class Filer(ResourceGetter):

    name = 'filer'
    description = """
    Finds resources on a (locally mounted) remote filer and caches them
    locally.

    This assumes that the filer is mounted on the local machine (e.g. as a
    samba share).

    """
    parameters = [
        Parameter('remote_path', global_alias='remote_assets_path', default='',
                  description="""
                  Path, on the local system, where the assets are located.
                  """),
        Parameter('always_fetch', kind=boolean, default=False,
                  global_alias='always_fetch_remote_assets',
                  description="""
                  If ``True``, will always attempt to fetch assets from the
                  remote, even if a local cached copy is available.
                  """),
    ]

    def register(self, resolver):
        resolver.register(self.get, SourcePriority.lan)

    def get(self, resource):
        if resource.owner:
            remote_path = os.path.join(self.remote_path, resource.owner.name)
            local_path = os.path.join(settings.dependencies_directory, '__filer',
                                      resource.owner.dependencies_directory)
            return self.try_get_resource(resource, remote_path, local_path)
        else:  # No owner
            result = None
            for entry in os.listdir(remote_path):
                remote_path = os.path.join(self.remote_path, entry)
                local_path = os.path.join(settings.dependencies_directory, '__filer',
                                          settings.dependencies_directory, entry)
                result = self.try_get_resource(resource, remote_path, local_path)
                if result:
                    break
            return result

    def try_get_resource(self, resource, remote_path, local_path):
        if not self.always_fetch:
            result = get_from_location(local_path, resource)
            if result:
                return result
        if not os.path.exists(local_path):
            return None
        if os.path.exists(remote_path):
            # Didn't find it cached locally; now check the remoted
            result = get_from_location(remote_path, resource)
            if not result:
                return result
        else:  # remote path is not set
            return None
        # Found it remotely, cache locally, then return it
        local_full_path = os.path.join(_d(local_path), os.path.basename(result))
        self.logger.debug('cp {} {}'.format(result, local_full_path))
        shutil.copy(result, local_full_path)
