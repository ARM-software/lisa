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
import logging
import os
import re

from devlib.utils.android import ApkInfo

from wa.framework import pluginloader
from wa.framework.plugin import Plugin
from wa.framework.exception import ResourceError
from wa.framework.configuration import settings
from wa.utils import log
from wa.utils.misc import get_object_name
from wa.utils.types import enum, list_or_string, prioritylist



SourcePriority = enum(['package', 'remote', 'lan', 'external_package', 'local',
                       'perferred'], start=0, step=10)


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


class Resource(object):
    """
    Represents a resource that needs to be resolved. This can be pretty much
    anything: a file, environment variable, a Python object, etc. The only
    thing a resource *has* to have is an owner (which would normally be the
    Workload/Instrument/Device/etc object that needs the resource). In
    addition, a resource have any number of attributes to identify, but all of
    them are resource type specific.

    """

    kind = None

    def __init__(self, owner=NO_ONE):
        self.owner = owner

    def match(self, path):
        return self.match_path(path)

    def match_path(self, path):
        raise NotImplementedError()

    def __str__(self):
        return '<{}\'s {}>'.format(self.owner, self.kind)


class File(Resource):

    kind = 'file'

    def __init__(self, owner, path):
        super(File, self).__init__(owner)
        self.path = path

    def match_path(self, path):
        return self.path == path

    def __str__(self):
        return '<{}\'s {} {} file>'.format(self.owner, self.kind, self.path)


class Executable(Resource):

    kind = 'executable'

    def __init__(self, owner, abi, filename):
        super(Executable, self).__init__(owner)
        self.abi = abi
        self.filename = filename

    def match_path(self, path):
        return self.filename == os.path.basename(path)

    def __str__(self):
        return '<{}\'s {} {} executable>'.format(self.owner, self.abi, self.filename)


class ReventFile(Resource):

    kind = 'revent'

    def __init__(self, owner, stage, target):
        super(ReventFile, self).__init__(owner)
        self.stage = stage
        self.target = target

    def match_path(self, path):
        filename = os.path.basename(path)
        parts = filename.split('.')
        if len(parts) > 2:
            target, stage = parts[:2]
            return target == self.target and stage == self.stage
        else:
            stage = parts[0]
            return stage == self.stage


class JarFile(Resource):

    kind = 'jar'

    def match_path(self, path):
        # An owner always  has at most one jar file, so
        # always match
        return True


class ApkFile(Resource):

    kind = 'apk'

    def __init__(self, owner, variant=None, version=None,
                 package=None, uiauto=False, exact_abi=False,
                 supported_abi=None):
        super(ApkFile, self).__init__(owner)
        self.variant = variant
        self.version = version
        self.package = package
        self.uiauto = uiauto
        self.exact_abi = exact_abi
        self.supported_abi = supported_abi

    def match_path(self, path):
        ext = os.path.splitext(path)[1].lower()
        return ext == '.apk'

    def match(self, path):
        name_matches = True
        version_matches = True
        package_matches = True
        abi_matches = True
        uiauto_matches = uiauto_test_matches(path, self.uiauto)
        if self.version is not None:
            version_matches = apk_version_matches(path, self.version)
        if self.variant is not None:
            name_matches = file_name_matches(path, self.variant)
        if self.package is not None:
            package_matches = package_name_matches(path, self.package)
        if self.supported_abi is not None:
            abi_matches = apk_abi_matches(path, self.supported_abi,
                                          self.exact_abi)
        return name_matches and version_matches and \
               uiauto_matches and package_matches and \
               abi_matches


    def __str__(self):
        text = '<{}\'s apk'.format(self.owner)
        if self.variant:
            text += ' {}'.format(self.variant)
        if self.version:
            text += ' {}'.format(self.version)
        if self.uiauto:
            text += 'uiautomator test'
        text += '>'
        return text


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
    :priority: Priority with which this getter will be invoked. This should
               be one of the standard priorities specified in
               ``GetterPriority`` enumeration. If not set, this will default
               to ``GetterPriority.environment``.

    """

    name = None
    kind = 'resource_getter'

    def register(self, resolver):
        raise NotImplementedError()

    def initialize(self):
        pass

    def __str__(self):
        return '<ResourceGetter {}>'.format(self.name)


class ResourceResolver(object):
    """
    Discovers and registers getters, and then handles requests for
    resources using registered getters.

    """

    def __init__(self, loader=pluginloader):
        self.loader = loader
        self.logger = logging.getLogger('resolver')
        self.getters = []
        self.sources = prioritylist()

    def load(self):
        for gettercls in self.loader.list_plugins('resource_getter'):
            self.logger.debug('Loading getter {}'.format(gettercls.name))
            getter = self.loader.get_plugin(name=gettercls.name,
                                            kind="resource_getter")
            log.indent()
            try:
                getter.initialize()
                getter.register(self)
            finally:
                log.dedent()
            self.getters.append(getter)

    def register(self, source, priority=SourcePriority.local):
        msg = 'Registering "{}" with priority "{}"'
        self.logger.debug(msg.format(get_object_name(source), priority))
        self.sources.add(source, priority)

    def get(self, resource, strict=True):
        """
        Uses registered getters to attempt to discover a resource of the specified
        kind and matching the specified criteria. Returns path to the resource that
        has been discovered. If a resource has not been discovered, this will raise
        a ``ResourceError`` or, if ``strict`` has been set to ``False``, will return
        ``None``.

        """
        self.logger.debug('Resolving {}'.format(resource))
        for source in self.sources:
            source_name = get_object_name(source)
            self.logger.debug('Trying {}'.format(source_name))
            result = source(resource)
            if result is not None:
                msg = 'Resource {} found using {}:'
                self.logger.debug(msg.format(resource, source_name))
                self.logger.debug('\t{}'.format(result))
                return result
        if strict:
            raise ResourceError('{} could not be found'.format(resource))
        self.logger.debug('Resource {} not found.'.format(resource))
        return None


def apk_version_matches(path, version):
    info = ApkInfo(path)
    if info.version_name == version or info.version_code == version:
        return True
    return loose_version_matching(version, info.version_name)

def loose_version_matching(config_version, apk_version):
    config_version = config_version.split('.')
    apk_version = apk_version.split('.')

    if len(apk_version) < len(config_version):
        return False  # More specific version requested than available

    for i in xrange(len(config_version)):
        if config_version[i] != apk_version[i]:
            return False
    return True


def file_name_matches(path, pattern):
    filename = os.path.basename(path)
    if pattern in filename:
        return True
    if re.search(pattern, filename):
        return True
    return False

def uiauto_test_matches(path, uiauto):
    info = ApkInfo(path)
    return uiauto == ('com.arm.wa.uiauto' in info.package)

def package_name_matches(path, package):
    info = ApkInfo(path)
    return info.package == package

def apk_abi_matches(path, supported_abi, exact_abi=False):
    supported_abi = list_or_string(supported_abi)
    info = ApkInfo(path)
    # If no native code present, suitable for all devices.
    if not info.native_code:
        return True

    if exact_abi:  # Only check primary
        return supported_abi[0] in info.native_code
    else:
        for abi in supported_abi:
            if abi in info.native_code:
                return True
    return False
