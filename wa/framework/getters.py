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
import inspect
import json
import logging
import os
import re
import shutil
import sys

import requests

from devlib.utils.android import ApkInfo

from wa import Parameter, settings, __file__ as __base_filepath
from wa.framework.resource import ResourceGetter, SourcePriority, NO_ONE 
from wa.framework.exception import ResourceError
from wa.utils.misc import (ensure_directory_exists as _d, 
                           ensure_file_directory_exists as _f, sha256, urljoin)
from wa.utils.types import boolean, caseless_string


logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger('resource')


def get_by_extension(path, ext):
    if not ext.startswith('.'):
        ext = '.' + ext
    ext = caseless_string(ext)

    found = []
    for entry in os.listdir(path):
        entry_ext = os.path.splitext(entry)
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
        return ResourceError(msg.format(resource, matches))
    return matches[0]


class Package(ResourceGetter):

    name = 'package'

    def register(self, resolver):
        resolver.register(self.get, SourcePriority.package)

    def get(self, resource):
        if resource.owner == NO_ONE:
            basepath = os.path.join(os.path.dirname(__base_filepath), 'assets')
        else:
            modname = resource.owner.__module__
            basepath  = os.path.dirname(sys.modules[modname].__file__)

        if resource.kind == 'file':
            path = os.path.join(basepath, resource.path)
            if os.path.exists(path):
                return path
        elif resource.kind == 'executable':
            path = os.path.join(basepath, 'bin', resource.abi, resource.filename)
            if os.path.exists(path):
                return path
        elif resource.kind in ['apk', 'jar', 'revent']:
            files = get_by_extension(basepath, resource.kind)
            return get_generic_resource(resource, files)

        return None


