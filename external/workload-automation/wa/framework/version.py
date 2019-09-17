#    Copyright 2014-2018 ARM Limited
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
from collections import namedtuple
from subprocess import Popen, PIPE


VersionTuple = namedtuple('Version', ['major', 'minor', 'revision', 'dev'])

version = VersionTuple(3, 2, 0, 'dev1')

required_devlib_version = VersionTuple(1, 2, 0, 'dev1')


def format_version(v):
    version_string = '{}.{}.{}'.format(
        v.major, v.minor, v.revision)
    if v.dev:
        version_string += '.{}'.format(v.dev)
    return version_string


def get_wa_version():
    return format_version(version)


def get_wa_version_with_commit():
    version_string = get_wa_version()
    commit = get_commit()
    if commit:
        return '{}+{}'.format(version_string, commit)
    else:
        return version_string


def get_commit():
    p = Popen(['git', 'rev-parse', 'HEAD'],
              cwd=os.path.dirname(__file__), stdout=PIPE, stderr=PIPE)
    std, _ = p.communicate()
    p.wait()
    if p.returncode:
        return None
    if sys.version_info[0] == 3 and isinstance(std, bytes):
        return std[:8].decode(sys.stdout.encoding or 'utf-8')
    else:
        return std[:8]
