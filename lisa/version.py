# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
:mod:`lisa` version identification.
"""

# Re-export
from lisa._version import version_tuple, __version__, format_version, parse_version

def _compute_version_token():
    import os
    import hashlib
    from subprocess import CalledProcessError
    from pathlib import Path

    from lisa._git import get_sha1, get_uncommited_patch

    plain_version_token = f'v{format_version(version_tuple)}'

    forced = os.getenv('_LISA_FORCE_VERSION_TOKEN')

    # When in devmode, use the commit SHA1 and the SHA1 of the patch of
    # uncommitted changes
    if forced:
        return forced
    elif int(os.getenv('LISA_DEVMODE', '0')):
        # pylint: disable=import-outside-toplevel
        import lisa
        repo = Path(list(lisa.__path__)[0])
        repo = repo / '..'

        try:
            sha1 = get_sha1(repo)
            # Get uncommitted content of the LISA sources only, not the entire
            # repo as it would include things like target_conf.yml
            patch = get_uncommited_patch(repo, path='lisa/')
        # Git is not installed, just use the regular version
        except (FileNotFoundError, CalledProcessError):
            return plain_version_token

        # Dirty tree
        if patch:
            patch_sha1 = hashlib.sha1(patch.encode()).hexdigest()
            patch_sha1 = f'-dirty-{patch_sha1}'
        else:
            patch_sha1 = ''

        return f'git-{sha1}{patch_sha1}'
    else:
        return plain_version_token


VERSION_TOKEN = _compute_version_token()
"""
Unique token related to code version.

When ``LISA_DEVMODE`` environment variable is set to 1, the git sha1 followed
by the uncommitted patch's sha1 will be used, so that the code of LISA can
uniquely be identified even in development state.
"""
