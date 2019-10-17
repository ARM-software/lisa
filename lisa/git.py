# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

import logging
import os
import subprocess

def find_shortest_symref(repo_path, sha1):
    """
    Find the shortest symbolic reference (branch/tag) to a Git SHA1

    :param repo_path: the path of a valid git repository
    :type repo_path: str

    :param sha1: the SAH1 of a commit to lookup the reference for
    :type sha1: str

    Returns None if nothing points to the requested SHA1
    """
    repo_path = os.path.expanduser(repo_path)
    branches = subprocess.check_output([
            'git', 'for-each-ref',
            '--points-at={}'.format(sha1),
            '--format=%(refname:short)',
        ],
        universal_newlines=True,
        cwd=repo_path)

    possibles = branches.splitlines()

    if not possibles:
        raise ValueError('No symbolic reference found for SHA1 {} in {}'.format(sha1, repo_path))

    return min(possibles, key=len)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
