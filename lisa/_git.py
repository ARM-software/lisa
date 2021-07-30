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

import os
import subprocess


def git(repo, *args):
    """
    Call git in the given repo with the given arguments
    """

    return subprocess.check_output(['git', '-C', repo, *args]).decode()

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
        f'--points-at={sha1}',
        '--format=%(refname:short)',
    ],
        universal_newlines=True,
        cwd=repo_path)

    possibles = branches.splitlines()

    if not possibles:
        raise ValueError(f'No symbolic reference found for SHA1 {sha1} in {repo_path}')

    return min(possibles, key=len)

def get_sha1(repo, ref='HEAD'):
    """
    Get the currently checked-out sha1 in the given repository
    """
    return git(repo, 'rev-list', '-1', ref).strip()

def get_uncommited_patch(repo):
    """
    Return the patch of non commited changes, both staged and not staged yet.
    """
    return git(repo, 'diff', 'HEAD')

def get_commit_message(repo, ref='HEAD', format='%s'):
    """
    Get the reference commit message.

    :returns: String containing the commit message.

    :param format: Format string passed to ``git show --format``. Default to subject.
    :type format: str
    """
    # pylint: disable=redefined-builtin

    return git(repo, 'show', '--format=' + format, '-s', ref)

def find_commits(repo, ref='HEAD', grep=None, regex=False):
    """
    Find git commits.

    :returns: List of matching commits' SHA1.

    :param ref: Git reference passed to ``git log``
    :type ref: str

    :param grep: Passed to ``git log --grep``
    :type grep: str or None

    :param regex: If ``True``, the ``grep`` pattern will be used as an extended
        regex. If ``False``, a fixed string will be searched for.
    :type regex: bool
    """
    opts = []
    if grep:
        opts += [
            '--grep', grep,
            '-E' if regex else '-F'
        ]

    commits = git(repo, 'log', '--format=%H', *opts, ref, '--')
    return commits.splitlines()

def log(repo, ref='HEAD', format=None, commits_nr=1):
    """
    Run git log and collect its output unmodified.

    :param format: Format string passed to ``git log --format``.
    :type format: str or None

    :param commits_nr: Number of commits to display. ``None`` means no limit.
    :type commits_nr: int or None
    """
    # pylint: disable=redefined-builtin

    opts = []
    if format:
        opts += [f'--format={format}']

    if commits_nr is not None:
        opts += [f'-n{commits_nr}']

    return git(repo, 'log', *opts, ref, '--')

def find_tags(repo, pattern):
    """
    Find the git tags matching the given pattern.
    """
    tags = git(repo, 'tag', '--list', '--sort=-creatordate', '--', pattern)
    return tags.splitlines()


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
