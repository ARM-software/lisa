#! /usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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
import shlex
import argparse
import subprocess
import sys
from collections.abc import Mapping, Iterable
from collections import Counter
import tempfile
from pathlib import Path
import datetime
import shutil
import functools
import textwrap
import logging
import contextlib
import typing
import json

def info(msg):
    logging.info(msg)

def warn(msg):
    logging.warning(msg)

def error(msg):
    logging.error(msg)


def get_nested_key(conf, key):
    for k in key:
        conf = conf[k]
    return conf

def load_conf(path):

    def postprocess(conf):
        return conf['rebase-conf']

    def load_yaml(path):
        try:
            from ruamel.yaml import YAML
        except ImportError:
            raise RuntimeError('ruamel.yaml is not installed, YAML support disabled')
        else:
            yaml = YAML(typ='safe')
            with open(path) as f:
                return yaml.load(f)

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    loaders = [
        ('JSON', load_json),
        ('YAML', load_yaml),
    ]

    last_excep = ValueError('No loader selected')
    for fmt, loader in loaders:
        try:
            conf = loader(path)
        except Exception as e:
            last_excep = e
            error(f'Error while trying to load the configuration as {fmt}: {e}')
        else:
            info(f'Configuration successfully loaded as {fmt}')
            return postprocess(conf)

    raise last_excep


def dump_conf(conf, path):
    def convert(value):
        if isinstance(value, Mapping):
            return {
                convert(k): convert(v)
                for k, v in value.items()
            }
        elif isinstance(value, (str, Path)):
            return str(value)
        elif isinstance(value, Iterable):
            return [convert(x) for x in value]
        else:
            return value

    conf = {'rebase-conf': conf}
    conf = convert(conf)
    with open(path, 'w') as f:
        json.dump(conf, f)


RESUME_MANIFEST_NAME = '.batch-rebase-state'

def call_git(git_args, silent_stderr=False, capture=False, merge_stderr=False, quiet=False):
    git_args = [str(arg) for arg in git_args]

    cmd = ('git', *git_args)

    env = {
        **os.environ,
        # Allow inspecting the output more robustly against i18n
        'LC_ALL': 'C',
    }

    if not quiet:
        info(' '.join(shlex.quote(x) for x in cmd))

    if capture:
        def run(cmd):
            return subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT if merge_stderr else None,
                encoding='utf-8',
                universal_newlines=True,
                env=env,
            ).strip()
    else:
        if silent_stderr:
            stdout = subprocess.DEVNULL
        else:
            stdout = None
        stderr = stdout
        def run(cmd):
            return subprocess.check_call(
                cmd,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )

    return run(cmd)

def make_git_func(repo, **kwargs):
    """
    Make a function calling git on a specific repo
    """
    @functools.wraps(call_git)
    def wrapper(git_args, **wrapper_kwargs):
        return call_git(['-C', repo, *git_args], **wrapper_kwargs, **kwargs)

    return wrapper

def copy_content(src, dst):
    """
    Copy content of src folder to dst, overwriting files with the same name.
    """
    for path in src.rglob('*'):
        if path.is_dir():
            continue
        dst_path = dst/path.relative_to(src)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            str(path),
            str(dst_path)
        )

def has_unresolved(repo):
    git = make_git_func(repo)
    try:
        git(['diff', '--quiet', '--name-only', '--diff-filter=U'], silent_stderr=True)
    except subprocess.CalledProcessError:
        return True
    else:
        return False

def empty_staging_area(repo):
    git = make_git_func(repo)
    try:
        git(['diff', '--quiet', '--exit-code', '--cached'])
    except subprocess.CalledProcessError:
        return False
    else:
        return True

def do_create(conf_folder, repo, temp_repo, new_branch, conf, persistent_tags, tags_suffix):
    conf = conf.copy()

    # Use --shared to avoid copying the git objects, so we only pay for a
    # checkout
    call_git(['clone', '--shared', '--', repo, temp_repo])

    # From now on, only act on the temp repo
    git = make_git_func(temp_repo)

    # rr-cache
    rr_cache = conf.get('rr-cache')
    if rr_cache is not None:
        rr_cache = Path(conf.get('rr-cache'))
        if not rr_cache.is_absolute():
            rr_cache = conf_folder / rr_cache
        rr_cache = rr_cache.resolve()

        # Copy the rr-cache content
        copy_content(rr_cache, temp_repo/ '.git' / 'rr-cache')

    git_config = {
        'rerere.enabled': 'true',
        'rerere.autoupdate': '99999',
        'gc.rerereUnresolved': '99999',
    }

    for opt, val in git_config.items():
        git(['config', opt, val], silent_stderr=True)

    # Add all relevant remotes and fetch them
    conf['remotes']['localrepo'] = {
        'url': repo,
    }
    for name, remote in conf['remotes'].items():
        git(['remote', 'add', '--', name, remote['url']])

    # Create the new branch at the beginning of the cherry picking session
    base = conf['base']
    base_ref = base['ref']
    remote = base['remote']
    remote_base = remote_tracking(remote, base_ref)
    git(['fetch', '--', remote, f'{base_ref}:{remote_base}'])
    git(['checkout', '-b', new_branch, 'FETCH_HEAD'])

    # Start cherry picking topics
    return do_cherry_pick(repo, temp_repo, conf, persistent_tags, tags_suffix, new_branch, rr_cache)

def do_resume(temp_repo, conf):
    repo = conf['resume']['repo']
    persistent_tags = conf['resume']['tags']['persistent']
    tags_suffix = conf['resume']['tags']['suffix']
    branch = conf['resume']['branch']
    rr_cache = Path(conf['resume'].get('rr-cache'))

    try:
        return do_cherry_pick(repo, temp_repo, conf, persistent_tags, tags_suffix, branch, rr_cache)
    # If there was an error, make sure we do not delete the temp repo
    except subprocess.CalledProcessError:
        return (False, True)

def do_cherry_pick(repo, temp_repo, conf, persistent_tags, tags_suffix, branch, rr_cache):
    has_conflict, conf, persistent_refs = _do_cherry_pick(temp_repo, conf, persistent_tags, tags_suffix)

    if has_conflict:
        conf = conf.copy()
        conf['resume'].update({
            'repo': str(repo),
            'rr-cache': str(rr_cache) if rr_cache else rr_cache,
            'branch': branch,
        })
        # Save the augmented manifest for later resumption
        dump_conf(conf, temp_repo/RESUME_MANIFEST_NAME)
    else:
        # Copy back the rr-cache
        if rr_cache:
            copy_content(temp_repo / '.git' / 'rr-cache', rr_cache)

        # Push back the persistent references
        persistent_refs.add(branch)
        joiner = '\n\t* '
        info('Will save the following refs:{}{}'.format(
            joiner,
            joiner.join(sorted(persistent_refs)),
        ))
        for ref in persistent_refs:
            call_git(['-C', temp_repo, 'push', '-f', repo, ref])

    return (not has_conflict, 1 if has_conflict else 0)


def remote_tracking(remote, tip):
    return f'refs/remotes/{remote}/batch-rebase-tips/{tip}'


def _do_cherry_pick(repo, conf, persistent_tags, tags_suffix):
    git = make_git_func(repo)

    def add_tag(name, prefix='topic-', suffix=tags_suffix):
        tag_name = '{prefix}{name}{suffix}'.format(
            name=name,
            prefix=prefix,
            suffix='-{}'.format(suffix) if suffix else '',
        )
        # Force create the tag
        git(['tag', '-f', '--', tag_name])
        return tag_name

    def resolve_action(topic):
        return topic.get('action', 'cherry-pick')

    def resolve_name(topic):
        action = resolve_action(topic)
        if action == 'tag':
            return topic['name']
        elif action == 'cherry-pick':
            return topic.get('name', topic['tip'])
        else:
            raise ValueError(f'Could not handle action={action} in topic: {topic}')

    persistent_refs = set()
    topics = conf['topics'].copy()

    for topic in topics:
        topic.update(
            action=resolve_action(topic),
            name=resolve_name(topic),
        )

    for key, cnt in Counter(topic['name'] for topic in topics).items():
        if cnt > 1:
            raise ValueError(
                f'Found {cnt} topics named "{key}", but names must be unique.'
            )

    # If we are resuming after a conflict
    if 'conflict-topic' in conf.get('resume', {}):
        resume_topic = conf['resume']['conflict-topic']
        persistent_refs.update(conf['resume']['persistent-refs'])

        # Check that everything was committed
        try:
            git(['diff-index', '--quiet', 'HEAD', '--'])
        except subprocess.CalledProcessError:
            info('Please commit all files before running batch-rebase resume')
            return (True, conf, persistent_refs)

        # Finish cherry picking the topic with the conflict.
        # If the commit was the last in the topic, this will fail as the `git
        # commit` issued by the user finished the cherry-picking session.
        try:
            git(['cherry-pick', '--continue'], capture=True, merge_stderr=True)
        except subprocess.CalledProcessError as e:
            if 'no cherry-pick or revert in progress' in e.output:
                pass
            else:
                raise

        tag_name = add_tag(resume_topic)
        persistent_refs.add(tag_name)

        # Skip the topics that were already create successfully
        for i, topic in enumerate(topics):
            if topic['name'] == resume_topic:
                break
        try:
            topics = topics[i + 1:]
        except IndexError:
            topics = []

    # Cherry pick topic branches in the specified order
    for topic in topics:
        action = topic['action']
        name = topic['name']

        if action == 'tag':
            suffix = topic.get('suffix', tags_suffix)
            tag_name = add_tag(name, prefix='', suffix=suffix)
            info('Added a tag: {}'.format(tag_name))
            persistent_refs.add(tag_name)

        elif action == 'cherry-pick':
            tip = topic['tip']
            remote = topic['remote']

            base = topic.get('base')
            nr_commits = topic.get('nr-commits')

            # Fetch the topic base and tip
            remote_tip = remote_tracking(remote, tip)
            git(['fetch', '--', remote, f'{tip}:{remote_tip}'])
            tip = remote_tip

            if base is not None:
                remote_base = remote_tracking(remote, base)
                git(['fetch', '--', remote, f'{base}:{remote_base}'])
                base = remote_base
            elif nr_commits is not None:
                base = f'{tip}~{nr_commits}'
            else:
                raise ValueError(f'base or nr-commits need to be set on topic "{name}"')

            range_ref = f'{base}..{tip}'
            range_sha1s = list(reversed(git(['rev-list', range_ref], capture=True).splitlines()))

            nr_commits = len(range_sha1s)
            info('Cherry-picking topic "{name}" from {remote} ({nr_commits} commits)\nremote: {remote}\nbase: {base}\ntip: {tip}\n'.format(
                name=name,
                remote=remote,
                base=base,
                tip=tip,
                nr_commits=nr_commits,
            ))


            if not cherry_pick_ref(repo, range_sha1s):
                # Save the current state for later resumption
                conf = {
                    **conf,
                    'resume': {
                        'conflict-topic': name,
                        'persistent-refs': sorted(persistent_refs),
                        'tags': {
                            'persistent': persistent_tags,
                            'suffix': tags_suffix,
                        }
                    }
                }
                info(textwrap.dedent("""
                    A conflict occured while cherry-picking topic "{topic}" ({range_ref})
                    In order to fix it, please:
                        1. Resolve the conflict in:
                        {repo}

                        2. Finish cherry picking the topic and fix any
                           remaining conflicts.

                        3. Run:
                        batch-rebase resume {repo}
                    """.format(
                        topic=name,
                        repo=repo,
                        range_ref=range_ref,
                    )).strip())
                return (True, conf, persistent_refs)

            tag_name = add_tag(name)
            if persistent_tags:
                persistent_refs.add(tag_name)

        else:
            raise ValueError('Unknown action: {}'.format(action))

    return (False, conf, persistent_refs)


def cherry_pick_ref(repo, refs):
    git = make_git_func(repo)
    for ref in refs:
        try:
            git(['cherry-pick', '--', ref])
        # There is a conflict
        except subprocess.CalledProcessError:
            # Let Git rerere do its job
            if not rerere_autocommit(repo):
                return False

    return True

def rerere_autocommit(repo):
    git = make_git_func(repo)
    while True:
        # If there is an unsolvable conflict, bail out
        if has_unresolved(repo):
            return False
        # If rerere did its job, commit and carry on
        else:
            if empty_staging_area(repo):
                # If the commit would be empty, just skip it
                curr_sha1 = git(['rev-list', '-n1', 'CHERRY_PICK_HEAD'], capture=True)
                warn('Empty commit, skipping it: {}'.format(curr_sha1))
                git(['reset'])
            else:
                info('Git rerere supplied solution, carrying on')
                git(['commit', '--no-edit'])

        try:
            git(['cherry-pick', '--continue'])
        except subprocess.CalledProcessError:
            # If there is no cherry-pick in progress, just exit the loop
            try:
                git(['rev-parse', '--verify', '--quiet', 'CHERRY_PICK_HEAD'])
            except subprocess.CalledProcessError:
                break
            else:
                continue
        else:
            break

    return True


def do_stat(repo, tip, base):
    git = make_git_func(repo=repo, capture=True, quiet=True)

    git_range = '{}..{}'.format(base, tip)
    sha1s = git(['rev-list', '--simplify-by-decoration', '--topo-order', git_range]).splitlines()
    tag_map = {
        sha1: ' + '.join(git(['tag', '--sort=taggerdate', '--points-at', '--', sha1]).splitlines())
        for sha1 in sha1s
    }

    total = None
    def count(base, tip, display=False):
        nr = int(git(['rev-list', '--count', '{}..{}'.format(base, tip)]))
        if display:
            pct = 100 * nr/total
            return '{nr}/{pct:.0f}%'.format(nr=nr, pct=pct).rjust(8)
        else:
            return nr

    total = count(base, tip)

    max_tag_len = max(map(len, tag_map.values()))
    sha1s = sha1s + [base]
    for ref, prev_ref in zip(sha1s, sha1s[1:]):
        tag = tag_map[ref]
        print('{tag:<{len}}: {prev} commits ({base} from base, {tip} to tip)'.format(
            tag=tag,
            prev=count(prev_ref, ref, display=True),
            base=count(base, ref, display=True),
            tip=count(ref, tip, display=True),
            len=max_tag_len + 1,
        ))


    print('\nTotal commits: {}'.format(total))
    return 0

def _main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s]  %(message)s'
    )

    parser = argparse.ArgumentParser(description=textwrap.dedent(
    '''
    DESCRIPTION

        Create a git branch by rebasing a series of other branches on top of
        each other.

        Git rerere is enabled and used automatically when possible.
        The remote "localrepo" can be used to get a branch that is not actually
        on a remote.

        Note: All the operations are carried out in a temporary git repository
        that is cloned using git clone --shared. That means it will not touch
        the staging area of the original repository. If anything goes wrong,
        deleting the temporary repository is enough to wipe everything for a
        fresh start.

    CONFLICT RESOLUTION

        When a conflict cannot be solved by git rerere alone, batch-rebase will
        give instructions to follow.

    MANIFEST

        The manifest is a YAML file following that structure:

        ├ rebase-conf: Batch rebase configuration
            ├ topics (list):
              List of topics. Each topic is described by a mapping with
              "name", "remote", "base" (or "nr-commits) and "tip" git
              references keys. Also, a tag can be added between topics
              with "action: tag" key and "name: tag-name"..
            ├ remotes (Mapping):
              Git remotes. Keys are remote name, values are a mapping with
              an "url" key.
            ├ rr-cache (str or None):
              Path to git rr-cache. Relative paths are relative to that
              manifest file.
            ├ base: Base branche spec
                ├ remote (str): remote where the base branch is located.
                └ ref (str): Name of the base branch.
            ├ tip: New branche spec
                ├ ref (str):
                  Name of the base branch. --refs-suffix can be used to tweak
                  the name.
                └ tags (bool): Default of --tags.
            └ resume: Internal state used for resuming after a conflict
                ├ conflict-topic (str): Topic where the conflict happened.
                ├ persistent-refs (typing.Sequence[str]):
                  List of references that needs to be pushed back to the main
                  repo.
                ├ tags: Topic branch tags
                    ├ persistent (bool): whether tags should be pushed back or not.
                    └ suffix (str): suffix to use for tags, or None.
                ├ repo (str): Path to the main repo.
                ├ branch (str): Branch that is being created.
                └ rr-cache (str or None): Source of the rr-cache.

    EXAMPLE

        rebase-conf:
            # relative paths are relative to the manifest file
            rr-cache: ./rr-cache
            base:
                remote: localrepo
                ref: sched/core

            topics:
                -
                    name: foo # The name will default to the value of "tip" if not provided
                    remote: myremote
                    base: mybase
                    tip: mytip

               # Add a tag between topics
                -
                    action: tag
                    name: mytag

                -
                    name: bar
                    remote: myremote
                    base: mybase2
                    tip: mytip2


            remotes:
                tip:
                    url: git://git.kernel.org/pub/scm/linux/kernel/git/tip/tip.git
                arm-power:
                    url: git://linux-arm.org/linux-power.git


        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter,

    )

    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand', required=True)
    create_parser = subparsers.add_parser('create',
        help='Create a branch'
    )
    resume_parser = subparsers.add_parser('resume',
        help='Resume creation after resolving a conflict'
    )

    stat_parser = subparsers.add_parser('stat',
        help='Inspect a branch created by "create" subcommand with --tags and dump some statistics',
    )

    # Common optiosn
    for subparser in (create_parser, resume_parser, stat_parser):
        subparser.add_argument('GIT_TREE',
            help='Git tree to work on',
        )

    for subparser in (create_parser, resume_parser):
        subparser.add_argument('--keep-temp', action='store_true',
            help='Keep the temporary git repo instead of removing it when the cherry picking is finished'
        )

        subparser.add_argument('--delete-temp', action='store_true',
            help='Always delete the temporary git repo, even when there is a cherry pick conflict.'
        )

    # Create options
    create_parser.add_argument('--manifest', required=True,
        help='Config file'
    )

    create_parser.add_argument('--create-branch',
        help='Name of the new branch to create'
    )

    create_parser.add_argument('--tags', action='store_true',
        help='Create a tag per topic'
    )

    create_parser.add_argument('--tags-suffix',
        default='',
        help='Replace the default tag suffix (current date). Implies --tags'
    )

    create_parser.add_argument('--refs-suffix',
        default='',
        help='Used as a suffix for both the created branch and tags (if enabled with --tags)'
    )

    create_parser.add_argument('--checkout', action='store_true',
        help='Attempt to checkout the newly created branch.'
    )

    # Stat action
    stat_parser.add_argument('--base', required=True,
        help='Git ref pointing at the base',
    )
    stat_parser.add_argument('--tip', required=True,
        help='Git ref pointing at the tip',
    )

    args = parser.parse_args()

    ret = 0
    repo = Path(args.GIT_TREE).resolve()

    if args.subcommand == 'stat':
        return do_stat(repo, tip=args.tip, base=args.base)
    else:
        keep_temp = args.keep_temp
        delete_temp_repo = True
        temp_repo = None
        try:
            if args.subcommand == 'create':
                conf = load_conf(args.manifest)
                def default_from_conf(x, key):
                    try:
                        default = get_nested_key(conf, key)
                    except KeyError:
                        return x
                    else:
                        return x or default

                refs_suffix = args.refs_suffix
                new_branch_stem = default_from_conf(args.create_branch, ['tip', 'ref'])
                if not new_branch_stem:
                    create_parser.error('--create-branch or tip/ref key in config file is mandatory')

                new_branch = '{}-{}'.format(new_branch_stem, refs_suffix) if refs_suffix else new_branch_stem
                persistent_tags = bool(
                    default_from_conf(args.tags, ['tip', 'tags']) or args.tags_suffix
                )

                # If the user passed an explicit suffix, use it rather than the date
                if args.tags_suffix or args.refs_suffix:
                    tags_suffix = '{}{}'.format(args.tags_suffix, args.refs_suffix)
                else:
                    tags_suffix = datetime.datetime.now().strftime("%Y%m%d")

                # Make sure we use resolved absolute path
                conf_folder = Path(args.manifest).parent
                temp_repo = Path(tempfile.mkdtemp())
                delete_temp_repo, ret = do_create(
                    conf_folder,
                    repo, temp_repo,
                    new_branch,
                    conf,
                    persistent_tags, tags_suffix,
                )

                if args.delete_temp:
                    info('Deleting temporary clone as requested with --delete-temp')
                    delete_temp_repo = True

                ###
                # Functions that could raise must be used after setting
                # delete_temp_repo
                ###

                git = make_git_func(repo)
                if args.checkout:
                    git(['checkout', new_branch, '--'])

            elif args.subcommand == 'resume':
                temp_repo = repo
                conf = load_conf(temp_repo/RESUME_MANIFEST_NAME)
                delete_temp_repo, ret = do_resume(temp_repo, conf)

        finally:
            if delete_temp_repo and temp_repo and not keep_temp:
                shutil.rmtree(str(temp_repo))

    return ret


def main():
    sys.exit(_main())

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
