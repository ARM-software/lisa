#! /usr/bin/env python3

import argparse
import configparser
import subprocess
import shlex
import os

def format_cmd_line(cmd):
    return ' '.join(shlex.quote(arg) for arg in cmd)

def call_subprocess(cmd):
    print(format_cmd_line(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print('Failed with exit code {}'.format(e.returncode))

def main():
    parser = argparse.ArgumentParser(description="""
    Apply git subtree command to the subtree config list.
    """)

    parser.add_argument('subtree_conf', help='subtree configuration file')
    parser.add_argument('--fetch', action='store_true',
            help='Fetch the remote before issuing the git subtree command. That ensures all the remote commits objects are available for git subtree split and push.')
    parser.add_argument('git_cmd',
            choices=['pull', 'push'],
            help='git subtree command')

    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
            help='Extra arguments to git subtree')
    args = parser.parse_args()

    parser = configparser.ConfigParser()
    subtree_conf = args.subtree_conf
    git_cmd = args.git_cmd
    extra_args = args.extra_args
    do_fetch = args.fetch

    parser.read(subtree_conf)

    for section, options in parser.items():
        if section == configparser.DEFAULTSECT:
            continue
        url = options['url']
        # Allow env var to be used in the path
        path = os.path.expandvars(options['path'])
        git_ref = options['ref']

        if do_fetch:
            fetch_cmd = ['git', 'fetch', url, git_ref]
            call_subprocess(cmd)

        if git_cmd in ('pull', 'push'):
            git_args = [git_cmd] + extra_args +['-P', path, url, git_ref]

        cmd = ['git', 'subtree'] + git_args
        call_subprocess(cmd)
        print()


if __name__ == '__main__':
    main()

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
