#! /usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, Arm Limited and contributors.
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

import argparse
import subprocess
import sys
import time
import contextlib
import textwrap
import logging
import tempfile
import shlex
import itertools
import pathlib

from lisa.target import Target
from lisa._kmod import LISADynamicKmod, KmodSrc
from lisa.utils import ignore_exceps


def lisa_kmod(logger, target, args, reset_config):
    if (features := args.feature) is None:
        if args.no_enable_all:
            logger.info('No feature will be enabled')
            config = {}
        else:
            logger.info('All features will be enabled on a best-effort basis')
            config = {
                'all': {
                    'best-effort': True,
                },
            }
    else:
        config = dict.fromkeys(features)

    config = {
        'features': config
    }
    kmod = target.get_kmod(LISADynamicKmod)

    @contextlib.contextmanager
    def cm():
        with kmod.run(config=config, reset_config=reset_config) as _kmod:
            pretty_events = ', '.join(_kmod._defined_events)
            logger.info(f'Kernel module provides the following ftrace events: {pretty_events}')
            yield _kmod

    return cm()


def main():
    params = {
        'feature': dict(
            action='append',
            help='Enable a specific module feature. Can be repeated. By default, the module will try to enable all features and will log in dmesg the ones that failed to enable'
        ),
        'no-enable-all': dict(
            action='store_true',
            help='Do not attempt to enable all features, only enable the features that are specified using --feature or in the configuration if no --feature option is passed'
        ),
        'cmd': dict(
            nargs=argparse.REMAINDER,
            help='Load the module, run the given command then unload the module. If not command is provided, just load the module and exit.'
        )
    }

    args, target = Target.from_custom_cli(
        description=textwrap.dedent('''
        Compile and load the LISA kernel module, then unloads it when the
        command finishes.

        If no command is passed, it will simply compile and load the module then return.

        EXAMPLES

        $ lisa-load-kmod --conf target_conf.yml -- echo hello world

        '''),
        params=params,
    )

    with target.closing() as target:
        return _main(args, target)


def _main(args, target):
    logger = logging.getLogger('lisa-load-kmod')

    keep_loaded = not bool(args.cmd)

    cmd = args.cmd or []
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    kmod_cm = lisa_kmod(
        logger=logger,
        target=target,
        args=args,
        # If we keep the module loaded after exiting, we treat that as
        # resetting the state of the module so we reset the config.
        #
        # Otherwise, we will simply push our bit of config, and then pop it
        # upon exiting.
        reset_config=keep_loaded,
    )

    def run_cmd():
        if cmd:
            pretty_cmd = ' '.join(map(shlex.quote, cmd))
            logger.info(f'Running command: {pretty_cmd}')
            return subprocess.run(cmd).returncode
        else:
            return 0

    if keep_loaded:
        @contextlib.contextmanager
        def cm():
            logger.info('Loading kernel module ...')
            kmod = kmod_cm.__enter__()
            yield
            logger.info(f'Loaded kernel module as "{kmod.mod_name}"')
    else:
        @contextlib.contextmanager
        def cm():
            with kmod_cm:
                logger.info('Loading kernel module ...')
                try:
                    yield
                finally:
                    logger.info('Unloading kernel module')

    with cm():
        ret = run_cmd()

    return ret



if __name__ == '__main__':
    sys.exit(main())
