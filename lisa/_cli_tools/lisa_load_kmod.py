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

from lisa.target import Target
from lisa._kmod import LISADynamicKmod
from lisa.utils import ignore_exceps

def main():
    params = {
        'feature': dict(
            action='append',
            help='Enable a specific module feature. Can be repeated. By default, the module will try to enable all features and will log in dmesg the ones that failed to enable'
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

    features = args.feature
    keep_loaded = not bool(args.cmd)

    cmd = args.cmd or []
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    kmod_params = {}
    if features is not None:
        kmod_params['features'] = list(features)

    kmod = target.get_kmod(LISADynamicKmod)
    pretty_events = ', '.join(kmod.defined_events)
    logger.info(f'Kernel module provides the following ftrace events: {pretty_events}')

    _kmod_cm = kmod.run(kmod_params=kmod_params)

    if keep_loaded:
        @contextlib.contextmanager
        def cm():
            logger.info('Loading kernel module ...')
            yield _kmod_cm.__enter__()
            logger.info(f'Loaded kernel module as "{kmod.mod_name}"')
    else:
        @contextlib.contextmanager
        def cm():
            with _kmod_cm:
                logger.info('Loading kernel module ...')
                try:
                    yield
                finally:
                    logger.info('Unloading kernel module')
    kmod_cm = cm()

    def run_cmd():
        if cmd:
            pretty_cmd = ' '.join(map(shlex.quote, cmd))
            logger.info(f'Running command: {pretty_cmd}')
            return subprocess.run(cmd).returncode
        else:
            return 0

    with kmod_cm:
        ret = run_cmd()

    return ret


if __name__ == '__main__':
    sys.exit(main())
