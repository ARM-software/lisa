#! /usr/bin/env python3

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
        'feature-param': dict(
            action='append',
            metavar=('FEATURE_NAME', 'PARAM_NAME', 'PARAM_VALUE'),
            nargs=3,
            help='Set a feature parameter value.'
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

    features = args.feature or []
    features_params = args.feature_param or {}
    keep_loaded = not bool(args.cmd)

    cmd = args.cmd or []
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    features = {
        feature: {}
        for feature in features
    }
    for feature, param_name, param_value in features_params:
        features.setdefault(feature, {})[param_name] = param_value

    kmod = target.get_kmod(LISADynamicKmod)
    _kmod_cm = kmod.run(features=features)

    if keep_loaded:
        @contextlib.contextmanager
        def cm():
            logging.info('Compiling and loading kernel module ...')
            yield _kmod_cm.__enter__()
            logging.info(f'Loaded kernel module as "{kmod.mod_name}"')
    else:
        @contextlib.contextmanager
        def cm():
            with _kmod_cm:
                logging.info('Compiling and loading kernel module ...')
                try:
                    yield
                finally:
                    logging.info('Unloading kernel module')
    kmod_cm = cm()

    def run_cmd():
        if cmd:
            pretty_cmd = ' '.join(map(shlex.quote, cmd))
            logging.info(f'Running command: {pretty_cmd}')
            return subprocess.run(cmd).returncode
        else:
            return 0

    with kmod_cm:
        ret = run_cmd()

    return ret


if __name__ == '__main__':
    sys.exit(main())
