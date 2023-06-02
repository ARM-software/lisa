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
from lisa.trace import DmesgCollector
from lisa._kmod import LISAFtraceDynamicKmod
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

    features = args.feature
    keep_loaded = not bool(args.cmd)

    cmd = args.cmd or []
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    kmod_params = {}
    if features is not None:
        kmod_params['features'] = list(features)

    kmod = target.get_kmod(LISAFtraceDynamicKmod)
    _kmod_cm = kmod.run(kmod_params=kmod_params)

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

    @contextlib.contextmanager
    def dmesg_cm():
        with tempfile.NamedTemporaryFile() as f:
            dmesg_path = f.name
            coll = DmesgCollector(target, output_path=dmesg_path)

            def log_err(when, cm, excep):
                logging.error(f'Encounted exceptions while {when}ing dmesg collector: {excep}')

            coll = ignore_exceps(Exception, coll, log_err)

            with coll as coll:
                yield

            if coll:
                dmesg_entries = [
                    entry
                    for entry in coll.entries
                    if entry.msg.startswith(kmod.mod_name)
                ]
                if dmesg_entries:
                    sep = '\n    '
                    dmesg = sep.join(map(str, dmesg_entries))
                    logging.info(f'Module dmesg output:{sep}{dmesg}')

    def run_cmd():
        if cmd:
            pretty_cmd = ' '.join(map(shlex.quote, cmd))
            logging.info(f'Running command: {pretty_cmd}')
            return subprocess.run(cmd).returncode
        else:
            return 0


    with dmesg_cm(), kmod_cm:
        ret = run_cmd()

    return ret


if __name__ == '__main__':
    sys.exit(main())
