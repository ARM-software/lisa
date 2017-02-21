#    Copyright 2013-2015 ARM Limited
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
import argparse
import logging
import subprocess

from wa.framework import pluginloader, log
from wa.framework.configuration import settings
from wa.framework.exception import WAError
from wa.utils.doc import format_body
from wa.utils.misc import init_argument_parser


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='zope')


logger = logging.getLogger('wa')


def init_settings():
    settings.load_environment()
    if not os.path.isdir(settings.user_directory):
        settings.initialize_user_directory()
    settings.load_user_config()


def get_argument_parser():
    description = ("Execute automated workloads on a remote device and process "
                    "the resulting output.\n\nUse \"wa <subcommand> -h\" to see "
                    "help for individual subcommands.")
    parser = argparse.ArgumentParser(description=format_body(description, 80),
                                        prog='wa',
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        )
    init_argument_parser(parser)
    return parser


def load_commands(subparsers):
    commands = {}
    for command in pluginloader.list_commands():
        commands[command.name] = pluginloader.get_command(command.name, subparsers=subparsers)
    return commands


def main():
    try:
        log.init()
        init_settings()
        parser = get_argument_parser()
        commands = load_commands(parser.add_subparsers(dest='command'))  # each command will add its own subparser
        args = parser.parse_args()
        settings.set('verbosity', args.verbose)
        if args.config:
            settings.load_config_file(args.config)
        log.set_level(settings.verbosity)
        command = commands[args.command]
        sys.exit(command.execute(args))
    except KeyboardInterrupt:
        logging.info('Got CTRL-C. Aborting.')
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        log_error(e, logger, critical=True)
        if isinstance(e, WAError):
            sys.exit(2)
        else:
            sys.exit(3)

