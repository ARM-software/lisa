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


import sys
import argparse
import logging
import os
import warnings

from wa.framework import pluginloader
from wa.framework.command import init_argument_parser
from wa.framework.configuration import settings
from wa.framework.configuration.execution import ConfigManager
from wa.framework.host import init_user_directory
from wa.framework.exception import ConfigError
from wa.utils import log
from wa.utils.doc import format_body

warnings.filterwarnings(action='ignore', category=UserWarning, module='zope')

# Disable this to avoid false positive from dynamically-created attributes.
# pylint: disable=no-member

logger = logging.getLogger('command_line')


def load_commands(subparsers):
    commands = {}
    for command in pluginloader.list_commands():
        commands[command.name] = pluginloader.get_command(command.name,
                                                          subparsers=subparsers)
    return commands


# ArgumentParser.parse_known_args() does not correctly deal with concatenated
# single character options. See https://bugs.python.org/issue16142 for the
# description of the issue (with a fix attached since 2013!). To get around
# this problem, this will pre-process sys.argv to detect such joined options
# and split them.
def split_joined_options(argv):
    output = []
    for part in argv:
        if len(part) > 1 and part[0] == '-' and part[1] != '-':
            for c in part[1:]:
                output.append('-' + c)
        else:
            output.append(part)
    return output


def main():
    if not os.path.exists(settings.user_directory):
        init_user_directory()

    try:

        description = ("Execute automated workloads on a remote device and process "
                       "the resulting output.\n\nUse \"wa <subcommand> -h\" to see "
                       "help for individual subcommands.")
        parser = argparse.ArgumentParser(description=format_body(description, 80),
                                         prog='wa',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         )
        init_argument_parser(parser)

        # load_commands will trigger plugin enumeration, and we want logging
        # to be enabled for that, which requires the verbosity setting; however
        # full argument parse cannot be complted until the commands are loaded; so
        # parse just the base args for know so we can get verbosity.
        argv = split_joined_options(sys.argv[1:])

        # 'Parse_known_args' automatically displays the default help and exits
        # if '-h' is detected, we want our custom help messages so ensure this
        # is never passed as a parameter.
        filtered_argv = list(argv)
        if '-h' in filtered_argv:
            filtered_argv.remove('-h')

        args, _ = parser.parse_known_args(filtered_argv)
        settings.set("verbosity", args.verbose)
        log.init(settings.verbosity)

        # each command will add its own subparser
        commands = load_commands(parser.add_subparsers(dest='command'))
        args = parser.parse_args(argv)

        config = ConfigManager()
        config.load_config_file(settings.user_config_file)
        for config_file in args.config:
            if not os.path.exists(config_file):
                raise ConfigError("Config file {} not found".format(config_file))
            config.load_config_file(config_file)

        command = commands[args.command]
        sys.exit(command.execute(config, args))

    except KeyboardInterrupt:
        logging.info('Got CTRL-C. Aborting.')
        sys.exit(3)
    except Exception as e:  # pylint: disable=broad-except
        if not getattr(e, 'logged', None):
            log.log_error(e, logger)
        sys.exit(2)
