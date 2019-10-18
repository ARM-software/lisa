#    Copyright 2013-2018 ARM Limited
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
import locale
import logging
import os
import warnings

import devlib
try:
    from devlib.utils.version import version as installed_devlib_version
except ImportError:
    installed_devlib_version = None

from wa.framework import pluginloader
from wa.framework.command import init_argument_parser
from wa.framework.configuration import settings
from wa.framework.configuration.execution import ConfigManager
from wa.framework.host import init_user_directory, init_config
from wa.framework.exception import ConfigError, HostError
from wa.framework.version import (get_wa_version_with_commit, format_version,
                                  required_devlib_version)
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


# Instead of presenting an obscure error due to a version mismatch explicitly warn the user.
def check_devlib_version():
    if not installed_devlib_version or installed_devlib_version < required_devlib_version:
        msg = 'WA requires Devlib version >={}. Please update the currently installed version {}'
        raise HostError(msg.format(format_version(required_devlib_version), devlib.__version__))


# If the default encoding is not UTF-8 warn the user as this may cause compatibility issues
# when parsing files.
def check_system_encoding():
    system_encoding = locale.getpreferredencoding()
    msg = 'System Encoding: {}'.format(system_encoding)
    if 'UTF-8' not in system_encoding:
        logger.warning(msg)
        logger.warning('To prevent encoding issues please use a locale setting which supports UTF-8')
    else:
        logger.debug(msg)


def main():
    if not os.path.exists(settings.user_directory):
        init_user_directory()
    if not os.path.exists(os.path.join(settings.user_directory, 'config.yaml')):
        init_config()

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
        # full argument parsing cannot be completed until the commands are loaded; so
        # parse just the base args for now so we can get verbosity.
        argv = split_joined_options(sys.argv[1:])

        # 'Parse_known_args' automatically displays the default help and exits
        # if '-h' or '--help' is detected, we want our custom help messages so
        # ensure these are never passed as parameters.
        filtered_argv = list(argv)
        if '-h' in filtered_argv:
            filtered_argv.remove('-h')
        elif '--help' in filtered_argv:
            filtered_argv.remove('--help')

        args, _ = parser.parse_known_args(filtered_argv)
        settings.set("verbosity", args.verbose)
        log.init(settings.verbosity)
        logger.debug('Version: {}'.format(get_wa_version_with_commit()))
        logger.debug('devlib version: {}'.format(devlib.__full_version__))
        logger.debug('Command Line: {}'.format(' '.join(sys.argv)))
        check_devlib_version()
        check_system_encoding()

        # each command will add its own subparser
        subparsers = parser.add_subparsers(dest='command')
        subparsers.required = True
        commands = load_commands(subparsers)
        args = parser.parse_args(argv)

        config = ConfigManager()
        config.load_config_file(settings.user_config_file)
        for config_file in args.config:
            if not os.path.exists(config_file):
                raise ConfigError("Config file {} not found".format(config_file))
            config.load_config_file(config_file)

        command = commands[args.command]
        sys.exit(command.execute(config, args))

    except KeyboardInterrupt as e:
        log.log_error(e, logger)
        sys.exit(3)
    except Exception as e:  # pylint: disable=broad-except
        log.log_error(e, logger)
        sys.exit(2)
