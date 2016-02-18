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
import subprocess
import warnings

from wlauto.core.bootstrap import settings
from wlauto.core.extension_loader import ExtensionLoader
from wlauto.exceptions import WAError, ConfigError
from wlauto.utils.misc import get_traceback
from wlauto.utils.log import init_logging
from wlauto.utils.cli import init_argument_parser
from wlauto.utils.doc import format_body


warnings.filterwarnings(action='ignore', category=UserWarning, module='zope')


logger = logging.getLogger('command_line')


def load_commands(subparsers):
    ext_loader = ExtensionLoader(paths=settings.extension_paths)
    for command in ext_loader.list_commands():
        settings.commands[command.name] = ext_loader.get_command(command.name, subparsers=subparsers)


def main():
    try:
        description = ("Execute automated workloads on a remote device and process "
                       "the resulting output.\n\nUse \"wa <subcommand> -h\" to see "
                       "help for individual subcommands.")
        parser = argparse.ArgumentParser(description=format_body(description, 80),
                                         prog='wa',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         )
        init_argument_parser(parser)
        load_commands(parser.add_subparsers(dest='command'))  # each command will add its own subparser
        args = parser.parse_args()
        settings.verbosity = args.verbose
        settings.debug = args.debug
        if args.config:
            if not os.path.exists(args.config):
                raise ConfigError("Config file {} not found".format(args.config))
            settings.update(args.config)
        init_logging(settings.verbosity)

        command = settings.commands[args.command]
        sys.exit(command.execute(args))

    except KeyboardInterrupt:
        logging.info('Got CTRL-C. Aborting.')
        sys.exit(3)
    except WAError as e:
        logging.critical(e)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        tb = get_traceback()
        logging.critical(tb)
        command = e.cmd
        if e.args:
            command = '{} {}'.format(command, ' '.join(e.args))
        message = 'Command \'{}\' returned non-zero exit status {}\nOUTPUT:\n{}\n'
        logging.critical(message.format(command, e.returncode, e.output))
        sys.exit(2)
    except SyntaxError as e:
        tb = get_traceback()
        logging.critical(tb)
        message = 'Syntax Error in {}, line {}, offset {}:'
        logging.critical(message.format(e.filename, e.lineno, e.offset))
        logging.critical('\t{}'.format(e.msg))
        sys.exit(2)
    except Exception as e:  # pylint: disable=broad-except
        tb = get_traceback()
        logging.critical(tb)
        logging.critical('{}({})'.format(e.__class__.__name__, e))
        sys.exit(2)
