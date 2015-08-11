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
import stat
import string
import textwrap
import argparse
import shutil
import getpass
import subprocess
from collections import OrderedDict

import yaml

from wlauto import ExtensionLoader, Command, settings
from wlauto.exceptions import CommandError, ConfigError
from wlauto.utils.cli import init_argument_parser
from wlauto.utils.misc import (capitalize, check_output,
                               ensure_file_directory_exists as _f, ensure_directory_exists as _d)
from wlauto.utils.types import identifier
from wlauto.utils.doc import format_body


__all__ = ['create_workload']


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

UIAUTO_BUILD_SCRIPT = """#!/bin/bash

class_dir=bin/classes/com/arm/wlauto/uiauto
base_class=`python -c "import os, wlauto; print os.path.join(os.path.dirname(wlauto.__file__), 'common', 'android', 'BaseUiAutomation.class')"`
mkdir -p $$class_dir
cp $$base_class $$class_dir

ant build

if [[ -f bin/${package_name}.jar ]]; then
    cp bin/${package_name}.jar ..
fi
"""


class CreateSubcommand(object):

    name = None
    help = None
    usage = None
    description = None
    epilog = None
    formatter_class = None

    def __init__(self, logger, subparsers):
        self.logger = logger
        self.group = subparsers
        parser_params = dict(help=(self.help or self.description), usage=self.usage,
                             description=format_body(textwrap.dedent(self.description), 80),
                             epilog=self.epilog)
        if self.formatter_class:
            parser_params['formatter_class'] = self.formatter_class
        self.parser = subparsers.add_parser(self.name, **parser_params)
        init_argument_parser(self.parser)  # propagate top-level options
        self.initialize()

    def initialize(self):
        pass


class CreateWorkloadSubcommand(CreateSubcommand):

    name = 'workload'
    description = '''Create a new workload. By default, a basic workload template will be
                     used but you can use options to specify a different template.'''

    def initialize(self):
        self.parser.add_argument('name', metavar='NAME',
                                 help='Name of the workload to be created')
        self.parser.add_argument('-p', '--path', metavar='PATH', default=None,
                                 help='The location at which the workload will be created. If not specified, ' +
                                      'this defaults to "~/.workload_automation/workloads".')
        self.parser.add_argument('-f', '--force', action='store_true',
                                 help='Create the new workload even if a workload with the specified ' +
                                      'name already exists.')

        template_group = self.parser.add_mutually_exclusive_group()
        template_group.add_argument('-A', '--android-benchmark', action='store_true',
                                    help='Use android benchmark template. This template allows you to specify ' +
                                         ' an APK file that will be installed and run on the device. You should ' +
                                         ' place the APK file into the workload\'s directory at the same level ' +
                                         'as the __init__.py.')
        template_group.add_argument('-U', '--ui-automation', action='store_true',
                                    help='Use UI automation template. This template generates a UI automation ' +
                                         'Android project as well as the Python class. This a more general ' +
                                         'version of the android benchmark template that makes no assumptions ' +
                                         'about the nature of your workload, apart from the fact that you need ' +
                                         'UI automation. If you need to install an APK, start an app on device, ' +
                                         'etc., you will need to do that explicitly in your code.')
        template_group.add_argument('-B', '--android-uiauto-benchmark', action='store_true',
                                    help='Use android uiauto benchmark template. This generates a UI automation ' +
                                         'project as well as a Python class. This template should be used ' +
                                         'if you have a APK file that needs to be run on the device. You ' +
                                         'should place the APK file into the workload\'s directory at the ' +
                                         'same level as the __init__.py.')

    def execute(self, args):  # pylint: disable=R0201
        where = args.path or 'local'
        check_name = not args.force

        if args.android_benchmark:
            kind = 'android'
        elif args.ui_automation:
            kind = 'uiauto'
        elif args.android_uiauto_benchmark:
            kind = 'android_uiauto'
        else:
            kind = 'basic'

        try:
            create_workload(args.name, kind, where, check_name)
        except CommandError, e:
            print "ERROR:", e


class CreatePackageSubcommand(CreateSubcommand):

    name = 'package'
    description = '''Create a new empty Python package for WA extensions. On installation,
                     this package will "advertise" itself to WA so that Extensions with in it will
                     be loaded by WA when it runs.'''

    def initialize(self):
        self.parser.add_argument('name', metavar='NAME',
                                 help='Name of the package to be created')
        self.parser.add_argument('-p', '--path', metavar='PATH', default=None,
                                 help='The location at which the new pacakge will be created. If not specified, ' +
                                      'current working directory will be used.')
        self.parser.add_argument('-f', '--force', action='store_true',
                                 help='Create the new package even if a file or directory with the same name '
                                      'already exists at the specified location.')

    def execute(self, args):  # pylint: disable=R0201
        package_dir = args.path or os.path.abspath('.')
        template_path = os.path.join(TEMPLATES_DIR, 'setup.template')
        self.create_extensions_package(package_dir, args.name, template_path, args.force)

    def create_extensions_package(self, location, name, setup_template_path, overwrite=False):
        package_path = os.path.join(location, name)
        if os.path.exists(package_path):
            if overwrite:
                self.logger.info('overwriting existing "{}"'.format(package_path))
                shutil.rmtree(package_path)
            else:
                raise CommandError('Location "{}" already exists.'.format(package_path))
        actual_package_path = os.path.join(package_path, name)
        os.makedirs(actual_package_path)
        setup_text = render_template(setup_template_path, {'package_name': name, 'user': getpass.getuser()})
        with open(os.path.join(package_path, 'setup.py'), 'w') as wfh:
            wfh.write(setup_text)
        touch(os.path.join(actual_package_path, '__init__.py'))


class CreateAgendaSubcommand(CreateSubcommand):

    name = 'agenda'
    description = """
    Create an agenda whith the specified extensions enabled. And parameters set to their
    default values.
    """

    def initialize(self):
        self.parser.add_argument('extensions', nargs='+',
                                 help='Extensions to be added')
        self.parser.add_argument('-i', '--iterations', type=int, default=1,
                                 help='Sets the number of iterations for all workloads')
        self.parser.add_argument('-r', '--include-runtime-params', action='store_true',
                                 help="""
                                 Adds runtime parameters to the global section of the generated
                                 agenda. Note: these do not have default values, so only name
                                 will be added. Also, runtime parameters are devices-specific, so
                                 a device must be specified (either in the list of extensions,
                                 or in the existing config).
                                 """)
        self.parser.add_argument('-o', '--output', metavar='FILE',
                                 help='Output file. If not specfied, STDOUT will be used instead.')

    def execute(self, args):  # pylint: disable=no-self-use,too-many-branches,too-many-statements
        loader = ExtensionLoader(packages=settings.extension_packages,
                                 paths=settings.extension_paths)
        agenda = OrderedDict()
        agenda['config'] = OrderedDict(instrumentation=[], result_processors=[])
        agenda['global'] = OrderedDict(iterations=args.iterations)
        agenda['workloads'] = []
        device = None
        device_config = None
        for name in args.extensions:
            extcls = loader.get_extension_class(name)
            config = loader.get_default_config(name)
            del config['modules']

            if extcls.kind == 'workload':
                entry = OrderedDict()
                entry['name'] = extcls.name
                if name != extcls.name:
                    entry['label'] = name
                entry['params'] = config
                agenda['workloads'].append(entry)
            elif extcls.kind == 'device':
                if device is not None:
                    raise ConfigError('Specifying multiple devices: {} and {}'.format(device.name, name))
                device = extcls
                device_config = config
                agenda['config']['device'] = name
                agenda['config']['device_config'] = config
            else:
                if extcls.kind == 'instrument':
                    agenda['config']['instrumentation'].append(name)
                if extcls.kind == 'result_processor':
                    agenda['config']['result_processors'].append(name)
                agenda['config'][name] = config

        if args.include_runtime_params:
            if not device:
                if settings.device:
                    device = loader.get_extension_class(settings.device)
                    device_config = loader.get_default_config(settings.device)
                else:
                    raise ConfigError('-r option requires for a device to be in the list of extensions')
            rps = OrderedDict()
            for rp in device.runtime_parameters:
                if hasattr(rp, 'get_runtime_parameters'):
                    # a core parameter needs to be expanded for each of the
                    # device's cores, if they're avialable
                    for crp in rp.get_runtime_parameters(device_config.get('core_names', [])):
                        rps[crp.name] = None
                else:
                    rps[rp.name] = None
            agenda['global']['runtime_params'] = rps

        if args.output:
            wfh = open(args.output, 'w')
        else:
            wfh = sys.stdout
        yaml.dump(agenda, wfh, indent=4, default_flow_style=False)
        if args.output:
            wfh.close()


class CreateCommand(Command):

    name = 'create'
    description = '''Used to create various WA-related objects (see positional arguments list for what
                     objects may be created).\n\nUse "wa create <object> -h" for object-specific arguments.'''
    formatter_class = argparse.RawDescriptionHelpFormatter
    subcmd_classes = [
        CreateWorkloadSubcommand,
        CreatePackageSubcommand,
        CreateAgendaSubcommand,
    ]

    def initialize(self, context):
        subparsers = self.parser.add_subparsers(dest='what')
        self.subcommands = []  # pylint: disable=W0201
        for subcmd_cls in self.subcmd_classes:
            subcmd = subcmd_cls(self.logger, subparsers)
            self.subcommands.append(subcmd)

    def execute(self, args):
        for subcmd in self.subcommands:
            if subcmd.name == args.what:
                subcmd.execute(args)
                break
        else:
            raise CommandError('Not a valid create parameter: {}'.format(args.name))


def create_workload(name, kind='basic', where='local', check_name=True, **kwargs):
    if check_name:
        extloader = ExtensionLoader(packages=settings.extension_packages, paths=settings.extension_paths)
        if name in [wl.name for wl in extloader.list_workloads()]:
            raise CommandError('Workload with name "{}" already exists.'.format(name))

    class_name = get_class_name(name)
    if where == 'local':
        workload_dir = _d(os.path.join(settings.environment_root, 'workloads', name))
    else:
        workload_dir = _d(os.path.join(where, name))

    if kind == 'basic':
        create_basic_workload(workload_dir, name, class_name, **kwargs)
    elif kind == 'uiauto':
        create_uiautomator_workload(workload_dir, name, class_name, **kwargs)
    elif kind == 'android':
        create_android_benchmark(workload_dir, name, class_name, **kwargs)
    elif kind == 'android_uiauto':
        create_android_uiauto_benchmark(workload_dir, name, class_name, **kwargs)
    else:
        raise CommandError('Unknown workload type: {}'.format(kind))

    print 'Workload created in {}'.format(workload_dir)


def create_basic_workload(path, name, class_name):
    source_file = os.path.join(path, '__init__.py')
    with open(source_file, 'w') as wfh:
        wfh.write(render_template('basic_workload', {'name': name, 'class_name': class_name}))


def create_uiautomator_workload(path, name, class_name):
    uiauto_path = _d(os.path.join(path, 'uiauto'))
    create_uiauto_project(uiauto_path, name)
    source_file = os.path.join(path, '__init__.py')
    with open(source_file, 'w') as wfh:
        wfh.write(render_template('uiauto_workload', {'name': name, 'class_name': class_name}))


def create_android_benchmark(path, name, class_name):
    source_file = os.path.join(path, '__init__.py')
    with open(source_file, 'w') as wfh:
        wfh.write(render_template('android_benchmark', {'name': name, 'class_name': class_name}))


def create_android_uiauto_benchmark(path, name, class_name):
    uiauto_path = _d(os.path.join(path, 'uiauto'))
    create_uiauto_project(uiauto_path, name)
    source_file = os.path.join(path, '__init__.py')
    with open(source_file, 'w') as wfh:
        wfh.write(render_template('android_uiauto_benchmark', {'name': name, 'class_name': class_name}))


def create_uiauto_project(path, name, target='1'):
    sdk_path = get_sdk_path()
    android_path = os.path.join(sdk_path, 'tools', 'android')
    package_name = 'com.arm.wlauto.uiauto.' + name.lower()

    # ${ANDROID_HOME}/tools/android create uitest-project -n com.arm.wlauto.uiauto.linpack -t 1 -p ../test2
    command = '{} create uitest-project --name {} --target {} --path {}'.format(android_path,
                                                                                package_name,
                                                                                target,
                                                                                path)
    try:
        check_output(command, shell=True)
    except subprocess.CalledProcessError as e:
        if 'is is not valid' in e.output:
            message = 'No Android SDK target found; have you run "{} update sdk" and download a platform?'
            raise CommandError(message.format(android_path))

    build_script = os.path.join(path, 'build.sh')
    with open(build_script, 'w') as wfh:
        template = string.Template(UIAUTO_BUILD_SCRIPT)
        wfh.write(template.substitute({'package_name': package_name}))
    os.chmod(build_script, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    source_file = _f(os.path.join(path, 'src',
                                  os.sep.join(package_name.split('.')[:-1]),
                                  'UiAutomation.java'))
    with open(source_file, 'w') as wfh:
        wfh.write(render_template('UiAutomation.java', {'name': name, 'package_name': package_name}))


# Utility functions

def get_sdk_path():
    sdk_path = os.getenv('ANDROID_HOME')
    if not sdk_path:
        raise CommandError('Please set ANDROID_HOME environment variable to point to ' +
                           'the locaton of Android SDK')
    return sdk_path


def get_class_name(name, postfix=''):
    name = identifier(name)
    return ''.join(map(capitalize, name.split('_'))) + postfix


def render_template(name, params):
    filepath = os.path.join(TEMPLATES_DIR, name)
    with open(filepath) as fh:
        text = fh.read()
        template = string.Template(text)
        return template.substitute(params)


def touch(path):
    with open(path, 'w') as _:
        pass
