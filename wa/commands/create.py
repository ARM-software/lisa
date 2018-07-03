#    Copyright 2018 ARM Limited
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
import shutil
import string
import getpass
from collections import OrderedDict
from distutils.dir_util import copy_tree

from wa import ComplexCommand, SubCommand, pluginloader, settings
from wa.framework.target.descriptor import list_target_descriptions
from wa.framework.exception import ConfigError, CommandError
from wa.utils.misc import (ensure_directory_exists as _d, capitalize,
                           ensure_file_directory_exists as _f)
from wa.utils.serializer import yaml

from devlib.utils.types import identifier


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')


class CreateAgendaSubcommand(SubCommand):

    name = 'agenda'
    description = """
    Create an agenda with the specified extensions enabled. And parameters set
    to their default values.
    """

    def initialize(self, context):
        self.parser.add_argument('plugins', nargs='+',
                                 help='Plugins to be added to the agendas')
        self.parser.add_argument('-i', '--iterations', type=int, default=1,
                                 help='Sets the number of iterations for all workloads')
        self.parser.add_argument('-o', '--output', metavar='FILE',
                                 help='Output file. If not specfied, STDOUT will be used instead.')

    def execute(self, state, args):
        agenda = OrderedDict()
        agenda['config'] = OrderedDict(augmentations=[], iterations=args.iterations)
        agenda['workloads'] = []
        target_desc = None

        targets = {td.name: td for td in list_target_descriptions()}

        for name in args.plugins:
            if name in targets:
                if target_desc is not None:
                    raise ConfigError('Specifying multiple devices: {} and {}'.format(target_desc.name, name))
                target_desc = targets[name]
                agenda['config']['device'] = name
                agenda['config']['device_config'] = target_desc.get_default_config()
                continue

            extcls = pluginloader.get_plugin_class(name)
            config = pluginloader.get_default_config(name)

            if extcls.kind == 'workload':
                entry = OrderedDict()
                entry['name'] = extcls.name
                if name != extcls.name:
                    entry['label'] = name
                entry['params'] = config
                agenda['workloads'].append(entry)
            else:
                if extcls.kind == 'instrument':
                    agenda['config']['augmentations'].append(name)
                if extcls.kind == 'output_processor':
                    agenda['config']['augmentations'].append(name)
                agenda['config'][name] = config

        if args.output:
            wfh = open(args.output, 'w')
        else:
            wfh = sys.stdout
        yaml.dump(agenda, wfh, indent=4, default_flow_style=False)
        if args.output:
            wfh.close()


class CreateWorkloadSubcommand(SubCommand):

    name = 'workload'
    description = '''Create a new workload. By default, a basic workload template will be
                     used but you can specify the `KIND` to choose a different template.'''

    def initialize(self, context):
        self.parser.add_argument('name', metavar='NAME',
                                 help='Name of the workload to be created')
        self.parser.add_argument('-p', '--path', metavar='PATH', default=None,
                                 help='The location at which the workload will be created. If not specified, ' +
                                      'this defaults to "~/.workload_automation/plugins".')
        self.parser.add_argument('-f', '--force', action='store_true',
                                 help='Create the new workload even if a workload with the specified ' +
                                      'name already exists.')
        self.parser.add_argument('-k', '--kind', metavar='KIND', default='basic', choices=list(create_funcs.keys()),
                                 help='The type of workload to be created. The available options ' +
                                      'are: {}'.format(', '.join(list(create_funcs.keys()))))

    def execute(self, state, args):  # pylint: disable=R0201
        where = args.path or 'local'
        check_name = not args.force

        try:
            create_workload(args.name, args.kind, where, check_name)
        except CommandError as e:
            self.logger.error('ERROR: {}'.format(e))


class CreatePackageSubcommand(SubCommand):

    name = 'package'
    description = '''Create a new empty Python package for WA extensions. On installation,
                     this package will "advertise" itself to WA so that Plugins within it will
                     be loaded by WA when it runs.'''

    def initialize(self, context):
        self.parser.add_argument('name', metavar='NAME',
                                 help='Name of the package to be created')
        self.parser.add_argument('-p', '--path', metavar='PATH', default=None,
                                 help='The location at which the new package will be created. If not specified, ' +
                                      'current working directory will be used.')
        self.parser.add_argument('-f', '--force', action='store_true',
                                 help='Create the new package even if a file or directory with the same name '
                                      'already exists at the specified location.')

    def execute(self, state, args):  # pylint: disable=R0201
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


class CreateCommand(ComplexCommand):

    name = 'create'
    description = '''
    Used to create various WA-related objects (see positional arguments list
    for what objects may be created).\n\nUse "wa create <object> -h" for
    object-specific arguments.
    '''
    subcmd_classes = [
        CreateWorkloadSubcommand,
        CreateAgendaSubcommand,
        CreatePackageSubcommand,
    ]


def create_workload(name, kind='basic', where='local', check_name=True, **kwargs):

    if check_name:
        if name in [wl.name for wl in pluginloader.list_plugins('workload')]:
            raise CommandError('Workload with name "{}" already exists.'.format(name))

    class_name = get_class_name(name)
    if where == 'local':
        workload_dir = _d(os.path.join(settings.plugins_directory, name))
    else:
        workload_dir = _d(os.path.join(where, name))

    try:
        # Note: `create_funcs` mapping is listed below
        create_funcs[kind](workload_dir, name, kind, class_name, **kwargs)
    except KeyError:
        raise CommandError('Unknown workload type: {}'.format(kind))

    # pylint: disable=superfluous-parens
    print('Workload created in {}'.format(workload_dir))


def create_template_workload(path, name, kind, class_name):
    source_file = os.path.join(path, '__init__.py')
    with open(source_file, 'w') as wfh:
        wfh.write(render_template('{}_workload'.format(kind), {'name': name, 'class_name': class_name}))


def create_uiautomator_template_workload(path, name, kind, class_name):
    uiauto_path = os.path.join(path, 'uiauto')
    create_uiauto_project(uiauto_path, name)
    create_template_workload(path, name, kind, class_name)


def create_uiauto_project(path, name):
    package_name = 'com.arm.wa.uiauto.' + name.lower()

    copy_tree(os.path.join(TEMPLATES_DIR, 'uiauto', 'uiauto_workload_template'), path)

    manifest_path = os.path.join(path, 'app', 'src', 'main')
    mainifest = os.path.join(_d(manifest_path), 'AndroidManifest.xml')
    with open(mainifest, 'w') as wfh:
        wfh.write(render_template(os.path.join('uiauto', 'uiauto_AndroidManifest.xml'),
                                  {'package_name': package_name}))

    build_gradle_path = os.path.join(path, 'app')
    build_gradle = os.path.join(_d(build_gradle_path), 'build.gradle')
    with open(build_gradle, 'w') as wfh:
        wfh.write(render_template(os.path.join('uiauto', 'uiauto_build.gradle'),
                                  {'package_name': package_name}))

    build_script = os.path.join(path, 'build.sh')
    with open(build_script, 'w') as wfh:
        wfh.write(render_template(os.path.join('uiauto', 'uiauto_build_script'),
                                  {'package_name': package_name}))
    os.chmod(build_script, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    source_file = _f(os.path.join(path, 'app', 'src', 'main', 'java',
                                  os.sep.join(package_name.split('.')[:-1]),
                                  'UiAutomation.java'))
    with open(source_file, 'w') as wfh:
        wfh.write(render_template(os.path.join('uiauto', 'UiAutomation.java'),
                                  {'name': name, 'package_name': package_name}))

# Mapping of workload types to their corresponding creation method
create_funcs = {
    'basic': create_template_workload,
    'apk': create_template_workload,
    'revent': create_template_workload,
    'apkrevent': create_template_workload,
    'uiauto': create_uiautomator_template_workload,
    'apkuiauto': create_uiautomator_template_workload,
}


# Utility functions
def render_template(name, params):
    filepath = os.path.join(TEMPLATES_DIR, name)
    with open(filepath) as fh:
        text = fh.read()
        template = string.Template(text)
        return template.substitute(params)


def get_class_name(name, postfix=''):
    name = identifier(name)
    return ''.join(map(capitalize, name.split('_'))) + postfix


def touch(path):
    with open(path, 'w') as _:
        pass
