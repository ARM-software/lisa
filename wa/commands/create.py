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
import re
import uuid
import getpass
from collections import OrderedDict
from distutils.dir_util import copy_tree  # pylint: disable=no-name-in-module, import-error

from devlib.utils.types import identifier
try:
    import psycopg2
    from psycopg2 import connect, OperationalError, extras
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError as e:
    psycopg2 = None
    import_error_msg = e.args[0] if e.args else str(e)

from wa import ComplexCommand, SubCommand, pluginloader, settings
from wa.framework.target.descriptor import list_target_descriptions
from wa.framework.exception import ConfigError, CommandError
from wa.instruments.energy_measurement import EnergyInstrumentBackend
from wa.utils.misc import (ensure_directory_exists as _d, capitalize,
                           ensure_file_directory_exists as _f)
from wa.utils.serializer import yaml


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')


class CreateDatabaseSubcommand(SubCommand):

    name = 'database'
    description = """
    Create a Postgresql database which is compatible with the WA Postgres
    output processor.
    """

    schemafilepath = 'postgres_schema.sql'

    def __init__(self, *args, **kwargs):
        super(CreateDatabaseSubcommand, self).__init__(*args, **kwargs)
        self.sql_commands = None
        self.schemaversion = None
        self.schema_major = None
        self.schema_minor = None

    def initialize(self, context):
        self.parser.add_argument(
            '-a', '--postgres-host', default='localhost',
            help='The host on which to create the database.')
        self.parser.add_argument(
            '-k', '--postgres-port', default='5432',
            help='The port on which the PostgreSQL server is running.')
        self.parser.add_argument(
            '-u', '--username', default='postgres',
            help='The username with which to connect to the server.')
        self.parser.add_argument(
            '-p', '--password',
            help='The password for the user account.')
        self.parser.add_argument(
            '-d', '--dbname', default='wa',
            help='The name of the database to create.')
        self.parser.add_argument(
            '-f', '--force', action='store_true',
            help='Force overwrite the existing database if one exists.')
        self.parser.add_argument(
            '-F', '--force-update-config', action='store_true',
            help='Force update the config file if an entry exists.')
        self.parser.add_argument(
            '-r', '--config-file', default=settings.user_config_file,
            help='Path to the config file to be updated.')
        self.parser.add_argument(
            '-x', '--schema-version', action='store_true',
            help='Display the current schema version.')

    def execute(self, state, args):  # pylint: disable=too-many-branches
        if not psycopg2:
            raise CommandError(
                'The module psycopg2 is required for the wa ' +
                'create database command.')
        self.get_schema(self.schemafilepath)

        # Display the version if needed and exit
        if args.schema_version:
            self.logger.info(
                'The current schema version is {}'.format(self.schemaversion))
            return

        if args.dbname == 'postgres':
            raise ValueError('Databasename to create cannot be postgres.')

        # Open user configuration
        with open(args.config_file, 'r') as config_file:
            config = yaml.load(config_file)
            if 'postgres' in config and not args.force_update_config:
                raise CommandError(
                    "The entry 'postgres' already exists in the config file. " +
                    "Please specify the -F flag to force an update.")

        possible_connection_errors = [
            (
                re.compile('FATAL:  role ".*" does not exist'),
                'Username does not exist or password is incorrect'
            ),
            (
                re.compile('FATAL:  password authentication failed for user'),
                'Password was incorrect'
            ),
            (
                re.compile('fe_sendauth: no password supplied'),
                'Passwordless connection is not enabled. '
                'Please enable trust in pg_hba for this host '
                'or use a password'
            ),
            (
                re.compile('FATAL:  no pg_hba.conf entry for'),
                'Host is not allowed to connect to the specified database '
                'using this user according to pg_hba.conf. Please change the '
                'rules in pg_hba or your connection method'
            ),
            (
                re.compile('FATAL:  pg_hba.conf rejects connection'),
                'Connection was rejected by pg_hba.conf'
            ),
        ]

        def predicate(error, handle):
            if handle[0].match(str(error)):
                raise CommandError(handle[1] + ': \n' + str(error))

        # Attempt to create database
        try:
            self.create_database(args)
        except OperationalError as e:
            for handle in possible_connection_errors:
                predicate(e, handle)
            raise e

        # Update the configuration file
        _update_configuration_file(args, config)

    def create_database(self, args):
        _check_database_existence(args)

        _create_database_postgres(args)

        _apply_database_schema(args, self.sql_commands, self.schema_major, self.schema_minor)

        self.logger.debug(
            "Successfully created the database {}".format(args.dbname))

    def get_schema(self, schemafilepath):
        postgres_output_processor_dir = os.path.dirname(__file__)
        sqlfile = open(os.path.join(
            postgres_output_processor_dir, schemafilepath))
        self.sql_commands = sqlfile.read()
        sqlfile.close()
        # Extract schema version
        if self.sql_commands.startswith('--!VERSION'):
            splitcommands = self.sql_commands.split('!ENDVERSION!\n')
            self.schemaversion = splitcommands[0].strip('--!VERSION!')
            (self.schema_major, self.schema_minor) = self.schemaversion.split('.')
            self.sql_commands = splitcommands[1]


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

    # pylint: disable=too-many-branches
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

            # Handle special case for EnergyInstrumentBackends
            if issubclass(extcls, EnergyInstrumentBackend):
                if 'energy_measurement' not in agenda['config']['augmentations']:
                    energy_config = pluginloader.get_default_config('energy_measurement')
                    agenda['config']['augmentations'].append('energy_measurement')
                    agenda['config']['energy_measurement'] = energy_config
                agenda['config']['energy_measurement']['instrument'] = name
                agenda['config']['energy_measurement']['instrument_parameters'] = config
            elif extcls.kind == 'workload':
                entry = OrderedDict()
                entry['name'] = extcls.name
                if name != extcls.name:
                    entry['label'] = name
                entry['params'] = config
                agenda['workloads'].append(entry)
            else:
                if extcls.kind in ('instrument', 'output_processor'):
                    if name not in agenda['config']['augmentations']:
                        agenda['config']['augmentations'].append(name)

                if name not in agenda['config']:
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
        CreateDatabaseSubcommand,
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
    with open(path, 'w') as _: # NOQA
        pass


def _check_database_existence(args):
    try:
        connect(dbname=args.dbname, user=args.username,
                password=args.password, host=args.postgres_host, port=args.postgres_port)
    except OperationalError as e:
        # Expect an operational error (database's non-existence)
        if not re.compile('FATAL:  database ".*" does not exist').match(str(e)):
            raise e
    else:
        if not args.force:
            raise CommandError(
                "Database {} already exists. ".format(args.dbname) +
                "Please specify the -f flag to create it from afresh."
            )


def _create_database_postgres(args):  # pylint: disable=no-self-use
    conn = connect(dbname='postgres', user=args.username,
                   password=args.password, host=args.postgres_host, port=args.postgres_port)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute('DROP DATABASE IF EXISTS ' + args.dbname)
    cursor.execute('CREATE DATABASE ' + args.dbname)
    conn.commit()
    cursor.close()
    conn.close()


def _apply_database_schema(args, sql_commands, schema_major, schema_minor):
    conn = connect(dbname=args.dbname, user=args.username,
                   password=args.password, host=args.postgres_host, port=args.postgres_port)
    cursor = conn.cursor()
    cursor.execute(sql_commands)

    extras.register_uuid()
    cursor.execute("INSERT INTO DatabaseMeta VALUES (%s, %s, %s)",
                   (
                       uuid.uuid4(),
                       schema_major,
                       schema_minor
                   )
                   )

    conn.commit()
    cursor.close()
    conn.close()


def _update_configuration_file(args, config):
    ''' Update the user configuration file with the newly created database's
        configuration.
        '''
    config['postgres'] = OrderedDict(
        [('host', args.postgres_host), ('port', args.postgres_port),
         ('dbname', args.dbname), ('username', args.username), ('password', args.password)])
    with open(args.config_file, 'w+') as config_file:
        yaml.dump(config, config_file)
