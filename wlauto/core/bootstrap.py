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
import shutil
import sys
import re
from collections import namedtuple, OrderedDict

from wlauto.exceptions import ConfigError
from wlauto.utils.misc import merge_dicts, normalize, unique
from wlauto.utils.misc import load_struct_from_yaml, load_struct_from_python, LoadSyntaxError
from wlauto.utils.types import identifier


_this_dir = os.path.dirname(__file__)
_user_home = os.path.expanduser('~')

# loading our external packages over those from the environment
sys.path.insert(0, os.path.join(_this_dir, '..', 'external'))


# Defines extension points for the WA framework. This table is used by the
# ExtensionLoader (among other places) to identify extensions it should look
# for.
# Parameters that need to be specified in a tuple for each extension type:
#     name: The name of the extension type. This will be used to resolve get_
#           and list_methods in the extension loader.
#     class: The base class for the extension type. Extension loader will check
#            whether classes it discovers are subclassed from this.
#     default package: This is the package that will be searched for extensions
#                      of that type by default (if not other packages are
#                      specified when creating the extension loader). This
#                      package *must* exist.
#    default path: This is the subdirectory under the environment_root which
#                  will be searched for extensions of this type by default (if
#                  no other paths are specified when creating the extension
#                  loader). This directory will be automatically created if it
#                  does not exist.

#pylint: disable=C0326
_EXTENSION_TYPE_TABLE = [
    # name,               class,                                    default package,            default path
    ('command',           'wlauto.core.command.Command',            'wlauto.commands',          'commands'),
#    ('device',            'wlauto.core.device.Device',              'wlauto.devices',           'devices'),
    ('device_manager',    'wlauto.core.device.DeviceManager',       'wlauto.managers',          'devices'),
    ('instrument',        'wlauto.core.instrumentation.Instrument', 'wlauto.instrumentation',   'instruments'),
    ('module',            'wlauto.core.extension.Module',           'wlauto.modules',           'modules'),
    ('resource_getter',   'wlauto.core.resource.ResourceGetter',    'wlauto.resource_getters',  'resource_getters'),
    ('result_processor',  'wlauto.core.result.ResultProcessor',     'wlauto.result_processors', 'result_processors'),
    ('workload',          'wlauto.core.workload.Workload',          'wlauto.workloads',         'workloads'),
]
_Extension = namedtuple('_Extension', 'name, cls, default_package, default_path')
_extensions = [_Extension._make(ext) for ext in _EXTENSION_TYPE_TABLE]  # pylint: disable=W0212


class ConfigLoader(object):
    """
    This class is responsible for loading and validating config files.

    """

    def __init__(self):
        self._loaded = False
        self._config = {}
        self.config_count = 0
        self.loaded_files = []
        self.environment_root = None
        self.output_directory = 'wa_output'
        self.reboot_after_each_iteration = True
        self.dependencies_directory = None
        self.agenda = None
        self.extension_packages = []
        self.extension_paths = []
        self.extensions = []
        self.verbosity = 0
        self.debug = False
        self.package_directory = os.path.dirname(_this_dir)
        self.commands = {}

    @property
    def meta_directory(self):
        return os.path.join(self.output_directory, '__meta')

    @property
    def log_file(self):
        return os.path.join(self.output_directory, 'run.log')

    def update(self, source):
        if isinstance(source, dict):
            self.update_from_dict(source)
        else:
            self.config_count += 1
            self.update_from_file(source)

    def update_from_file(self, source):
        ext = os.path.splitext(source)[1].lower()  # pylint: disable=redefined-outer-name
        try:
            if ext in ['.py', '.pyo', '.pyc']:
                new_config = load_struct_from_python(source)
            elif ext == '.yaml':
                new_config = load_struct_from_yaml(source)
            else:
                raise ConfigError('Unknown config format: {}'.format(source))
        except LoadSyntaxError as e:
            raise ConfigError(e)

        self._config = merge_dicts(self._config, new_config,
                                   list_duplicates='first',
                                   match_types=False,
                                   dict_type=OrderedDict)
        self.loaded_files.append(source)
        self._loaded = True

    def update_from_dict(self, source):
        normalized_source = dict((identifier(k), v) for k, v in source.iteritems())
        self._config = merge_dicts(self._config, normalized_source, list_duplicates='first',
                                   match_types=False, dict_type=OrderedDict)
        self._loaded = True

    def get_config_paths(self):
        return [lf.rstrip('c') for lf in self.loaded_files]

    def _check_loaded(self):
        if not self._loaded:
            raise ConfigError('Config file not loaded.')

    def __getattr__(self, name):
        self._check_loaded()
        return self._config.get(normalize(name))


def init_environment(env_root, dep_dir, extension_paths, overwrite_existing=False):  # pylint: disable=R0914
    """Initialise a fresh user environment creating the workload automation"""
    if os.path.exists(env_root):
        if not overwrite_existing:
            raise ConfigError('Environment {} already exists.'.format(env_root))
        shutil.rmtree(env_root)

    os.makedirs(env_root)
    with open(os.path.join(_this_dir, '..', 'config_example.py')) as rf:
        text = re.sub(r'""".*?"""', '', rf.read(), 1, re.DOTALL)
        with open(os.path.join(_env_root, 'config.py'), 'w') as wf:
            wf.write(text)

    os.makedirs(dep_dir)
    for path in extension_paths:
        os.makedirs(path)

    if os.getenv('USER') == 'root':
        # If running with sudo on POSIX, change the ownership to the real user.
        real_user = os.getenv('SUDO_USER')
        if real_user:
            import pwd  # done here as module won't import on win32
            user_entry = pwd.getpwnam(real_user)
            uid, gid = user_entry.pw_uid, user_entry.pw_gid
            os.chown(env_root, uid, gid)
            # why, oh why isn't there a recusive=True option for os.chown?
            for root, dirs, files in os.walk(env_root):
                for d in dirs:
                    os.chown(os.path.join(root, d), uid, gid)
                for f in files:  # pylint: disable=W0621
                    os.chown(os.path.join(root, f), uid, gid)


_env_root = os.getenv('WA_USER_DIRECTORY', os.path.join(_user_home, '.workload_automation'))
_dep_dir = os.path.join(_env_root, 'dependencies')
_extension_paths = [os.path.join(_env_root, ext.default_path) for ext in _extensions]
_env_var_paths = os.getenv('WA_EXTENSION_PATHS', '')
if _env_var_paths:
    _extension_paths.extend(_env_var_paths.split(os.pathsep))

_env_configs = []
for filename in ['config.py', 'config.yaml']:
    filepath = os.path.join(_env_root, filename)
    if os.path.isfile(filepath):
        _env_configs.append(filepath)

if not os.path.isdir(_env_root):
    init_environment(_env_root, _dep_dir, _extension_paths)
elif not _env_configs:
    filepath = os.path.join(_env_root, 'config.py')
    with open(os.path.join(_this_dir, '..', 'config_example.py')) as f:
        f_text = re.sub(r'""".*?"""', '', f.read(), 1, re.DOTALL)
        with open(filepath, 'w') as f:
            f.write(f_text)
        _env_configs.append(filepath)

settings = ConfigLoader()
settings.environment_root = _env_root
settings.dependencies_directory = _dep_dir
settings.extension_paths = _extension_paths
settings.extensions = _extensions

_packages_file = os.path.join(_env_root, 'packages')
if os.path.isfile(_packages_file):
    with open(_packages_file) as fh:
        settings.extension_packages = unique(fh.read().split())

for config in _env_configs:
    settings.update(config)
