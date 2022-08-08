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
import shutil

from wa.framework import pluginloader
from wa.framework.configuration.core import (settings, ConfigurationPoint,
                                             MetaConfiguration, RunConfiguration)
from wa.framework.configuration.default import (generate_default_config,
                                                write_param_yaml)
from wa.framework.configuration.plugin_cache import PluginCache
from wa.utils.misc import load_struct_from_python
from wa.utils.serializer import yaml
from wa.utils.types import identifier


# Have to disable this due to dynamic attributes
# pylint: disable=no-member

def init_user_directory(overwrite_existing=False):  # pylint: disable=R0914
    """
    Initialise a fresh user directory.
    """
    if os.path.exists(settings.user_directory):
        if not overwrite_existing:
            raise RuntimeError('Environment {} already exists.'.format(settings.user_directory))
        shutil.rmtree(settings.user_directory)

    os.makedirs(settings.user_directory)
    os.makedirs(settings.dependencies_directory)
    os.makedirs(settings.plugins_directory)
    os.makedirs(settings.cache_directory)

    generate_default_config(os.path.join(settings.user_directory, 'config.yaml'))

    if os.getenv('USER') == 'root':
        # If running with sudo on POSIX, change the ownership to the real user.
        real_user = os.getenv('SUDO_USER')
        if real_user:
            # pylint: disable=import-outside-toplevel
            import pwd  # done here as module won't import on win32
            user_entry = pwd.getpwnam(real_user)
            uid, gid = user_entry.pw_uid, user_entry.pw_gid
            os.chown(settings.user_directory, uid, gid)
            # why, oh why isn't there a recusive=True option for os.chown?
            for root, dirs, files in os.walk(settings.user_directory):
                for d in dirs:
                    os.chown(os.path.join(root, d), uid, gid)
                for f in files:
                    os.chown(os.path.join(root, f), uid, gid)


def init_config():
    """
    If configuration file is missing try to convert WA2 config if present
    otherwise initialize fresh config file
    """
    wa2_config_file = os.path.join(settings.user_directory, 'config.py')
    wa3_config_file = os.path.join(settings.user_directory, 'config.yaml')
    if os.path.exists(wa2_config_file):
        convert_wa2_agenda(wa2_config_file, wa3_config_file)
    else:
        generate_default_config(wa3_config_file)


def convert_wa2_agenda(filepath, output_path):
    """
    Convert WA2 .py config file to a WA3 .yaml config file.
    """

    orig_agenda = load_struct_from_python(filepath)
    new_agenda = {'augmentations': []}
    config_points = MetaConfiguration.config_points + RunConfiguration.config_points

    # Add additional config points to extract from config file.
    # Also allows for aliasing of renamed parameters
    config_points.extend([
        ConfigurationPoint(
            'augmentations',
            aliases=["instruments", "processors", "instrumentation",
                     "output_processors", "augment", "result_processors"],
            description='''
                The augmentations enabled by default.
                This combines the "instrumentation"
                and "result_processors" from previous
                versions of WA (the old entries are
                now aliases for this).
            '''),
        ConfigurationPoint(
            'device_config',
            description='''Generic configuration for device.''',
            default={}),
        ConfigurationPoint(
            'cleanup_assets',
            aliases=['clean_up'],
            description='''Specify whether to clean up assets
                            deployed to the target''',
            default=True),
    ])

    for param in list(orig_agenda.keys()):
        for cfg_point in config_points:
            if param == cfg_point.name or param in cfg_point.aliases:
                if cfg_point.name == 'augmentations':
                    new_agenda['augmentations'].extend(orig_agenda.pop(param))
                else:
                    new_agenda[cfg_point.name] = format_parameter(orig_agenda.pop(param))

    with open(output_path, 'w') as output:
        for param in config_points:
            entry = {param.name: new_agenda.get(param.name, param.default)}
            write_param_yaml(entry, param, output)

        # Convert plugin configuration
        output.write("# Plugin Configuration\n")
        for param in list(orig_agenda.keys()):
            if pluginloader.has_plugin(param):
                entry = {param: orig_agenda.pop(param)}
                yaml.dump(format_parameter(entry), output, default_flow_style=False)
                output.write("\n")

        # Write any additional aliased parameters into new config
        plugin_cache = PluginCache()
        output.write("# Additional global aliases\n")
        for param in list(orig_agenda.keys()):
            if plugin_cache.is_global_alias(param):
                entry = {param: orig_agenda.pop(param)}
                yaml.dump(format_parameter(entry), output, default_flow_style=False)
                output.write("\n")


def format_parameter(param):
    if isinstance(param, dict):
        return {identifier(k): v for k, v in param.items()}
    else:
        return param
