#    Copyright 2015 ARM Limited
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

from wlauto.exceptions import ConfigError
from wlauto.utils.serializer import read_pod, SerializerSyntaxError
from wlauto.utils.types import toggle_set, counter
from wlauto.core.configuration.configuration import JobSpec

########################
### Helper functions ###
########################

DUPLICATE_ENTRY_ERROR = 'Only one of {} may be specified in a single entry'


def get_aliased_param(cfg_point, d, default=None, pop=True):
    """
    Given a ConfigurationPoint and a dict, this function will search the dict for
    the ConfigurationPoint's name/aliases. If more than one is found it will raise
    a ConfigError. If one (and only one) is found then it will return the value
    for the ConfigurationPoint. If the name or aliases are present in the dict it will
    return the "default" parameter of this function.
    """
    aliases = [cfg_point.name] + cfg_point.aliases
    alias_map = [a for a in aliases if a in d]
    if len(alias_map) > 1:
        raise ConfigError(DUPLICATE_ENTRY_ERROR.format(aliases))
    elif alias_map:
        if pop:
            return d.pop(alias_map[0])
        else:
            return d[alias_map[0]]
    else:
        return default


def _load_file(filepath, error_name):
    if not os.path.isfile(filepath):
        raise ValueError("{} does not exist".format(filepath))
    try:
        raw = read_pod(filepath)
    except SerializerSyntaxError as e:
        raise ConfigError('Error parsing {} {}: {}'.format(error_name, filepath, e))
    if not isinstance(raw, dict):
        message = '{} does not contain a valid {} structure; top level must be a dict.'
        raise ConfigError(message.format(filepath, error_name))
    return raw


def get_workload_entry(w):
    if isinstance(w, basestring):
        w = {'name': w}
    elif not isinstance(w, dict):
        raise ConfigError('Invalid workload entry: "{}"')
    return w


def merge_result_processors_instruments(raw):
    instruments = toggle_set(get_aliased_param(JobSpec.configuration['instrumentation'],
                                               raw, default=[]))
    result_processors = toggle_set(raw.pop('result_processors', []))
    if instruments and result_processors:
        conflicts = instruments.conflicts_with(result_processors)
        if conflicts:
            msg = '"instrumentation" and "result_processors" have conflicting entries: {}'
            entires = ', '.join('"{}"'.format(c.strip("~")) for c in conflicts)
            raise ConfigError(msg.format(entires))
    raw['instrumentation'] = instruments.merge_with(result_processors)


def _construct_valid_entry(raw, seen_ids, counter_name):
    entries = {}

    # Generate an automatic ID if the entry doesn't already have one
    if "id" not in raw:
        while True:
            new_id = "{}{}".format(counter_name, counter(name=counter_name))
            if new_id not in seen_ids:
                break
        entries["id"] = new_id
    else:
        entries["id"] = raw.pop("id")

    merge_result_processors_instruments(raw)

    # Validate all entries
    for cfg_point in JobSpec.configuration.itervalues():
        value = get_aliased_param(cfg_point, raw)
        if value is not None:
            value = cfg_point.kind(value)
            cfg_point.validate_value(cfg_point.name, value)
            entries[cfg_point] = value

    # error if there are unknown entries
    if raw:
        msg = 'Invalid entry(ies) in "{}": "{}"'
        raise ConfigError(msg.format(entries['id'], ', '.join(raw.keys())))

    return entries

###############
### Parsers ###
###############


class ConfigParser(object):

    def __init__(self, wa_config, run_config, jobs_config, plugin_cache):
        self.wa_config = wa_config
        self.run_config = run_config
        self.jobs_config = jobs_config
        self.plugin_cache = plugin_cache

    def load_from_path(self, filepath):
        self.load(_load_file(filepath, "Config"), filepath)

    def load(self, raw, source):  # pylint: disable=too-many-branches
        try:
            if 'run_name' in raw:
                msg = '"run_name" can only be specified in the config section of an agenda'
                raise ConfigError(msg)
            if 'id' in raw:
                raise ConfigError('"id" cannot be set globally')

            merge_result_processors_instruments(raw)

            for cfg_point in self.wa_config.configuration.itervalues():
                value = get_aliased_param(cfg_point, raw)
                if value is not None:
                    self.wa_config.set(cfg_point.name, value)

            for cfg_point in self.run_config.configuration.itervalues():
                value = get_aliased_param(cfg_point, raw)
                if value is not None:
                    self.run_config.set(cfg_point.name, value)

            for cfg_point in JobSpec.configuration.itervalues():
                value = get_aliased_param(cfg_point, raw)
                if value is not None:
                    #TODO: runtime_params and boot_params
                    if cfg_point.name == "workload_parameters":
                        self.plugin_cache.add_plugin_config("workload_parameters", value, source)
                    else:
                        self.jobs_config.set_global_value(cfg_point.name, value)

            device_config = raw.pop('device_config', None)
            if device_config:
                self.plugin_cache.add_device_config('device_config', device_config, source)

            for name, value in raw.iteritems():
                if self.plugin_cache.is_global_alias(name):
                    self.plugin_cache.add_global_alias(name, value, source)
                else:
                    # Assume that all leftover config is for a plug-in
                    # it is up to PluginCache to assert this assumption
                    self.plugin_cache.add_plugin_config(name, value, source)

        except ConfigError as e:
            raise ConfigError('Error in "{}":\n{}'.format(source, str(e)))


class AgendaParser(object):

    def __init__(self, wa_config, run_config, jobs_config, plugin_cache):
        self.wa_config = wa_config
        self.run_config = run_config
        self.jobs_config = jobs_config
        self.plugin_cache = plugin_cache

    def load_from_path(self, filepath):
        raw = _load_file(filepath, 'Agenda')
        self.load(raw, filepath)

    def load(self, raw, source):  # pylint: disable=too-many-branches, too-many-locals
        try:
            if not isinstance(raw, dict):
                raise ConfigError('Invalid agenda, top level entry must be a dict')

            # PHASE 1: Populate and validate configuration.
            for name in ['config', 'global']:
                entry = raw.pop(name, {})
                if not isinstance(entry, dict):
                    raise ConfigError('Invalid entry "{}" - must be a dict'.format(name))
                if 'run_name' in entry:
                    self.run_config.set('run_name', entry.pop('run_name'))
                config_parser = ConfigParser(self.wa_config, self.run_config,
                                             self.jobs_config, self.plugin_cache)
                config_parser.load(entry, source)

            # PHASE 2: Getting "section" and "workload" entries.
            sections = raw.pop("sections", [])
            if not isinstance(sections, list):
                raise ConfigError('Invalid entry "sections" - must be a list')
            global_workloads = raw.pop("workloads", [])
            if not isinstance(global_workloads, list):
                raise ConfigError('Invalid entry "workloads" - must be a list')
            if raw:
                msg = 'Invalid top level agenda entry(ies): "{}"'
                raise ConfigError(msg.format('", "'.join(raw.keys())))

            # PHASE 3: Collecting existing workload and section IDs
            seen_section_ids = set()
            for section in sections:
                entry_id = section.get("id")
                if entry_id is None:
                    continue
                if entry_id in seen_section_ids:
                    raise ConfigError('Duplicate section ID "{}".'.format(entry_id))
                # "-" is reserved for joining section and workload IDs
                if "-" in entry_id:
                    msg = 'Invalid ID "{}"; IDs cannot contain a "-"'
                    raise ConfigError(msg.format(entry_id))
                seen_section_ids.add(entry_id)

            seen_workload_ids = set()
            for workload in global_workloads:
                entry_id = workload.get("id")
                if entry_id is None:
                    continue
                if entry_id in seen_workload_ids:
                    raise ConfigError('Duplicate workload ID "{}".'.format(entry_id))
                # "-" is reserved for joining section and workload IDs
                if "-" in entry_id:
                    msg = 'Invalid ID "{}"; IDs cannot contain a "-"'
                    raise ConfigError(msg.format(entry_id))
                if entry_id == "global":
                    raise ConfigError(('The ID "global" is reserved'))
                seen_workload_ids.add(entry_id)

            # PHASE 4: Assigning IDs and validating entries
            # TODO: Error handling for workload errors vs section errors ect
            for workload in global_workloads:
                self.jobs_config.add_workload(self._process_entry(workload, seen_workload_ids))

            for section in sections:
                workloads = []
                for workload in section.pop("workloads", []):
                    workloads.append(self._process_entry(workload, seen_workload_ids))

                if "params" in section:
                    if "runtime_params" in section:
                        raise ConfigError(DUPLICATE_ENTRY_ERROR.format(["params", "runtime_params"]))
                    section["runtime_params"] = section.pop("params")
                section = _construct_valid_entry(section, seen_section_ids, "s")
                self.jobs_config.add_section(section, workloads)

        except (ConfigError, SerializerSyntaxError) as e:
            raise ConfigError('Error in "{}":\n\t{}'.format(source, str(e)))

    def _process_entry(self, entry, seen_workload_ids):
        workload = get_workload_entry(entry)
        if "params" in workload:
            if "workload_params" in workload:
                raise ConfigError(DUPLICATE_ENTRY_ERROR.format(["params", "workload_params"]))
            workload["workload_params"] = workload.pop("params")
        return _construct_valid_entry(entry, seen_workload_ids, "wk")


class EnvironmentVarsParser(object):
    #TODO: podable
    def __init__(self, wa_config, environ):
        user_directory = environ.pop('WA_USER_DIRECTORY', '')
        if user_directory:
            wa_config.set('user_directory', user_directory)
        plugin_paths = environ.pop('WA_PLUGIN_PATHS', '')
        if plugin_paths:
            wa_config.set('plugin_paths', plugin_paths.split(os.pathsep))
        ext_paths = environ.pop('WA_EXTENSION_PATHS', '')
        if ext_paths:
            wa_config.set('plugin_paths', ext_paths.split(os.pathsep))


# Command line options are parsed in the "run" command. This is used to send
# certain arguments to the correct configuration points and keep a record of
# how WA was invoked
class CommandLineArgsParser(object):
    #TODO: podable
    def __init__(self, cmd_args, wa_config, run_config, jobs_config):
        wa_config.set("verbosity", cmd_args.verbosity)
        # TODO: Is this correct? Does there need to be a third output dir param
        disabled_instruments = toggle_set(["~{}".format(i) for i in cmd_args.instruments_to_disable])
        jobs_config.disable_instruments(disabled_instruments)
        jobs_config.only_run_ids(cmd_args.only_run_ids)
