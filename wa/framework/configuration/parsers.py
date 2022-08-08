#    Copyright 2015-2018 ARM Limited
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
# pylint: disable=no-self-use

import os
import logging
from functools import reduce  # pylint: disable=redefined-builtin

from devlib.utils.types import identifier

from wa.framework.configuration.core import JobSpec
from wa.framework.exception import ConfigError
from wa.utils import log
from wa.utils.serializer import json, read_pod, SerializerSyntaxError
from wa.utils.types import toggle_set, counter
from wa.utils.misc import merge_config_values, isiterable


logger = logging.getLogger('config')


class ConfigParser(object):

    def load_from_path(self, state, filepath):
        raw, includes = _load_file(filepath, "Config")
        self.load(state, raw, filepath)
        return includes

    def load(self, state, raw, source, wrap_exceptions=True):  # pylint: disable=too-many-branches
        logger.debug('Parsing config from "{}"'.format(source))
        log.indent()
        try:
            state.plugin_cache.add_source(source)
            if 'run_name' in raw:
                msg = '"run_name" can only be specified in the config '\
                      'section of an agenda'
                raise ConfigError(msg)

            if 'id' in raw:
                raise ConfigError('"id" cannot be set globally')

            merge_augmentations(raw)

            # Get WA core configuration
            for cfg_point in state.settings.configuration.values():
                value = pop_aliased_param(cfg_point, raw)
                if value is not None:
                    logger.debug('Setting meta "{}" to "{}"'.format(cfg_point.name, value))
                    state.settings.set(cfg_point.name, value)

            # Get run specific configuration
            for cfg_point in state.run_config.configuration.values():
                value = pop_aliased_param(cfg_point, raw)
                if value is not None:
                    logger.debug('Setting run "{}" to "{}"'.format(cfg_point.name, value))
                    state.run_config.set(cfg_point.name, value)

            # Get global job spec configuration
            for cfg_point in JobSpec.configuration.values():
                value = pop_aliased_param(cfg_point, raw)
                if value is not None:
                    logger.debug('Setting global "{}" to "{}"'.format(cfg_point.name, value))
                    state.jobs_config.set_global_value(cfg_point.name, value)

            for name, values in raw.items():
                # Assume that all leftover config is for a plug-in or a global
                # alias it is up to PluginCache to assert this assumption
                logger.debug('Caching "{}" with "{}"'.format(identifier(name), values))
                state.plugin_cache.add_configs(identifier(name), values, source)

        except ConfigError as e:
            if wrap_exceptions:
                raise ConfigError('Error in "{}":\n{}'.format(source, str(e)))
            else:
                raise e
        finally:
            log.dedent()


class AgendaParser(object):

    def load_from_path(self, state, filepath):
        raw, includes = _load_file(filepath, 'Agenda')
        self.load(state, raw, filepath)
        return includes

    def load(self, state, raw, source):
        logger.debug('Parsing agenda from "{}"'.format(source))
        log.indent()
        try:
            if not isinstance(raw, dict):
                raise ConfigError('Invalid agenda, top level entry must be a dict')

            self._populate_and_validate_config(state, raw, source)
            sections = self._pop_sections(raw)
            global_workloads = self._pop_workloads(raw)
            if not global_workloads:
                msg = 'No jobs avaliable. Please ensure you have specified at '\
                      'least one workload to run.'
                raise ConfigError(msg)

            if raw:
                msg = 'Invalid top level agenda entry(ies): "{}"'
                raise ConfigError(msg.format('", "'.join(list(raw.keys()))))

            sect_ids, wkl_ids = self._collect_ids(sections, global_workloads)
            self._process_global_workloads(state, global_workloads, wkl_ids)
            self._process_sections(state, sections, sect_ids, wkl_ids)

            state.agenda = source

        except (ConfigError, SerializerSyntaxError) as e:
            raise ConfigError('Error in "{}":\n\t{}'.format(source, str(e)))
        finally:
            log.dedent()

    def _populate_and_validate_config(self, state, raw, source):
        for name in ['config', 'global']:
            entry = raw.pop(name, None)
            if entry is None:
                continue

            if not isinstance(entry, dict):
                msg = 'Invalid entry "{}" - must be a dict'
                raise ConfigError(msg.format(name))

            if 'run_name' in entry:
                value = entry.pop('run_name')
                logger.debug('Setting run name to "{}"'.format(value))
                state.run_config.set('run_name', value)

            state.load_config(entry, '{}/{}'.format(source, name))

    def _pop_sections(self, raw):
        sections = raw.pop("sections", [])
        if not isinstance(sections, list):
            raise ConfigError('Invalid entry "sections" - must be a list')
        for section in sections:
            if not hasattr(section, 'items'):
                raise ConfigError('Invalid section "{}" - must be a dict'.format(section))
        return sections

    def _pop_workloads(self, raw):
        workloads = raw.pop("workloads", [])
        if not isinstance(workloads, list):
            raise ConfigError('Invalid entry "workloads" - must be a list')
        return workloads

    def _collect_ids(self, sections, global_workloads):
        seen_section_ids = set()
        seen_workload_ids = set()

        for workload in global_workloads:
            workload = _get_workload_entry(workload)
            _collect_valid_id(workload.get("id"), seen_workload_ids, "workload")

        for section in sections:
            _collect_valid_id(section.get("id"), seen_section_ids, "section")
            for workload in section["workloads"] if "workloads" in section else []:
                workload = _get_workload_entry(workload)
                _collect_valid_id(workload.get("id"), seen_workload_ids,
                                  "workload")

        return seen_section_ids, seen_workload_ids

    def _process_global_workloads(self, state, global_workloads, seen_wkl_ids):
        for workload_entry in global_workloads:
            workload = _process_workload_entry(workload_entry, seen_wkl_ids,
                                               state.jobs_config)
            state.jobs_config.add_workload(workload)

    def _process_sections(self, state, sections, seen_sect_ids, seen_wkl_ids):
        for section in sections:
            workloads = []
            for workload_entry in section.pop("workloads", []):
                workload = _process_workload_entry(workload_entry, seen_wkl_ids,
                                                   state.jobs_config)
                workloads.append(workload)

            if 'params' in section:
                if 'runtime_params' in section:
                    msg = 'both "params" and "runtime_params" specified in a '\
                          'section: "{}"'
                    raise ConfigError(msg.format(json.dumps(section, indent=None)))
                section['runtime_params'] = section.pop('params')

            group = section.pop('group', None)
            section = _construct_valid_entry(section, seen_sect_ids,
                                             "s", state.jobs_config)
            state.jobs_config.add_section(section, workloads, group)


########################
### Helper functions ###
########################

def pop_aliased_param(cfg_point, d, default=None):
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
        raise ConfigError('Duplicate entry: {}'.format(aliases))
    elif alias_map:
        return d.pop(alias_map[0])
    else:
        return default


def _load_file(filepath, error_name):
    if not os.path.isfile(filepath):
        raise ValueError("{} does not exist".format(filepath))
    try:
        raw = read_pod(filepath)
        includes = _process_includes(raw, filepath, error_name)
    except SerializerSyntaxError as e:
        raise ConfigError('Error parsing {} {}: {}'.format(error_name, filepath, e))
    if not isinstance(raw, dict):
        message = '{} does not contain a valid {} structure; top level must be a dict.'
        raise ConfigError(message.format(filepath, error_name))
    return raw, includes


def _process_includes(raw, filepath, error_name):
    if not raw:
        return []

    source_dir = os.path.dirname(filepath)
    included_files = []
    replace_value = None

    if hasattr(raw, 'items'):
        for key, value in raw.items():
            if key == 'include#':
                include_path = os.path.expanduser(os.path.join(source_dir, value))
                included_files.append(include_path)
                replace_value, includes = _load_file(include_path, error_name)
                included_files.extend(includes)
            elif hasattr(value, 'items') or isiterable(value):
                includes = _process_includes(value, filepath, error_name)
                included_files.extend(includes)
    elif isiterable(raw):
        for element in raw:
            if hasattr(element, 'items') or isiterable(element):
                includes = _process_includes(element, filepath, error_name)
                included_files.extend(includes)

    if replace_value is not None:
        del raw['include#']
        for key, value in replace_value.items():
            raw[key] = merge_config_values(value, raw.get(key, None))

    return included_files


def merge_augmentations(raw):
    """
    Since, from configuration perspective, output processors and instruments are
    handled identically, the configuration entries are now interchangeable. E.g. it is
    now valid to specify a output processor in an instruments list. This is to make things
    easier for the users, as, from their perspective, the distinction is somewhat arbitrary.

    For backwards compatibility, both entries are still valid, and this
    function merges them together into a single "augmentations" set, ensuring
    that there are no conflicts between the entries.

    """
    cfg_point = JobSpec.configuration['augmentations']
    names = [cfg_point.name, ] + cfg_point.aliases

    entries = []
    for n in names:
        if n not in raw:
            continue
        value = raw.pop(n)
        try:
            entries.append(toggle_set(value))
        except TypeError as exc:
            msg = 'Invalid value "{}" for "{}": {}'
            raise ConfigError(msg.format(value, n, exc))

    # Make sure none of the specified aliases conflict with each other
    to_check = list(entries)
    while len(to_check) > 1:
        check_entry = to_check.pop()
        for e in to_check:
            conflicts = check_entry.conflicts_with(e)
            if conflicts:
                msg = '"{}" and "{}" have conflicting entries: {}'
                conflict_string = ', '.join('"{}"'.format(c.strip("~"))
                                            for c in conflicts)
                raise ConfigError(msg.format(check_entry, e, conflict_string))

    if entries:
        raw['augmentations'] = reduce(lambda x, y: x.union(y), entries)


def _pop_aliased(d, names, entry_id):
    name_count = sum(1 for n in names if n in d)
    if name_count > 1:
        names_list = ', '.join(names)
        msg = 'Invalid workload entry "{}": at most one of ({}}) must be specified.'
        raise ConfigError(msg.format(entry_id, names_list))
    for name in names:
        if name in d:
            return d.pop(name)
    return None


def _construct_valid_entry(raw, seen_ids, prefix, jobs_config):
    workload_entry = {}

    # Generate an automatic ID if the entry doesn't already have one
    if 'id' not in raw:
        while True:
            new_id = '{}{}'.format(prefix, counter(name=prefix))
            if new_id not in seen_ids:
                break
        workload_entry['id'] = new_id
        seen_ids.add(new_id)
    else:
        workload_entry['id'] = raw.pop('id')

    # Process instruments
    merge_augmentations(raw)

    # Validate all workload_entry
    for name, cfg_point in JobSpec.configuration.items():
        value = pop_aliased_param(cfg_point, raw)
        if value is not None:
            value = cfg_point.kind(value)
            cfg_point.validate_value(name, value)
            workload_entry[name] = value

    if "augmentations" in workload_entry:
        if '~~' in workload_entry['augmentations']:
            msg = '"~~" can only be specfied in top-level config, and not for individual workloads/sections'
            raise ConfigError(msg)
        jobs_config.update_augmentations(workload_entry['augmentations'])

    # error if there are unknown workload_entry
    if raw:
        msg = 'Invalid entry(ies) in "{}": "{}"'
        raise ConfigError(msg.format(workload_entry['id'], ', '.join(list(raw.keys()))))

    return workload_entry


def _collect_valid_id(entry_id, seen_ids, entry_type):
    if entry_id is None:
        return
    entry_id = str(entry_id)
    if entry_id in seen_ids:
        raise ConfigError('Duplicate {} ID "{}".'.format(entry_type, entry_id))
    # "-" is reserved for joining section and workload IDs
    if "-" in entry_id:
        msg = 'Invalid {} ID "{}"; IDs cannot contain a "-"'
        raise ConfigError(msg.format(entry_type, entry_id))
    if entry_id == "global":
        msg = 'Invalid {} ID "global"; is a reserved ID'
        raise ConfigError(msg.format(entry_type))
    seen_ids.add(entry_id)


def _get_workload_entry(workload):
    if isinstance(workload, str):
        workload = {'name': workload}
    elif not isinstance(workload, dict):
        raise ConfigError('Invalid workload entry: "{}"')
    return workload


def _process_workload_entry(workload, seen_workload_ids, jobs_config):
    workload = _get_workload_entry(workload)
    workload = _construct_valid_entry(workload, seen_workload_ids,
                                      "wk", jobs_config)
    if "workload_name" not in workload:
        raise ConfigError('No workload name specified in entry {}'.format(workload['id']))
    return workload
