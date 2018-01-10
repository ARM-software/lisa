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

from wa.framework.configuration.core import JobSpec
from wa.framework.exception import ConfigError
from wa.utils.serializer import json, read_pod, SerializerSyntaxError
from wa.utils.types import toggle_set, counter


###############
### Parsers ###
###############

class ConfigParser(object):

    def load_from_path(self, state, filepath):
        self.load(state, _load_file(filepath, "Config"), filepath)

    def load(self, state, raw, source, wrap_exceptions=True):  # pylint: disable=too-many-branches
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
            for cfg_point in state.settings.configuration.itervalues():
                value = pop_aliased_param(cfg_point, raw)
                if value is not None:
                    state.settings.set(cfg_point.name, value)

            # Get run specific configuration
            for cfg_point in state.run_config.configuration.itervalues():
                value = pop_aliased_param(cfg_point, raw)
                if value is not None:
                    state.run_config.set(cfg_point.name, value)

            # Get global job spec configuration
            for cfg_point in JobSpec.configuration.itervalues():
                value = pop_aliased_param(cfg_point, raw)
                if value is not None:
                    state.jobs_config.set_global_value(cfg_point.name, value)

            for name, values in raw.iteritems():
                # Assume that all leftover config is for a plug-in or a global
                # alias it is up to PluginCache to assert this assumption
                state.plugin_cache.add_configs(name, values, source)

        except ConfigError as e:
            if wrap_exceptions:
                raise ConfigError('Error in "{}":\n{}'.format(source, str(e)))
            else:
                raise e


class AgendaParser(object):

    def load_from_path(self, state, filepath):
        raw = _load_file(filepath, 'Agenda')
        self.load(state, raw, filepath)

    def load(self, state, raw, source):
        try:
            if not isinstance(raw, dict):
                raise ConfigError('Invalid agenda, top level entry must be a dict')

            self._populate_and_validate_config(state, raw, source)
            sections = self._pop_sections(raw)
            global_workloads = self._pop_workloads(raw)

            if raw:
                msg = 'Invalid top level agenda entry(ies): "{}"'
                raise ConfigError(msg.format('", "'.join(raw.keys())))

            sect_ids, wkl_ids = self._collect_ids(sections, global_workloads)
            self._process_global_workloads(state, global_workloads, wkl_ids)
            self._process_sections(state, sections, sect_ids, wkl_ids)

            state.agenda = source

        except (ConfigError, SerializerSyntaxError) as e:
            raise ConfigError('Error in "{}":\n\t{}'.format(source, str(e)))

    def _populate_and_validate_config(self, state, raw, source):
        for name in ['config', 'global']:
            entry = raw.pop(name, None)
            if entry is None:
                continue

            if not isinstance(entry, dict):
                msg = 'Invalid entry "{}" - must be a dict'
                raise ConfigError(msg.format(name))

            if 'run_name' in entry:
                state.run_config.set('run_name', entry.pop('run_name'))

            state.load_config(entry, '{}/{}'.format(source, name), wrap_exceptions=False)

    def _pop_sections(self, raw):
        sections = raw.pop("sections", [])
        if not isinstance(sections, list):
            raise ConfigError('Invalid entry "sections" - must be a list')
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

            section = _construct_valid_entry(section, seen_sect_ids,
                                             "s", state.jobs_config)
            state.jobs_config.add_section(section, workloads)


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
    except SerializerSyntaxError as e:
        raise ConfigError('Error parsing {} {}: {}'.format(error_name, filepath, e))
    if not isinstance(raw, dict):
        message = '{} does not contain a valid {} structure; top level must be a dict.'
        raise ConfigError(message.format(filepath, error_name))
    return raw


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
    names = [cfg_point.name,] + cfg_point.aliases

    entries = [toggle_set(raw.pop(n)) for n in names if n in raw]

    # Make sure none of the specified aliases conflict with each other
    to_check = [e for e in entries]
    while len(to_check) > 1:
        check_entry = to_check.pop()
        for e in to_check:
            conflicts = check_entry.conflicts_with(e)
            if conflicts:
                msg = '"{}" and "{}" have conflicting entries: {}'
                conflict_string  = ', '.join('"{}"'.format(c.strip("~"))
                                             for c in conflicts)
                raise ConfigError(msg.format(check_entry, e, conflict_string))

    if entries:
        raw['augmentations'] = reduce(lambda x, y: x.merge_with(y), entries)


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
    for name, cfg_point in JobSpec.configuration.iteritems():
        value = pop_aliased_param(cfg_point, raw)
        if value is not None:
            value = cfg_point.kind(value)
            cfg_point.validate_value(name, value)
            workload_entry[name] = value

    if "augmentations" in workload_entry:
        jobs_config.update_augmentations(workload_entry["augmentations"])

    # error if there are unknown workload_entry
    if raw:
        msg = 'Invalid entry(ies) in "{}": "{}"'
        raise ConfigError(msg.format(workload_entry['id'], ', '.join(raw.keys())))

    return workload_entry


def _collect_valid_id(entry_id, seen_ids, entry_type):
    if entry_id is None:
        return
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
    if isinstance(workload, basestring):
        workload = {'name': workload}
    elif not isinstance(workload, dict):
        raise ConfigError('Invalid workload entry: "{}"')
    return workload


def _process_workload_entry(workload, seen_workload_ids, jobs_config):
    workload = _get_workload_entry(workload)
    workload = _construct_valid_entry(workload, seen_workload_ids,
                                      "wk", jobs_config)
    return workload

