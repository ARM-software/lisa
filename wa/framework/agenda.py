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
from copy import copy
from collections import OrderedDict, defaultdict

from wa.framework.exception import ConfigError, SerializerSyntaxError
from wa.utils.serializer import yaml
from wa.utils import counter


def get_aliased_param(d, aliases, default=None, pop=True):
    alias_map = [i for i, a in enumerate(aliases) if a in d]
    if len(alias_map) > 1:
        message = 'Only one of {} may be specified in a single entry'
        raise ConfigError(message.format(aliases))
    elif alias_map:
        if pop:
            return d.pop(aliases[alias_map[0]])
        else:
            return d[aliases[alias_map[0]]]
    else:
        return default


class AgendaEntry(object):

    def to_dict(self):
        return copy(self.__dict__)

    def __str__(self):
        name = self.__class__.__name__.split('.')[-1]
        if hasattr(self, 'id'):
            # pylint: disable=no-member
            return '{}({})'.format(name, self.id)
        else:
            return name

    __repr__ = __str__


class AgendaWorkloadEntry(AgendaEntry):
    """
    Specifies execution of a workload, including things like the number of
    iterations, device runtime_parameters configuration, etc.

    """

    def __init__(self, **kwargs):
        super(AgendaWorkloadEntry, self).__init__()
        self.id = kwargs.pop('id')
        self.workload_name = get_aliased_param(kwargs, ['workload_name', 'name'])
        if not self.workload_name:
            raise ConfigError('No workload name specified in entry {}'.format(self.id))
        self.label = kwargs.pop('label', self.workload_name)
        self.number_of_iterations = kwargs.pop('iterations', None)
        self.boot_parameters = get_aliased_param(kwargs,
                                                 ['boot_parameters', 'boot_params'],
                                                 default=OrderedDict())
        self.runtime_parameters = get_aliased_param(kwargs,
                                                    ['runtime_parameters', 'runtime_params'],
                                                    default=OrderedDict())
        self.workload_parameters = get_aliased_param(kwargs,
                                                     ['workload_parameters', 'workload_params', 'params'],
                                                     default=OrderedDict())
        self.instrumentation = kwargs.pop('instrumentation', [])
        self.flash = kwargs.pop('flash', OrderedDict())
        self.classifiers = kwargs.pop('classifiers', OrderedDict())
        if kwargs:
            raise ConfigError('Invalid entry(ies) in workload {}: {}'.format(self.id, ', '.join(kwargs.keys())))


class AgendaSectionEntry(AgendaEntry):
    """
    Specifies execution of a workload, including things like the number of
    iterations, device runtime_parameters configuration, etc.

    """

    def __init__(self, agenda, **kwargs):
        super(AgendaSectionEntry, self).__init__()
        self.id = kwargs.pop('id')
        self.number_of_iterations = kwargs.pop('iterations', None)
        self.boot_parameters = get_aliased_param(kwargs,
                                                 ['boot_parameters', 'boot_params'],
                                                 default=OrderedDict())
        self.runtime_parameters = get_aliased_param(kwargs,
                                                    ['runtime_parameters', 'runtime_params', 'params'],
                                                    default=OrderedDict())
        self.workload_parameters = get_aliased_param(kwargs,
                                                     ['workload_parameters', 'workload_params'],
                                                     default=OrderedDict())
        self.instrumentation = kwargs.pop('instrumentation', [])
        self.flash = kwargs.pop('flash', OrderedDict())
        self.classifiers = kwargs.pop('classifiers', OrderedDict())
        self.workloads = []
        for w in kwargs.pop('workloads', []):
            self.workloads.append(agenda.get_workload_entry(w))
        if kwargs:
            raise ConfigError('Invalid entry(ies) in section {}: {}'.format(self.id, ', '.join(kwargs.keys())))

    def to_dict(self):
        d = copy(self.__dict__)
        d['workloads'] = [w.to_dict() for w in self.workloads]
        return d


class AgendaGlobalEntry(AgendaEntry):
    """
    Workload configuration global to all workloads.

    """

    def __init__(self, **kwargs):
        super(AgendaGlobalEntry, self).__init__()
        self.number_of_iterations = kwargs.pop('iterations', None)
        self.boot_parameters = get_aliased_param(kwargs,
                                                 ['boot_parameters', 'boot_params'],
                                                 default=OrderedDict())
        self.runtime_parameters = get_aliased_param(kwargs,
                                                    ['runtime_parameters', 'runtime_params', 'params'],
                                                    default=OrderedDict())
        self.workload_parameters = get_aliased_param(kwargs,
                                                     ['workload_parameters', 'workload_params'],
                                                     default=OrderedDict())
        self.instrumentation = kwargs.pop('instrumentation', [])
        self.flash = kwargs.pop('flash', OrderedDict())
        self.classifiers = kwargs.pop('classifiers', OrderedDict())
        if kwargs:
            raise ConfigError('Invalid entries in global section: {}'.format(kwargs))


class Agenda(object):

    def __init__(self, source=None):
        self.filepath = None
        self.config = None
        self.global_ = None
        self.sections = []
        self.workloads = []
        self._seen_ids = defaultdict(set)
        if source:
            try:
                counter.reset('section')
                counter.reset('workload')
                self._load(source)
            except (ConfigError, SerializerSyntaxError, SyntaxError), e:
                raise ConfigError(str(e))

    def add_workload_entry(self, w):
        entry = self.get_workload_entry(w)
        self.workloads.append(entry)

    def get_workload_entry(self, w):
        if isinstance(w, basestring):
            w = {'name': w}
        if not isinstance(w, dict):
            raise ConfigError('Invalid workload entry: "{}" in {}'.format(w, self.filepath))
        self._assign_id_if_needed(w, 'workload')
        return AgendaWorkloadEntry(**w)

    def expand(self, target):
        # TODO: currently a no-op, this method is here to support future features, such
        #       as section cross products and sweeps.
        pass

    def _load(self, source):  # pylint: disable=too-many-branches
        try:
            raw = self._load_raw_from_source(source)
        except SerializerSyntaxError as e:
            name = getattr(source, 'name', '')
            raise ConfigError('Error parsing agenda {}: {}'.format(name, e))
        if not isinstance(raw, dict):
            message = '{} does not contain a valid agenda structure; top level must be a dict.'
            raise ConfigError(message.format(self.filepath))
        for k, v in raw.iteritems():
            if k == 'config':
                if not isinstance(v, dict):
                    raise ConfigError('Invalid agenda: "config" entry must be a dict')
                self.config = v
            elif k == 'global':
                self.global_ = AgendaGlobalEntry(**v)
            elif k == 'sections':
                self._collect_existing_ids(v, 'section')
                for s in v:
                    if not isinstance(s, dict):
                        raise ConfigError('Invalid section entry: "{}" in {}'.format(s, self.filepath))
                    self._collect_existing_ids(s.get('workloads', []), 'workload')
                for s in v:
                    self._assign_id_if_needed(s, 'section')
                    self.sections.append(AgendaSectionEntry(self, **s))
            elif k == 'workloads':
                self._collect_existing_ids(v, 'workload')
                for w in v:
                    self.workloads.append(self.get_workload_entry(w))
            else:
                raise ConfigError('Unexpected agenda entry "{}" in {}'.format(k, self.filepath))

    def _load_raw_from_source(self, source):
        if hasattr(source, 'read') and hasattr(source, 'name'):  # file-like object
            self.filepath = source.name
            raw = yaml.load(source)
        elif isinstance(source, basestring):
            if os.path.isfile(source):
                self.filepath = source
                with open(source, 'rb') as fh:
                    raw = yaml.load(fh)
            else:  # assume YAML text
                raw = yaml.loads(source)
        else:
            raise ConfigError('Unknown agenda source: {}'.format(source))
        return raw

    def _collect_existing_ids(self, ds, pool):
        # Collection needs to take place first  so that auto IDs can be
        # correctly assigned, e.g. if someone explicitly specified an ID
        # of '1' for one of the workloads.
        for d in ds:
            if isinstance(d, dict) and 'id' in d:
                did = str(d['id'])
                if did in self._seen_ids[pool]:
                    raise ConfigError('Duplicate {} ID: {}'.format(pool, did))
                self._seen_ids[pool].add(did)

    def _assign_id_if_needed(self, d, pool):
        # Also enforces string IDs
        if d.get('id') is None:
            did = str(counter.next(pool))
            while did in self._seen_ids[pool]:
                did = str(counter.next(pool))
            d['id'] = did
            self._seen_ids[pool].add(did)
        else:
            d['id'] = str(d['id'])
