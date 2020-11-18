#    Copyright 2016-2018 ARM Limited
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

import logging

from wa.utils import log


logger = logging.getLogger('config')


class JobSpecSource(object):

    kind = ""

    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent
        self._log_self()

    @property
    def id(self):
        return self.config['id']

    @property
    def name(self):
        raise NotImplementedError()

    def _log_self(self):
        logger.debug('Creating {} node'.format(self.kind))
        with log.indentcontext():
            for key, value in self.config.items():
                logger.debug('"{}" to "{}"'.format(key, value))


class WorkloadEntry(JobSpecSource):
    kind = "workload"

    @property
    def name(self):
        if self.parent.id == "global":
            return 'workload "{}"'.format(self.id)
        else:
            return 'workload "{}" from section "{}"'.format(self.id, self.parent.id)


class SectionNode(JobSpecSource):

    kind = "section"

    @property
    def name(self):
        if self.id == "global":
            return "globally specified configuration"
        else:
            return 'section "{}"'.format(self.id)

    @property
    def is_leaf(self):
        return not bool(self.children)

    def __init__(self, config, parent=None, group=None):
        super(SectionNode, self).__init__(config, parent=parent)
        self.workload_entries = []
        self.children = []
        self.group = group

    def add_section(self, section, group=None):
        # Each level is the same group, only need to check first
        if not self.children or group == self.children[0].group:
            new_node = SectionNode(section, parent=self, group=group)
            self.children.append(new_node)
        else:
            for child in self.children:
                new_node = child.add_section(section, group)
        return new_node

    def add_workload(self, workload_config):
        self.workload_entries.append(WorkloadEntry(workload_config, self))

    def descendants(self):
        for child in self.children:
            for n in child.descendants():
                yield n
            yield child

    def ancestors(self):
        if self.parent is not None:
            yield self.parent
            for ancestor in self.parent.ancestors():
                yield ancestor

    def leaves(self):
        if self.is_leaf:
            yield self
        else:
            for n in self.descendants():
                if n.is_leaf:
                    yield n
