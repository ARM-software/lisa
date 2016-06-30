#    Copyright 2016 ARM Limited
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


class Node(object):
    @property
    def is_leaf(self):
        return not bool(self.children)

    def __init__(self, value, parent=None):
        self.workloads = []
        self.parent = parent
        self.children = []
        self.config = value

    def add_section(self, section):
        new_node = Node(section, parent=self)
        self.children.append(new_node)
        return new_node

    def add_workload(self, workload):
        self.workloads.append(workload)

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
