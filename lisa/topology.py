# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import abc
from collections import namedtuple, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from itertools import product, chain, zip_longest
import logging
import operator
from operator import itemgetter, attrgetter
import warnings
import re

import pandas as pd
import numpy as np

from devlib.utils.misc import mask_to_list, ranges_to_list
from devlib.exception import TargetStableError

from lisa.utils import Loggable, Serializable, memoized, groupby, get_subclasses, get_common_ancestor, deprecate, grouper, compose, deduplicate, take

"""
Various topology descriptions
"""


class _DefaultName:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        name = self._name
        if name is None:
            return '{}({})'.format(
                self.__class__.__qualname__,
                ','.join(
                    '{}={}'.format(attr, val)
                    for attr, val in sorted(self.__dict__.items())
                )
            )
        else:
            return name


class OPP(_DefaultName):
    def __init__(self, voltage, freq, dynamic_power, leakage_power=None, latencies=None, name=None):
        super().__init__(name=name)

        self.voltage = voltage
        self.freq = freq
        self.dynamic_power = dynamic_power
        self.leakage_power = leakage_power or 0
        self.latencies = latencies or (0, 0)

    @property
    def power(self):
        return self.dynamic_power + self.leakage_power

    @classmethod
    def from_capacitance(cls, capacitance, voltage, freq, **kwargs):
        if 'dynamic_power' not in kwargs:
            kwargs['dynamic_power'] = cls._compute_dynamic_power(
                capacitance=capacitance,
                voltage=voltage,
                freq=freq,
            )
            capacitance * v * v * freq

        return cls(
            voltage=voltage,
            freq=freq,
            **kwargs
        )


    @staticmethod
    def _compute_dynamic_power(capacitance, voltage, freq):
        return capacitance * (voltage ** 2) * freq


class _NodeBase(Loggable):
    NAME = 'node'

    def __init__(self, name, children=None):
        self._name = name
        self.parents = []
        # Use an immutable type so that we cannot mistakenly change the
        # children without fixing up the parents link too
        self._children = tuple()
        self.children = children or []

    @property
    def name(self):
        name = self._name
        if name is None:
            return self._get_default_name()
        else:
            return name

    def _get_default_name(self):
        def get_name(dom):
            name = dom._name
            if name:
                return name
            else:
                try:
                    parent = dom.parents[0]
                except IndexError:
                    return '<root>'
                else:
                    child_nr = parent.children.index(dom)
                    return '<child{}>'.format(child_nr)

        ancestors = [self] + self.ancestors
        path = list(map(get_name, ancestors))

        return '{}:{}'.format(
            self.NAME,
            '-'.join(map(str, reversed(path)))
        )

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, val):
        """
        Direct assignment to ``children`` will ensure the ``parents`` link is
        preserved correctly.
        """
        old_children = set(self._children)
        new_val = tuple(val)
        self._children = new_val
        new_val = set(new_val)

        for child in (old_children - new_val):
            child.parents.remove(self)

        for child in (new_val - old_children):
            if self not in child.parents:
                child.parents.append(self)

    def _get_ancestors(self, visited):
        if self in visited:
            return []
        else:
            visited.add(self)

            ancestors = list(chain.from_iterable(
                parent._get_ancestors(visited)
                for parent in self.parents
            ))
            return [self] + ancestors

    @property
    def ancestors(self):
        ancestors = self._get_ancestors(set())
        ancestors.remove(self)
        return ancestors

    @staticmethod
    def _indent(s):
        return s.replace('\n', '\n' + ' ' * 4)

    def __str__(self):
        return self._indent(self._str())

    def _str(self):
        indent = self._indent

        def get_str(val):
            if isinstance(val, Mapping):
                def f(val):
                    if val:
                        return '{{{nl}{items}{nl}}}'.format(
                            items=indent(
                                ',\n'.join(
                                    '{}: {}'.format(get_str(key), get_str(val))
                                    for key, val in val.items()
                                )
                            ),
                            nl=indent('\n'),
                        )
                    else:
                        return '{}'
            elif isinstance(val, Sequence) and not isinstance(val, str):
                def f(val):
                    if val:
                        return '[{nl}{items}{nl}]'.format(
                            items=indent(
                                ',\n'.join(
                                    map(get_str, val)
                                )
                            ),
                            nl=indent('\n'),
                        )
                    else:
                        return '[]'
            else:
                try:
                    f = val.__class__._str
                except AttributeError:
                    f = repr
                f = compose(f, indent)

            return f(val)

        hidden = {'parents'}
        def format_val(attr, val):
            if attr == 'domains':
                return {
                    key: dom.name
                    for key, dom in val.items()
                }
            else:
                return val

        return '{cls}(\n{param}\n)'.format(
            cls=self.__class__.__qualname__,
            param = ',\n'.join(
                '{attr}={val}'.format(
                    # Remove leading underscore for better readability
                    attr=attr.lstrip('_'),
                    val=get_str(format_val(attr, val))
                )
                for attr, val in self.__dict__.items()
                if attr not in hidden
            ),
        )

    def iter_nodes(self):
        """Iterate over nodes depth-first, post-order"""
        return self.iter_dfs(only_leaves=False)

    def _get_root_leaves(self, edge_attr):
        node_list = getattr(self, edge_attr)
        if node_list:
            return deduplicate(
                subnode
                for node in node_list
                for subnode in node._get_root_leaves(edge_attr)
            )
        else:
            return [self]

    @property
    def roots(self):
        """
        Roots of the domain graph
        """
        return self._get_root_leaves('parents')

    @property
    def leaves(self):
        """
        Leaves of the domain graph
        """
        return self._get_root_leaves('children')


class DomainStateBase(_NodeBase, abc.ABC):
    EVALUATION_ORDER = 'up'

    def __init__(self, domain, children):
        super().__init__(
            children=children,
            name=None,
        )
        self.domain = domain

    @property
    def name(self):
        return self.domain.name

    def __getattr__(self, attr):
        return getattr(self.domain, attr)

    @abc.abstractmethod
    def compute(self, input_):
        pass


class DomainBase(_NodeBase):
    NAME = 'domain'
    DERIVE_FROM = None
    STATE_CLS = DomainStateBase

    def __init__(self, name=None, children=None, topo_nodes=None):
        self._provided_children = children is not None

        super().__init__(
            name=name,
            children=children,
        )
        self.topo_nodes = list(topo_nodes or [])

    @property
    def _name(self):
        """
        Inherit from the topo name if no name was provided.
        """
        real_name = self.__dict__['_name']
        if real_name:
            return real_name
        else:
            try:
                topo_node, = self.topo_nodes
            except ValueError:
                return None
            else:
                return topo_node._name

    @_name.setter
    def _name(self, val):
        self.__dict__['_name'] = val

    @classmethod
    def make_default_domain(cls, topo_node):
        return cls(topo_nodes=[topo_node])

    def add_topo_node(self, node):
        if node not in self.topo_nodes:
            self.topo_nodes.append(node)

    @classmethod
    def finalize_graph(cls, dom_nodes):
        for dom_node in dom_nodes:
            # Attach as children domain the domains of the children of the
            # topo nodes
            if not dom_node._provided_children:
                topo_children = set(
                    topo_subnode
                    for topo_node in dom_node.topo_nodes
                    for topo_subnode in topo_node.children
                )
                dom_node_children = set(
                    dom_node
                    for topo_node in topo_children
                    for dom_node in (
                        dom_node
                        for dom_node in topo_node.domains.values()
                        if isinstance(dom_node, cls)
                    )
                )

                dom_node.children = sorted(
                    dom_node_children,
                    key=attrgetter('name')
                )

class DomainGraph:
    def __init__(self, roots):
        self.roots = roots

    @property
    def leaves(self):
        return deduplicate(
            dom
            # Backtrack up to the roots to have visibility on the full graph
            for root in self.roots
            for dom in root.leaves
        )

    @property
    @memoized
    def domain_cls(self):
        cls = get_common_ancestor(map(type, self.roots))
        assert issubclass(cls, DomainBase)
        return cls

    @property
    @memoized
    def name(self):
        return self.domain_cls.NAME

    @property
    def input_domains(self):
        order = self.domain_cls.STATE_CLS.EVALUATION_ORDER
        if order == 'down':
            return self.roots
        else:
            return self.leaves

    def _make_states(self):
        visited = {}
        def make_state(dom):
            try:
                return visited[dom]
            except KeyError:
                state = dom.STATE_CLS(
                    domain=dom,
                    children=[
                        make_state(dom)
                        for dom in dom.children
                    ],
                )
                visited[dom] = state
                return state

        for dom in self.roots:
            make_state(dom)
        return visited

    def compute(self, inputs):
        states = self._make_states()
        input_domains = self.input_domains

        for dom in input_domains:
            states[dom].compute(inputs[dom.name])

        return {
            dom: states[dom]
            for dom in input_domains
        }

class PowerDomainState(DomainStateBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visited=False

    def compute(self, input_, idt=0):
        print(idt * 2 * ' ' + '{} ({})=> {}'.format(self.name, self.visited, input_))

        self.visited=True
        for state in self.parents:
            state.compute(input_, idt=idt+1)


class PowerDomain(DomainBase):
    """
    :Edges: A child cannot be powered if the parent is not powered too.
    :Nodes: The voltage for all managed :class:`TopoNodeBase` is the same.


    .. note:: Following kernel's ``genpd`` kernel framework, the only link
        between child and parent domains is that the parent domain must have a
        non-zero voltage in order to have a non-zero voltage in the child
        domain.
    """
    NAME = 'power'
    DERIVE_FROM = 'freq'
    STATE_CLS = PowerDomainState

    def __init__(self, voltages=None, **kwargs):
        super().__init__(**kwargs)
        self.voltages = voltages


class FreqDomain(DomainBase):
    """
    :Edges: No meaning.
    :Nodes: The frequency for all managed :class:`TopoNodeBase` is the same.
    """
    NAME = 'freq'
    DERIVE_FROM = 'power'


class TopoNodeBase(_NodeBase):
    NAME = 'topo_node'

    def __init__(self, name, domains=None, children=None):
        super().__init__(
            name=name,
            children=children,
        )
        domains = domains or {}
        for dom in domains.values():
            dom.add_topo_node(self)

        self.domains = domains


class ComponentNode(TopoNodeBase):
    NAME = 'component'

    def __init__(self, children, name=None, domains=None):
        # A domain without children is illegal. It must either contain other
        # domains or CPU nodes
        children = list(children)
        assert children
        super().__init__(
            name=name,
            domains=domains,
            children=children,
        )

    @property
    @memoized
    def cpus(self):
        """
        Tuple of CPUs this node is spanning over, recursively.
        """
        return tuple(sorted(set(
            cpu
            for node in self.children
            for cpu in node.cpus
        )))

    def iter_dfs(self, only_leaves=False):
        return self._iter_dfs(visited=set(), only_leaves=only_leaves)

    def _iter_dfs(self, visited, only_leaves=False):
        # Avoid getting caught in case the graph has loops, which is allowed
        if self in visited:
            return
        else:
            if not only_leaves:
                yield from (
                    subnode
                    for node in self.children
                    for subnode in node.iter_dfs(only_leaves=only_leaves)
                )

            visited.add(self)
            yield self


class CPUNode(TopoNodeBase):
    NAME = 'cpu'

    def __init__(self, cpu, name=None, domains=None):
        # Always provide a name rather than using _get_default_name() since it
        # is user-provided information, which can therfore be used to create
        # default domain names
        name = name or '{}{}'.format(self.NAME, cpu)
        super().__init__(
            name=name,
            domains=domains,
        )
        self.cpu = cpu

    @property
    def cpus(self):
        return [self.cpu]

    def iter_dfs(self, only_leaves=False):
        yield self


class TopoGraph:
    def __init__(self, roots, ensure_domains=None):
        self.roots = roots

        nodes = set(root.iter_dfs())

        self._add_missing_domains(nodes, ensure_domains)

        dom_nodes = set(
            (name, dom_node)
            for node in nodes
            for name, dom_node in node.domains.items()
        )

        dom_node_groups = {}
        for name, dom_node in dom_nodes:
            dom_node_groups.setdefault(name, set()).add(dom_node)

        dom_graphs = {}
        # Finalize each domain, per class
        for group_name, dom_nodes in dom_node_groups.items():
            dom_nodes = list(dom_nodes)
            dom_cls = get_common_ancestor(map(type, dom_nodes))
            dom_cls.finalize_graph(dom_nodes)
            dom_graphs[dom_cls.NAME] = DomainGraph(
                roots=dom_nodes[0].roots,
            )

        self.domain_graphs = dom_graphs

    @staticmethod
    def _mirror_domain(dom_cls, source_dom):
        """
        Mirror the source domain
        """
        name = 'derived-from:{}'.format(source_dom.name)
        dom = dom_cls(name=name)
        dom_cls_name = dom.NAME

        for topo_node in source_dom.topo_nodes:
            try:
                existing_dom = topo_node.domains[dom_cls_name]
            # If the node did not have a domain registered
            # already, add this one
            except KeyError:
                dom.add_topo_node(topo_node)
                topo_node.domains[dom_cls_name] = dom
            # The user provided a domain explicitly, so we skip it
            else:
                continue

        return dom

    @classmethod
    def _add_missing_domains(cls, nodes, ensure_domains):
        # Add the missing domains by either deriving it from a provided one, or
        # creating one for the topo node
        dom_cls_map = {
            dom_cls.NAME: dom_cls
            for dom_cls in get_subclasses(DomainBase, only_leaves=True)
            if (
                (ensure_domains is None) or
                (dom_cls.NAME in ensure_domains)
            )
        }

        mirror_map = {}

        for node in nodes:
            provided_domains = node.domains.copy()

            missing_doms = dom_cls_map.keys() - node.domains.keys()
            for dom_cls_name in missing_doms:
                dom_cls = dom_cls_map[dom_cls_name]

                # Only derive from domains that were provided by the user, not
                # the default ones
                source_dom = provided_domains.get(dom_cls.DERIVE_FROM)
                if source_dom is None:
                    dom = dom_cls.make_default_domain(node)
                else:
                    # Mirror the source domain
                    dom = cls._mirror_domain(dom_cls, source_dom)
                    mirror_map[dom] = source_dom

                node.domains[dom_cls_name] = dom

        # Now that all the domains are created, add the edges of the mirrors
        # like their source
        for dom, source_dom in mirror_map.items():
            children = [
                topo_node.domains.get(dom_cls_name)
                for child in source_dom.children
                for topo_node in child.topo_nodes
            ]
            children = [child for child in children if child is not None]
            dom.children = deduplicate(children)

    def __str__(self):
        return '\n'.join(map(str, self.roots))




freq1 = FreqDomain(name='freq1')
# freq1 = FreqDomain()

root=ComponentNode(
    name='root',
    children=[
        ComponentNode(
            # name='cluster0',
            children=[
                CPUNode(cpu=0),
                CPUNode(cpu=1),
            ],
            domains=dict(freq=freq1),
        ),
        ComponentNode(
            name='cluster1',
            children=[
                CPUNode(cpu=2, domains=dict(freq=freq1)),
                # CPUNode(cpu=2),
                CPUNode(cpu=3),
            ]
        ),
    ]
)

g = TopoGraph(
    roots=[
        ComponentNode(
            name='preroot',
            children=[root]
        ),
    ],
)
print(g)
dom_g = g.domain_graphs['power']
print(dom_g.roots[0].name)

print('computing state')
inputs = defaultdict(lambda: None)
inputs = {
    'cpu0': 0,
    'cpu1': 1,
    'cpu2': 2,
    'cpu3': 2,
}
print(dom_g.compute(inputs))

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
