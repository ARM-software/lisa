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

from collections import namedtuple, OrderedDict
from itertools import product
import logging
import operator
import re

import pandas as pd
import numpy as np

from devlib.utils.misc import memoized, mask_to_list
from devlib import TargetError

"""Classes for modeling and estimating energy usage of CPU systems"""

def read_multiple_oneline_files(target, glob_patterns):
    """
    Quickly read many single-line files that match a glob pattern

    Finds all the files that match any of the glob patterns and, assuming that
    they each contain exactly 1 line of text, read them all at once. When the
    target or connection is slow this saves a lot of time when reading a large
    number of files.

    :param target: devlib target object to read from
    :param glob_pattern: Unix glob pattern matching the files to read
    :returns: A dictionary mapping matched paths to the values read. ``{}`` if
              no paths matched the globs.
    """
    try:
        paths = target.execute('find ' + ' '.join(glob_patterns)).split()
    except TargetError:
        return {}

    cmd = 'cat ' + ' '.join(glob_patterns)
    contents = target.execute(cmd).splitlines()

    return dict(zip(paths, contents))

class EnergyModelCapacityError(Exception):
    """Used by :meth:`EnergyModel.get_optimal_placements`"""
    pass

class ActiveState(namedtuple('ActiveState', ['capacity', 'power'])):
    """Represents power and compute capacity at a given frequency

    :param capacity: Relative compute capacity at frequency
    :param power: Power usage at frequency
    """
    def __new__(cls, capacity=None, power=None):
        return super(ActiveState, cls).__new__(cls, capacity, power)

class _CpuTree(object):
    """Internal class. Abstract representation of a CPU topology.

    Each node contains either a single CPU or a set of child nodes.
    """
    def __init__(self, cpu, children):
        if (cpu is None) == (children is None):
            raise ValueError('Provide exactly one of: cpu or children')

        self.parent = None
        self.cpu = cpu

        if cpu is not None:
            self.cpus = (cpu,)
            self.children = []
        else:
            if len(children) == 0:
                raise ValueError('children cannot be empty')
            self.cpus = tuple(sorted(set().union(*[n.cpus for n in children])))
            self.children = children
            for child in children:
                child.parent = self

        self.name = None

    def __repr__(self):
        name_bit = ''
        if self.name:
            name_bit = 'name="{}", '.format(self.name)

        if self.children:
            return '{}({}children={})'.format(
                self.__class__.__name__, name_bit, self.children)
        else:
            return '{}({}cpus={})'.format(
                self.__class__.__name__, name_bit, self.cpus)

    def _iter(self, include_non_leaves):
        for child in self.children:
            for child_i in child._iter(include_non_leaves):
                yield child_i
        if include_non_leaves or not self.children:
            yield self

    def iter_nodes(self):
        """Iterate over nodes depth-first, post-order"""
        return self._iter(True)

    def iter_leaves(self):
        """Iterate over leaves"""
        return self._iter(False)

class EnergyModelNode(_CpuTree):
    """Describes topology and energy data for an EnergyModel.

    Represents a CPU topology with energy data. The active and idle state data
    represents the power usage of just the hardware resources of this topology
    level, not its children. e.g. If the node represents a cluster, the power
    numbers should not include power used by the CPU - that power should be
    included the data of the child nodes.

    Exactly one of ``cpu`` and ``children`` must be given.

    :param active_states: Dict mapping frequencies to :class:`ActiveState`
                          values. Compute capacity data is optional for
                          non-leaf nodes.
    :param idle_states: Dict mapping idle state names to power usage values
    :param cpu: The CPU this node represents. If provided, this is a leaf node.
    :type cpus: tuple(int)
    :param children: Non-empty list of child :class:`EnergyModelNode` objects
    :param name: Optional human-readable name for this node. Leaf (CPU) nodes
                 have a default name of "cpuN" where N is the cpu number.

    :ivar cpus: CPUs contained in this node. Includes those of child nodes.
    :ivar cpu: For convenience, this holds the single CPU contained by leaf
               nodes. ``None`` for non-leaf nodes.
    """
    def __init__(self, active_states, idle_states,
                 cpu=None, children=None, name=None):
        super(EnergyModelNode, self).__init__(cpu, children)

        self._log = logging.getLogger('EnergyModel')

        def is_monotonic(l, decreasing=False):
            op = operator.ge if decreasing else operator.le
            return all(op(a, b) for a, b in zip(l, l[1:]))

        if active_states:
            # Sanity check for active_states's frequencies
            freqs = active_states.keys()
            if not is_monotonic(freqs):
                self._log.warning(
                    'Active states frequencies are expected to be '
                    'monotonically increasing. Freqs: {}'.format(freqs))

            # Sanity check for active_states's powers
            power_vals = [s.power for s in active_states.values()]
            if not is_monotonic(power_vals):
                self._log.warning(
                    'Active states powers are expected to be '
                    'monotonically increasing. Values: {}'.format(power_vals))

        # Sanity check for idle_states powers
        if idle_states:
            power_vals = idle_states.values()
            if not is_monotonic(power_vals, decreasing=True):
                self._log.warning(
                    'Idle states powers are expected to be '
                    'monotonically decreasing. Values: {}'.format(power_vals))

        if cpu is not None and not name:
            name = 'cpu' + str(cpu)

        self.name = name
        self.active_states = active_states
        self.idle_states = idle_states

    @property
    def max_capacity(self):
        """Compute capacity at highest frequency"""
        return max(s.capacity for s in self.active_states.values())

class EnergyModelRoot(EnergyModelNode):
    """
    Convenience class for root of an EnergyModelNode tree.

    Just like EnergyModelNode except that ``active_states`` and ``idle_states``
    aren't required.
    """
    def __init__(self, active_states=None, idle_states=None,
                 cpu=None, children=None, name=None):
        return super(EnergyModelRoot, self).__init__(
            active_states, idle_states, cpu, children, name)

class PowerDomain(_CpuTree):
    """Describes the power domain hierarchy for an EnergyModel.

    Power domains are a description of the topological dependencies in hardware
    for entering idle states. "Composite" states such as cluster-sleep states
    require a set of CPUs to all be idle before that state can be entered. In
    that case those CPUs can be grouped into a power domain, and that composite
    state attached to the power domain. Note that cpuidle is not aware of these
    dependencies; they are typically handled by the platform firmware.

    Exactly one of ``cpu`` and ``children`` must be given. That is, leaves of
    the PowerDomain tree always contain exactly one CPU - each CPU is
    represented as being in a power domain of its own. This represents the
    assumption that all CPUs have at least one idle state (such as ARM WFI) that
    they can enter independently of other CPUs.

    :param idle_states: List of names of idle states for this power domain. Does
                        not store power data - these names are used as keys into
                        the ``idle_states`` field of :class:`EnergyModelNode`
                        objects.
    :type idle_states: list(str)
    :param cpu: The CPU this node represents. If provided, this is a leaf node.
    :type cpu:  int
    :param children: Non-empty list of child :class:`PowerDomain` objects
    :type children:  list(PowerDomain)

    :ivar cpus: CPUs contained in this node. Includes those of child nodes.
    :type cpus: tuple(int)
    """
    def __init__(self, idle_states, cpu=None, children=None):
        if idle_states is None:
            raise ValueError('idle_states cannot be None (but may be empty)')
        super(PowerDomain, self).__init__(cpu, children)
        self.idle_states = idle_states

class EnergyModel(object):
    """Represents hierarchical CPU topology with power and capacity data

    An energy model consists of

    - A CPU topology, representing the physical (cache/interconnect) topology of
      the CPUs.  Each node stores the energy usage of that node's hardware when
      it is in each active or idle state. They also store a compute capacity at
      each frequency, but this is only meaningful for leaf nodes (CPUs) and may
      be None at higher levels. These capacity values are relative; the maximum
      capacity would usually be 1024, the value of SCHED_CAPACITY_SCALE in the
      Linux kernel scheduler.  Use EnergyModelNodes to describe this.

    - A power domain topology, representing the hierarchy of areas that can be
      powered down (idled).
      The power domains are a single tree. Leaf nodes must contain exactly one
      CPU and the root node must indirectly contain every CPU. Each power domain
      has a list (maybe empty) of names of idle states that that domain can
      enter.
      Use PowerDomains to describe this.

    - A set of frequency domains, representing groups of CPUs whose clock
      frequencies must be equal (probably because they share a clock). The
      frequency domains must be a partition of the CPUs.

    :ivar cpu_nodes: List of leaf (CPU) :class:`EnergyModelNode`
    :ivar cpus: List of logical CPU numbers in the system

    :param root_node: Root of :class:`EnergyModelNode` tree
    :param root_power_domain: Root of :class:`PowerDomain` tree
    :param freq_domains: Collection of collections of logical CPU numbers
                         representing frequency (clock) domains.

    .. note::
      The most signficant shortcomings of the model are:

        1. Voltage domains are assumed to be congruent to frequency domains

        2. Idle state power is assumed to be independent of voltage

        3. Temperature is ignored entirely

    .. _cpu-utils:

    .. admonition:: ``cpu_utils``: CPU util distributions

        Used throughout this module: A ``cpu_utils`` is a list ``u`` where
        ``u[N]`` is the sum of the frequency-invariant, capacity-invariant
        utilization of tasks placed on CPU N. That is, the quantity represented
        by a CPU runqueue's util_avg in the Linux kernel scheduler's
        load-tracking system with EAS features enabled.

        The range of utilization values is 0 -
        :attr:`EnergyModel.capacity_scale`.

        This represents a static utilization, assuming that tasks don't change
        in size (for example representing a set of fixed periodic RT-App
        workloads). For workloads that change over time, a series of
        ``cpu_utils`` items would be needed to describe the utilization, with a
        distinct estimation for each item in the series.
    """

    capacity_scale = 1024
    """The relative computational capacity of the most powerful CPU at its
    highest available frequency.
    """

    def __init__(self, root_node, root_power_domain, freq_domains):
        self.cpus = root_node.cpus
        if self.cpus != tuple(range(len(self.cpus))):
            raise ValueError('CPU IDs [{}] are sparse'.format(self.cpus))

        # Check that freq_domains is a partition of the CPUs
        fd_intersection = set().intersection(*freq_domains)
        if fd_intersection:
            raise ValueError('CPUs {} exist in multiple freq domains'.format(
                fd_intersection))
        fd_difference = set(self.cpus) - set().union(*freq_domains)
        if fd_difference:
            raise ValueError('CPUs {} not in any frequency domain'.format(
                fd_difference))
        self.freq_domains = freq_domains

        # Check that nodes with energy data are all within a frequency domain
        for node in root_node.iter_nodes():
            if not node.active_states or node.idle_states:
                continue
            cpu_freq_doms = []
            for cpu in node.cpus:
                [cpu_freq_dom] = [d for d in freq_domains if cpu in d]
                cpu_freq_doms.append(cpu_freq_dom)
            if not all(d == cpu_freq_doms[0] for d in cpu_freq_doms[1:]):
                raise ValueError(
                    'Node {} (CPUs {}) '
                    'has energy data and overlaps freq domains'.format(
                        node.name, node.cpus))

        def sorted_leaves(root):
            # Get a list of the leaf (cpu) nodes of a _CpuTree in order of the
            # CPU ID
            ret = sorted(list(root.iter_leaves()), key=lambda n: n.cpus[0])
            assert all(len(n.cpus) == 1 for n in ret)
            return ret

        self.root = root_node
        self.cpu_nodes = sorted_leaves(root_node)
        self.cpu_pds = sorted_leaves(root_power_domain)
        assert len(self.cpu_pds) == len(self.cpu_nodes)

        self._log = logging.getLogger('EnergyModel')

        max_cap = max(n.max_capacity for n in self.cpu_nodes)
        if max_cap != self.capacity_scale:
            self._log.warning(
                'Unusual max capacity (%s), overriding capacity_scale', max_cap)
            self.capacity_scale = max_cap

    def _cpus_with_capacity(self, cap):
        """
        Helper method to find the CPUs whose max capacity equals cap
        """
        return [c for c in self.cpus
                if self.cpu_nodes[c].max_capacity == cap]

    @property
    @memoized
    def biggest_cpus(self):
        """
        The CPUs with the highest compute capacity at their highest frequency
        """
        return self._cpus_with_capacity(self.capacity_scale)

    @property
    @memoized
    def littlest_cpus(self):
        """
        The CPUs with the lowest compute capacity at their highest frequency
        """
        min_cap = min(n.max_capacity for n in self.cpu_nodes)
        return self._cpus_with_capacity(min_cap)

    @property
    @memoized
    def is_heterogeneous(self):
        """
        True iff CPUs do not all have the same efficiency and OPP range
        """
        states = self.cpu_nodes[0].active_states
        return any(c.active_states != states for c in self.cpu_nodes[1:])

    @property
    @memoized
    def cpu_groups(self):
        """
        List of lists of CPUs who share the same active state values
        """
        groups = []
        for node in self.cpu_nodes:
            for group in groups:
                group_states = self.cpu_nodes[group[0]].active_states
                if node.active_states == group_states:
                    group.append(node.cpu)
                    break
            else:
                groups.append([node.cpu])
        return groups

    def _guess_idle_states(self, cpus_active):
        def find_deepest(pd):
            if not any(cpus_active[c] for c in pd.cpus):
                if pd.parent:
                    parent_state = find_deepest(pd.parent)
                    if parent_state:
                        return parent_state
                return pd.idle_states[-1] if len(pd.idle_states) else None
            return None

        return [find_deepest(pd) for pd in self.cpu_pds]

    def get_cpu_capacity(self, cpu, freq=None):
        """Convenience method to get the capacity of a CPU at a given frequency

        :param cpu: CPU to get capacity for
        :param freq: Frequency to get the CPU capacity at. Default is max
                     capacity.
        """
        if freq is None:
            return self.cpu_nodes[cpu].max_capacity
        return self.cpu_nodes[cpu].active_states[freq].capacity

    def guess_idle_states(self, cpus_active):
        """Pessimistically guess the idle states that each CPU may enter

        If a CPU has any tasks it is estimated that it may only enter its
        shallowest idle state in between task activations. If all the CPUs
        within a power domain have no tasks, they will all be judged able to
        enter that domain's deepest idle state. If any CPU in a domain has work,
        no CPUs in that domain are assumed to enter any domain shared state.

        e.g. Consider a system with

        - two power domains PD0 and PD1

        - 4 CPUs, with CPUs [0, 1] in PD0 and CPUs [2, 3] in PD1

        - 4 idle states: "WFI", "cpu-sleep", "cluster-sleep-0" and
          "cluster-sleep-1", where the "cluster-sleep-*" states domain states,
          i.e. a CPU can only enter those states when both CPUs in the domain
          are idle.

        Then here are some example inputs and outputs:

        ::

          # All CPUs idle:
          [0, 0, 0, 0] -> ["cluster-sleep-1", "cluster-sleep-1",
                           "cluster-sleep-1", "cluster-sleep-1"]

          # All CPUs have work
          [1, 1, 1, 1] -> ["WFI","WFI","WFI", "WFI"]

          # One power domain active, the other idle
          [0, 0, 1, 1] -> ["cluster-sleep-1", "cluster-sleep-1", "WFI","WFI"]

          # One CPU active.
          # Note that CPU 2 has no work but is assumed to never be able to enter
          # any "cluster" state.
          [0, 0, 0, 1] -> ["cluster-sleep-1", "cluster-sleep-1",
                           "cpu-sleep","WFI"]

        :param cpus_active: list where bool(cpus_active[N]) is False iff no
                            tasks will run on CPU N.
        :returns: List ``ret`` where ``ret[N]`` is the name of the estimated
                  idle state that CPU N can enter during idle periods.

        """
        states = self._guess_idle_states(cpus_active)
        return [s or c.idle_states.keys()[0]
                for s, c in zip(states, self.cpu_nodes)]

    def _guess_freqs(self, cpu_utils):
        overutilized = False
        # Find what frequency each CPU would need if it was alone in its
        # frequency domain
        ideal_freqs = [0 for _ in self.cpus]
        for node in self.cpu_nodes:
            [cpu] = node.cpus
            required_cap = cpu_utils[cpu]

            possible_freqs = [f for f, s in node.active_states.iteritems()
                              if s.capacity >= required_cap]

            if possible_freqs:
                ideal_freqs[cpu] = min(possible_freqs)
            else:
                # CPU cannot provide required capacity, use max freq
                ideal_freqs[cpu] = max(node.active_states.keys())
                overutilized = True

        # Rectify the frequencies among domains
        freqs = [0 for _ in ideal_freqs]
        for domain in self.freq_domains:
            domain_freq = max(ideal_freqs[c] for c in domain)
            for cpu in domain:
                freqs[cpu] = domain_freq

        return freqs, overutilized

    def guess_freqs(self, cpu_utils):
        """Work out CPU frequencies required to execute a workload

        Find the lowest possible frequency for each CPU that provides enough
        capacity to satisfy the utilization, taking into account frequency
        domains.

        :param cpu_utils: Utilization distribution, see
                             :ref:`cpu_utils <cpu-utils>`
        :returns: List ``ret`` where ``ret[N]`` is the frequency that CPU N must
                  run at
        """
        freqs, _ = self._guess_freqs(cpu_utils)
        return freqs

    def _estimate_from_active_time(self, cpu_active_time, freqs, idle_states,
                                   combine):
        """Helper for estimate_from_cpu_util

        Like estimate_from_cpu_util but uses active time i.e. proportion of time
        spent not-idle in the range 0.0 - 1.0.

        If combine=False, return idle and active power as separate components.
        """
        power = 0
        ret = {}

        assert all(0.0 <= a <= 1.0 for a in cpu_active_time)

        for node in self.root.iter_nodes():
            # Some nodes might not have energy model data, they could just be
            # used to group other nodes (likely the root node, for example).
            if not node.active_states or not node.idle_states:
                continue

            cpus = tuple(node.cpus)
            # For now we assume topology nodes with energy models do not overlap
            # with frequency domains
            freq = freqs[cpus[0]]
            assert all(freqs[c] == freq for c in cpus[1:])

            # The active time of a node is estimated as the max of the active
            # times of its children.
            # This works great for the synthetic periodic workloads we use in
            # LISA (where all threads wake up at the same time) but is probably
            # no good for real workloads.
            active_time = max(cpu_active_time[c] for c in cpus)
            active_power = node.active_states[freq].power * active_time

            _idle_power = max(node.idle_states[idle_states[c]] for c in cpus)
            idle_power = _idle_power * (1 - active_time)

            if combine:
                ret[cpus] = active_power + idle_power
            else:
                ret[cpus] = {}
                ret[cpus]["active"] = active_power
                ret[cpus]["idle"] = idle_power

        return ret

    def estimate_from_cpu_util(self, cpu_utils, freqs=None, idle_states=None):
        """
        Estimate the energy usage of the system under a utilization distribution

        Optionally also take freqs; a list of frequencies at which each CPU is
        assumed to run, and idle_states, the idle states that each CPU can enter
        between activations. If not provided, they will be estimated assuming an
        ideal selection system (i.e. perfect cpufreq & cpuidle governors).

        :param cpu_utils: Utilization distribution, see
                             :ref:`cpu_utils <cpu-utils>`
        :param freqs: List of CPU frequencies. Got from :meth:`guess_freqs` by
                      default.
        :param idle_states: List of CPU frequencies. Got from
                            :meth:`guess_idle_states` by default.

        :returns: Dict with power in bogo-Watts (bW), with contributions from
                  each system component keyed with a tuple of the CPUs
                  comprising that component (i.e. :attr:EnergyModelNode.cpus)

                  ::

                    {
                        (0,)    : 10,
                        (1,)    : 10,
                        (0, 1)  : 5,
                    }

                  This represents CPUs 0 and 1 each using 10bW and their shared
                  resources using 5bW for a total of 25bW.
        """
        if len(cpu_utils) != len(self.cpus):
            raise ValueError(
                'cpu_utils length ({}) must equal CPU count ({})'.format(
                    len(cpu_utils), len(self.cpus)))

        if freqs is None:
            freqs = self.guess_freqs(cpu_utils)
        if idle_states is None:
            idle_states = self.guess_idle_states(cpu_utils)

        cpu_active_time = []
        for cpu, node in enumerate(self.cpu_nodes):
            assert (cpu,) == node.cpus
            cap = node.active_states[freqs[cpu]].capacity
            cpu_active_time.append(min(float(cpu_utils[cpu]) / cap, 1.0))

        return self._estimate_from_active_time(cpu_active_time,
                                               freqs, idle_states, combine=True)

    def get_optimal_placements(self, capacities):
        """Find the optimal distribution of work for a set of tasks

        Find a list of candidates which are estimated to be optimal in terms of
        power consumption, but that do not result in any CPU becoming
        over-utilized.

        If no such candidates exist, i.e. the system being modeled cannot
        satisfy the workload's throughput requirements, an
        :class:`EnergyModelCapacityError` is raised. For example, if e was an
        EnergyModel modeling two CPUs with capacity 1024, this error would be
        raised by:

        ::

          e.get_optimal_placements({"t1": 800, "t2": 800, "t3: "800"})

        This estimation assumes an ideal system of selecting OPPs and idle
        states for CPUs.

        .. note::
            This is a brute force search taking time exponential wrt. the number
            of tasks.

        :param capacities: Dict mapping tasks to expected utilization
                           values. These tasks are assumed not to change; they
                           have a single static utilization value. A set of
                           single-phase periodic RT-App tasks is an example of a
                           suitable workload for this model.
        :returns: List of ``cpu_utils`` items representing distributions of work
                  under optimal task placements, see
                  :ref:`cpu_utils <cpu-utils>`. Multiple task placements
                  that result in the same CPU utilizations are considered
                  equivalent.
        """
        tasks = capacities.keys()

        num_candidates = len(self.cpus) ** len(tasks)
        self._log.debug(
            '%14s - Searching %d configurations for optimal task placement...',
            'EnergyModel', num_candidates)

        candidates = {}
        excluded = []
        for cpus in product(self.cpus, repeat=len(tasks)):
            placement = {task: cpu for task, cpu in zip(tasks, cpus)}

            util = [0 for _ in self.cpus]
            for task, cpu in placement.items():
                util[cpu] += capacities[task]
            util = tuple(util)

            # Filter out candidate placements that have tasks greater than max
            # or that we have already determined that we cannot place.
            if (any(u > self.capacity_scale for u in util) or util in excluded):
                continue

            if util not in candidates:
                freqs, overutilized = self._guess_freqs(util)
                if overutilized:
                    # This isn't a valid placement
                    excluded.append(util)
                else:
                    power = self.estimate_from_cpu_util(util, freqs=freqs)
                    candidates[util] = sum(power.values())

        if not candidates:
            # The system can't provide full throughput to this workload.
            raise EnergyModelCapacityError(
                "Can't handle workload - total cap = {}".format(
                    sum(capacities.values())))

        # Whittle down to those that give the lowest energy estimate
        min_power = min(p for p in candidates.itervalues())
        ret = [u for u, p in candidates.iteritems() if p == min_power]

        self._log.debug('%14s - Done', 'EnergyModel')
        return ret

    @classmethod
    def _find_core_groups(cls, target):
        """
        Read the core_siblings masks for each CPU from sysfs

        :param target: Devlib Target object to read masks from
        :returns: A list of tuples of ints, representing the partition of core
                  siblings
        """
        cpus = range(target.number_of_cpus)

        topology_base = '/sys/devices/system/cpu/'

        # We only care about core_siblings, but let's check *_siblings, so we
        # can throw an error if a CPU's thread_siblings isn't just itself, or if
        # there's a topology level we don't understand.

        # Since we might have to read a lot of files, read everything we need in
        # one go to avoid taking too long.
        mask_glob = topology_base + 'cpu**/topology/*_siblings'
        file_values = read_multiple_oneline_files(target, [mask_glob])

        regex = re.compile(
            topology_base + r'cpu([0-9]+)/topology/([a-z]+)_siblings')

        ret = set()

        for path, mask_str in file_values.iteritems():
            match = regex.match(path)
            cpu = int(match.groups()[0])
            level = match.groups()[1]
            # mask_to_list returns the values in descending order, so we'll sort
            # them ascending. This isn't strictly necessary but it's nicer.
            siblings = tuple(sorted(mask_to_list(int(mask_str, 16))))

            if level == 'thread':
                if siblings != (cpu,):
                    # SMT systems aren't supported
                    raise RuntimeError('CPU{} thread_siblings is {}. '
                                       'expected {}'.format(cpu, siblings, [cpu]))
                continue
            if level != 'core':
                # The only other levels we should expect to find are 'book' and
                # 'shelf', which are not used by architectures we support.
                raise RuntimeError(
                    'Unrecognised topology level "{}"'.format(level))

            ret.add(siblings)

        # Sort core groups so that the lowest-numbered cores are first
        # Again, not strictly necessary, just more pleasant.
        return sorted(ret, key=lambda x: x[0])

    @classmethod
    def from_target(cls, target):
        """
        Create an EnergyModel by reading a target filesystem

        This uses the sysctl added by EAS pathces to exposes the cap_states and
        idle_states fields for each sched_group. This feature depends on
        CONFIG_SCHED_DEBUG, and is not upstream in mainline Linux (as of v4.11),
        so this method is only tested with Android kernels.

        The kernel doesn't have an power domain data, so this method assumes
        that all CPUs are totally independent wrt. idle states - the EnergyModel
        constructed won't be aware of the topological dependencies for entering
        "cluster" idle states.

        Assumes the energy model has two-levels (plus the root) - a level for
        CPUs and a level for 'clusters'.

        :param target: Devlib target object to read filesystem from. Must have
                       cpufreq and cpuidle modules enabled.
        :returns: Constructed EnergyModel object based on the parameters
                  reported by the target.
        """
        if 'cpufreq' not in target.modules:
            raise TargetError('Requires cpufreq devlib module. Please ensure '
                               '"cpufreq" is listed in your target/test modules')
        if 'cpuidle' not in target.modules:
            raise TargetError('Requires cpuidle devlib module. Please ensure '
                               '"cpuidle" is listed in your target/test modules')

        def sge_path(cpu, domain, group, field):
            f = '/proc/sys/kernel/sched_domain/cpu{}/domain{}/group{}/energy/{}'
            return f.format(cpu, domain, group, field)

        # Read all the files we might need in one go, otherwise this will take
        # ages.
        sge_globs = [sge_path('**', '**', '**', 'cap_states'),
                     sge_path('**', '**', '**', 'idle_states')]
        sge_file_values = read_multiple_oneline_files(target, sge_globs)

        if not sge_file_values:
            raise TargetError('Energy Model not exposed in sysfs. '
                              'Check CONFIG_SCHED_DEBUG is enabled.')

        # These functions read the cap_states and idle_states vectors for the
        # first sched_group in the sched_domain for a given CPU at a given
        # level. That first group will include the given CPU. So
        # read_active_states(0, 0) will give the CPU-level active_states for
        # CPU0 and read_active_states(0, 1) will give the "cluster"-level
        # active_states for the "cluster" that contains CPU0.

        def read_sge_file(path):
            try:
                return sge_file_values[path]
            except KeyError as e:
                raise TargetError('No such file: {}'.format(e))

        def read_active_states(cpu, domain_level):
            cap_states_path = sge_path(cpu, domain_level, 0, 'cap_states')
            cap_states_strs = read_sge_file(cap_states_path).split()

            # cap_states lists the capacity of each state followed by its power,
            # in increasing order. The `zip` call does this:
            #   [c0, p0, c1, p1, c2, p2] -> [(c0, p0), (c1, p1), (c2, p2)]
            cap_states = [ActiveState(capacity=int(c), power=int(p))
                          for c, p in zip(cap_states_strs[0::2],
                                          cap_states_strs[1::2])]
            freqs = target.cpufreq.list_frequencies(cpu)
            return OrderedDict(zip(sorted(freqs), cap_states))

        def read_idle_states(cpu, domain_level):
            idle_states_path = sge_path(cpu, domain_level, 0, 'idle_states')
            idle_states_strs = read_sge_file(idle_states_path).split()

            # get_states should return the state names in increasing depth order
            names = [s.name for s in target.cpuidle.get_states(cpu)]
            # idle_states is a list of power values in increasing order of
            # idle-depth/decreasing order of power.
            return OrderedDict(zip(names, [int(p) for p in idle_states_strs]))

        # Read the CPU-level data from sched_domain level 0
        cpus = range(target.number_of_cpus)
        cpu_nodes = []
        for cpu in cpus:
            node = EnergyModelNode(
                cpu=cpu,
                active_states=read_active_states(cpu, 0),
                idle_states=read_idle_states(cpu, 0))
            cpu_nodes.append(node)

        # Read the "cluster" level data from sched_domain level 1
        core_group_nodes = []
        for core_group in cls._find_core_groups(target):
            node=EnergyModelNode(
                children=[cpu_nodes[c] for c in core_group],
                active_states=read_active_states(core_group[0], 1),
                idle_states=read_idle_states(core_group[0], 1))
            core_group_nodes.append(node)

        root = EnergyModelRoot(children=core_group_nodes)

        # Use cpufreq to figure out the frequency domains
        freq_domains = []
        remaining_cpus = set(cpus)
        while remaining_cpus:
            cpu = next(iter(remaining_cpus))
            dom = target.cpufreq.get_domain_cpus(cpu)
            freq_domains.append(dom)
            remaining_cpus = remaining_cpus.difference(dom)

        # We don't have a way to read the power domains from sysfs (the kernel
        # isn't even aware of them) so we'll just have to assume each CPU is its
        # own power domain and all idle states are independent of each other.
        cpu_pds = []
        for cpu in cpus:
            names = [s.name for s in target.cpuidle.get_states(cpu)]
            cpu_pds.append(PowerDomain(cpu=cpu, idle_states=names))

        root_pd=PowerDomain(children=cpu_pds, idle_states=[])

        return cls(root_node=root,
                   root_power_domain=root_pd,
                   freq_domains=freq_domains)
