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
"""Classes for modeling and estimating energy usage of CPU systems"""

from collections import namedtuple, OrderedDict
from itertools import product
import operator
import re

import pandas

from devlib.utils.misc import mask_to_list, ranges_to_list
from devlib.exception import TargetStableError

from lisa.utils import Loggable, Serializable, memoized, groupby, get_subclasses, deprecate, grouper
from lisa.datautils import df_deduplicate
from lisa.analysis.frequency import FrequencyAnalysis



def _read_multiple_oneline_files(target, glob_patterns):
    """
    Quickly read many single-line files that match a glob pattern

    Finds all the files that match any of the glob patterns and, assuming that
    they each contain exactly 1 line of text, read them all at once. When the
    target or connection is slow this saves a lot of time when reading a large
    number of files.

    This will only work safely on stationary files, don't try to use it where
    the glob expansion will change often - for example ``/proc/**/autogroup`` would
    not work because /proc/ entries will likely appear & disappear while we're
    reading them.

    :param target: devlib target object to read from
    :param glob_pattern: Unix glob pattern matching the files to read
    :returns: A dictionary mapping matched paths to the values read. ``{}`` if
              no paths matched the globs.
    """
    find_cmd = 'find ' + ' '.join(glob_patterns)
    try:
        paths = target.execute(find_cmd).split()
    except TargetStableError:
        return {}

    cmd = f'{find_cmd} | {target.busybox} xargs cat'
    contents = target.execute(cmd).splitlines()

    if len(contents) != len(paths):
        raise RuntimeError('File count mismatch while reading multiple files')

    return dict(zip(paths, contents))


class EnergyModelCapacityError(Exception):
    """Used by :meth:`EnergyModel.get_optimal_placements`"""


class ActiveState(namedtuple('ActiveState', ['capacity', 'power'])):
    """Represents power and compute capacity at a given frequency

    :param capacity: Relative compute capacity at frequency
    :param power: Power usage at frequency
    """
    def __new__(cls, capacity=None, power=None):
        return super().__new__(cls, capacity, power)


class _CpuTree(Loggable):
    """
    :meta public:

    Internal class. Abstract representation of a CPU topology.

    Each node contains either a single CPU or a set of child nodes.

    :Attributes:
        * ``cpus``: CPUs contained in this node. Includes those of child nodes.
        * ``cpu``: For convenience, this holds the single CPU contained by leaf
          nodes. ``None`` for non-leaf nodes.
    """

    def __init__(self, cpu, children):
        if (cpu is None) == (children is None):
            raise ValueError('Provide exactly one of: cpu or children')

        self.parent = None
        #: Test yolo
        self.cpu = cpu

        if cpu is not None:
            #: This is another thingie
            self.cpus = (cpu,)
            self.children = []
        else:
            if len(children) == 0:
                raise ValueError('children cannot be empty')
            self.cpus = tuple(sorted({
                cpu
                for node in children
                for cpu in node.cpus
            }))
            self.children = children
            for child in children:
                child.parent = self

        self.name = None

    def __repr__(self):
        name_bit = ''
        if self.name:
            name_bit = f'name="{self.name}", '

        if self.children:
            return f'{self.__class__.__name__}({name_bit}children={self.children})'
        else:
            return f'{self.__class__.__name__}({name_bit}cpus={self.cpus})'

    def _iter(self, include_non_leaves):
        for child in self.children:
            yield from child._iter(include_non_leaves)
        if include_non_leaves or not self.children:
            yield self

    def iter_nodes(self):
        """Iterate over nodes depth-first, post-order"""
        return self._iter(True)

    def iter_leaves(self):
        """Iterate over leaves"""
        return self._iter(False)


class EnergyModelNode(_CpuTree):
    """
    Describes topology and energy data for an EnergyModel.

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
    :type idle_states: dict

    :param cpu: The CPU this node represents. If provided, this is a leaf node.
    :type cpu: tuple(int)

    :param children: Non-empty list of child :class:`EnergyModelNode` objects
    :type children: list(EnergyModelNode)

    :param name: Optional human-readable name for this node. Leaf (CPU) nodes
                 have a default name of "cpuN" where N is the cpu number.
    :type name: str
    """

    def __init__(self, active_states, idle_states,
                 cpu=None, children=None, name=None):
        super().__init__(cpu, children)
        logger = self.logger

        def is_monotonic(l, decreasing=False):
            op = operator.ge if decreasing else operator.le
            return all(op(a, b) for a, b in zip(l, l[1:]))

        if active_states:
            # Sanity check for active_states's frequencies
            freqs = list(active_states.keys())
            if not is_monotonic(freqs):
                logger.warning(
                    f'Active states frequencies are expected to be monotonically increasing. Freqs: {freqs}')

            # Sanity check for active_states's powers
            power_vals = [s.power for s in list(active_states.values())]
            if not is_monotonic(power_vals):
                logger.warning(
                    f'Active states powers are expected to be monotonically increasing. Values: {power_vals}')

        if idle_states:
            # This is needed for idle_state_by_idx to work.
            if not isinstance(idle_states, OrderedDict):
                f = 'idle_states is {}, must be collections.OrderedDict'
                raise ValueError(f.format(type(idle_states)))

            # Sanity check for idle_states powers
            power_vals = list(idle_states.values())
            if not is_monotonic(power_vals, decreasing=True):
                logger.warning(
                    f'Idle states powers are expected to be monotonically decreasing. Values: {power_vals}')

        if cpu is not None and not name:
            name = 'cpu' + str(cpu)

        self.name = name
        self.active_states = active_states
        self.idle_states = idle_states

    @property
    def max_capacity(self):
        """Compute capacity at highest frequency"""
        return max(s.capacity for s in list(self.active_states.values()))

    def idle_state_by_idx(self, idx):
        """Return the idle state with index ``idx``"""
        # NB self.idle_states must be ordered for this to work. __init__
        # enforces that it is an OrderedDict
        if self.idle_states and idx < len(self.idle_states):
            return list(self.idle_states.keys())[idx]

        raise KeyError(f'No idle state with index {idx}')


class EnergyModelRoot(EnergyModelNode):
    """
    Convenience class for root of an EnergyModelNode tree.

    Just like EnergyModelNode except that ``active_states`` and ``idle_states``
    aren't required.
    """

    def __init__(self, active_states=None, idle_states=None,
                 cpu=None, children=None, name=None):
        super().__init__(active_states, idle_states, cpu, children, name)


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

    :Attributes:
        * ``cpus`` (`tuple(int)`): CPUs contained in this node. Includes
          those of child nodes.
    """

    def __init__(self, idle_states, cpu=None, children=None):
        if idle_states is None:
            raise ValueError('idle_states cannot be None (but may be empty)')
        super().__init__(cpu, children)
        self.idle_states = idle_states


class EnergyModel(Serializable, Loggable):
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

    :Attributes:
        * ``cpu_nodes``: List of leaf (CPU) :class`:`EnergyModelNode`
        * ``cpus``: List of logical CPU numbers in the system
        * ``capacity_scale``: The relative computational capacity of the most
          powerful CPU at its highest available frequency. Utilisation is in
          the interval ``[0, capacity_scale]``.

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

        This represents a static utilization, assuming that tasks don't change
        in size (for example representing a set of fixed periodic RT-App
        workloads). For workloads that change over time, a series of
        ``cpu_utils`` items would be needed to describe the utilization, with a
        distinct estimation for each item in the series.
    """

    _PROBE_ORDER = None
    """
    Order in which subclasses are tried when auto-detecting the kind of energy
    model to load from a target.
    """

    def __init__(self, root_node, root_power_domain, freq_domains):
        self.cpus = root_node.cpus
        if self.cpus != tuple(range(len(self.cpus))):
            raise ValueError(f'CPU IDs [{self.cpus}] are sparse')

        domains_as_set = [set(dom) for dom in freq_domains]

        # Check that freq_domains is a partition of the CPUs
        fd_intersection = set.intersection(*domains_as_set)
        if len(domains_as_set) > 1 and fd_intersection:
            raise ValueError(f'CPUs {fd_intersection} exist in multiple freq domains')
        fd_difference = set(self.cpus) - set.union(*domains_as_set)
        if fd_difference:
            raise ValueError(f'CPUs {fd_difference} not in any frequency domain')
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
                    f'Node {node.name} (CPUs {node.cpus}) has energy data and overlaps freq domains')

        def sorted_leaves(root):
            # Get a list of the leaf (cpu) nodes of a _CpuTree in order of the
            # CPU ID
            ret = sorted(list(root.iter_leaves()), key=lambda n: n.cpus[0])
            assert all(len(n.cpus) == 1 for n in ret)
            return ret

        self.root = root_node
        self.cpu_nodes = sorted_leaves(root_node)
        self.pd = root_power_domain
        self.cpu_pds = sorted_leaves(root_power_domain)
        assert len(self.cpu_pds) == len(self.cpu_nodes)

        self.capacity_scale = max(
            node.max_capacity
            for node in self.cpu_nodes
        )

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
        return [
            [node.cpu for node in group]
            for group in self.node_groups
        ]

    @property
    @memoized
    def node_groups(self):
        """
        List of lists of CPUs nodes who share the same active state values
        """
        def key(node):
            return sorted(node.active_states.items())

        return [
            list(group)
            for active_states, group in groupby(self.cpu_nodes, key=key)
        ]

    def _deepest_idle_idxs(self, cpus_active):
        def find_deepest(pd):
            if any(cpus_active[c] for c in pd.cpus):
                return -1
            if pd.parent:
                parent_idx = find_deepest(pd.parent)
            else:
                parent_idx = -1
            ret = parent_idx + len(pd.idle_states)
            return ret
        return [find_deepest(pd) for pd in self.cpu_pds]

    def _guess_idle_states(self, cpus_active):
        idxs = self._deepest_idle_idxs(cpus_active)
        return [n.idle_state_by_idx(max(i, 0)) for n, i in zip(self.cpu_nodes, idxs)]

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
        r"""Pessimistically guess the idle states that each CPU may enter

        If a CPU has any tasks it is estimated that it may only enter its
        shallowest idle state in between task activations. If all the CPUs
        within a power domain have no tasks, they will all be judged able to
        enter that domain's deepest idle state. If any CPU in a domain has work,
        no CPUs in that domain are assumed to enter any domain shared state.

        e.g. Consider a system with

        - two power domains PD0 and PD1

        - 4 CPUs, with CPUs [0, 1] in PD0 and CPUs [2, 3] in PD1

        - 4 idle states: "WFI", "cpu-sleep", "cluster-sleep-0" and
          "cluster-sleep-1", where the "cluster-sleep-\*" states domain states,
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
        return [s or list(c.idle_states.keys())[0]
                for s, c in zip(states, self.cpu_nodes)]

    def _guess_freqs(self, cpu_utils, capacity_margin_pct):
        overutilized = False
        # Find what frequency each CPU would need if it was alone in its
        # frequency domain
        ideal_freqs = [0 for _ in self.cpus]
        for node in self.cpu_nodes:
            [cpu] = node.cpus

            # A capacity margin should be provided to meet the demand at a
            # given utilizaton level as per the scheduler's 'capacity_margin'
            # coefficient so reflect that here
            margin = 100 / (100 - capacity_margin_pct)
            required_cap = cpu_utils[cpu] * margin

            possible_freqs = [f for f, s in node.active_states.items()
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

    def guess_freqs(self, cpu_utils, capacity_margin_pct=0):
        """Work out CPU frequencies required to execute a workload

        Find the lowest possible frequency for each CPU that provides enough
        capacity to satisfy the utilization, taking into account frequency
        domains.

        :param cpu_utils: Utilization distribution, see
                             :ref:`cpu_utils <cpu-utils>`
        :param capacity_margin_pct: Capacity margin before overutilizing a CPU
        :returns: List ``ret`` where ``ret[N]`` is the frequency that CPU N must
                  run at
        """
        freqs, _ = self._guess_freqs(cpu_utils, capacity_margin_pct)
        return freqs

    def _estimate_from_active_time(self, cpu_active_time, freqs, idle_states,
                                   combine):
        """Helper for estimate_from_cpu_util

        Like estimate_from_cpu_util but uses active time i.e. proportion of time
        spent not-idle in the range 0.0 - 1.0.

        If combine=False, return idle and active power as separate components.
        """
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

            # The active time of a node is estimated as the max of the active
            # times of its children.
            # This works great for the synthetic periodic workloads we use in
            # LISA (where all threads wake up at the same time) but is probably
            # no good for real workloads.
            active_time = max(cpu_active_time[c] for c in cpus)
            active_power = node.active_states[freq].power * active_time

            _idle_power = max(node.idle_states[idle_states[c]] for c in cpus)
            idle_power = _idle_power * (1 - active_time)

            if cpus not in ret:
                if combine:
                    ret[cpus] = 0
                else:
                    ret[cpus] = {}
                    ret[cpus]["active"] = 0
                    ret[cpus]["idle"] = 0

            if combine:
                ret[cpus] += active_power + idle_power
            else:
                ret[cpus]["active"] += active_power
                ret[cpus]["idle"] += idle_power

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
                f'cpu_utils length ({len(cpu_utils)}) must equal CPU count ({len(self.cpus)})')

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

    def get_optimal_placements(self, capacities, capacity_margin_pct=0):
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
        :param capacity_margin_pct: Capacity margin before overutilizing a CPU
        :returns: List of ``cpu_utils`` items representing distributions of work
                  under optimal task placements, see
                  :ref:`cpu_utils <cpu-utils>`. Multiple task placements
                  that result in the same CPU utilizations are considered
                  equivalent.
        """
        tasks = list(capacities.keys())

        num_candidates = len(self.cpus) ** len(tasks)

        logger = self.logger
        logger.debug(
            f'Searching {num_candidates} configurations for optimal task placement...')

        candidates = {}
        excluded = []
        for cpus in product(self.cpus, repeat=len(tasks)):
            placement = dict(zip(tasks, cpus))

            util = [0] * len(self.cpus)
            for task, cpu in placement.items():
                util[cpu] += capacities[task]
            util = tuple(util)

            # Filter out candidate placements that have tasks greater than max
            # or that we have already determined that we cannot place.
            if (any(u > self.capacity_scale for u in util) or util in excluded):
                continue

            if util not in candidates:
                freqs, overutilized = self._guess_freqs(util, capacity_margin_pct)
                if overutilized:
                    # This isn't a valid placement
                    excluded.append(util)
                else:
                    power = self.estimate_from_cpu_util(util, freqs=freqs)
                    candidates[util] = sum(power.values())

        if not candidates:
            # The system can't provide full throughput to this workload.
            raise EnergyModelCapacityError(
                f"Can't handle workload: total capacity = {sum(capacities.values())}")

        # Whittle down to those that give the lowest energy estimate
        min_power = min(p for p in iter(candidates.values()))
        ret = [u for u, p in candidates.items() if p == min_power]

        logger.debug('done')
        return ret

    @classmethod
    def probe_target(cls, target):
        """
        Check if an :class:`EnergyModel` can be loaded from the target.

        :param target: Target to look at.
        :type target: devlib.target.Target
        """
        try:
            cls._find_subcls(target)
        except TargetStableError:
            return False
        else:
            return True

    @classmethod
    def _find_subcls(cls, target):
        subcls_list = sorted(
            get_subclasses(cls, only_leaves=True),
            key=lambda cls: cls._PROBE_ORDER
        )

        for subcls in subcls_list:
            if subcls.probe_target(target):
                return subcls

        raise TargetStableError('Unable to probe for energy model on target.')

    @classmethod
    def from_target(cls, target):
        """
        Create an instance of (a subclass of) :class:``EnergyModel`` by reading
        a target filesystem.

        :param target: Target object to read filesystem from.
        :type target: devlib.target.Target
        :returns: A instance of a subclass of :class:`EnergyModel`.

        .. seealso:: :meth:`LinuxEnergyModel.from_target`
           and :meth:`LegacyEnergyModel.from_target`
        """
        logger = cls.get_logger('from_target')

        subcls = cls._find_subcls(target)
        logger.info(f'Attempting to load EM using {subcls.__name__}')
        em = subcls.from_target(target)

        cpu_missing_idle_states = sorted(
            node.cpu
            for node in em.root.iter_leaves()
            if not node.idle_states
        )
        if cpu_missing_idle_states:
            logger.warning(f'CPUs missing idle states in cpuidle framework: {cpu_missing_idle_states}')

        return em

    @deprecate(replaced_by='lisa.energy_model.LinuxEnergyModel.from_target', deprecated_in='2.0', removed_in='4.0')
    @staticmethod
    def from_debugfsEM_target(*args, **kwargs):
        """
        See :meth:`LinuxEnergyModel.from_target`
        """
        return LinuxEnergyModel.from_target(*args, **kwargs)

    @deprecate(replaced_by='lisa.energy_model.LegacyEnergyModel.from_target', deprecated_in='2.0', removed_in='4.0')
    @staticmethod
    def from_sd_target(*args, **kwargs):
        """
        See :meth:`LegacyEnergyModel.from_target`
        """
        return LegacyEnergyModel.from_target(*args, **kwargs)

    @FrequencyAnalysis.df_cpus_frequency.used_events
    def estimate_from_trace(self, trace):
        """
        Estimate the energy consumption of the system by looking at a trace

        Usese the EAS energy model data, and the idle and DVFS conditions
        reported in the trace, to estimate the energy usage of the system at
        every given moment.

        Takes into account knowledge of power domains - where cpuidle makes
        impossible claims about idle states (e.g. a CPU in 'cluster sleep' while
        its cluster siblings are running), the states will be minimised.

        The accuracy of this is otherwise totally dependent on the accuracy of
        the EAS energy model and the kernel's information. This does not take
        into account cost of idle state of DVFS transitions, nor any other
        conditions that are invisible to the kernel. The effect any power
        decisions that the platform makes independently of the kernel cannot be
        seen in this data. Examples of this _might_ include firmware thermal
        management invisibly restricting CPU frequencies, or secure-world
        software with real-time constraints preventing deep CPU idle states.

        :param trace: The trace
        :type trace: lisa.trace.Trace

        :returns: A DataFrame with a column for each node in the energy model,
                  labelled with the CPU members of the node joined by  '-'s.
                  Shows the energy use by each node at each given moment.
                  If you don't care about those details, call ``.sum(axis=1)`` on
                  the returned DataFrame to get a Series that shows overall
                  estimated power usage over time.
        """
        idle = trace.ana.idle.df_cpus_idle().pivot(columns='cpu')['state']
        freqs = trace.ana.frequency.df_cpus_frequency().pivot(columns='cpu')['frequency']

        inputs = pandas.concat([idle, freqs], axis=1, keys=['idle', 'freq'], sort=True).ffill()

        # Drop stuff at the beginning where we don't have the inputs
        # (e.g. where we have had our first cpu_idle event but no cpu_frequency)
        inputs = inputs.dropna()
        # Convert to int wholesale so we can do things like use the values in
        # the inputs DataFrame as list indexes. The only reason we had floats
        # was to make room for NaN, but we've just dropped all the NaNs, so
        # that's fine.
        inputs = inputs.astype(int)
        inputs = df_deduplicate(inputs, keep='first', consecutives=True)

        memo_cache = {}

        def f(input_row):
            # The code in this module is slow. Try not to call it too much.
            memo_key = tuple(input_row)
            if memo_key in memo_cache:
                return memo_cache[memo_key]

            # cpuidle doesn't understand shared resources so it will claim to
            # put a CPU into e.g. 'cluster sleep' while its cluster siblings are
            # active. Rectify those false claims.
            cpus_active = input_row['idle'] == -1
            deepest_possible = self._deepest_idle_idxs(cpus_active)
            idle_idxs = [min(i, j) for i, j in zip(deepest_possible,
                                                   input_row['idle'])]

            # Convert indexes to state names
            idle_states = [n.idle_state_by_idx(max(i, 0))
                           for n, i in zip(self.cpu_nodes, idle_idxs)]

            # We don't use tracked load, we just treat a CPU as active or idle,
            # so set util to 0 or 100%.
            utils = cpus_active * self.capacity_scale

            nrg = self.estimate_from_cpu_util(cpu_utils=utils,
                                              idle_states=idle_states,
                                              freqs=input_row['freq'])

            # nrg is a dict mapping CPU group tuples to energy values.
            # Unfortunately tuples don't play nicely as pandas column labels
            # because parts of its API treat that as nested indexing
            # (i.e. df[(0, 1)] sometimes means df[0][1]). So we'll give them
            # awkward names.

            nrg = {'-'.join(str(c) for c in k): v for k, v in iter(nrg.items())}

            ret = pandas.Series(nrg)
            memo_cache[memo_key] = ret
            return ret

        return inputs.apply(f, axis=1)

    @classmethod
    @memoized
    def _get_idle_states_name(cls, target, cpu, only_real=False):
        def placeholder():
            if only_real:
                raise ValueError('idle state detection requires the cpuidle devlib module, and populated sysfs idle states')
            else:
                return ['placeholder-idle-state']

        try:
            cpuidle = target.cpuidle
        except AttributeError:
            return placeholder()
        else:
            states = [s.name for s in cpuidle.get_states(cpu)]
            return states or placeholder()


class LinuxEnergyModel(EnergyModel):
    """
    Mainline Linux kernel energy model, available since linux 5.0 .

    The energy model information is stored in debugfs.
    """

    _PROBE_ORDER = 1

    @staticmethod
    def probe_target(target):
        directory = '/sys/kernel/debug/energy_model'
        return target.file_exists(directory)

    @classmethod
    def from_target(cls, target, directory='/sys/kernel/debug/energy_model'):
        """
        Create an :class:`EnergyModel` by reading a target filesystem on a
        device with the new Simplified Energy Model present in debugfs.

        :param target: Target object to read filesystem from.
        :type target: lisa.target.Target

        This uses the energy_model debugfs used usptream to expose the
        performance domains, their frequencies and power costs. This feature is
        upstream as of Linux 5.1. It is also available on Android 4.19 and
        later.

        Wrt. idle states - the EnergyModel constructed won't be aware of
        any power data or topological dependencies for entering "cluster"
        idle states since the simplified model has no such concept.

        .. note:: Initialises only class:`ActiveStates` for CPUs and clears all
            other levels.

        The recognized debugfs hierarchy pointed at by ``directory`` is::

            |-- cpu0
            |   |-- cpus
            |   |-- ps:450000
            |   |   |-- cost
            |   |   |-- frequency
            |   |   `-- power
            |   |-- ps:575000
            |   |   |-- cost
            |   |   |-- frequency
            |   |   `-- power
            |   `-- ps:850000
            |       |-- cost
            |       |-- frequency
            |       `-- power
            `-- cpu1
                |-- cpus
                |-- ps:1100000
                |   |-- cost
                |   |-- frequency
                |   `-- power
                |-- ps:450000
                |   |-- cost
                |   |-- frequency
                |   `-- power
                `-- ps:950000
                    |-- cost
                    |-- frequency
                    `-- power
        """
        cpu_cap_sysfs = '/sys/devices/system/cpu/cpu{}/cpu_capacity'

        # Format of a debugfs cstate entry
        cstate_format = {
            'cost': int,
            'frequency': int,
            'power': int,
        }

        def has_keys(mapping, keys):
            try:
                return set(keys) <= mapping.keys()
            except AttributeError:
                return False

        def parse_pd_attr(pd_em):
            """
            Parse the power domain attributes from debugfs sub-hierarchy
            """
            cstate_keys = cstate_format.keys()
            cstates = [
                {
                    # Convert from string to appropriate type
                    key: cstate_format.get(key, lambda x: x)(val)
                    for key, val in entry.items()
                }
                for key, entry in pd_em.items()
                # Filter-out folders that don't contain the files we need, so
                # we don't depend on the name of the folder itself which can
                # change from one kernel version to another
                if has_keys(entry, cstate_keys)
            ]

            # Read the CPUMask
            cpus = ranges_to_list(pd_em['cpus'])

            # Add the capacity to the cstate
            max_freq = max(cs['frequency'] for cs in cstates)
            pd_cap = target.read_value(cpu_cap_sysfs.format(cpus[0]), int)
            cstates = [
                {
                    'capacity': cs['frequency'] * pd_cap / max_freq,
                    **cs
                }
                for cs in cstates
            ]

            return {
                'cpus': cpus,
                'cstates': cstates,
            }

        debugfs_em = target.read_tree_values(directory, depth=3, tar=True)
        if not debugfs_em:
            raise TargetStableError(f'Energy Model not exposed at {directory} in debugfs.')

        pd_attr = {
            pd: parse_pd_attr(pd_em)
            for pd, pd_em in debugfs_em.items()
        }
        cpu_attr = {
            cpu: attr
            for attr in pd_attr.values()
            for cpu in attr['cpus']
        }

        return cls(
            root_node=EnergyModelRoot(
                children=[
                    EnergyModelNode(
                        cpu=cpu,
                        active_states=OrderedDict(
                            (cs['frequency'], ActiveState(cs['capacity'], cs['power']))
                            for cs in sorted(
                                attr['cstates'],
                                key=operator.itemgetter('frequency'),
                            )
                        ),
                        # Idle states are not supported in the simple model, so record
                        # 0 power for them all, but name them according to target
                        idle_states=OrderedDict(
                            (name, 0)
                            for name in cls._get_idle_states_name(target, cpu)
                        ),
                    )
                    for cpu, attr in cpu_attr.items()
                ]
            ),
            # We don't have a way to read the idle power domains from sysfs (the
            # kernel isn't even aware of them) so we'll just have to assume each CPU
            # is its own power domain and all idle states are independent of each
            # other.
            root_power_domain=PowerDomain(
                children=[
                    PowerDomain(
                        cpu=cpu,
                        idle_states=cls._get_idle_states_name(target, cpu),
                    )
                    for cpu in sorted(cpu_attr.keys())
                ],
                idle_states=[]
            ),
            freq_domains=sorted(attr['cpus'] for attr in pd_attr.values()),
        )


class LegacyEnergyModel(EnergyModel):
    """
    Legacy energy model used on Android kernels prior 4.19.

    The energy model information is stored in sysfs and contains detailed
    information about idle states.
    """

    _PROBE_ORDER = 2

    @staticmethod
    def probe_target(target):
        filename = '/proc/sys/kernel/sched_domain/cpu{}/domain{}/group{}/energy/{}'
        cpu = target.list_online_cpus()[0]
        f = filename.format(cpu, 0, 0, 'cap_states')
        return target.file_exists(f)

    @classmethod
    def from_target(cls, target, filename='/proc/sys/kernel/sched_domain/cpu{}/domain{}/group{}/energy/{}'):
        """
        Create an EnergyModel by reading a target filesystem

        This uses the sysctl added by EAS patches to exposes the cap_states and
        idle_states fields for each sched_group. This feature depends on
        CONFIG_SCHED_DEBUG, and is not upstream in mainline Linux.

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

        cpus = set(range(target.number_of_cpus))

        def sge_path(cpu, domain, group, field):
            return filename.format(cpu, domain, group, field)

        # Read all the files we might need in one go, otherwise this will take
        # ages.
        sge_globs = [
            sge_path('**', '**', '**', 'cap_states'),
            sge_path('**', '**', '**', 'nr_cap_states'),
            sge_path('**', '**', '**', 'idle_states'),
        ]
        sge_file_values = _read_multiple_oneline_files(target, sge_globs)

        if not sge_file_values:
            raise TargetStableError(
                'Energy Model not exposed in sysfs. Check CONFIG_SCHED_DEBUG is enabled.'
            )

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
                raise TargetStableError(f'No such file: {e}') from e

        def read_active_states(cpu, domain_level):
            cap_states_path = sge_path(cpu, domain_level, 0, 'cap_states')
            cap_states_strs = read_sge_file(cap_states_path).split()
            nr_cap_states_path = sge_path(cpu, domain_level, 0, 'nr_cap_states')
            nr_cap_states_strs = read_sge_file(nr_cap_states_path).split()
            # there are potentially two formats for this data which can be
            # differentiated by knowing how many strings were obtained when
            # we split cap_states *and* how many cap states there are.
            # If the split has 2x the number of states, the reported states
            # are from a kernel without frequency-model support and there
            # two values per state. If the split has 3x the number of states
            # then the reported states are from a kernel which *has*
            # frequency model support, and each state has three values to parse.
            nr_values = len(cap_states_strs)
            nr_states = int(nr_cap_states_strs[0])
            em_member_count = int(nr_values / nr_states)
            if em_member_count not in (2, 3):
                raise ValueError(f'Unsupported cap_states format cpu={cpu} domain_level={domain_level} path={cap_states_path}')

            # Here we split the incoming cap_states_strs list into em_member_count lists, so that
            # we can use the first one (representing capacity) and the last one (representing power)
            # to build the EM class. What we get is
            # for a 2-element list:
            #   [c0, p0, c1, p1, c2, p2] -> [(c0, p0), (c1, p1), (c2, p2)]
            # or for a 3-element list:
            #   [c0, f0, p0, c1, f1, p1, c2, f2, p2] -> [(c0, f0, p0), (c1, f1, p1), (c2, f2, p2)]
            # it's generic, and doesn't care if the EM gets any more values in between so long as the
            # capacity is first and power is last.
            cap_states = [
                ActiveState(capacity=int(c), power=int(p))
                for c, p in map(
                    lambda x: (x[0], x[-1]),
                    grouper(cap_states_strs, em_member_count)
                )
            ]

            freqs = target.cpufreq.list_frequencies(cpu)
            return OrderedDict(zip(sorted(freqs), cap_states))

        def read_idle_states(cpu, domain_level):
            idle_states_path = sge_path(cpu, domain_level, 0, 'idle_states')
            idle_states_strs = read_sge_file(idle_states_path).split()

            # get_states should return the state names in increasing depth order
            names = cls._get_idle_states_name(target, cpu, only_real=True)
            # idle_states is a list of power values in increasing order of
            # idle-depth/decreasing order of power.
            return OrderedDict(zip(names, map(int, idle_states_strs)))

        def _find_core_groups(target):
            """
            Read the core_siblings masks for each CPU from sysfs

            :returns: A list of tuples of ints, representing the partition of core
                    siblings
            """
            topology_base = '/sys/devices/system/cpu/'

            # We only care about core_siblings, but let's check *_siblings, so we
            # can throw an error if a CPU's thread_siblings isn't just itself, or if
            # there's a topology level we don't understand.

            # Since we might have to read a lot of files, read everything we need in
            # one go to avoid taking too long.
            mask_glob = topology_base + 'cpu**/topology/*_siblings'
            file_values = _read_multiple_oneline_files(target, [mask_glob])

            regex = re.compile(
                topology_base + r'cpu(?P<cpu>[0-9]+)/topology/(?P<level>[a-z]+)_siblings')

            ret = set()

            for path, mask_str in file_values.items():
                match = regex.match(path)
                cpu = int(match.group('cpu'))
                level = match.group('level')
                # mask_to_list returns the values in descending order, so we'll sort
                # them ascending. This isn't strictly necessary but it's nicer.
                siblings = tuple(sorted(mask_to_list(int(mask_str, base=16))))

                if level == 'thread':
                    if siblings != (cpu,):
                        # SMT systems aren't supported
                        raise ValueError(
                            f'CPU{cpu} thread_siblings is {siblings}. Expected: {[cpu]}'
                        )
                elif level != 'core':
                    # The only other levels we should expect to find are 'book' and
                    # 'shelf', which are not used by architectures we support.
                    raise ValueError(
                        f'Unrecognised topology level "{level}"')
                else:
                    ret.add(siblings)

            # Sort core groups so that the lowest-numbered cores are first
            # Again, not strictly necessary, just more pleasant.
            return sorted(ret, key=operator.itemgetter(0))

        return cls(
            root_node=EnergyModelRoot(
                children=[
                    EnergyModelNode(
                        children=[
                            EnergyModelNode(
                                cpu=cpu,
                                active_states=read_active_states(cpu, 0),
                                idle_states=read_idle_states(cpu, 0)
                            )
                            # Read the CPU-level data from sched_domain level 0
                            for cpu in core_group
                        ],
                        active_states=read_active_states(core_group[0], 1),
                        idle_states=read_idle_states(core_group[0], 1)
                    )
                    # Read the "cluster" level data from sched_domain level 1
                    for core_group in _find_core_groups(target)
                ]
            ),
            # We don't have a way to read the power domains from sysfs (the kernel
            # isn't even aware of them) so we'll just have to assume each CPU is its
            # own power domain and all idle states are independent of each other.
            root_power_domain=PowerDomain(
                children=[
                    PowerDomain(
                        cpu=cpu,
                        idle_states=cls._get_idle_states_name(target, cpu, only_real=True),
                    )
                    for cpu in cpus
                ],
                idle_states=[],
            ),
            freq_domains=sorted(target.cpufreq.iter_domains()),
        )



# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
