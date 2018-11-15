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

import logging
import re

from past.builtins import basestring

from devlib.module import Module
from devlib.utils.misc import memoized


class SchedProcFSNode(object):
    """
    Represents a sched_domain procfs node

    :param nodes: Dictionnary view of the underlying procfs nodes
        (as returned by devlib.read_tree_values())
    :type nodes: dict


    Say you want to represent this path/data:
    $ cat /proc/sys/kernel/sched_domain/cpu0/domain*/name
    MC
    DIE

    Taking cpu0 as a root, this can be defined as:
    >>> data = {"domain0" : {"name" : "MC"}, "domain1" : {"name" : "DIE"}}

    >>> repr = SchedProcFSNode(data)
    >>> print repr.domains[0].name
    MC

    The "raw" dict remains available under the `procfs` field:
    >>> print repr.procfs["domain0"]["name"]
    MC
    """

    _re_procfs_node = re.compile(r"(?P<name>.*\D)(?P<digits>\d+)$")

    @staticmethod
    def _ends_with_digits(node):
        if not isinstance(node, basestring):
            return False

        return re.search(SchedProcFSNode._re_procfs_node, node) != None

    @staticmethod
    def _node_digits(node):
        """
        :returns: The ending digits of the procfs node
        """
        return int(re.search(SchedProcFSNode._re_procfs_node, node).group("digits"))

    @staticmethod
    def _node_name(node):
        """
        :returns: The name of the procfs node
        """
        return re.search(SchedProcFSNode._re_procfs_node, node).group("name")

    @staticmethod
    def _packable(node, entries):
        """
        :returns: Whether it makes sense to pack a node into a common entry
        """
        return (SchedProcFSNode._ends_with_digits(node) and
                any([SchedProcFSNode._ends_with_digits(x) and
                     SchedProcFSNode._node_digits(x) != SchedProcFSNode._node_digits(node) and
                     SchedProcFSNode._node_name(x) == SchedProcFSNode._node_name(node)
                     for x in entries]))

    @staticmethod
    def _build_directory(node_name, node_data):
        if node_name.startswith("domain"):
            return SchedDomain(node_data)
        else:
            return SchedProcFSNode(node_data)

    @staticmethod
    def _build_entry(node_data):
        value = node_data

        # Most nodes just contain numerical data, try to convert
        try:
            value = int(value)
        except ValueError:
            pass

        return value

    @staticmethod
    def _build_node(node_name, node_data):
        if isinstance(node_data, dict):
            return SchedProcFSNode._build_directory(node_name, node_data)
        else:
            return SchedProcFSNode._build_entry(node_data)

    def __getattr__(self, name):
        return self._dyn_attrs[name]

    def __init__(self, nodes):
        self.procfs = nodes
        # First, reduce the procs fields by packing them if possible
        # Find which entries can be packed into a common entry
        packables = {
            node : SchedProcFSNode._node_name(node) + "s"
            for node in list(nodes.keys()) if SchedProcFSNode._packable(node, list(nodes.keys()))
        }

        self._dyn_attrs = {}

        for dest in set(packables.values()):
            self._dyn_attrs[dest] = {}

        # Pack common entries
        for key, dest in packables.items():
            i = SchedProcFSNode._node_digits(key)
            self._dyn_attrs[dest][i] = self._build_node(key, nodes[key])

        # Build the other nodes
        for key in nodes.keys():
            if key in packables:
                continue

            self._dyn_attrs[key] = self._build_node(key, nodes[key])


class SchedDomain(SchedProcFSNode):
    """
    Represents a sched domain as seen through procfs
    """
    # pylint: disable=bad-whitespace
    # Domain flags obtained from include/linux/sched/topology.h on v4.17
    # https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/v4.17/include/linux/sched/topology.h#20
    SD_LOAD_BALANCE        = 0x0001  # Do load balancing on this domain.
    SD_BALANCE_NEWIDLE     = 0x0002  # Balance when about to become idle
    SD_BALANCE_EXEC        = 0x0004  # Balance on exec
    SD_BALANCE_FORK        = 0x0008  # Balance on fork, clone
    SD_BALANCE_WAKE        = 0x0010  # Balance on wakeup
    SD_WAKE_AFFINE         = 0x0020  # Wake task to waking CPU
    SD_ASYM_CPUCAPACITY    = 0x0040  # Groups have different max cpu capacities
    SD_SHARE_CPUCAPACITY   = 0x0080  # Domain members share cpu capacity
    SD_SHARE_POWERDOMAIN   = 0x0100  # Domain members share power domain
    SD_SHARE_PKG_RESOURCES = 0x0200  # Domain members share cpu pkg resources
    SD_SERIALIZE           = 0x0400  # Only a single load balancing instance
    SD_ASYM_PACKING        = 0x0800  # Place busy groups earlier in the domain
    SD_PREFER_SIBLING      = 0x1000  # Prefer to place tasks in a sibling domain
    SD_OVERLAP             = 0x2000  # sched_domains of this level overlap
    SD_NUMA                = 0x4000  # cross-node balancing
    # Only defined in Android
    # https://android.googlesource.com/kernel/common/+/android-4.14/include/linux/sched/topology.h#29
    SD_SHARE_CAP_STATES    = 0x8000  # Domain members share capacity state

    # Checked to be valid from v4.4
    SD_FLAGS_REF_PARTS = (4, 4, 0)

    @staticmethod
    def check_version(target, logger):
        """
        Check the target and see if its kernel version matches our view of the world
        """
        parts = target.kernel_version.parts
        if parts < SchedDomain.SD_FLAGS_REF_PARTS:
            logger.warn(
                "Sched domain flags are defined for kernels v{} and up, "
                "but target is running v{}".format(SchedDomain.SD_FLAGS_REF_PARTS, parts)
            )

    def has_flags(self, flags):
        """
        :returns: Whether 'flags' are set on this sched domain
        """
        return self.flags & flags == flags


class SchedProcFSData(SchedProcFSNode):
    """
    Root class for creating & storing SchedProcFSNode instances
    """
    _read_depth = 6
    sched_domain_root = '/proc/sys/kernel/sched_domain'

    @staticmethod
    def available(target):
        path = SchedProcFSData.sched_domain_root
        cpus = target.list_directory(path) if target.file_exists(path) else []

        if not cpus:
            return False

        # Even if we have a CPU entry, it can be empty (e.g. hotplugged out)
        # Make sure some data is there
        for cpu in cpus:
            if target.file_exists(target.path.join(path, cpu, "domain0", "name")):
                return True

        return False

    def __init__(self, target, path=None):
        if not path:
            path = self.sched_domain_root

        procfs = target.read_tree_values(path, depth=self._read_depth)
        super(SchedProcFSData, self).__init__(procfs)


class SchedModule(Module):

    name = 'sched'

    cpu_sysfs_root = '/sys/devices/system/cpu'

    @staticmethod
    def probe(target):
        logger = logging.getLogger(SchedModule.name)
        SchedDomain.check_version(target, logger)

        return SchedProcFSData.available(target)

    def get_cpu_sd_info(self, cpu):
        """
        :returns: An object view of /proc/sys/kernel/sched_domain/cpu<cpu>/*
        """
        path = self.target.path.join(
            SchedProcFSData.sched_domain_root,
            "cpu{}".format(cpu)
        )

        return SchedProcFSData(self.target, path)

    def get_sd_info(self):
        """
        :returns: An object view of /proc/sys/kernel/sched_domain/*
        """
        return SchedProcFSData(self.target)

    def get_capacity(self, cpu):
        """
        :returns: The capacity of 'cpu'
        """
        return self.get_capacities()[cpu]

    @memoized
    def has_em(self, cpu, sd=None):
        """
        :returns: Whether energy model data is available for 'cpu'
        """
        if not sd:
            sd = SchedProcFSData(self.target, cpu)

        return sd.procfs["domain0"].get("group0", {}).get("energy", {}).get("cap_states") != None

    @memoized
    def has_dmips_capacity(self, cpu):
        """
        :returns: Whether dmips capacity data is available for 'cpu'
        """
        return self.target.file_exists(
            self.target.path.join(self.cpu_sysfs_root, 'cpu{}/cpu_capacity'.format(cpu))
        )

    @memoized
    def get_em_capacity(self, cpu, sd=None):
        """
        :returns: The maximum capacity value exposed by the EAS energy model
        """
        if not sd:
            sd = SchedProcFSData(self.target, cpu)

        cap_states = sd.domains[0].groups[0].energy.cap_states
        return int(cap_states.split('\t')[-2])

    @memoized
    def get_dmips_capacity(self, cpu):
        """
        :returns: The capacity value generated from the capacity-dmips-mhz DT entry
        """
        return self.target.read_value(
            self.target.path.join(
                self.cpu_sysfs_root,
                'cpu{}/cpu_capacity'.format(cpu)
            ),
            int
        )

    @memoized
    def get_capacities(self, default=None):
        """
        :param default: Default capacity value to find if no data is
        found in procfs

        :returns: a dictionnary of the shape {cpu : capacity}

        :raises RuntimeError: Raised when no capacity information is
        found and 'default' is None
        """
        cpus = list(range(self.target.number_of_cpus))

        capacities = {}
        sd_info = self.get_sd_info()

        for cpu in cpus:
            if self.has_em(cpu, sd_info.cpus[cpu]):
                capacities[cpu] = self.get_em_capacity(cpu, sd_info.cpus[cpu])
            elif self.has_dmips_capacity(cpu):
                capacities[cpu] = self.get_dmips_capacity(cpu)
            else:
                if default != None:
                    capacities[cpu] = default
                else:
                    raise RuntimeError('No capacity data for cpu{}'.format(cpu))

        return capacities

    @memoized
    def get_hz(self):
        """
        :returns: The scheduler tick frequency on the target
        """
        return int(self.target.config.get('CONFIG_HZ', strict=True))
