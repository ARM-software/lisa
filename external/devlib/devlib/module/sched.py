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
from enum import Enum

from past.builtins import basestring

from devlib.module import Module
from devlib.utils.misc import memoized
from devlib.utils.types import boolean


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

    PACKABLE_ENTRIES = [
        "cpu",
        "domain",
        "group"
    ]

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
        match = re.search(SchedProcFSNode._re_procfs_node, node)
        if match:
            return match.group("name")

        return node

    @classmethod
    def _packable(cls, node):
        """
        :returns: Whether it makes sense to pack a node into a common entry
        """
        return (SchedProcFSNode._ends_with_digits(node) and
                SchedProcFSNode._node_name(node) in cls.PACKABLE_ENTRIES)

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
            for node in list(nodes.keys()) if SchedProcFSNode._packable(node)
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


class DocInt(int):

    # See https://stackoverflow.com/a/50473952/5096023
    def __new__(cls, value, doc):
        new = super(DocInt, cls).__new__(cls, value)
        new.__doc__ = doc
        return new


class SchedDomainFlag(DocInt, Enum):
    """
    Represents a sched domain flag
    """
    # pylint: disable=bad-whitespace
    # Domain flags obtained from include/linux/sched/topology.h on v4.17
    # https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/v4.17/include/linux/sched/topology.h#20
    SD_LOAD_BALANCE =        0x0001, "Do load balancing on this domain"
    SD_BALANCE_NEWIDLE =     0x0002, "Balance when about to become idle"
    SD_BALANCE_EXEC =        0x0004, "Balance on exec"
    SD_BALANCE_FORK =        0x0008, "Balance on fork, clone"
    SD_BALANCE_WAKE =        0x0010, "Balance on wakeup"
    SD_WAKE_AFFINE =         0x0020, "Wake task to waking CPU"
    SD_ASYM_CPUCAPACITY =    0x0040, "Groups have different max cpu capacities"
    SD_SHARE_CPUCAPACITY =   0x0080, "Domain members share cpu capacity"
    SD_SHARE_POWERDOMAIN =   0x0100, "Domain members share power domain"
    SD_SHARE_PKG_RESOURCES = 0x0200, "Domain members share cpu pkg resources"
    SD_SERIALIZE =           0x0400, "Only a single load balancing instance"
    SD_ASYM_PACKING =        0x0800, "Place busy groups earlier in the domain"
    SD_PREFER_SIBLING =      0x1000, "Prefer to place tasks in a sibling domain"
    SD_OVERLAP =             0x2000, "Sched_domains of this level overlap"
    SD_NUMA =                0x4000, "Cross-node balancing"
    # Only defined in Android
    # https://android.googlesource.com/kernel/common/+/android-4.14/include/linux/sched/topology.h#29
    SD_SHARE_CAP_STATES =    0x8000, "(Android only) Domain members share capacity state"

    @classmethod
    def check_version(cls, target, logger):
        """
        Check the target and see if its kernel version matches our view of the world
        """
        parts = target.kernel_version.parts
        # Checked to be valid from v4.4
        # Not saved as a class attribute else it'll be converted to an enum
        ref_parts = (4, 4, 0)
        if parts < ref_parts:
            logger.warn(
                "Sched domain flags are defined for kernels v{} and up, "
                "but target is running v{}".format(ref_parts, parts)
            )


    def __str__(self):
        return self.name


class SchedDomain(SchedProcFSNode):
    """
    Represents a sched domain as seen through procfs
    """
    def __init__(self, nodes):
        super(SchedDomain, self).__init__(nodes)

        obj_flags = set()
        for flag in list(SchedDomainFlag):
            if self.flags & flag.value == flag.value:
                obj_flags.add(flag)

        self.flags = obj_flags


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
            if target.file_exists(target.path.join(path, cpu, "domain0", "flags")):
                return True

        return False

    def __init__(self, target, path=None):
        if path is None:
            path = self.sched_domain_root

        procfs = target.read_tree_values(path, depth=self._read_depth)
        super(SchedProcFSData, self).__init__(procfs)


class SchedModule(Module):

    name = 'sched'

    cpu_sysfs_root = '/sys/devices/system/cpu'

    @staticmethod
    def probe(target):
        logger = logging.getLogger(SchedModule.name)
        SchedDomainFlag.check_version(target, logger)

        # It makes sense to load this module if at least one of those
        # functionalities is enabled
        schedproc = SchedProcFSData.available(target)
        debug = SchedModule.target_has_debug(target)
        dmips = any([target.file_exists(SchedModule.cpu_dmips_capacity_path(target, cpu))
                     for cpu in target.list_online_cpus()])

        logger.info("Scheduler sched_domain procfs entries %s",
                    "found" if schedproc else "not found")
        logger.info("Detected kernel compiled with SCHED_DEBUG=%s",
                    "y" if debug else "n")
        logger.info("CPU capacity sysfs entries %s",
                    "found" if dmips else "not found")

        return schedproc or debug or dmips

    def get_kernel_attributes(self, matching=None, check_exit_code=True):
        """
        Get the value of scheduler attributes.

        :param matching: an (optional) substring to filter the scheduler
        attributes to be returned.

        The scheduler exposes a list of tunable attributes under:
            /proc/sys/kernel
        all starting with the "sched_" prefix.

        This method returns a dictionary of all the "sched_" attributes exposed
        by the target kernel, within the prefix removed.
        It's possible to restrict the list of attributes by specifying a
        substring to be matched.

        returns: a dictionary of scheduler tunables
        """
        command = 'sched_get_kernel_attributes {}'.format(
            matching if matching else ''
        )
        output = self.target._execute_util(command, as_root=self.target.is_rooted,
                                           check_exit_code=check_exit_code)
        result = {}
        for entry in output.strip().split('\n'):
            if ':' not in entry:
                continue
            path, value = entry.strip().split(':', 1)
            if value in ['0', '1']:
                value = bool(int(value))
            elif value.isdigit():
                value = int(value)
            result[path] = value
        return result

    def set_kernel_attribute(self, attr, value, verify=True):
        """
        Set the value of a scheduler attribute.

        :param attr: the attribute to set, without the "sched_" prefix
        :param value: the value to set
        :param verify: true to check that the requested value has been set

        :raise TargetError: if the attribute cannot be set
        """
        if isinstance(value, bool):
            value = '1' if value else '0'
        elif isinstance(value, int):
            value = str(value)
        path = '/proc/sys/kernel/sched_' + attr
        self.target.write_value(path, value, verify)

    @classmethod
    def target_has_debug(cls, target):
        if target.config.get('SCHED_DEBUG') != 'y':
            return False
        return target.file_exists('/sys/kernel/debug/sched_features')

    @property
    @memoized
    def has_debug(self):
        return self.target_has_debug(self.target)

    def get_features(self):
        """
        Get the status of each sched feature

        :returns: a dictionary of features and their "is enabled" status
        """
        if not self.has_debug:
            raise RuntimeError("sched_features not available")
        feats = self.target.read_value('/sys/kernel/debug/sched_features')
        features = {}
        for feat in feats.split():
            value = True
            if feat.startswith('NO'):
                feat = feat.replace('NO_', '', 1)
                value = False
            features[feat] = value
        return features

    def set_feature(self, feature, enable, verify=True):
        """
        Set the status of a specified scheduler feature

        :param feature: the feature name to set
        :param enable: true to enable the feature, false otherwise

        :raise ValueError: if the specified enable value is not bool
        :raise RuntimeError: if the specified feature cannot be set
        """
        if not self.has_debug:
            raise RuntimeError("sched_features not available")
        feature = feature.upper()
        feat_value = feature
        if not boolean(enable):
            feat_value = 'NO_' + feat_value
        self.target.write_value('/sys/kernel/debug/sched_features',
                                feat_value, verify=False)
        if not verify:
            return
        msg = 'Failed to set {}, feature not supported?'.format(feat_value)
        features = self.get_features()
        feat_value = features.get(feature, not enable)
        if feat_value != enable:
            raise RuntimeError(msg)

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
            sd = self.get_cpu_sd_info(cpu)

        return sd.procfs["domain0"].get("group0", {}).get("energy", {}).get("cap_states") != None

    @classmethod
    def cpu_dmips_capacity_path(cls, target, cpu):
        """
        :returns: The target sysfs path where the dmips capacity data should be
        """
        return target.path.join(
            cls.cpu_sysfs_root,
            'cpu{}/cpu_capacity'.format(cpu))

    @memoized
    def has_dmips_capacity(self, cpu):
        """
        :returns: Whether dmips capacity data is available for 'cpu'
        """
        return self.target.file_exists(
            self.cpu_dmips_capacity_path(self.target, cpu)
        )

    @memoized
    def get_em_capacity(self, cpu, sd=None):
        """
        :returns: The maximum capacity value exposed by the EAS energy model
        """
        if not sd:
            sd = self.get_cpu_sd_info(cpu)

        cap_states = sd.domains[0].groups[0].energy.cap_states
        cap_states_list = cap_states.split('\t')
        num_cap_states = sd.domains[0].groups[0].energy.nr_cap_states
        max_cap_index = -1 * int(len(cap_states_list) / num_cap_states)
        return int(cap_states_list[max_cap_index])

    @memoized
    def get_dmips_capacity(self, cpu):
        """
        :returns: The capacity value generated from the capacity-dmips-mhz DT entry
        """
        return self.target.read_value(
            self.cpu_dmips_capacity_path(self.target, cpu), int
        )

    def get_capacities(self, default=None):
        """
        :param default: Default capacity value to find if no data is
        found in procfs

        :returns: a dictionnary of the shape {cpu : capacity}

        :raises RuntimeError: Raised when no capacity information is
        found and 'default' is None
        """
        cpus = self.target.list_online_cpus()

        capacities = {}
        sd_info = self.get_sd_info()

        for cpu in cpus:
            if self.has_dmips_capacity(cpu):
                capacities[cpu] = self.get_dmips_capacity(cpu)
            elif self.has_em(cpu, sd_info.cpus[cpu]):
                capacities[cpu] = self.get_em_capacity(cpu, sd_info.cpus[cpu])
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
