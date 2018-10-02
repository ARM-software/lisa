# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

import inspect
import contextlib
from collections import ChainMap
from collections.abc import Mapping
from lisa.utils import HideExekallID, MultiSrcConf, memoized, TypedDict, TypedList, DeferredValue
from lisa.energy_model import EnergyModel
from lisa.wlgen.rta import RTA

from trappy.stats.Topology import Topology
from devlib.target import KernelVersion
from devlib.exception import TargetStableError

class PlatformInfo(MultiSrcConf, HideExekallID):
    YAML_MAP_TOP_LEVEL_KEY = 'platform-info'

    # we could use mypy.subtypes.is_subtype and use the infrastructure provided
    # by typing module, but adding an external dependency is overkill for what
    # we need.
    STRUCTURE = {
        'rtapp': {
            'calib': TypedDict[int, int],
        },
        'nrg-model': EnergyModel,
        'kernel-version': KernelVersion,
        'abi': str,
        'os': str,
        'name': str,

        # TODO: remove that once no code depend on it anymore
        'topology': Topology,
        'clusters': TypedDict[str, TypedList[int]],
        'cpus-count': int,
        'freqs': TypedDict[str, TypedList[int]],
    }
    """Some keys have a reserved meaning with an associated type."""

    def __init__(self, conf=None, src='user'):
        super().__init__(conf=conf, src=src)

    def add_target_src(self, te, src='target', **kwargs):
        target = te.target
        info = {
            'nrg-model': self._nrg_model_from_target(target),
            'kernel-version': target.kernel_version,
            'abi': target.abi,
            'os': target.os,
            'rtapp': {
                # Since it is expensive to compute, use an on-demand DeferredValue
                'calib': DeferredValue(RTA.get_cpu_calibrations, te)
            }
        }

        #TODO: kill that once code depending on this has been converted to
        # using the appropriate "root" data, instead of these derived values.

        # Derive all the deprecated keys from the nrg_model
        nrg_model = info['nrg-model']
        # If the EnergyModel was not read properly
        if nrg_model is not None:
            node_groups = nrg_model.node_groups

            # Sort according to max capacity found in the group
            def max_capacity(group):
                return max(
                    s.capacity
                    for node in group
                    for s in node.active_states.values()
                )
            node_groups = sorted(node_groups, key=max_capacity)
            cpu_groups = [
                [node.cpu for node in group]
                for group in node_groups
            ]

            # big.LITTLE platform
            if len(cpu_groups) == 2:
                cluster_names = ['little', 'big']
            # SMP platform
            else:
                cluster_names = [str(i) for i in range(len(cpu_groups))]
            clusters = {
                name: group
                for name, group in zip(cluster_names, cpu_groups)
            }

            topology = Topology(clusters=cpu_groups)
            cpus_count = sum(len(group) for group in cpu_groups)

            def freq_list(group):
                return sorted(set(
                    freq
                    for node in group
                    for freq in node.active_states.keys()
                ))

            freqs = {
                cluster_name: freq_list(group)
                for cluster_name, group in zip(cluster_names, node_groups)
            }

            info.update({
                'clusters': clusters,
                'topology': topology,
                'cpus-count': cpus_count,
                'freqs': freqs,
            })

        return self.add_src(src, info, filter_none=True, **kwargs)

    @property
    def tags(self):
        try:
            name = self['name']
        except KeyError:
            return []
        else:
            return [name]

    # Internal methods used to compute some keys from a live devlib Target

    @classmethod
    def _nrg_model_from_target(cls, target):
        logger = cls.get_logger()
        logger.info('Attempting to read energy model from target')
        try:
            return EnergyModel.from_target(target)
        except (TargetStableError, RuntimeError, ValueError) as err:
            logger.error("Couldn't read target energy model: %s", err)
            return None

    #  @classmethod
    #  def _topology_from_target(cls, target):
        #  logger = cls.get_logger()
        #  # Initialize target Topology for behavior analysis
        #  clusters = []

        #  # Build topology for a big.LITTLE systems
        #  if target.big_core and \
           #  (target.abi == 'arm64' or target.abi == 'armeabi'):
            #  # Populate cluster for a big.LITTLE platform
            #  if target.big_core:
                #  # Load cluster of LITTLE cores
                #  clusters.append(
                    #  [i for i,t in enumerate(target.core_names)
                                #  if t == target.little_core])
                #  # Load cluster of big cores
                #  clusters.append(
                    #  [i for i,t in enumerate(target.core_names)
                                #  if t == target.big_core])
        #  # Build topology for an SMP systems
        #  elif not target.big_core or \
             #  target.abi == 'x86_64':
            #  for core in set(target.core_clusters):
                #  clusters.append(
                    #  [i for i,v in enumerate(target.core_clusters)
                     #  if v == core])
        #  logger.info('Topology:')
        #  logger.info('   %s', clusters)

        #  return Topology(clusters=clusters)

    # ANY CODE RELYING ON THAT SHOULD BE UPDATED TO USE THE Topology OBJECT
    # The following 2 methods provide platform keys:
    # * 'clusters'
    # * 'freqs'
    # * 'cpus_count'
    #  @staticmethod
    #  def _init_plat_info[cpus-count]bl(target):
        #  #TODO: replace with PlatformInfo
        #  platform = {
            #  'clusters' : {
                #  'little'    : target.bl.littles,
                #  'big'       : target.bl.bigs
            #  },
            #  'freqs' : {
                #  'little'    : target.bl.list_littles_frequencies(),
                #  'big'       : target.bl.list_bigs_frequencies()
            #  }
        #  }
        #  plat_info['cpus-count'] = \
            #  len(platform['clusters']['little']) + \
            #  len(platform['clusters']['big'])

        #  return platform

    #  @classmethod
    #  def _init_plat_info[cpus-count]smp(cls, target):
        #  logger = cls.get_logger()
        #  #TODO: replace with PlatformInfo
        #  clusters = {}
        #  freqs = {}

        #  for cpu_id, cluster_id in enumerate(target.core_clusters):
            #  clusters.setdefault(cluster_id, []).append(cpu_id)

        #  if 'cpufreq' in target.modules:
            #  # Try loading frequencies using the cpufreq module
            #  for cluster_id, cluster in clusters.items():
                #  core_id = cluster[0]
                #  freqs[cluster_id] = \
                    #  target.cpufreq.list_frequencies(core_id)
        #  else:
            #  logger.warning('Unable to identify cluster frequencies')

        #  # TODO: get the performance boundaries in case of intel_pstate driver

        #  platform = dict(
            #  clusters = clusters,
            #  freqs = freqs,
            #  cpus_count = len(target.core_clusters)
        #  }
        #  return platform

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
