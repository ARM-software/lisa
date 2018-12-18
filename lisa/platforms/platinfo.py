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

from collections.abc import Mapping

from lisa.utils import HideExekallID, memoized
from lisa.conf import DeferredValue, IntIntDict, IntListList, IntIntListDict, StrIntListDict, MultiSrcConf, KeyDesc, LevelKeyDesc, TopLevelKeyDesc
from lisa.energy_model import EnergyModel
from lisa.wlgen.rta import RTA

from devlib.target import KernelVersion
from devlib.exception import TargetStableError

class PlatformInfo(MultiSrcConf, HideExekallID):
    """
    Platform-specific information made available to tests.

    .. warning::
        The follwing keys are here for compatibility with old code only, do not
        write new code depending on them:

            * topology
            * clusters
            * freqs

    {generated_help}

    """

    # we could use mypy.subtypes.is_subtype and use the infrastructure provided
    # by typing module, but adding an external dependency is overkill for what
    # we need.
    STRUCTURE = TopLevelKeyDesc('platform-info', 'Platform-specific information', (
        LevelKeyDesc('rtapp', 'RTapp configuration', (
            KeyDesc('calib', 'RTapp calibration dictionary', [IntIntDict]),
        )),
        KeyDesc('nrg-model', 'Energy model object', [EnergyModel]),
        KeyDesc('cpu-capacities', 'Dictionary of CPU ID to capacity value', [IntIntDict]),
        KeyDesc('kernel-version', '', [KernelVersion]),
        KeyDesc('abi', 'ABI, e.g. "arm64"', [str]),
        KeyDesc('os', 'OS being used, e.g. "linux"', [str]),
        KeyDesc('name', 'Free-form name of the board', [str]),
        KeyDesc('cpus-count', 'Number of CPUs', [int]),
        KeyDesc('freq-domains', 'Frequency domains modeled by a list of CPU id for each domain', [IntListList]),
        KeyDesc('freqs', 'Dictionnary of CPU ID to list of frequencies', [IntIntListDict]),
    ))
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
            },
            'cpus-count': te.target.number_of_cpus
        }

        if hasattr(target, 'cpufreq'):
            info['freq-domains'] = list(target.cpufreq.iter_domains())
            info['freqs'] = {cpu : target.cpufreq.list_frequencies(cpu)
                             for cpu in range(target.number_of_cpus)}

        if hasattr(target, 'sched'):
            info['cpu-capacities'] = target.sched.get_capacities(default=1024)

        return self.add_src(src, info, filter_none=True, **kwargs)

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

    # OLD CODE AFTER REFACTORING, FOR REFERENCE OF WHAT IS THE EXPECTED FORMAT
    # AND CONTENT OF THE COMPATIBILITY KEYS
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
