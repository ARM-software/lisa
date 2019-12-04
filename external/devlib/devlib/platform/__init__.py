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


BIG_CPUS = ['A15', 'A57', 'A72', 'A73']


class Platform(object):

    @property
    def number_of_clusters(self):
        return len(set(self.core_clusters))

    def __init__(self,
                 name=None,
                 core_names=None,
                 core_clusters=None,
                 big_core=None,
                 model=None,
                 modules=None,
                 ):
        self.name = name
        self.core_names = core_names or []
        self.core_clusters = core_clusters or []
        self.big_core = big_core
        self.little_core = None
        self.model = model
        self.modules = modules or []
        self.logger = logging.getLogger(self.name)
        if not self.core_clusters and self.core_names:
            self._set_core_clusters_from_core_names()

    def init_target_connection(self, target):
        # May be ovewritten by subclasses to provide target-specific
        # connection initialisation.
        pass

    def update_from_target(self, target):
        if not self.core_names:
            self.core_names = target.cpuinfo.cpu_names
            self._set_core_clusters_from_core_names()
        if not self.big_core and self.number_of_clusters == 2:
            self.big_core = self._identify_big_core()
        if not self.core_clusters and self.core_names:
            self._set_core_clusters_from_core_names()
        if not self.model:
            self._set_model_from_target(target)
        if not self.name:
            self.name = self.model
        self._validate()

    def setup(self, target):
        # May be overwritten by subclasses to provide platform-specific
        # setup procedures.
        pass

    def _set_core_clusters_from_core_names(self):
        self.core_clusters = []
        clusters = []
        for cn in self.core_names:
            if cn not in clusters:
                clusters.append(cn)
            self.core_clusters.append(clusters.index(cn))

    def _set_model_from_target(self, target):
        if target.os == 'android':
            try:
                self.model = target.getprop(prop='ro.product.device')
            except KeyError:
                self.model = target.getprop('ro.product.model')
        elif target.file_exists("/proc/device-tree/model"):
            # There is currently no better way to do this cross platform.
            # ARM does not have dmidecode
            raw_model = target.execute("cat /proc/device-tree/model")
            device_model_to_return = '_'.join(raw_model.split()[:2])
            return device_model_to_return.rstrip(' \t\r\n\0')
        elif target.is_rooted:
            try:
                self.model = target.execute('dmidecode -s system-version',
                                            as_root=True).strip()
            except Exception:  # pylint: disable=broad-except
                pass  # this is best-effort

    def _identify_big_core(self):
        for core in self.core_names:
            if core.upper() in BIG_CPUS:
                return core
        big_idx = self.core_clusters.index(max(self.core_clusters))
        return self.core_names[big_idx]

    def _validate(self):
        if len(self.core_names) != len(self.core_clusters):
            raise ValueError('core_names and core_clusters are of different lengths.')
        if self.big_core and self.number_of_clusters != 2:
            raise ValueError('attempting to set big_core on non-big.LITTLE device. '
                             '(number of clusters  is not 2)')
        if self.big_core and self.big_core not in self.core_names:
            message = 'Invalid big_core value "{}"; must be in [{}]'
            raise ValueError(message.format(self.big_core,
                                            ', '.join(set(self.core_names))))
        if self.big_core:
            for core in self.core_names:
                if core != self.big_core:
                    self.little_core = core
                    break
