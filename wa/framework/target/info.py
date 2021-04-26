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
# pylint: disable=protected-access

import os

from devlib import AndroidTarget, TargetError
from devlib.target import KernelConfig, KernelVersion, Cpuinfo
from devlib.utils.android import AndroidProperties

from wa.framework.configuration.core import settings
from wa.framework.exception import ConfigError
from wa.utils.serializer import read_pod, write_pod, Podable
from wa.utils.misc import atomic_write_path


def cpuinfo_from_pod(pod):
    cpuinfo = Cpuinfo('')
    cpuinfo.sections = pod['cpuinfo']
    lines = []
    for section in cpuinfo.sections:
        for key, value in section.items():
            line = '{}: {}'.format(key, value)
            lines.append(line)
        lines.append('')
    cpuinfo.text = '\n'.join(lines)
    return cpuinfo


def kernel_version_from_pod(pod):
    release_string = pod['kernel_release']
    version_string = pod['kernel_version']
    if release_string:
        if version_string:
            kernel_string = '{} #{}'.format(release_string, version_string)
        else:
            kernel_string = release_string
    else:
        kernel_string = '#{}'.format(version_string)
    return KernelVersion(kernel_string)


def kernel_config_from_pod(pod):
    config = KernelConfig('')
    config.typed_config._config = pod['kernel_config']
    lines = []
    for key, value in config.items():
        if value == 'n':
            lines.append('# {} is not set'.format(key))
        else:
            lines.append('{}={}'.format(key, value))
    config.text = '\n'.join(lines)
    return config


class CpufreqInfo(Podable):

    _pod_serialization_version = 1

    @staticmethod
    def from_pod(pod):
        pod = CpufreqInfo._upgrade_pod(pod)
        return CpufreqInfo(**pod)

    def __init__(self, **kwargs):
        super(CpufreqInfo, self).__init__()
        self.available_frequencies = kwargs.pop('available_frequencies', [])
        self.available_governors = kwargs.pop('available_governors', [])
        self.related_cpus = kwargs.pop('related_cpus', [])
        self.driver = kwargs.pop('driver', None)
        self._pod_version = kwargs.pop('_pod_version', self._pod_serialization_version)

    def to_pod(self):
        pod = super(CpufreqInfo, self).to_pod()
        pod.update(self.__dict__)
        return pod

    @staticmethod
    def _pod_upgrade_v1(pod):
        pod['_pod_version'] = pod.get('_pod_version', 1)
        return pod

    def __repr__(self):
        return 'Cpufreq({} {})'.format(self.driver, self.related_cpus)

    __str__ = __repr__


class IdleStateInfo(Podable):

    _pod_serialization_version = 1

    @staticmethod
    def from_pod(pod):
        pod = IdleStateInfo._upgrade_pod(pod)
        return IdleStateInfo(**pod)

    def __init__(self, **kwargs):
        super(IdleStateInfo, self).__init__()
        self.name = kwargs.pop('name', None)
        self.desc = kwargs.pop('desc', None)
        self.power = kwargs.pop('power', None)
        self.latency = kwargs.pop('latency', None)
        self._pod_version = kwargs.pop('_pod_version', self._pod_serialization_version)

    def to_pod(self):
        pod = super(IdleStateInfo, self).to_pod()
        pod.update(self.__dict__)
        return pod

    @staticmethod
    def _pod_upgrade_v1(pod):
        pod['_pod_version'] = pod.get('_pod_version', 1)
        return pod

    def __repr__(self):
        return 'IdleState({}/{})'.format(self.name, self.desc)

    __str__ = __repr__


class CpuidleInfo(Podable):

    _pod_serialization_version = 1

    @staticmethod
    def from_pod(pod):
        pod = CpuidleInfo._upgrade_pod(pod)
        instance = CpuidleInfo()
        instance._pod_version = pod['_pod_version']
        instance.governor = pod['governor']
        instance.driver = pod['driver']
        instance.states = [IdleStateInfo.from_pod(s) for s in pod['states']]
        return instance

    @property
    def num_states(self):
        return len(self.states)

    def __init__(self):
        super(CpuidleInfo, self).__init__()
        self.governor = None
        self.driver = None
        self.states = []

    def to_pod(self):
        pod = super(CpuidleInfo, self).to_pod()
        pod['governor'] = self.governor
        pod['driver'] = self.driver
        pod['states'] = [s.to_pod() for s in self.states]
        return pod

    @staticmethod
    def _pod_upgrade_v1(pod):
        pod['_pod_version'] = pod.get('_pod_version', 1)
        return pod

    def __repr__(self):
        return 'Cpuidle({}/{} {} states)'.format(
            self.governor, self.driver, self.num_states)

    __str__ = __repr__


class CpuInfo(Podable):

    _pod_serialization_version = 1

    @staticmethod
    def from_pod(pod):
        instance = super(CpuInfo, CpuInfo).from_pod(pod)
        instance.id = pod['id']
        instance.name = pod['name']
        instance.architecture = pod['architecture']
        instance.features = pod['features']
        instance.cpufreq = CpufreqInfo.from_pod(pod['cpufreq'])
        instance.cpuidle = CpuidleInfo.from_pod(pod['cpuidle'])
        return instance

    def __init__(self):
        super(CpuInfo, self).__init__()
        self.id = None
        self.name = None
        self.architecture = None
        self.features = []
        self.cpufreq = CpufreqInfo()
        self.cpuidle = CpuidleInfo()

    def to_pod(self):
        pod = super(CpuInfo, self).to_pod()
        pod['id'] = self.id
        pod['name'] = self.name
        pod['architecture'] = self.architecture
        pod['features'] = self.features
        pod['cpufreq'] = self.cpufreq.to_pod()
        pod['cpuidle'] = self.cpuidle.to_pod()
        return pod

    @staticmethod
    def _pod_upgrade_v1(pod):
        pod['_pod_version'] = pod.get('_pod_version', 1)
        return pod

    def __repr__(self):
        return 'Cpu({} {})'.format(self.id, self.name)

    __str__ = __repr__


def get_target_info(target):
    info = TargetInfo()
    info.target = target.__class__.__name__
    info.modules = target.modules
    info.os = target.os
    info.os_version = target.os_version
    info.system_id = target.system_id
    info.abi = target.abi
    info.is_rooted = target.is_rooted
    info.kernel_version = target.kernel_version
    info.kernel_config = target.config
    info.hostname = target.hostname
    info.hostid = target.hostid

    try:
        info.sched_features = target.read_value('/sys/kernel/debug/sched_features').split()
    except TargetError:
        # best effort -- debugfs might not be mounted
        pass

    for i, name in enumerate(target.cpuinfo.cpu_names):
        cpu = CpuInfo()
        cpu.id = i
        cpu.name = name
        cpu.features = target.cpuinfo.get_cpu_features(i)
        cpu.architecture = target.cpuinfo.architecture

        if target.has('cpufreq'):
            cpu.cpufreq.available_governors = target.cpufreq.list_governors(i)
            cpu.cpufreq.available_frequencies = target.cpufreq.list_frequencies(i)
            cpu.cpufreq.related_cpus = target.cpufreq.get_related_cpus(i)
            cpu.cpufreq.driver = target.cpufreq.get_driver(i)

        if target.has('cpuidle'):
            cpu.cpuidle.driver = target.cpuidle.get_driver()
            cpu.cpuidle.governor = target.cpuidle.get_governor()
            for state in target.cpuidle.get_states(i):
                state_info = IdleStateInfo()
                state_info.name = state.name
                state_info.desc = state.desc
                state_info.power = state.power
                state_info.latency = state.latency
                cpu.cpuidle.states.append(state_info)

        info.cpus.append(cpu)

    info.page_size_kb = target.page_size_kb

    if isinstance(target, AndroidTarget):
        info.screen_resolution = target.screen_resolution
        info.prop = target.getprop()
        info.android_id = target.android_id

    return info


def read_target_info_cache():
    if not os.path.exists(settings.cache_directory):
        os.makedirs(settings.cache_directory)
    if not os.path.isfile(settings.target_info_cache_file):
        return {}
    return read_pod(settings.target_info_cache_file)


def write_target_info_cache(cache):
    if not os.path.exists(settings.cache_directory):
        os.makedirs(settings.cache_directory)
    with atomic_write_path(settings.target_info_cache_file) as at_path:
        write_pod(cache, at_path)


def get_target_info_from_cache(system_id, cache=None):
    if cache is None:
        cache = read_target_info_cache()
    pod = cache.get(system_id, None)

    if not pod:
        return None

    _pod_version = pod.get('_pod_version', 0)
    if _pod_version != TargetInfo._pod_serialization_version:
        msg = 'Target info version mismatch. Expected {}, but found {}.\nTry deleting {}'
        raise ConfigError(msg.format(TargetInfo._pod_serialization_version, _pod_version,
                                     settings.target_info_cache_file))
    return TargetInfo.from_pod(pod)


def cache_target_info(target_info, overwrite=False, cache=None):
    if cache is None:
        cache = read_target_info_cache()
    if target_info.system_id in cache and not overwrite:
        raise ValueError('TargetInfo for {} is already in cache.'.format(target_info.system_id))
    cache[target_info.system_id] = target_info.to_pod()
    write_target_info_cache(cache)


class TargetInfo(Podable):

    _pod_serialization_version = 5

    @staticmethod
    def from_pod(pod):
        instance = super(TargetInfo, TargetInfo).from_pod(pod)
        instance.target = pod['target']
        instance.modules = pod['modules']
        instance.abi = pod['abi']
        instance.cpus = [CpuInfo.from_pod(c) for c in pod['cpus']]
        instance.os = pod['os']
        instance.os_version = pod['os_version']
        instance.system_id = pod['system_id']
        instance.hostid = pod['hostid']
        instance.hostname = pod['hostname']
        instance.abi = pod['abi']
        instance.is_rooted = pod['is_rooted']
        instance.kernel_version = kernel_version_from_pod(pod)
        instance.kernel_config = kernel_config_from_pod(pod)
        instance.sched_features = pod['sched_features']
        instance.page_size_kb = pod.get('page_size_kb')
        if instance.os == 'android':
            instance.screen_resolution = pod['screen_resolution']
            instance.prop = AndroidProperties('')
            instance.prop._properties = pod['prop']
            instance.android_id = pod['android_id']

        return instance

    def __init__(self):
        super(TargetInfo, self).__init__()
        self.target = None
        self.modules = []
        self.cpus = []
        self.os = None
        self.os_version = None
        self.system_id = None
        self.hostid = None
        self.hostname = None
        self.abi = None
        self.is_rooted = None
        self.kernel_version = None
        self.kernel_config = None
        self.sched_features = None
        self.screen_resolution = None
        self.prop = None
        self.android_id = None
        self.page_size_kb = None

    def to_pod(self):
        pod = super(TargetInfo, self).to_pod()
        pod['target'] = self.target
        pod['modules'] = self.modules
        pod['abi'] = self.abi
        pod['cpus'] = [c.to_pod() for c in self.cpus]
        pod['os'] = self.os
        pod['os_version'] = self.os_version
        pod['system_id'] = self.system_id
        pod['hostid'] = self.hostid
        pod['hostname'] = self.hostname
        pod['abi'] = self.abi
        pod['is_rooted'] = self.is_rooted
        pod['kernel_release'] = self.kernel_version.release
        pod['kernel_version'] = self.kernel_version.version
        pod['kernel_config'] = dict(self.kernel_config.iteritems())
        pod['sched_features'] = self.sched_features
        pod['page_size_kb'] = self.page_size_kb
        if self.os == 'android':
            pod['screen_resolution'] = self.screen_resolution
            pod['prop'] = self.prop._properties
            pod['android_id'] = self.android_id

        return pod

    @staticmethod
    def _pod_upgrade_v1(pod):
        pod['_pod_version'] = pod.get('_pod_version', 1)
        pod['cpus'] = pod.get('cpus', [])
        pod['system_id'] = pod.get('system_id')
        pod['hostid'] = pod.get('hostid')
        pod['hostname'] = pod.get('hostname')
        pod['sched_features'] = pod.get('sched_features')
        pod['screen_resolution'] = pod.get('screen_resolution', (0, 0))
        pod['prop'] = pod.get('prop')
        pod['android_id'] = pod.get('android_id')
        return pod

    @staticmethod
    def _pod_upgrade_v2(pod):
        pod['page_size_kb'] = pod.get('page_size_kb')
        pod['_pod_version'] = pod.get('format_version', 0)
        return pod

    @staticmethod
    def _pod_upgrade_v3(pod):
        config = {}
        for key, value in pod['kernel_config'].items():
            config[key.upper()] = value
        pod['kernel_config'] = config
        return pod

    @staticmethod
    def _pod_upgrade_v4(pod):
        return TargetInfo._pod_upgrade_v3(pod)

    @staticmethod
    def _pod_upgrade_v5(pod):
        pod['modules'] = pod.get('modules') or []
        return pod
