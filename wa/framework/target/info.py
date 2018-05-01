from copy import copy

from devlib import AndroidTarget, TargetError
from devlib.target import KernelConfig, KernelVersion, Cpuinfo
from devlib.utils.android import AndroidProperties


def cpuinfo_from_pod(pod):
    cpuinfo = Cpuinfo('')
    cpuinfo.sections = pod['cpuinfo']
    lines = []
    for section in cpuinfo.sections:
        for key, value in section.iteritems():
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
            kerne_string = release_string
    else:
        kernel_string = '#{}'.format(version_string)
    return KernelVersion(kernel_string)


def kernel_config_from_pod(pod):
    config = KernelConfig('')
    config._config = pod['kernel_config']
    lines = []
    for key, value in config._config.iteritems():
        if value == 'n':
            lines.append('# {} is not set'.format(key))
        else:
            lines.append('{}={}'.format(key, value))
    config.text = '\n'.join(lines)
    return config


class CpufreqInfo(object):

    @staticmethod
    def from_pod(pod):
        return CpufreqInfo(**pod)

    def __init__(self, **kwargs):
        self.available_frequencies = kwargs.pop('available_frequencies', [])
        self.available_governors = kwargs.pop('available_governors', [])
        self.related_cpus = kwargs.pop('related_cpus', [])
        self.driver = kwargs.pop('driver', None)

    def to_pod(self):
        return copy(self.__dict__)

    def __repr__(self):
        return 'Cpufreq({} {})'.format(self.driver, self.related_cpus)

    __str__ = __repr__


class IdleStateInfo(object):

    @staticmethod
    def from_pod(pod):
        return IdleStateInfo(**pod)

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name', None)
        self.desc = kwargs.pop('desc', None)
        self.power = kwargs.pop('power', None)
        self.latency = kwargs.pop('latency', None)

    def to_pod(self):
        return copy(self.__dict__)

    def __repr__(self):
        return 'IdleState({}/{})'.format(self.name, self.desc)

    __str__ = __repr__


class CpuidleInfo(object):

    @staticmethod
    def from_pod(pod):
        instance = CpuidleInfo()
        instance.governor = pod['governor']
        instance.driver = pod['driver']
        instance.states = [IdleStateInfo.from_pod(s) for s in pod['states']]
        return instance

    @property
    def num_states(self):
        return len(self.states)

    def __init__(self):
        self.governor = None
        self.driver = None
        self.states = []

    def to_pod(self):
        pod = {}
        pod['governor'] = self.governor
        pod['driver'] = self.driver
        pod['states'] = [s.to_pod() for s in self.states]
        return pod

    def __repr__(self):
        return 'Cpuidle({}/{} {} states)'.format(
                self.governor, self.driver, self.num_states)

    __str__ = __repr__


class CpuInfo(object):

    @staticmethod
    def from_pod(pod):
        instance = CpuInfo()
        instance.id = pod['id']
        instance.name = pod['name']
        instance.architecture = pod['architecture']
        instance.features = pod['features']
        instance.cpufreq = CpufreqInfo.from_pod(pod['cpufreq'])
        instance.cpuidle = CpuidleInfo.from_pod(pod['cpuidle'])
        return instance

    def __init__(self):
        self.id = None
        self.name = None
        self.architecture = None
        self.features = []
        self.cpufreq = CpufreqInfo()
        self.cpuidle = CpuidleInfo()

    def to_pod(self):
        pod = {}
        pod['id'] = self.id
        pod['name'] = self.name
        pod['architecture'] = self.architecture
        pod['features'] = self.features
        pod['cpufreq'] = self.cpufreq.to_pod()
        pod['cpuidle'] = self.cpuidle.to_pod()
        return pod

    def __repr__(self):
        return 'Cpu({} {})'.format(self.id, self.name)

    __str__ = __repr__


def get_target_info(target):
    info = TargetInfo()
    info.target = target.__class__.__name__
    info.os = target.os
    info.os_version = target.os_version
    info.abi = target.abi
    info.is_rooted = target.is_rooted
    info.kernel_version = target.kernel_version
    info.kernel_config = target.config
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

    if isinstance(target, AndroidTarget):
        info.screen_resolution = target.screen_resolution
        info.prop = target.getprop()
        info.android_id = target.android_id

    return info


class TargetInfo(object):

    @staticmethod
    def from_pod(pod):
        instance = TargetInfo()
        instance.target = pod['target']
        instance.abi = pod['abi']
        instance.cpus = [CpuInfo.from_pod(c) for c in pod['cpus']]
        instance.os = pod['os']
        instance.os_version = pod['os_version']
        instance.abi = pod['abi']
        instance.is_rooted = pod['is_rooted']
        instance.kernel_version = kernel_version_from_pod(pod)
        instance.kernel_config = kernel_config_from_pod(pod)
        instance.sched_features = pod['sched_features']
        if instance.os == 'android':
            instance.screen_resolution = pod['screen_resolution']
            instance.prop = AndroidProperties('')
            instance.prop._properties = pod['prop']
            instance.android_id = pod['android_id']

        return instance

    def __init__(self):
        self.target = None
        self.cpus = []
        self.os = None
        self.os_version = None
        self.abi = None
        self.is_rooted = None
        self.kernel_version = None
        self.kernel_config = None
        self.sched_features = None

    def to_pod(self):
        pod = {}
        pod['target'] = self.target
        pod['abi'] = self.abi
        pod['cpus'] = [c.to_pod() for c in self.cpus]
        pod['os'] = self.os
        pod['os_version'] = self.os_version
        pod['abi'] = self.abi
        pod['is_rooted'] = self.is_rooted
        pod['kernel_release'] = self.kernel_version.release
        pod['kernel_version'] = self.kernel_version.version
        pod['kernel_config'] = dict(self.kernel_config.iteritems())
        pod['sched_features'] = self.sched_features
        if self.os == 'android':
            pod['screen_resolution'] = self.screen_resolution
            pod['prop'] = self.prop._properties
            pod['android_id'] = self.android_id

        return pod
