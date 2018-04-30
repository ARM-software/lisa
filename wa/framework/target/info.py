from devlib import AndroidTarget
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


def get_target_info(target):
    info = TargetInfo()
    info.target = target.__class__.__name__
    info.cpuinfo = target.cpuinfo
    info.os = target.os
    info.os_version = target.os_version
    info.abi = target.abi
    info.is_rooted = target.is_rooted
    info.kernel_version = target.kernel_version
    info.kernel_config = target.config

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
        instance.cpuinfo = cpuinfo_from_pod(pod)
        instance.os = pod['os']
        instance.os_version = pod['os_version']
        instance.abi = pod['abi']
        instance.is_rooted = pod['is_rooted']
        instance.kernel_version = kernel_version_from_pod(pod)
        instance.kernel_config = kernel_config_from_pod(pod)
        if instance.os == 'android':
            instance.screen_resolution = pod['screen_resolution']
            instance.prop = AndroidProperties('')
            instance.prop._properties = pod['prop']
            instance.android_id = pod['android_id']

        return instance

    def __init__(self):
        self.target = None
        self.cpuinfo = None
        self.os = None
        self.os_version = None
        self.abi = None
        self.is_rooted = None
        self.kernel_version = None
        self.kernel_config = None

    def to_pod(self):
        pod = {}
        pod['target'] = self.target
        pod['abi'] = self.abi
        pod['cpuinfo'] = self.cpuinfo.sections
        pod['os'] = self.os
        pod['os_version'] = self.os_version
        pod['abi'] = self.abi
        pod['is_rooted'] = self.is_rooted
        pod['kernel_release'] = self.kernel_version.release
        pod['kernel_version'] = self.kernel_version.version
        pod['kernel_config'] = dict(self.kernel_config.iteritems())
        if self.os == 'android':
            pod['screen_resolution'] = self.screen_resolution
            pod['prop'] = self.prop._properties
            pod['android_id'] = self.android_id

        return pod
