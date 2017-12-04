from devlib import AndroidTarget
from devlib.target import KernelConfig, KernelVersion, Cpuinfo
from devlib.utils.android import AndroidProperties


class TargetInfo(object):

    @staticmethod
    def from_pod(pod):
        instance = TargetInfo()
        instance.target = pod['target']
        instance.abi = pod['abi']
        instance.cpuinfo = Cpuinfo(pod['cpuinfo'])
        instance.os = pod['os']
        instance.os_version = pod['os_version']
        instance.abi = pod['abi']
        instance.is_rooted = pod['is_rooted']
        instance.kernel_version = KernelVersion(pod['kernel_release'], 
                                                pod['kernel_version'])
        instance.kernel_config = KernelConfig(pod['kernel_config'])

        if pod["target"] == "AndroidTarget":
            instance.screen_resolution = pod['screen_resolution']
            instance.prop = AndroidProperties('')
            instance.prop._properties = pod['prop']
            instance.android_id = pod['android_id']

        return instance

    def __init__(self, target=None):
        if target:
            self.target = target.__class__.__name__
            self.cpuinfo = target.cpuinfo
            self.os = target.os
            self.os_version = target.os_version
            self.abi = target.abi
            self.is_rooted = target.is_rooted
            self.kernel_version = target.kernel_version
            self.kernel_config = target.config

            if isinstance(target, AndroidTarget):
                self.screen_resolution = target.screen_resolution
                self.prop = target.getprop()
                self.android_id = target.android_id

        else:
            self.target = None
            self.cpuinfo = None
            self.os = None
            self.os_version = None
            self.abi = None
            self.is_rooted = None
            self.kernel_version = None
            self.kernel_config = None

            if isinstance(target, AndroidTarget):
                self.screen_resolution = None
                self.prop = None
                self.android_id = None

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

        if self.target == "AndroidTarget":
            pod['screen_resolution'] = self.screen_resolution
            pod['prop'] = self.prop._properties
            pod['android_id'] = self.android_id

        return pod
