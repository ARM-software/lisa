import string
from copy import copy

from wlauto.core.plugin import Plugin, Parameter
from wlauto.core.configuration.configuration import RuntimeParameter
from wlauto.exceptions import ConfigError
from wlauto.utils.types import list_of_integers, list_of, caseless_string

from devlib.platform import Platform
from devlib.target import AndroidTarget, Cpuinfo, KernelVersion, KernelConfig

__all__ = ['RuntimeParameter', 'CoreParameter', 'DeviceManager', 'TargetInfo']

UNKOWN_RTP = 'Unknown runtime parameter "{}"'


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
            instance.prop = pod['prop']
            instance.prop = pod['android_id']

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
            pod['prop'] = self.prop
            pod['android_id'] = self.android_id

        return pod


class DeviceManager(Plugin):

    kind = "manager"
    name = None
    target_type = None
    platform_type = Platform
    has_gpu = None
    path_module = None
    info = None

    parameters = [
        Parameter('core_names', kind=list_of(caseless_string),
                  description="""
                  This is a list of all cpu cores on the device with each
                  element being the core type, e.g. ``['a7', 'a7', 'a15']``. The
                  order of the cores must match the order they are listed in
                  ``'/sys/devices/system/cpu'``. So in this case, ``'cpu0'`` must
                  be an A7 core, and ``'cpu2'`` an A15.'
                  """),
        Parameter('core_clusters', kind=list_of_integers,
                  description="""
                  This is a list indicating the cluster affinity of the CPU cores,
                  each element correponding to the cluster ID of the core coresponding
                  to its index. E.g. ``[0, 0, 1]`` indicates that cpu0 and cpu1 are on
                  cluster 0, while cpu2 is on cluster 1. If this is not specified, this
                  will be inferred from ``core_names`` if possible (assuming all cores with
                  the same name are on the same cluster).
                  """),
        Parameter('working_directory',
                  description='''
                  Working directory to be used by WA. This must be in a location where the specified user
                  has write permissions. This will default to /home/<username>/wa (or to /root/wa, if
                  username is 'root').
                  '''),
        Parameter('binaries_directory',
                  description='Location of executable binaries on this device (must be in PATH).'),
    ]
    modules = []

    runtime_parameter_managers = [
    ]

    def __init__(self):
        super(DeviceManager, self).__init__()
        self.runtime_parameter_values = None

    # Framework

    def connect(self):
        raise NotImplementedError("connect method must be implemented for device managers")

    def initialize(self, context):
        super(DeviceManager, self).initialize(context)
        self.info = TargetInfo(self.target)
        self.target.setup()

    def start(self):
        pass

    def stop(self):
        pass

    def validate(self):
        pass

    # Runtime Parameters

    def merge_runtime_parameters(self, params):
        merged_values = {}
        for source, values in params.iteritems():
            for name, value in values:
                for rtpm in self.runtime_parameter_managers:
                    if rtpm.match(name):
                        rtpm.update_value(name, value, source, merged_values)
                        break
                else:
                    msg = 'Unknown runtime parameter "{}" in "{}"'
                    raise ConfigError(msg.format(name, source))
        return merged_values

    def static_runtime_parameter_validation(self, params):
        params = copy(params)
        for rtpm in self.runtime_parameters_managers:
            rtpm.static_validation(params)
        if params:
            msg = 'Unknown runtime_parameters for "{}": "{}"'
            raise ConfigError(msg.format(self.name, '", "'.join(params.iterkeys())))

    def dynamic_runtime_parameter_validation(self, params):
        for rtpm in self.runtime_parameters_managers:
            rtpm.dynamic_validation(params)

    def commit_runtime_parameters(self, params):
        params = copy(params)
        for rtpm in self.runtime_parameters_managers:
            rtpm.commit(params)

    #Runtime parameter getters/setters
    def get_sysfile_values(self):
        return self._written_sysfiles

    def set_sysfile_values(self, params):
        for sysfile, value in params.iteritems():
            verify = not sysfile.endswith('!')
            sysfile = sysfile.rstrip('!')
            self._written_sysfiles.append((sysfile, value))
            self.target.write_value(sysfile, value, verify=verify)
