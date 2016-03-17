import string
from collections import OrderedDict

from wlauto.core.plugin import Plugin, Parameter
from wlauto.exceptions import ConfigError
from wlauto.utils.types import list_of_integers, list_of, caseless_string

from devlib.platform import Platform
from devlib.target import AndroidTarget, Cpuinfo, KernelVersion, KernelConfig

__all__ = ['RuntimeParameter', 'CoreParameter', 'DeviceManager', 'TargetInfo']


class RuntimeParameter(object):
    """
    A runtime parameter which has its getter and setter methods associated it
    with it.

    """

    def __init__(self, name, getter, setter,
                 getter_args=None, setter_args=None,
                 value_name='value', override=False):
        """
        :param name: the name of the parameter.
        :param getter: the getter method which returns the value of this parameter.
        :param setter: the setter method which sets the value of this parameter. The setter
                       always expects to be passed one argument when it is called.
        :param getter_args: keyword arguments to be used when invoking the getter.
        :param setter_args: keyword arguments to be used when invoking the setter.
        :param override: A ``bool`` that specifies whether a parameter of the same name further up the
                            hierarchy should be overridden. If this is ``False`` (the default), an exception
                            will be raised by the ``AttributeCollection`` instead.

        """
        self.name = name
        self.getter = getter
        self.setter = setter
        self.getter_args = getter_args or {}
        self.setter_args = setter_args or {}
        self.value_name = value_name
        self.override = override

    def __str__(self):
        return self.name

    __repr__ = __str__


class CoreParameter(RuntimeParameter):
    """A runtime parameter that will get expanded into a RuntimeParameter for each core type."""

    def get_runtime_parameters(self, core_names):
        params = []
        for core in set(core_names):
            name = string.Template(self.name).substitute(core=core)
            getter = string.Template(self.getter).substitute(core=core)
            setter = string.Template(self.setter).substitute(core=core)
            getargs = dict(self.getter_args.items() + [('core', core)])
            setargs = dict(self.setter_args.items() + [('core', core)])
            params.append(RuntimeParameter(name, getter, setter, getargs, setargs, self.value_name, self.override))
        return params


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
        instance.kernel_version = KernelVersion(pod['kernel_version'])
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
        pod['target'] = self.target.__class__.__name__
        pod['abi'] = self.abi
        pod['cpuinfo'] = self.cpuinfo.text
        pod['os'] = self.os
        pod['os_version'] = self.os_version
        pod['abi'] = self.abi
        pod['is_rooted'] = self.is_rooted
        pod['kernel_version'] = self.kernel_version.version
        pod['kernel_config'] = self.kernel_config.text

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

    runtime_parameters = [
        RuntimeParameter('sysfile_values', 'get_sysfile_values', 'set_sysfile_values', value_name='params'),
        CoreParameter('${core}_cores', 'get_number_of_online_cpus', 'set_number_of_online_cpus',
                      value_name='number'),
        CoreParameter('${core}_min_frequency', 'get_core_min_frequency', 'set_core_min_frequency',
                      value_name='freq'),
        CoreParameter('${core}_max_frequency', 'get_core_max_frequency', 'set_core_max_frequency',
                      value_name='freq'),
        CoreParameter('${core}_frequency', 'get_core_cur_frequency', 'set_core_cur_frequency',
                      value_name='freq'),
        CoreParameter('${core}_governor', 'get_core_governor', 'set_core_governor',
                      value_name='governor'),
        CoreParameter('${core}_governor_tunables', 'get_core_governor_tunables', 'set_core_governor_tunables',
                      value_name='tunables'),
    ]

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

    def get_runtime_parameter_names(self):
        return [p.name for p in self._expand_runtime_parameters()]

    def get_runtime_parameters(self):
        """ returns the runtime parameters that have been set. """
        # pylint: disable=cell-var-from-loop
        runtime_parameters = OrderedDict()
        for rtp in self._expand_runtime_parameters():
            if not rtp.getter:
                continue
            getter = getattr(self, rtp.getter)
            rtp_value = getter(**rtp.getter_args)
            runtime_parameters[rtp.name] = rtp_value
        return runtime_parameters

    def set_runtime_parameters(self, params):
        """
        The parameters are taken from the keyword arguments and are specific to
        a particular device. See the device documentation.

        """
        runtime_parameters = self._expand_runtime_parameters()
        rtp_map = {rtp.name.lower(): rtp for rtp in runtime_parameters}

        params = OrderedDict((k.lower(), v) for k, v in params.iteritems() if v is not None)

        expected_keys = rtp_map.keys()
        if not set(params.keys()).issubset(set(expected_keys)):
            unknown_params = list(set(params.keys()).difference(set(expected_keys)))
            raise ConfigError('Unknown runtime parameter(s): {}'.format(unknown_params))

        for param in params:
            self.logger.debug('Setting runtime parameter "{}"'.format(param))
            rtp = rtp_map[param]
            setter = getattr(self, rtp.setter)
            args = dict(rtp.setter_args.items() + [(rtp.value_name, params[rtp.name.lower()])])
            setter(**args)

    def _expand_runtime_parameters(self):
        expanded_params = []
        for param in self.runtime_parameters:
            if isinstance(param, CoreParameter):
                expanded_params.extend(param.get_runtime_parameters(self.target.core_names))  # pylint: disable=no-member
            else:
                expanded_params.append(param)
        return expanded_params

    #Runtime parameter getters/setters

    _written_sysfiles = []

    def get_sysfile_values(self):
        return self._written_sysfiles

    def set_sysfile_values(self, params):
        for sysfile, value in params.iteritems():
            verify = not sysfile.endswith('!')
            sysfile = sysfile.rstrip('!')
            self._written_sysfiles.append((sysfile, value))
            self.target.write_value(sysfile, value, verify=verify)

    # pylint: disable=E1101

    def _get_core_online_cpu(self, core):
        try:
            return self.target.list_online_core_cpus(core)[0]
        except IndexError:
            raise ValueError("No {} cores are online".format(core))

    def get_number_of_online_cpus(self, core):
        return len(self._get_core_online_cpu(core))

    def set_number_of_online_cpus(self, core, number):
        for cpu in self.target.core_cpus(core)[:number]:
            self.target.hotplug.online(cpu)

    def get_core_min_frequency(self, core):
        return self.target.cpufreq.get_min_frequency(self._get_core_online_cpu(core))

    def set_core_min_frequency(self, core, frequency):
        self.target.cpufreq.set_min_frequency(self._get_core_online_cpu(core), frequency)

    def get_core_max_frequency(self, core):
        return self.target.cpufreq.get_max_frequency(self._get_core_online_cpu(core))

    def set_core_max_frequency(self, core, frequency):
        self.target.cpufreq.set_max_frequency(self._get_core_online_cpu(core), frequency)

    def get_core_frequency(self, core):
        return self.target.cpufreq.get_frequency(self._get_core_online_cpu(core))

    def set_core_frequency(self, core, frequency):
        self.target.cpufreq.set_frequency(self._get_core_online_cpu(core), frequency)

    def get_core_governor(self, core):
        return self.target.cpufreq.get_cpu_governor(self._get_core_online_cpu(core))

    def set_core_governor(self, core, governor):
        self.target.cpufreq.set_cpu_governor(self._get_core_online_cpu(core), governor)

    def get_core_governor_tunables(self, core):
        return self.target.cpufreq.get_governor_tunables(self._get_core_online_cpu(core))

    def set_core_governor_tunables(self, core, tunables):
        self.target.cpufreq.set_governor_tunables(self._get_core_online_cpu(core),
                                                  *tunables)
