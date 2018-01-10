import logging
import time
from collections import defaultdict, OrderedDict
from copy import copy

from wa.framework.exception import ConfigError
from wa.framework.plugin import Plugin, Parameter
from wa.utils.misc import resolve_cpus, resolve_unique_domain_cpus
from wa.utils.types import caseless_string, enum

from devlib.exception import TargetError
from devlib.utils.misc import unique
from devlib.utils.types import integer

logger = logging.getLogger('RuntimeConfig')


class RuntimeParameter(Parameter):
    def __init__(self, name, setter, setter_params=None, **kwargs):
        super(RuntimeParameter, self).__init__(name, **kwargs)
        self.setter = setter
        self.setter_params = setter_params or {}

    def set(self, obj, value):
        self.validate_value(self.name, value)
        self.setter(obj, value, **self.setter_params)


class RuntimeConfig(Plugin):

    name = None
    kind = 'runtime-config'

    @property
    def supported_parameters(self):
        return self._runtime_params.values()

    @property
    def core_names(self):
        return unique(self.target.core_names)

    def __init__(self, target, **kwargs):
        super(RuntimeConfig, self).__init__(**kwargs)
        self.target = target
        self._target_checked = False
        self._runtime_params = {}
        try:
            self.initialize()
        except TargetError:
            msg = 'Failed to initialize: "{}"'
            self.logger.debug(msg.format(self.name))
            self._runtime_params = {}

    def initialize(self):
        raise NotImplementedError()

    def commit(self):
        raise NotImplementedError()

    def set_runtime_parameter(self, name, value):
        if not self._target_checked:
            self.check_target()
            self._target_checked = True
        self._runtime_params[name].set(self, value)

    def set_defaults(self):
        for p in self.supported_parameters:
            if p.default:
                self.set_runtime_parameter(p.name, p.default)

    def validate_parameters(self):
        raise NotImplementedError()

    def check_target(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


class HotplugRuntimeConfig(RuntimeConfig):
    '''
    NOTE: Currently will fail if trying to hotplug back a core that
    was hotplugged out when the devlib target was created.
    '''

    name = 'rt-hotplug'

    @staticmethod
    def set_num_cores(obj, value, core):
        cpus = resolve_cpus(core, obj.target)
        max_cores = len(cpus)
        value = integer(value)
        if value > max_cores:
            msg = 'Cannot set number of {}\'s to {}; max is {}'
            raise ValueError(msg.format(core, value, max_cores))

        msg = 'CPU{} Hotplugging already configured'
        # Set cpus to be enabled
        for cpu in cpus[:value]:
            if cpu in obj.num_cores:
                raise ConfigError(msg.format(cpu))
            obj.num_cores[cpu] = True
        # Set the remaining cpus to be disabled.
        for cpu in cpus[value:]:
            if cpu in obj.num_cores:
                raise ConfigError(msg.format(cpu))
            obj.num_cores[cpu] = False

    def __init__(self, target):
        self.num_cores = defaultdict(dict)
        super(HotplugRuntimeConfig, self).__init__(target)

    def initialize(self):
        if not self.target.has('hotplug'):
            return
        param_name = 'num_cores'
        self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=int,
                                 constraint=lambda x:
                                         0 <= x <= self.target.number_of_cpus,
                                 description="""
                                 The number of cpu cores to be online
                                 """,
                                 setter=self.set_num_cores,
                                 setter_params={'core': None})

        for name in unique(self.target.platform.core_names):
            param_name = 'num_{}_cores'.format(name)
            self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=int,
                                     constraint=lambda x, name=name:
                                             0 <= x <= len(self.target.core_cpus(name)),
                                     description="""
                                     The number of {} cores to be online
                                     """.format(name),
                                     setter=self.set_num_cores,
                                     setter_params={'core': name})

        for cpu_no in range(self.target.number_of_cpus):
            param_name = 'cpu{}_online'.format(cpu_no)
            self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=bool,
                                     description="""
                                     Specify whether cpu{} should be online
                                     """.format(cpu_no),
                                     setter=self.set_num_cores,
                                     setter_params={'core': cpu_no})

        if self.target.has('bl'):
            for cluster in ['big', 'little']:
                param_name = 'num_{}_cores'.format(cluster)
                self._runtime_params[param_name] = \
                        RuntimeParameter(param_name, kind=int,
                                         constraint=lambda x, cluster=cluster:
                                                   0 <= x <= len(resolve_cpus(cluster, self.target)),
                                         description="""
                                         The number of cores on the {} cluster to be online
                                         """.format(cluster),
                                         setter=self.set_num_cores,
                                         setter_params={'core': cluster})

    def check_target(self):
        if not self.target.has('hotplug'):
            raise TargetError('Target does not appear to support hotplug')

    def validate_parameters(self):
        if len(self.num_cores) == self.target.number_of_cpus:
            if all(v is False for v in self.num_cores.values()):
                raise ValueError('Cannot set number of all cores to 0')

    def commit(self):
        '''Online all CPUs required in order before then off-lining'''
        num_cores = sorted(self.num_cores.iteritems())
        for cpu, online in num_cores:
            if online:
                self.target.hotplug.online(cpu)
        for cpu, online in reversed(num_cores):
            if not online:
                self.target.hotplug.offline(cpu)

    def clear(self):
        self.num_cores = defaultdict(dict)


class SysfileValuesRuntimeConfig(RuntimeConfig):

    name = 'rt-sysfiles'

    #pylint: disable=unused-argument
    @staticmethod
    def set_sysfile(obj, value, core):
        for path, value in value.iteritems():
            verify = True
            if path.endswith('!'):
                verify = False
                path = path[:-1]

            if path in obj.sysfile_values:
                msg = 'Syspath "{}:{}" already specified with a value of "{}"'
                raise ConfigError(msg.format(path, value, obj.sysfile_values[path][0]))

            obj.sysfile_values[path] = (value, verify)

    def __init__(self, target):
        self.sysfile_values = OrderedDict()
        super(SysfileValuesRuntimeConfig, self).__init__(target)

    def initialize(self):
        self._runtime_params['sysfile_values'] = \
            RuntimeParameter('sysfile_values', kind=dict, merge=True,
                              setter=self.set_sysfile,
                              setter_params={'core': None},
                              description="""
                              Sysfile path to be set
                              """)

    def check_target(self):
        return True

    def validate_parameters(self):
        return

    def commit(self):
        for path, (value, verify) in self.sysfile_values.iteritems():
            self.target.write_value(path, value, verify=verify)

    def clear(self):
        self.sysfile_values = OrderedDict()

    def check_exists(self, path):
        if not self.target.file_exists(path):
            raise ConfigError('Sysfile "{}" does not exist.'.format(path))


class FreqValue(object):

    def __init__(self, values):
        if values is None:
            self.values = values
        else:
            self.values = sorted(values)

    def __call__(self, value):
        '''
        `self.values` can be `None` if the device's supported values could not be retrieved
        for some reason e.g. the cluster was offline, in this case we assume
        the user values will be available and allow any integer values.
        '''
        if self.values is None:
            if isinstance(value, int):
                return value
            else:
                msg = 'CPU frequency values could not be retrieved, cannot resolve "{}"'
                raise TargetError(msg.format(value))
        elif isinstance(value, int) and value in self.values:
            return value
        elif isinstance(value, basestring):
            value = caseless_string(value)
            if value in ['min', 'max']:
                return value

        msg = 'Invalid frequency value: {}; Must be in {}'
        raise ValueError(msg.format(value, self.values))

    def __str__(self):
        return 'valid frequency value: {}'.format(self.values)


class CpufreqRuntimeConfig(RuntimeConfig):

    name = 'rt-cpufreq'

    @staticmethod
    def set_frequency(obj, value, core):
        obj.set_param(obj, value, core, 'frequency')

    @staticmethod
    def set_max_frequency(obj, value, core):
        obj.set_param(obj, value, core, 'max_frequency')

    @staticmethod
    def set_min_frequency(obj, value, core):
        obj.set_param(obj, value, core, 'min_frequency')

    @staticmethod
    def set_governor(obj, value, core):
        obj.set_param(obj, value, core, 'governor')

    @staticmethod
    def set_governor_tunables(obj, value, core):
        obj.set_param(obj, value, core, 'governor_tunables')

    @staticmethod
    def set_param(obj, value, core, parameter):
        '''Method to store passed parameter if it is not already specified for that cpu'''
        cpus = resolve_unique_domain_cpus(core, obj.target)
        for cpu in cpus:
            if parameter in obj.config[cpu]:
                msg = 'Cannot set "{}" for core "{}"; Parameter for CPU{} has already been set'
                raise ConfigError(msg.format(parameter, core, cpu))
            obj.config[cpu][parameter] = value

    def __init__(self, target):
        self.config = defaultdict(dict)
        self.supported_cpu_freqs = {}
        self.supported_cpu_governors = {}
        super(CpufreqRuntimeConfig, self).__init__(target)

    def initialize(self):
        if not self.target.has('cpufreq'):
            return

        self._retrive_cpufreq_info()
        all_freqs, common_freqs, common_gov = self._get_common_values()

        # Add common parameters if available.
        freq_val = FreqValue(all_freqs)
        param_name = 'frequency'
        self._runtime_params[param_name] = \
            RuntimeParameter(param_name, kind=freq_val,
                        setter=self.set_frequency,
                        setter_params={'core': None},
                        description="""
                        The desired frequency for all cores
                        """)
        param_name = 'max_frequency'
        self._runtime_params[param_name] = \
            RuntimeParameter(param_name, kind=freq_val,
                        setter=self.set_max_frequency,
                        setter_params={'core': None},
                        description="""
                        The maximum frequency for all cores
                        """)
        param_name = 'min_frequency'
        self._runtime_params[param_name] = \
            RuntimeParameter(param_name, kind=freq_val,
                        setter=self.set_min_frequency,
                        setter_params={'core': None},
                        description="""
                        The minimum frequency for all cores
                        """)

        if common_gov:
            param_name = 'governor'
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=str,
                          allowed_values=common_gov,
                          setter=self.set_governor,
                          setter_params={'core': None},
                          description="""
                          The governor to be set for all cores
                          """)

        param_name = 'governor_tunables'
        self._runtime_params[param_name] = \
            RuntimeParameter(param_name, kind=dict,
                           merge=True,
                           setter=self.set_governor_tunables,
                           setter_params={'core': None},
                           description="""
                           The governor tunables to be set for all cores
                           """)

        # Add core name parameters
        for name in unique(self.target.platform.core_names):
            cpu = resolve_unique_domain_cpus(name, self.target)[0]
            freq_val = FreqValue(self.supported_cpu_freqs.get(cpu))
            avail_govs = self.supported_cpu_governors.get(cpu)

            param_name = '{}_frequency'.format(name)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=freq_val,
                          setter=self.set_frequency,
                          setter_params={'core': name},
                          description="""
                          The desired frequency for the {} cores
                          """.format(name))
            param_name = '{}_max_frequency'.format(name)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=freq_val,
                          setter=self.set_max_frequency,
                          setter_params={'core': name},
                          description="""
                          The maximum frequency for the {} cores
                          """.format(name))
            param_name = '{}_min_frequency'.format(name)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=freq_val,
                          setter=self.set_min_frequency,
                          setter_params={'core': name},
                          description="""
                          The minimum frequency for the {} cores
                          """.format(name))
            param_name = '{}_governor'.format(name)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=str,
                          allowed_values=avail_govs,
                          setter=self.set_governor,
                          setter_params={'core': name},
                          description="""
                          The governor to be set for the {} cores
                          """.format(name))
            param_name = '{}_gov_tunables'.format(name)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=dict,
                          setter=self.set_governor_tunables,
                          setter_params={'core': name},
                          merge=True,
                          description="""
                          The governor tunables to be set for the {} cores
                          """.format(name))

        # Add cpuX parameters.
        for cpu_no in range(self.target.number_of_cpus):
            freq_val = FreqValue(self.supported_cpu_freqs.get(cpu_no))
            avail_govs = self.supported_cpu_governors.get(cpu_no)

            param_name = 'cpu{}_frequency'.format(cpu_no)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=freq_val,
                          setter=self.set_frequency,
                          setter_params={'core': cpu_no},
                          description="""
                          The desired frequency for cpu{}
                          """.format(cpu_no))
            param_name = 'cpu{}_max_frequency'.format(cpu_no)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=freq_val,
                          setter=self.set_max_frequency,
                          setter_params={'core': cpu_no},
                          description="""
                          The maximum frequency for cpu{}
                          """.format(cpu_no))
            param_name = 'cpu{}_min_frequency'.format(cpu_no)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=freq_val,
                          setter=self.set_min_frequency,
                          setter_params={'core': cpu_no},
                          description="""
                          The minimum frequency for cpu{}
                          """.format(cpu_no))
            param_name = 'cpu{}_governor'.format(cpu_no)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=str,
                          allowed_values=avail_govs,
                          setter=self.set_governor,
                          setter_params={'core': cpu_no},
                          description="""
                          The governor to be set for cpu{}
                          """.format(cpu_no))
            param_name = 'cpu{}_gov_tunables'.format(cpu_no)
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=dict,
                          setter=self.set_governor_tunables,
                          setter_params={'core': cpu_no},
                          merge=True,
                          description="""
                          The governor tunables to be set for cpu{}
                          """.format(cpu_no))

        # Add big.little cores if present on device.
        if self.target.has('bl'):
            for cluster in ['big', 'little']:
                cpu = resolve_unique_domain_cpus(cluster, self.target)[0]
                freq_val = FreqValue(self.supported_cpu_freqs.get(cpu))
                avail_govs = self.supported_cpu_governors.get(cpu)
                param_name = '{}_frequency'.format(cluster)

                self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=freq_val,
                              setter=self.set_frequency,
                              setter_params={'core': cluster},
                              description="""
                              The desired frequency for the {} cluster
                              """.format(cluster))
                param_name = '{}_max_frequency'.format(cluster)
                self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=freq_val,
                              setter=self.set_max_frequency,
                              setter_params={'core': cluster},
                              description="""
                              The maximum frequency for the {} cluster
                              """.format(cluster))
                param_name = '{}_min_frequency'.format(cluster)
                self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=freq_val,
                              setter=self.set_min_frequency,
                              setter_params={'core': cluster},
                              description="""
                              The minimum frequency for the {} cluster
                              """.format(cluster))
                param_name = '{}_governor'.format(cluster)
                self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=str,
                              allowed_values=avail_govs,
                              setter=self.set_governor,
                              setter_params={'core': cluster},
                              description="""
                              The governor to be set for the {} cores
                              """.format(cluster))
                param_name = '{}_gov_tunables'.format(cluster)
                self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=dict,
                              setter=self.set_governor_tunables,
                              setter_params={'core': cluster},
                              merge=True,
                              description="""
                              The governor tunables to be set for the {} cores
                              """.format(cluster))

    def check_target(self):
        if not self.target.has('cpufreq'):
            raise TargetError('Target does not appear to support cpufreq')

    def validate_parameters(self):
        '''Method to validate parameters against each other'''
        for cpu in self.config:
            config = self.config[cpu]
            minf = config.get('min_frequency')
            maxf = config.get('max_frequency')
            freq = config.get('frequency')

            if freq and minf:
                msg = 'CPU{}: Can\'t set both cpu frequency and minimum frequency'
                raise ConfigError(msg.format(cpu))
            if freq and maxf:
                msg = 'CPU{}: Can\'t set both cpu frequency and maximum frequency'
                raise ConfigError(msg.format(cpu))

            if maxf and minf > maxf:
                msg = 'CPU{}: min_frequency "{}" cannot be greater than max_frequency "{}"'
                raise ConfigError(msg.format(cpu, minf, maxf))
            if maxf and freq > maxf:
                msg = 'CPU{}: cpu frequency "{}" cannot be greater than max_frequency "{}"'
                raise ConfigError(msg.format(cpu, freq, maxf))

    def commit(self):
        for cpu in self.config:
            config = self.config[cpu]
            self.configure_governor(cpu,
                                    config.get('governor'),
                                    config.get('governor_tunables'))

            frequency = config.get('frequency')
            if frequency == 'min':
                frequency = self.target.cpufreq.get_min_frequency(cpu)
            elif frequency == 'max':
                frequency = self.target.cpufreq.get_max_frequency(cpu)
            self.configure_frequency(cpu,
                                     frequency,
                                     config.get('min_frequency'),
                                     config.get('max_frequency'),
                                     config.get('governor'))

    def clear(self):
        self.config = defaultdict(dict)

    def configure_governor(self, cpu, governor=None, gov_tunables=None):
        if not governor and not gov_tunables:
            return
        if cpu not in self.target.list_online_cpus():
            msg = 'Cannot configure governor for {} as no CPUs are online.'
            raise TargetError(msg.format(cpu))
        if not governor:
            governor = self.target.get_governor(cpu)
        if not gov_tunables:
            gov_tunables = {}
        self.target.cpufreq.set_governor(cpu, governor, **gov_tunables)

    def configure_frequency(self, cpu, freq=None, min_freq=None, max_freq=None, governor=None):
        if freq and (min_freq or max_freq):
            msg = 'Cannot specify both frequency and min/max frequency'
            raise ConfigError(msg)

        if cpu not in self.target.list_online_cpus():
            msg = 'Cannot configure frequencies for CPU{} as no CPUs are online.'
            raise TargetError(msg.format(cpu))

        if freq:
            self._set_frequency(cpu, freq, governor)
        else:
            self._set_min_max_frequencies(cpu, min_freq, max_freq)

    def _set_frequency(self, cpu, freq, governor):
        if not governor:
            governor = self.target.cpufreq.get_governor(cpu)
        has_userspace = governor == 'userspace'

        # Sets all frequency to be to desired frequency
        if freq < self.target.cpufreq.get_frequency(cpu):
            self.target.cpufreq.set_min_frequency(cpu, freq)
            if has_userspace:
                self.target.cpufreq.set_frequency(cpu, freq)
            self.target.cpufreq.set_max_frequency(cpu, freq)
        else:
            self.target.cpufreq.set_max_frequency(cpu, freq)
            if has_userspace:
                self.target.cpufreq.set_frequency(cpu, freq)
            self.target.cpufreq.set_min_frequency(cpu, freq)

    def _set_min_max_frequencies(self, cpu, min_freq, max_freq):
        min_freq_set = False
        current_min_freq = self.target.cpufreq.get_min_frequency(cpu)
        current_max_freq = self.target.cpufreq.get_max_frequency(cpu)
        if max_freq:
            if max_freq < current_min_freq:
                if min_freq:
                    self.target.cpufreq.set_min_frequency(cpu, min_freq)
                    self.target.cpufreq.set_max_frequency(cpu, max_freq)
                    min_freq_set = True
                else:
                    msg = 'CPU {}: Cannot set max_frequency ({}) below current min frequency ({}).'
                    raise ConfigError(msg.format(cpu, max_freq, current_min_freq))
            else:
                self.target.cpufreq.set_max_frequency(cpu, max_freq)
        if min_freq and not min_freq_set:
            current_max_freq = max_freq or current_max_freq
            if min_freq > current_max_freq:
                msg = 'CPU {}: Cannot set min_frequency ({}) above current max frequency ({}).'
                raise ConfigError(msg.format(cpu, min_freq, current_max_freq))
            self.target.cpufreq.set_min_frequency(cpu, min_freq)

    def _retrive_cpufreq_info(self):
        '''
        Tries to retrieve cpu freq information for all cpus on device.
        For each cpu domain, only one cpu is queried for information and
        duplicated across related cpus. This is to reduce calls to the target
        and as long as one core per domain is online the remaining cpus information
        can still be populated.
        '''
        for cluster_cpu in resolve_unique_domain_cpus('all', self.target):
            domain_cpus = self.target.cpufreq.get_related_cpus(cluster_cpu)
            for cpu in domain_cpus:
                if cpu in self.target.list_online_cpus():
                    supported_cpu_freqs = self.target.cpufreq.list_frequencies(cpu)
                    supported_cpu_governors = self.target.cpufreq.list_governors(cpu)
                    break
            else:
                msg = 'CPUFreq information could not be retrieved for{};'\
                      'Will not be validated against device.'
                logger.debug(msg.format(' CPU{},'.format(cpu for cpu in domain_cpus)))
                return

            for cpu in domain_cpus:
                self.supported_cpu_freqs[cpu] = supported_cpu_freqs
                self.supported_cpu_governors[cpu] = supported_cpu_governors

    def _get_common_values(self):
        ''' Find common values for frequency and governors across all cores'''
        common_freqs = None
        common_gov = None
        all_freqs = None
        initialized = False
        for cpu in resolve_unique_domain_cpus('all', self.target):
            if not initialized:
                initialized = True
                common_freqs = set(self.supported_cpu_freqs.get(cpu) or [])
                all_freqs = copy(common_freqs)
                common_gov = set(self.supported_cpu_governors.get(cpu) or [])
            else:
                common_freqs = common_freqs.intersection(self.supported_cpu_freqs.get(cpu) or set())
                all_freqs = all_freqs.union(self.supported_cpu_freqs.get(cpu) or set())
                common_gov = common_gov.intersection(self.supported_cpu_governors.get(cpu))

        return all_freqs, common_freqs, common_gov

class IdleStateValue(object):

    def __init__(self, values):
        if values is None:
            self.values = values
        else:
            self.values = [(value.id, value.name, value.desc) for value in values]

    def __call__(self, value):
        if self.values is None:
            return value

        if isinstance(value, basestring):
            value = caseless_string(value)
            if value == 'all':
                return [state[0] for state in self.values]
            elif value == 'none':
                return []
            else:
                return [self._get_state_ID(value)]

        elif isinstance(value, list):
            valid_states = []
            for state in value:
                valid_states.append(self._get_state_ID(state))
            return valid_states
        else:
            raise ValueError('Invalid IdleState: "{}"'.format(value))

    def _get_state_ID(self, value):
        '''Checks passed state and converts to its ID'''
        value = caseless_string(value)
        for s_id, s_name, s_desc in self.values:
            if value == s_id or value == s_name or value == s_desc:
                return s_id
        msg = 'Invalid IdleState: "{}"; Must be in {}'
        raise ValueError(msg.format(value, self.values))

    def __str__(self):
        return 'valid idle state: "{}"'.format(self.values).replace('\'', '')


class CpuidleRuntimeConfig(RuntimeConfig):

    name = 'rt-cpuidle'

    @staticmethod
    def set_idle_state(obj, value, core):
        cpus = resolve_cpus(core, obj.target)
        for cpu in cpus:
            obj.config[cpu] = []
            for state in value:
                obj.config[cpu].append(state)

    def __init__(self, target):
        self.config = defaultdict(dict)
        self.supported_idle_states = {}
        super(CpuidleRuntimeConfig, self).__init__(target)

    def initialize(self):
        if not self.target.has('cpuidle'):
            return

        self._retrieve_device_idle_info()

        common_idle_states = self._get_common_idle_values()
        idle_state_val = IdleStateValue(common_idle_states)

        if common_idle_states:
            param_name = 'idle_states'
            self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=idle_state_val,
                                     setter=self.set_idle_state,
                                     setter_params={'core': None},
                                     description="""
                                     The idle states to be set for all cores
                                     """)

        for name in unique(self.target.platform.core_names):
            cpu = resolve_cpus(name, self.target)[0]
            idle_state_val = IdleStateValue(self.supported_idle_states.get(cpu))
            param_name = '{}_idle_states'.format(name)
            self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=idle_state_val,
                                     setter=self.set_idle_state,
                                     setter_params={'core': name},
                                     description="""
                                     The idle states to be set for {} cores
                                     """.format(name))

        for cpu_no in range(self.target.number_of_cpus):
            idle_state_val = IdleStateValue(self.supported_idle_states.get(cpu_no))
            param_name = 'cpu{}_idle_states'.format(cpu_no)
            self._runtime_params[param_name] = \
                    RuntimeParameter(param_name, kind=idle_state_val,
                                     setter=self.set_idle_state,
                                     setter_params={'core': cpu_no},
                                     description="""
                                     The idle states to be set for cpu{}
                                     """.format(cpu_no))

        if self.target.has('bl'):
            for cluster in ['big', 'little']:
                cpu = resolve_cpus(cluster, self.target)[0]
                idle_state_val = IdleStateValue(self.supported_idle_states.get(cpu))
                param_name = '{}_idle_states'.format(cluster)
                self._runtime_params[param_name] = \
                        RuntimeParameter(param_name, kind=idle_state_val,
                                         setter=self.set_idle_state,
                                         setter_params={'core': cluster},
                                         description="""
                                         The idle states to be set for the {} cores
                                         """.format(cluster))

    def check_target(self):
        if not self.target.has('cpuidle'):
            raise TargetError('Target does not appear to support cpuidle')

    def validate_parameters(self):
        return

    def clear(self):
        self.config = defaultdict(dict)

    def commit(self):
        for cpu in self.config:
            idle_states = set(state.id for state in self.supported_idle_states.get(cpu, []))
            enabled = self.config[cpu]
            disabled = idle_states.difference(enabled)
            for state in enabled:
                self.target.cpuidle.enable(state, cpu)
            for state in disabled:
                self.target.cpuidle.disable(state, cpu)

    def _retrieve_device_idle_info(self):
        for cpu in range(self.target.number_of_cpus):
            self.supported_idle_states[cpu] = self.target.cpuidle.get_states(cpu)

    def _get_common_idle_values(self):
        '''Find common values for cpu idle states across all cores'''
        common_idle_states = []
        for cpu in range(self.target.number_of_cpus):
            for state in self.supported_idle_states.get(cpu) or []:
                if state.name not in common_idle_states:
                    common_idle_states.append(state)
        return common_idle_states

ScreenOrientation = enum(['NATURAL', 'LEFT', 'INVERTED', 'RIGHT'])


class AndroidRuntimeConfig(RuntimeConfig):

    name = 'rt-android'

    @staticmethod
    def set_brightness(obj, value):
        if value is not None:
            obj.config['brightness'] = value

    @staticmethod
    def set_airplane_mode(obj, value):
        if value is not None:
            obj.config['airplane_mode'] = value

    @staticmethod
    def set_rotation(obj, value):
        if value is not None:
            obj.config['rotation'] = value.value

    @staticmethod
    def set_screen_state(obj, value):
        if value is not None:
            obj.config['screen_on'] = value

    def __init__(self, target):
        self.config = defaultdict(dict)
        super(AndroidRuntimeConfig, self).__init__(target)

    def initialize(self):
        if self.target.os not in ['android', 'chromeos']:
            return
        if self.target.os == 'chromeos' and not self.target.supports_android:
            return

        param_name = 'brightness'
        self._runtime_params[param_name] = \
            RuntimeParameter(param_name, kind=int,
                              constraint=lambda x: 0 <= x <= 255,
                              default=127,
                              setter=self.set_brightness,
                              description="""
                              Specify the screen brightness to be set for
                              the device
                              """)

        if self.target.os is 'android':
            param_name = 'airplane_mode'
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=bool,
                                  setter=self.set_airplane_mode,
                                  description="""
                                  Specify whether airplane mode should be
                                  enabled for the device
                                  """)

            param_name = 'rotation'
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=ScreenOrientation,
                                  setter=self.set_rotation,
                                  description="""
                                  Specify the screen orientation for the device
                                  """)

            param_name = 'screen_on'
            self._runtime_params[param_name] = \
                RuntimeParameter(param_name, kind=bool,
                                  default=True,
                                  setter=self.set_screen_state,
                                  description="""
                                  Specify whether the device screen should be on
                                  """)

    def check_target(self):
        if self.target.os != 'android' and self.target.os != 'chromeos':
            raise ConfigError('Target does not appear to be running Android')
        if self.target.os == 'chromeos' and not self.target.supports_android:
            raise ConfigError('Target does not appear to support Android')

    def validate_parameters(self):
        pass

    def commit(self):
        if 'airplane_mode' in self.config:
            new_airplane_mode = self.config['airplane_mode']
            old_airplane_mode = self.target.get_airplane_mode()
            self.target.set_airplane_mode(new_airplane_mode)

            # If we've just switched airplane mode off, wait a few seconds to
            # enable the network state to stabilise. That's helpful if we're
            # about to run a workload that is going to check for network
            # connectivity.
            if old_airplane_mode and not new_airplane_mode:
                self.logger.info('Disabled airplane mode, waiting up to 20 seconds for network setup')
                network_is_ready = False
                for _ in range(4):
                    time.sleep(5)
                    network_is_ready = self.target.is_network_connected()
                    if network_is_ready:
                        break
                if network_is_ready:
                    self.logger.info("Found a network")
                else:
                    self.logger.warning("Network unreachable")

        if 'brightness' in self.config:
            self.target.set_brightness(self.config['brightness'])
        if 'rotation' in self.config:
            self.target.set_rotation(self.config['rotation'])
        if 'screen_on' in self.config:
            if self.config['screen_on']:
                self.target.ensure_screen_is_on()
            else:
                self.target.ensure_screen_is_off()

    def clear(self):
        self.config = {}
