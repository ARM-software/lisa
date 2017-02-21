from collections import defaultdict, OrderedDict

from wa.framework.plugin import Plugin
from wa.framework.exception import ConfigError

from devlib.exception import TargetError
from devlib.utils.misc import unique
from devlib.utils.types import integer


class RuntimeConfig(Plugin):

    kind = 'runtime-config'

    parameters = [
    ]

# class RuntimeConfig(object):

    @property
    def supported_parameters(self):
        raise NotImplementedError()

    @property
    def core_names(self):
        return unique(self.target.core_names)

    def __init__(self, target):
        super(RuntimeConfig, self).__init__()
        self.target = target

    def initialize(self, context):
        pass

    def add(self, name, value):
        raise NotImplementedError()

    def validate(self):
        return True

    def set(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


class HotplugRuntimeConfig(RuntimeConfig):
##### NOTE: Currently if initialized with cores hotplugged, this will fail trying to hotplug back in
    @property
    def supported_parameters(self):
        # params = ['cores'.format(c) for c in self.target.core_names]
        # params = ['{}_cores'.format(c) for c in self.target.core_names]
        params = ['cores']
        return params

    def __init__(self, target):
        super(HotplugRuntimeConfig, self).__init__(target)
        self.num_cores = defaultdict(dict)

    def add(self, name, value):
        if not self.target.has('hotplug'):
            raise TargetError('Target does not support hotplug.')
        core, _ = split_parameter_name(name, self.supported_parameters)

        # cpus = cpusFromPrefix(core, self.target)
        # core = name.split('_')[0]
        value = integer(value)
        if core not in self.core_names:
            raise ValueError(name)
        max_cores = self.core_count(core)
        if value > max_cores:
            message = 'Cannot set number of {}\'s to {}; max is {}'
            raise ValueError(message.format(core, value, max_cores))
        self.num_cores[core] = value
        if all(v == 0 for v in self.num_cores.values()):
            raise ValueError('Cannot set number of all cores to 0')

    def set(self):
        for c, n in reversed(sorted(self.num_cores.iteritems(),
                                    key=lambda x: x[1])):
            self.set_num_online_cpus(c, n)

    def clear(self):
        self.num_cores = defaultdict(dict)

    def set_num_online_cpus(self, core, number):
        indexes = [i for i, c in enumerate(self.target.core_names) if c == core]
        self.target.hotplug.online(*indexes[:number])
        self.target.hotplug.offline(*indexes[number:])

    def core_count(self, core):
        return sum(1 for c in self.target.core_names if c == core)


class SysfileValuesRuntimeConfig(RuntimeConfig):

    @property
    def supported_parameters(self):
        return ['sysfile_values']

    def __init__(self, target):
        super(SysfileValuesRuntimeConfig, self).__init__(target)
        self.sysfile_values = OrderedDict()

    def add(self, name, value):
        for f, v in value.iteritems():
            if f.endswith('+'):
                f = f[:-1]
            elif f.endswith('+!'):
                f = f[:-2] + '!'
            else:
                if f.endswith('!'):
                    self._check_exists(f[:-1])
                else:
                    self._check_exists(f)
            self.sysfile_values[f] = v

    def set(self):
        for f, v in self.sysfile_values.iteritems():
            verify = True
            if f.endswith('!'):
                verify = False
                f = f[:-1]
            self.target.write_value(f, v, verify=verify)

    def clear(self):
        self.sysfile_values = OrderedDict()

    def _check_exists(self, path):
        if not self.target.file_exists(path):
            raise ConfigError('Sysfile "{}" does not exist.'.format(path))


class CpufreqRuntimeConfig(RuntimeConfig):

    @property
    def supported_parameters(self):
        # params = ['{}_frequency'.format(c) for c in self.core_names]
        # params.extend(['{}_max_frequency'.format(c) for c in self.core_names])
        # params.extend(['{}_min_frequency'.format(c) for c in self.core_names])
        # params.extend(['{}_governor'.format(c) for c in self.core_names])
        # params.extend(['{}_governor_tunables'.format(c) for c in self.core_names])

        params = ['frequency']
        params.extend(['max_frequency'])
        params.extend(['min_frequency'])
        params.extend(['governor'])
        params.extend(['governor_tunables'])

        return params

    def __init__(self, target):
        super(CpufreqRuntimeConfig, self).__init__(target)
        self.config = defaultdict(dict)
        self.supports_userspace = None
        self.supported_freqs = {}
        self.supported_govenors = {}
        self.min_supported_freq = {}
        self.max_supported_freq = {}

        for cpu in self.target.list_online_cpus():
            self.supported_freqs[cpu] = self.target.cpufreq.list_frequencies(cpu) or []
            self.supported_govenors[cpu] = self.target.cpufreq.list_governors(cpu) or []

    def add(self, name, value):
        if not self.target.has('cpufreq'):
            raise TargetError('Target does not support cpufreq.')

        prefix, parameter = split_parameter_name(name, self.supported_parameters)
        # Get list of valid cpus for a given prefix.
        cpus = uniqueDomainCpusFromPrefix(prefix, self.target)

        for cpu in cpus:
            # if cpu not in self.target.list_online_cpus():
            #     message = 'Unexpected core name "{}"; must be in {}'
            #     raise ConfigError(message.format(core, self.core_names))
            # try:
            #     cpu = self.target.list_online_cpus(core)[0]
            # except IndexError:
            #     message = 'Cannot retrieve frequencies for {} as no CPUs are online.'
            #     raise TargetError(message.format(core))
            if parameter.endswith('frequency'):
                try:
                    value = integer(value)
                except ValueError:
                    if value.upper() == 'MAX':
                        value = self.supported_freqs[cpu][-1]
                    elif value.upper() == 'MIN':
                        value = self.supported_freqs[cpu][0]
                    else:
                        msg = 'Invalid value {} specified for {}'
                        raise ConfigError(msg.format(value, parameter))
            self.config[cpu][parameter] = value

    def set(self):
        for cpu in self.config:
            config = self.config[cpu]
            if config.get('governor'):
                self.configure_governor(cpu,
                                        config.get('governor'),
                                        config.get('governor_tunables'))
            self.configure_frequency(cpu,
                                     config.get('frequency'),
                                     config.get('min_frequency'),
                                     config.get('max_frequency'))

    def clear(self):
        self.config = defaultdict(dict)

    def validate(self):
        for cpu in self.config:
            if cpu not in self.target.list_online_cpus():
                message = 'Cannot configure frequencies for {} as no CPUs are online.'
                raise TargetError(message.format(cpu))

            config = self.config[cpu]
            minf = config.get('min_frequency')
            maxf = config.get('max_frequency')
            freq = config.get('frequency')
            governor = config.get('governor')
            governor_tunables = config.get('governor_tunables')

            if maxf and minf > maxf:
                message = '{}: min_frequency ({}) cannot be greater than max_frequency ({})'
                raise ConfigError(message.format(cpu, minf, maxf))
            if maxf and freq > maxf:
                message = '{}: cpu frequency ({}) cannot be greater than max_frequency ({})'
                raise ConfigError(message.format(cpu, freq, maxf))
            if freq and minf > freq:
                message = '{}: min_frequency ({}) cannot be greater than cpu frequency ({})'
                raise ConfigError(message.format(cpu, minf, freq))

            # Check that either userspace governor is available or min and max do not differ to frequency
            if 'userspace' not in self.supported_govenors[cpu]:
                self.supports_userspace = False
                if minf and minf != freq:
                    message = '{}: "userspace" governor not available, min frequency ({}) cannot be different to frequency {}'
                    raise ConfigError(message.format(cpu, minf, freq))
                if maxf and maxf != freq:
                    message = '{}: "userspace" governor not available, max frequency ({}) cannot be different to frequency {}'
                    raise ConfigError(message.format(cpu, maxf, freq))
            else:
                self.supports_userspace = True

            # Check that specified values are available on the cpu
            if minf and not minf in self.supported_freqs[cpu]:
                msg = '{}: Minimum frequency {}Hz not available. Must be in {}'.format(cpu, minf, self.supported_freqs[cpu])
                raise TargetError(msg)
            if maxf and not maxf in self.supported_freqs[cpu]:
                msg = '{}: Maximum frequency {}Hz not available. Must be in {}'.format(cpu, maxf, self.supported_freqs[cpu])
                raise TargetError(msg)
            if freq and not freq in self.supported_freqs[cpu]:
                msg = '{}: Frequency {}Hz not available. Must be in {}'.format(cpu, freq, self.supported_freqs[cpu])
                raise TargetError(msg)
            if governor and governor not in self.supported_govenors[cpu]:
                raise TargetError('{}: {} governor not available'.format(cpu, governor))
            if governor_tunables and not governor:
                raise TargetError('{}: {} governor tunables cannot be provided without a governor'.format(cpu, governor))

            # Should check if governor is set to userspace if frequencies are being set?
            # Save a list of available frequencies on the device and check to see if matches?


    def configure_frequency(self, cpu, freq=None, min_freq=None, max_freq=None):
        if cpu not in self.target.list_online_cpus():
            message = 'Cannot configure frequencies for {} as no CPUs are online.'
            raise TargetError(message.format(cpu))


        current_min_freq = self.target.cpufreq.get_min_frequency(cpu)
        current_freq = self.target.cpufreq.get_frequency(cpu)
        current_max_freq = self.target.cpufreq.get_max_frequency(cpu)

        if freq:
            # If 'userspace' governor is not available 'spoof' functionality
            if not self.supports_userspace:
                min_freq = max_freq = freq
            else: ##############################-- Probably shouldn't do this.
                # Set min/max frequency if required
                if not min_freq:
                    min_freq = self.target.cpufreq.get_min_frequency(cpu)
                if not max_freq:
                    max_freq = self.target.cpufreq.get_max_frequency(cpu)

            if freq < current_freq:
                self.target.cpufreq.set_min_frequency(cpu, min_freq)
                if self.supports_userspace:
                    self.target.cpufreq.set_frequency(cpu, freq)
                self.target.cpufreq.set_max_frequency(cpu, max_freq)
            else:
                self.target.cpufreq.set_max_frequency(cpu, max_freq)
                if self.supports_userspace:
                    self.target.cpufreq.set_frequency(cpu, freq)
                self.target.cpufreq.set_min_frequency(cpu, min_freq)
            return

        if max_freq:
            if max_freq < current_min_freq:
                if min_freq:
                    self.target.cpufreq.set_min_frequency(cpu, min_freq)
                    self.target.cpufreq.set_max_frequency(cpu, max_freq)
                    min_freq_set = True
                else:
                    message = '{}: Cannot set max_frequency ({}) below current min frequency ({}).'
                    raise TargetError(message.format(cpu, max_freq, current_min_freq))
            else:
                self.target.cpufreq.set_max_frequency(cpu, max_freq)
        if min_freq and not min_freq_set:
            current_max_freq = max_freq or current_max_freq
            if min_freq > current_max_freq:
                message = '{}: Cannot set min_frequency ({}) below current max frequency ({}).'
                raise TargetError(message.format(cpu, max_freq, current_min_freq))
            self.target.cpufreq.set_min_frequency(cpu, min_freq)



        # if freq:
        #     if not min_freq:
        #         min_freq = self.target.cpufreq.get_min_frequency(cpu)
        #         min_freq = freq
        #     if not max_freq:
        #         max_freq = self.target.cpufreq.get_max_frequency(cpu)
        #         max_freq = freq
        #     self.target.cpufreq.set_min_frequency(cpu, min_freq)
        #     self.target.cpufreq.set_frequency(cpu, freq)
        #     self.target.cpufreq.set_max_frequency(cpu, max_freq)
    #     #     return
    #     min_freq_set = False
    #     if max_freq:
    #         current_min_freq = self.target.cpufreq.get_min_frequency(cpu)
    #         if max_freq < current_min_freq:
    #             if min_freq:
    #                 self.target.cpufreq.set_min_frequency(cpu, min_freq)
    #                 self.target.cpufreq.set_max_frequency(cpu, max_freq)
    #                 min_freq_set = True
    #             else:
    #                 message = '{}: Cannot set max_frequency ({}) below current min frequency ({}).'
    #                 raise TargetError(message.format(core, max_freq, current_min_freq))
    #         else:
    #             self.target.cpufreq.set_max_frequency(cpu, max_freq)
    #     if min_freq and not min_freq_set:
    #         current_max_freq = max_freq or self.target.cpufreq.get_max_frequency(cpu)
    #         if min_freq > current_max_freq:
    #             message = '{}: Cannot set min_frequency ({}) below current max frequency ({}).'
    #             raise TargetError(message.format(core, max_freq, current_min_freq))
    #         self.target.cpufreq.set_min_frequency(cpu, min_freq)

    def configure_governor(self, cpu, governor, governor_tunables=None):
        if cpu not in self.target.list_online_cpus():
            message = 'Cannot configure governor for {} as no CPUs are online.'
            raise TargetError(message.format(cpu))

        # for cpu in self.target.list_online_cpus(cpu): #All cpus or only online?
        if governor not in self.supported_govenors[cpu]:
            raise TargetError('{}: {} governor not available'.format(cpu, governor))
        if governor_tunables:
            self.target.cpufreq.set_governor(cpu, governor, **governor_tunables)
        else:
            self.target.cpufreq.set_governor(cpu, governor)



class CpuidleRuntimeConfig(RuntimeConfig):

    @property
    def supported_parameters(self):
        params = ['idle_states']
        return params

    def __init__(self, target):
        super(CpuidleRuntimeConfig, self).__init__(target)
        self.config = defaultdict(dict)
        self.aliases = ['ENABLE_ALL', 'DISABLE_ALL']
        self.available_states = {}

        for cpu in self.target.list_online_cpus():
            self.available_states[cpu] = self.target.cpuidle.get_states(cpu) or []

    def add(self, name, values):
        if not self.target.has('cpufreq'):
            raise TargetError('Target does not support cpufreq.')

        prefix, _ = split_parameter_name(name, self.supported_parameters)
        cpus = uniqueDomainCpusFromPrefix(prefix, self.target)
        # core, _ = name.split('_', 1)
        # if core not in self.core_names:
        #     message = 'Unexpected core name "{}"; must be in {}'
        #     raise ConfigError(message.format(core, self.core_names))

        for cpu in cpus:
            if values in self.aliases:
                self.config[cpu] = [values]
            else:
                self.config[cpu] = values

    def validate(self):
        for cpu in self.config:
            if cpu not in self.target.list_online_cpus():
                message = 'Cannot configure idle states for {} as no CPUs are online.'
                raise TargetError(message.format(cpu))
            for state in self.config[cpu]:
                state = state[1:] if state.startswith('~') else state
                # self.available_states.extend(self.aliases)
                if state not in self.available_states[cpu] + self.aliases:
                    message = 'Unexpected idle state "{}"; must be in {}'
                    raise ConfigError(message.format(state, self.available_states))

    def clear(self):
        self.config = defaultdict(dict)

    def set(self):
        for cpu in self.config:
            for state in self.config[cpu]:
                self.configure_idle_state(state, cpu)

    def configure_idle_state(self, state, cpu=None):
        if cpu is not None:
            if cpu not in self.target.list_online_cpus():
                message = 'Cannot configure idle state for {} as no CPUs are online {}.'
                raise TargetError(message.format(self.target.core_names[cpu], self.target.list_online_cpus()))
        else:
            cpu = 0

        # Check for aliases
        if state == 'ENABLE_ALL':
            self.target.cpuidle.enable_all(cpu)
        elif state == 'DISABLE_ALL':
            self.target.cpuidle.disable_all(cpu)
        elif state.startswith('~'):
            self.target.cpuidle.disable(state[1:], cpu)
        else:
            self.target.cpuidle.enable(state, cpu)


# def cpusFromPrefix(name, target, params):
#     prefix = ''
#     print prefix
#     for param in params:
#         if len(name.split(param)) > 1:
#             print name
#             print param
#             print name.split(param)
#             prefix, _ = name.split(param)
#             prefix = prefix.replace('_', '')
#             break








# TO BE MOVED TO UTILS FILE


# Function to return the cpu prefix without the trailing underscore if
# present from a given list of parameters, and its matching parameter
def split_parameter_name(name, params):
    for param in sorted(params, key=len)[::-1]: # Try matching longest parameter first
        if len(name.split(param)) > 1:
            prefix, _ = name.split(param)
            return prefix[:-1], param
    message = 'Cannot split {}, must in the form [core_]parameter'
    raise ConfigError(message.format(name))

import re
def cpusFromPrefix(prefix, target):   ##### DECIDE WHETHER TO INCLUDE OFFLINE CPUS?   ####

    # Deal with big little substitution
    if prefix.lower() == 'big':
        prefix = target.big_core
        if not prefix:
            raise ConfigError('big core name could not be retrieved')
    elif prefix.lower() == 'little':
        prefix = target.little_core
        if not prefix:
            raise ConfigError('little core name could not be retrieved')

    cpu_list = target.list_online_cpus() + target.list_offline_cpus()

    # Apply to all cpus
    if not prefix:
        cpus = cpu_list

    # Return all cores with specified name
    elif prefix in target.core_names:
        cpus = target.core_cpus(prefix)

    # Check if core number has been supplied.
    else:
        # core_no = prefix[4]
        core_no = re.match('cpu([0-9]+)', prefix, re.IGNORECASE)
        if core_no:
            cpus = [int(core_no.group(1))]
            if cpus[0] not in cpu_list:
                message = 'CPU{} is not available, must be in {}'
                raise ConfigError(message.format(cpus[0], cpu_list))

        else:
            message = 'Unexpected core name "{}"'
            raise ConfigError(message.format(prefix))
    # Should this be applied for everything or just all cpus?
    # Make sure not to include any cpus within the same frequency domain
    # for cpu in cpus:
    #     if cpu not in cpus: # Already removed
    #         continue
    #     cpus = [c for c in cpus if (c is cpu) or
    #                 (c not in target.cpufreq.get_domain_cpus(cpu))]
    # print 'Final results ' + str(cpus)
    # return cpus
    return cpus

# Function to only return cpus list on different frequency domains.
def uniqueDomainCpusFromPrefix(prefix, target):
    cpus = cpusFromPrefix(prefix, target)
    for cpu in cpus:
        if cpu not in cpus: # Already removed
            continue
        cpus = [c for c in cpus if (c is cpu) or
                    (c not in target.cpufreq.get_domain_cpus(cpu))]
    return cpus
