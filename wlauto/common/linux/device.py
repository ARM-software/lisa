#    Copyright 2014-2015 ARM Limited
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

# pylint: disable=E1101
import os
import re
from collections import namedtuple
from subprocess import CalledProcessError

from wlauto.core.extension import Parameter
from wlauto.core.device import Device, RuntimeParameter, CoreParameter
from wlauto.core.resource import NO_ONE
from wlauto.exceptions import ConfigError, DeviceError, TimeoutError, DeviceNotRespondingError
from wlauto.common.resources import Executable
from wlauto.utils.cpuinfo import Cpuinfo
from wlauto.utils.misc import convert_new_lines, escape_double_quotes
from wlauto.utils.ssh import SshShell
from wlauto.utils.types import boolean, list_of_strings


# a dict of governor name and a list of it tunables that can't be read
WRITE_ONLY_TUNABLES = {
    'interactive': ['boostpulse']
}

FstabEntry = namedtuple('FstabEntry', ['device', 'mount_point', 'fs_type', 'options', 'dump_freq', 'pass_num'])
PsEntry = namedtuple('PsEntry', 'user pid ppid vsize rss wchan pc state name')


class BaseLinuxDevice(Device):  # pylint: disable=abstract-method

    path_module = 'posixpath'
    has_gpu = True

    parameters = [
        Parameter('scheduler', kind=str, default='unknown',
                  allowed_values=['unknown', 'smp', 'hmp', 'iks', 'ea', 'other'],
                  description="""
                  Specifies the type of multi-core scheduling model utilized in the device. The value
                  must be one of the following:

                  :unknown: A generic Device interface is used to interact with the underlying device
                            and the underlying scheduling model is unkown.
                  :smp: A standard single-core or Symmetric Multi-Processing system.
                  :hmp: ARM Heterogeneous Multi-Processing system.
                  :iks: Linaro In-Kernel Switcher.
                  :ea: ARM Energy-Aware scheduler.
                  :other: Any other system not covered by the above.

                          .. note:: most currently-available systems would fall under ``smp`` rather than
                                    this value. ``other`` is there to future-proof against new schemes
                                    not yet covered by WA.

                  """),
        Parameter('iks_switch_frequency', kind=int, default=None,
                  description="""
                 This is the switching frequency, in kilohertz, of IKS devices. This parameter *MUST NOT*
                 be set for non-IKS device (i.e. ``scheduler != 'iks'``). If left unset for IKS devices,
                 it will default to ``800000``, i.e. 800MHz.
                 """),

    ]

    runtime_parameters = [
        RuntimeParameter('sysfile_values', 'get_sysfile_values', 'set_sysfile_values', value_name='params'),
        CoreParameter('${core}_cores', 'get_number_of_active_cores', 'set_number_of_active_cores',
                      value_name='number'),
        CoreParameter('${core}_min_frequency', 'get_core_min_frequency', 'set_core_min_frequency',
                      value_name='freq'),
        CoreParameter('${core}_max_frequency', 'get_core_max_frequency', 'set_core_max_frequency',
                      value_name='freq'),
        CoreParameter('${core}_governor', 'get_core_governor', 'set_core_governor',
                      value_name='governor'),
        CoreParameter('${core}_governor_tunables', 'get_core_governor_tunables', 'set_core_governor_tunables',
                      value_name='tunables'),
    ]

    @property
    def active_cpus(self):
        val = self.get_sysfile_value('/sys/devices/system/cpu/online')
        cpus = re.findall(r"([\d]\-[\d]|[\d])", val)
        active_cpus = []
        for cpu in cpus:
            if '-' in cpu:
                lo, hi = cpu.split('-')
                active_cpus.extend(range(int(lo), int(hi) + 1))
            else:
                active_cpus.append(int(cpu))
        return active_cpus

    @property
    def number_of_cores(self):
        """
        Added in version 2.1.4.

        """
        if self._number_of_cores is None:
            corere = re.compile('^\s*cpu\d+\s*$')
            output = self.execute('ls /sys/devices/system/cpu')
            self._number_of_cores = 0
            for entry in output.split():
                if corere.match(entry):
                    self._number_of_cores += 1
        return self._number_of_cores

    @property
    def resource_cache(self):
        return self.path.join(self.working_directory, '.cache')

    @property
    def file_transfer_cache(self):
        return self.path.join(self.working_directory, '.transfer')

    @property
    def cpuinfo(self):
        if not self._cpuinfo:
            self._cpuinfo = Cpuinfo(self.execute('cat /proc/cpuinfo'))
        return self._cpuinfo

    def __init__(self, **kwargs):
        super(BaseLinuxDevice, self).__init__(**kwargs)
        self.busybox = None
        self._is_initialized = False
        self._is_ready = False
        self._just_rebooted = False
        self._is_rooted = None
        self._available_frequencies = {}
        self._available_governors = {}
        self._available_governor_tunables = {}
        self._number_of_cores = None
        self._written_sysfiles = []
        self._cpuinfo = None

    def validate(self):
        if len(self.core_names) != len(self.core_clusters):
            raise ConfigError('core_names and core_clusters are of different lengths.')
        if self.iks_switch_frequency is not None and self.scheduler != 'iks':  # pylint: disable=E0203
            raise ConfigError('iks_switch_frequency must NOT be set for non-IKS devices.')
        if self.iks_switch_frequency is None and self.scheduler == 'iks':  # pylint: disable=E0203
            self.iks_switch_frequency = 800000  # pylint: disable=W0201

    def initialize(self, context, *args, **kwargs):
        self.execute('mkdir -p {}'.format(self.working_directory))
        if self.is_rooted:
            if not self.is_installed('busybox'):
                self.busybox = self.deploy_busybox(context)
            else:
                self.busybox = 'busybox'
        self.init(context, *args, **kwargs)

    def get_sysfile_value(self, sysfile, kind=None):
        """
        Get the contents of the specified sysfile.

        :param sysfile: The file who's contents will be returned.

        :param kind: The type of value to be expected in the sysfile. This can
                     be any Python callable that takes a single str argument.
                     If not specified or is None, the contents will be returned
                     as a string.

        """
        output = self.execute('cat \'{}\''.format(sysfile), as_root=True).strip()  # pylint: disable=E1103
        if kind:
            return kind(output)
        else:
            return output

    def set_sysfile_value(self, sysfile, value, verify=True):
        """
        Set the value of the specified sysfile. By default, the value will be checked afterwards.
        Can be overridden by setting ``verify`` parameter to ``False``.

        """
        value = str(value)
        self.execute('echo {} > \'{}\''.format(value, sysfile), check_exit_code=False, as_root=True)
        if verify:
            output = self.get_sysfile_value(sysfile)
            if not output.strip() == value:  # pylint: disable=E1103
                message = 'Could not set the value of {} to {}'.format(sysfile, value)
                raise DeviceError(message)
        self._written_sysfiles.append(sysfile)

    def get_sysfile_values(self):
        """
        Returns a dict mapping paths of sysfiles that were previously set to their
        current values.

        """
        values = {}
        for sysfile in self._written_sysfiles:
            values[sysfile] = self.get_sysfile_value(sysfile)
        return values

    def set_sysfile_values(self, params):
        """
        The plural version of ``set_sysfile_value``. Takes a single parameter which is a mapping of
        file paths to values to be set. By default, every value written will be verified. The can
        be disabled for individual paths by appending ``'!'`` to them.

        """
        for sysfile, value in params.iteritems():
            verify = not sysfile.endswith('!')
            sysfile = sysfile.rstrip('!')
            self.set_sysfile_value(sysfile, value, verify=verify)

    def deploy_busybox(self, context, force=False):
        """
        Deploys the busybox Android binary (hence in android module) to the
        specified device, and returns the path to the binary on the device.

        :param device: device to deploy the binary to.
        :param context: an instance of ExecutionContext
        :param force: by default, if the binary is already present on the
                    device, it will not be deployed again. Setting force
                    to ``True`` overrides that behavior and ensures that the
                    binary is always copied. Defaults to ``False``.

        :returns: The on-device path to the busybox binary.

        """
        on_device_executable = self.path.join(self.binaries_directory, 'busybox')
        if not force and self.file_exists(on_device_executable):
            return on_device_executable
        host_file = context.resolver.get(Executable(NO_ONE, self.abi, 'busybox'))
        return self.install(host_file)

    def list_file_systems(self):
        output = self.execute('mount')
        fstab = []
        for line in output.split('\n'):
            fstab.append(FstabEntry(*line.split()))
        return fstab

    # Process query and control

    def get_pids_of(self, process_name):
        """Returns a list of PIDs of all processes with the specified name."""
        result = self.execute('ps {}'.format(process_name[-15:]), check_exit_code=False).strip()
        if result and 'not found' not in result:
            return [int(x.split()[1]) for x in result.split('\n')[1:]]
        else:
            return []

    def ps(self, **kwargs):
        """
        Returns the list of running processes on the device. Keyword arguments may
        be used to specify simple filters for columns.

        Added in version 2.1.4

        """
        lines = iter(convert_new_lines(self.execute('ps')).split('\n'))
        lines.next()  # header
        result = []
        for line in lines:
            parts = line.split()
            if parts:
                result.append(PsEntry(*(parts[0:1] + map(int, parts[1:5]) + parts[5:])))
        if not kwargs:
            return result
        else:
            filtered_result = []
            for entry in result:
                if all(getattr(entry, k) == v for k, v in kwargs.iteritems()):
                    filtered_result.append(entry)
            return filtered_result

    def kill(self, pid, signal=None, as_root=False):  # pylint: disable=W0221
        """
        Kill the specified process.

            :param pid: PID of the process to kill.
            :param signal: Specify which singal to send to the process. This must
                           be a valid value for -s option of kill. Defaults to ``None``.

        Modified in version 2.1.4: added ``signal`` parameter.

        """
        signal_string = '-s {}'.format(signal) if signal else ''
        self.execute('kill {} {}'.format(signal_string, pid), as_root=as_root)

    def killall(self, process_name, signal=None, as_root=False):  # pylint: disable=W0221
        """
        Kill all processes with the specified name.

            :param process_name: The name of the process(es) to kill.
            :param signal: Specify which singal to send to the process. This must
                           be a valid value for -s option of kill. Defaults to ``None``.

        Modified in version 2.1.5: added ``as_root`` parameter.

        """
        for pid in self.get_pids_of(process_name):
            self.kill(pid, signal=signal, as_root=as_root)

    # cpufreq

    def list_available_cpu_governors(self, cpu):
        """Returns a list of governors supported by the cpu."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        if cpu not in self._available_governors:
            cmd = 'cat /sys/devices/system/cpu/{}/cpufreq/scaling_available_governors'.format(cpu)
            output = self.execute(cmd, check_exit_code=True)
            self._available_governors[cpu] = output.strip().split()  # pylint: disable=E1103
        return self._available_governors[cpu]

    def get_cpu_governor(self, cpu):
        """Returns the governor currently set for the specified CPU."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_governor'.format(cpu)
        return self.get_sysfile_value(sysfile)

    def set_cpu_governor(self, cpu, governor, **kwargs):
        """
        Set the governor for the specified CPU.
        See https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt

        :param cpu: The CPU for which the governor is to be set. This must be
                    the full name as it appears in sysfs, e.g. "cpu0".
        :param governor: The name of the governor to be used. This must be
                         supported by the specific device.

        Additional keyword arguments can be used to specify governor tunables for
        governors that support them.

        :note: On big.LITTLE all cores in a cluster must be using the same governor.
               Setting the governor on any core in a cluster will also set it on all
               other cores in that cluster.

        :raises: ConfigError if governor is not supported by the CPU.
        :raises: DeviceError if, for some reason, the governor could not be set.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        supported = self.list_available_cpu_governors(cpu)
        if governor not in supported:
            raise ConfigError('Governor {} not supported for cpu {}'.format(governor, cpu))
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_governor'.format(cpu)
        self.set_sysfile_value(sysfile, governor)
        self.set_cpu_governor_tunables(cpu, governor, **kwargs)

    def list_available_cpu_governor_tunables(self, cpu):
        """Returns a list of tunables available for the governor on the specified CPU."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        governor = self.get_cpu_governor(cpu)
        if governor not in self._available_governor_tunables:
            try:
                tunables_path = '/sys/devices/system/cpu/{}/cpufreq/{}'.format(cpu, governor)
                self._available_governor_tunables[governor] = self.listdir(tunables_path)
            except DeviceError:  # probably an older kernel
                try:
                    tunables_path = '/sys/devices/system/cpu/cpufreq/{}'.format(governor)
                    self._available_governor_tunables[governor] = self.listdir(tunables_path)
                except DeviceError:  # governor does not support tunables
                    self._available_governor_tunables[governor] = []
        return self._available_governor_tunables[governor]

    def get_cpu_governor_tunables(self, cpu):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        governor = self.get_cpu_governor(cpu)
        tunables = {}
        for tunable in self.list_available_cpu_governor_tunables(cpu):
            if tunable not in WRITE_ONLY_TUNABLES.get(governor, []):
                try:
                    path = '/sys/devices/system/cpu/{}/cpufreq/{}/{}'.format(cpu, governor, tunable)
                    tunables[tunable] = self.get_sysfile_value(path)
                except DeviceError:  # May be an older kernel
                    path = '/sys/devices/system/cpu/cpufreq/{}/{}'.format(governor, tunable)
                    tunables[tunable] = self.get_sysfile_value(path)
        return tunables

    def set_cpu_governor_tunables(self, cpu, governor, **kwargs):
        """
        Set tunables for the specified governor. Tunables should be specified as
        keyword arguments. Which tunables and values are valid depends on the
        governor.

        :param cpu: The cpu for which the governor will be set. This must be the
                    full cpu name as it appears in sysfs, e.g. ``cpu0``.
        :param governor: The name of the governor. Must be all lower case.

        The rest should be keyword parameters mapping tunable name onto the value to
        be set for it.

        :raises: ConfigError if governor specified is not a valid governor name, or if
                 a tunable specified is not valid for the governor.
        :raises: DeviceError if could not set tunable.


        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        valid_tunables = self.list_available_cpu_governor_tunables(cpu)
        for tunable, value in kwargs.iteritems():
            if tunable in valid_tunables:
                try:
                    path = '/sys/devices/system/cpu/{}/cpufreq/{}/{}'.format(cpu, governor, tunable)
                    self.set_sysfile_value(path, value)
                except DeviceError:  # May be an older kernel
                    path = '/sys/devices/system/cpu/cpufreq/{}/{}'.format(governor, tunable)
                    self.set_sysfile_value(path, value)
            else:
                message = 'Unexpected tunable {} for governor {} on {}.\n'.format(tunable, governor, cpu)
                message += 'Available tunables are: {}'.format(valid_tunables)
                raise ConfigError(message)

    def enable_cpu(self, cpu):
        """
        Enable the specified core.

        :param cpu: CPU core to enable. This must be the full name as it
                    appears in sysfs, e.g. "cpu0".

        """
        self.hotplug_cpu(cpu, online=True)

    def disable_cpu(self, cpu):
        """
        Disable the specified core.

        :param cpu: CPU core to disable. This must be the full name as it
                    appears in sysfs, e.g. "cpu0".
        """
        self.hotplug_cpu(cpu, online=False)

    def hotplug_cpu(self, cpu, online):
        """
        Hotplug the specified CPU either on or off.
        See https://www.kernel.org/doc/Documentation/cpu-hotplug.txt

        :param cpu: The CPU for which the governor is to be set. This must be
                    the full name as it appears in sysfs, e.g. "cpu0".
        :param online: CPU will be enabled if this value bool()'s to True, and
                       will be disabled otherwise.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        status = 1 if online else 0
        sysfile = '/sys/devices/system/cpu/{}/online'.format(cpu)
        self.set_sysfile_value(sysfile, status)

    def list_available_cpu_frequencies(self, cpu):
        """Returns a list of frequencies supported by the cpu or an empty list
        if not could be found."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        if cpu not in self._available_frequencies:
            try:
                cmd = 'cat /sys/devices/system/cpu/{}/cpufreq/scaling_available_frequencies'.format(cpu)
                output = self.execute(cmd)
                self._available_frequencies[cpu] = map(int, output.strip().split())  # pylint: disable=E1103
            except DeviceError:
                # we return an empty list because on some devices scaling_available_frequencies
                # is not generated. So we are returing an empty list as an indication
                # http://adrynalyne-teachtofish.blogspot.co.uk/2011/11/how-to-enable-scalingavailablefrequenci.html
                self._available_frequencies[cpu] = []
        return self._available_frequencies[cpu]

    def get_cpu_min_frequency(self, cpu):
        """
        Returns the min frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the minimum frequency and the following exception will be
        raised ::

        :raises: DeviceError if for some reason the frequency could not be read.

        """
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_min_freq'.format(cpu)
        return self.get_sysfile_value(sysfile)

    def set_cpu_min_frequency(self, cpu, frequency):
        """
        Set's the minimum value for CPU frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value should be
        either an int or a string representing an integer. The Value must also be
        supported by the device. The available frequencies can be obtained by calling
        get_available_frequencies() or examining

        /sys/devices/system/cpu/cpuX/cpufreq/scaling_available_frequencies

        on the device.

        :raises: ConfigError if the frequency is not supported by the CPU.
        :raises: DeviceError if, for some reason, frequency could not be set.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        available_frequencies = self.list_available_cpu_frequencies(cpu)
        try:
            value = int(frequency)
            if available_frequencies and value not in available_frequencies:
                raise ConfigError('Can\'t set {} frequency to {}\nmust be in {}'.format(cpu,
                                                                                        value,
                                                                                        available_frequencies))
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_min_freq'.format(cpu)
            self.set_sysfile_value(sysfile, value)
        except ValueError:
            raise ValueError('value must be an integer; got: "{}"'.format(value))

    def get_cpu_max_frequency(self, cpu):
        """
        Returns the max frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the maximum frequency and the following exception will be
        raised ::

        :raises: DeviceError if for some reason the frequency could not be read.
        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_max_freq'.format(cpu)
        return self.get_sysfile_value(sysfile)

    def set_cpu_max_frequency(self, cpu, frequency):
        """
        Set's the minimum value for CPU frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value should be
        either an int or a string representing an integer. The Value must also be
        supported by the device. The available frequencies can be obtained by calling
        get_available_frequencies() or examining

        /sys/devices/system/cpu/cpuX/cpufreq/scaling_available_frequencies

        on the device.

        :raises: ConfigError if the frequency is not supported by the CPU.
        :raises: DeviceError if, for some reason, frequency could not be set.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        available_frequencies = self.list_available_cpu_frequencies(cpu)
        try:
            value = int(frequency)
            if available_frequencies and value not in available_frequencies:
                raise DeviceError('Can\'t set {} frequency to {}\nmust be in {}'.format(cpu,
                                                                                        value,
                                                                                        available_frequencies))
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_max_freq'.format(cpu)
            self.set_sysfile_value(sysfile, value)
        except ValueError:
            raise ValueError('value must be an integer; got: "{}"'.format(value))

    def get_cpuidle_states(self, cpu=0):
        """
        Return map of cpuidle states with their descriptive names.
        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        cpuidle_states = {}
        statere = re.compile('^\s*state\d+\s*$')
        output = self.execute("ls /sys/devices/system/cpu/{}/cpuidle".format(cpu))
        for entry in output.split():
            if statere.match(entry):
                cpuidle_states[entry] = self.get_sysfile_value("/sys/devices/system/cpu/{}/cpuidle/{}/desc".format(cpu, entry))
        return cpuidle_states

    # Core- and cluster-level mapping for the above cpu-level APIs above. The
    # APIs make the following assumptions, which were True for all devices that
    # existed at the time of writing:
    #   1. A cluster can only contain cores of one type.
    #   2. All cores in a cluster are tied to the same DVFS domain, therefore
    #      changes to cpufreq for a core will affect all other cores on the
    #      same cluster.

    def get_core_clusters(self, core, strict=True):
        """Returns the list of clusters  that contain the specified core. if ``strict``
        is ``True``, raises ValueError if no clusters has been found (returns empty list
        if ``strict`` is ``False``)."""
        core_indexes = [i for i, c in enumerate(self.core_names) if c == core]
        clusters = sorted(list(set(self.core_clusters[i] for i in core_indexes)))
        if strict and not clusters:
            raise ValueError('No active clusters for core {}'.format(core))
        return clusters

    def get_cluster_cpu(self, cluster):
        """Returns the first *active* cpu for the cluster. If the entire cluster
        has been hotplugged, this will raise a ``ValueError``."""
        cpu_indexes = set([i for i, c in enumerate(self.core_clusters) if c == cluster])
        active_cpus = sorted(list(cpu_indexes.intersection(self.active_cpus)))
        if not active_cpus:
            raise ValueError('All cpus for cluster {} are offline'.format(cluster))
        return active_cpus[0]

    def list_available_cluster_governors(self, cluster):
        return self.list_available_cpu_governors(self.get_cluster_cpu(cluster))

    def get_cluster_governor(self, cluster):
        return self.get_cpu_governor(self.get_cluster_cpu(cluster))

    def set_cluster_governor(self, cluster, governor, **tunables):
        return self.set_cpu_governor(self.get_cluster_cpu(cluster), governor, **tunables)

    def list_available_cluster_governor_tunables(self, cluster):
        return self.list_available_cpu_governor_tunables(self.get_cluster_cpu(cluster))

    def get_cluster_governor_tunables(self, cluster):
        return self.get_cpu_governor_tunables(self.get_cluster_cpu(cluster))

    def set_cluster_governor_tunables(self, cluster, governor, **tunables):
        return self.set_cpu_governor_tunables(self.get_cluster_cpu(cluster), governor, **tunables)

    def get_cluster_min_frequency(self, cluster):
        return self.get_cpu_min_frequency(self.get_cluster_cpu(cluster))

    def set_cluster_min_frequency(self, cluster, freq):
        return self.set_cpu_min_frequency(self.get_cluster_cpu(cluster), freq)

    def get_cluster_max_frequency(self, cluster):
        return self.get_cpu_max_frequency(self.get_cluster_cpu(cluster))

    def set_cluster_max_frequency(self, cluster, freq):
        return self.set_cpu_max_frequency(self.get_cluster_cpu(cluster), freq)

    def get_core_cpu(self, core):
        for cluster in self.get_core_clusters(core):
            try:
                return self.get_cluster_cpu(cluster)
            except ValueError:
                pass
        raise ValueError('No active CPUs found for core {}'.format(core))

    def list_available_core_governors(self, core):
        return self.list_available_cpu_governors(self.get_core_cpu(core))

    def get_core_governor(self, core):
        return self.get_cpu_governor(self.get_core_cpu(core))

    def set_core_governor(self, core, governor, **tunables):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_governor(cluster, governor, **tunables)

    def list_available_core_governor_tunables(self, core):
        return self.list_available_cpu_governor_tunables(self.get_core_cpu(core))

    def get_core_governor_tunables(self, core):
        return self.get_cpu_governor_tunables(self.get_core_cpu(core))

    def set_core_governor_tunables(self, core, tunables):
        for cluster in self.get_core_clusters(core):
            governor = self.get_cluster_governor(cluster)
            self.set_cluster_governor_tunables(cluster, governor, **tunables)

    def get_core_min_frequency(self, core):
        return self.get_cpu_min_frequency(self.get_core_cpu(core))

    def set_core_min_frequency(self, core, freq):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_min_frequency(cluster, freq)

    def get_core_max_frequency(self, core):
        return self.get_cpu_max_frequency(self.get_core_cpu(core))

    def set_core_max_frequency(self, core, freq):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_max_frequency(cluster, freq)

    def get_number_of_active_cores(self, core):
        if core not in self.core_names:
            raise ValueError('Unexpected core: {}; must be in {}'.format(core, list(set(self.core_names))))
        active_cpus = self.active_cpus
        num_active_cores = 0
        for i, c in enumerate(self.core_names):
            if c == core and i in active_cpus:
                num_active_cores += 1
        return num_active_cores

    def set_number_of_active_cores(self, core, number):
        if core not in self.core_names:
            raise ValueError('Unexpected core: {}; must be in {}'.format(core, list(set(self.core_names))))
        core_ids = [i for i, c in enumerate(self.core_names) if c == core]
        max_cores = len(core_ids)
        if number > max_cores:
            message = 'Attempting to set the number of active {} to {}; maximum is {}'
            raise ValueError(message.format(core, number, max_cores))
        for i in xrange(0, number):
            self.enable_cpu(core_ids[i])
        for i in xrange(number, max_cores):
            self.disable_cpu(core_ids[i])

    # internal methods

    def _check_ready(self):
        if not self._is_ready:
            raise AttributeError('Device not ready.')

    def _get_core_cluster(self, core):
        """Returns the first cluster that has cores of the specified type. Raises
        value error if no cluster for the specified type has been found"""
        core_indexes = [i for i, c in enumerate(self.core_names) if c == core]
        core_clusters = set(self.core_clusters[i] for i in core_indexes)
        if not core_clusters:
            raise ValueError('No cluster found for core {}'.format(core))
        return sorted(list(core_clusters))[0]


class LinuxDevice(BaseLinuxDevice):

    platform = 'linux'

    default_timeout = 30
    delay = 2
    long_delay = 3 * delay
    ready_timeout = 60

    parameters = [
        Parameter('host', mandatory=True, description='Host name or IP address for the device.'),
        Parameter('username', mandatory=True, description='User name for the account on the device.'),
        Parameter('password', description='Password for the account on the device (for password-based auth).'),
        Parameter('keyfile', description='Keyfile to be used for key-based authentication.'),
        Parameter('port', kind=int, description='SSH port number on the device.'),

        Parameter('use_telnet', kind=boolean, default=False,
                  description='Optionally, telnet may be used instead of ssh, though this is discouraged.'),

        Parameter('working_directory', default=None,
                  description='''
                  Working directory to be used by WA. This must be in a location where the specified user
                  has write permissions. This will default to /home/<username>/wa (or to /root/wa, if
                  username is 'root').
                  '''),
        Parameter('binaries_directory', default='/usr/local/bin',
                  description='Location of executable binaries on this device (must be in PATH).'),
        Parameter('property_files', kind=list_of_strings,
                  default=['/proc/version', '/etc/debian_version', '/etc/lsb-release', '/etc/arch-release'],
                  description='''
                  A list of paths to files containing static OS properties. These will be pulled into the
                  __meta directory in output for each run in order to provide information about the platfrom.
                  These paths do not have to exist and will be ignored if the path is not present on a
                  particular device.
                  '''),
    ]

    @property
    def is_rooted(self):
        if self._is_rooted is None:
            try:
                self.execute('ls /', as_root=True)
                self._is_rooted = True
            except DeviceError:
                self._is_rooted = False
        return self._is_rooted

    def __init__(self, *args, **kwargs):
        super(LinuxDevice, self).__init__(*args, **kwargs)
        self.shell = None
        self.local_binaries_directory = None
        self._is_rooted = None

    def validate(self):
        if not self.password and not self.keyfile:
            raise ConfigError('Either a password or a keyfile must be provided.')
        if self.working_directory is None:  # pylint: disable=access-member-before-definition
            if self.username == 'root':
                self.working_directory = '/root/wa'  # pylint: disable=attribute-defined-outside-init
            else:
                self.working_directory = '/home/{}/wa'.format(self.username)  # pylint: disable=attribute-defined-outside-init
        self.local_binaries_directory = self.path.join(self.working_directory, 'bin')

    def initialize(self, context, *args, **kwargs):
        self.execute('mkdir -p {}'.format(self.local_binaries_directory))
        self.execute('export PATH={}:$PATH'.format(self.local_binaries_directory))
        super(LinuxDevice, self).initialize(context, *args, **kwargs)

    # Power control

    def reset(self):
        self._is_ready = False
        self.execute('reboot', as_root=True)

    def hard_reset(self):
        super(LinuxDevice, self).hard_reset()
        self._is_ready = False

    def boot(self, **kwargs):
        self.reset()

    def connect(self):  # NOQA pylint: disable=R0912
        self.shell = SshShell(timeout=self.default_timeout)
        self.shell.login(self.host, self.username, self.password, self.keyfile, self.port, telnet=self.use_telnet)
        self._is_ready = True

    def disconnect(self):  # NOQA pylint: disable=R0912
        self.shell.logout()
        self._is_ready = False

    # Execution

    def has_root(self):
        try:
            self.execute('ls /', as_root=True)
            return True
        except DeviceError as e:
            if 'not in the sudoers file' not in e.message:
                raise e
            return False

    def execute(self, command, timeout=default_timeout, check_exit_code=True, background=False,
                as_root=False, strip_colors=True, **kwargs):
        """
        Execute the specified command on the device using adb.

        Parameters:

            :param command: The command to be executed. It should appear exactly
                            as if you were typing it into a shell.
            :param timeout: Time, in seconds, to wait for adb to return before aborting
                            and raising an error. Defaults to ``AndroidDevice.default_timeout``.
            :param check_exit_code: If ``True``, the return code of the command on the Device will
                                    be check and exception will be raised if it is not 0.
                                    Defaults to ``True``.
            :param background: If ``True``, will execute create a new ssh shell rather than using
                               the default session and will return it immediately. If this is ``True``,
                               ``timeout``, ``strip_colors`` and (obvisously) ``check_exit_code`` will
                               be ignored; also, with this, ``as_root=True``  is only valid if ``username``
                               for the device was set to ``root``.
            :param as_root: If ``True``, will attempt to execute command in privileged mode. The device
                            must be rooted, otherwise an error will be raised. Defaults to ``False``.

                            Added in version 2.1.3

        :returns: If ``background`` parameter is set to ``True``, the subprocess object will
                  be returned; otherwise, the contents of STDOUT from the device will be returned.

        """
        self._check_ready()
        if background:
            if as_root and self.username != 'root':
                raise DeviceError('Cannot execute in background with as_root=True unless user is root.')
            return self.shell.background(command)
        else:
            return self.shell.execute(command, timeout, check_exit_code, as_root, strip_colors)

    def kick_off(self, command):
        """
        Like execute but closes adb session and returns immediately, leaving the command running on the
        device (this is different from execute(background=True) which keeps adb connection open and returns
        a subprocess object).

        """
        self._check_ready()
        command = 'sh -c "{}" 1>/dev/null 2>/dev/null &'.format(escape_double_quotes(command))
        return self.shell.execute(command)

    # File management

    def push_file(self, source, dest, as_root=False, timeout=default_timeout):  # pylint: disable=W0221
        self._check_ready()
        if not as_root or self.username == 'root':
            self.shell.push_file(source, dest, timeout=timeout)
        else:
            tempfile = self.path.join(self.working_directory, self.path.basename(dest))
            self.shell.push_file(source, tempfile, timeout=timeout)
            self.shell.execute('cp -r {} {}'.format(tempfile, dest), timeout=timeout, as_root=True)

    def pull_file(self, source, dest, as_root=False, timeout=default_timeout):  # pylint: disable=W0221
        self._check_ready()
        if not as_root or self.username == 'root':
            self.shell.pull_file(source, dest, timeout=timeout)
        else:
            tempfile = self.path.join(self.working_directory, self.path.basename(source))
            self.shell.execute('cp -r {} {}'.format(source, tempfile), timeout=timeout, as_root=True)
            self.shell.execute('chown -R {} {}'.format(self.username, tempfile), timeout=timeout, as_root=True)
            self.shell.pull_file(tempfile, dest, timeout=timeout)

    def delete_file(self, filepath, as_root=False):  # pylint: disable=W0221
        self.execute('rm -rf {}'.format(filepath), as_root=as_root)

    def file_exists(self, filepath):
        output = self.execute('if [ -e \'{}\' ]; then echo 1; else echo 0; fi'.format(filepath))
        return boolean(output.strip())  # pylint: disable=maybe-no-member

    def listdir(self, path, as_root=False, **kwargs):
        contents = self.execute('ls -1 {}'.format(path), as_root=as_root)
        return [x.strip() for x in contents.split('\n')]  # pylint: disable=maybe-no-member

    def install(self, filepath, timeout=default_timeout, with_name=None):  # pylint: disable=W0221
        if self.is_rooted:
            destpath = self.path.join(self.binaries_directory,
                                      with_name and with_name or self.path.basename(filepath))
            self.push_file(filepath, destpath, as_root=True)
            self.execute('chmod a+x {}'.format(destpath), timeout=timeout, as_root=True)
        else:
            destpath = self.path.join(self.local_binaries_directory,
                                      with_name and with_name or self.path.basename(filepath))
            self.push_file(filepath, destpath)
            self.execute('chmod a+x {}'.format(destpath), timeout=timeout)
        return destpath

    install_executable = install  # compatibility

    def uninstall(self, name):
        path = self.path.join(self.local_binaries_directory, name)
        self.delete_file(path)

    uninstall_executable = uninstall  # compatibility

    def is_installed(self, name):
        try:
            self.execute('which {}'.format(name))
            return True
        except DeviceError:
            return False

    # misc

    def ping(self):
        try:
            # May be triggered inside initialize()
            self.shell.execute('ls /', timeout=5)
        except (TimeoutError, CalledProcessError):
            raise DeviceNotRespondingError(self.host)

    def capture_screen(self, filepath):
        if not self.is_installed('scrot'):
            self.logger.debug('Could not take screenshot as scrot is not installed.')
            return
        try:
            tempfile = self.path.join(self.working_directory, os.path.basename(filepath))
            self.execute('DISPLAY=:0.0 scrot {}'.format(tempfile))
            self.pull_file(tempfile, filepath)
            self.delete_file(tempfile)
        except DeviceError as e:
            if "Can't open X dispay." not in e.message:
                raise e
            message = e.message.split('OUTPUT:', 1)[1].strip()
            self.logger.debug('Could not take screenshot: {}'.format(message))

    def is_screen_on(self):
        pass  # TODO

    def ensure_screen_is_on(self):
        pass  # TODO

    def get_properties(self, context):
        for propfile in self.property_files:
            if not self.file_exists(propfile):
                continue
            normname = propfile.lstrip(self.path.sep).replace(self.path.sep, '.')
            outfile = os.path.join(context.host_working_directory, normname)
            self.pull_file(propfile, outfile)
        return {}

