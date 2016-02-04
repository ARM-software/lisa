from wlauto import Module
from wlauto.exceptions import ConfigError, DeviceError


# a dict of governor name and a list of it tunables that can't be read
WRITE_ONLY_TUNABLES = {
    'interactive': ['boostpulse']
}


class CpufreqModule(Module):

    name = 'devcpufreq'
    description = """
    cpufreq-related functionality module for the device. Query and set frequencies, governors, etc.

    APIs in this module break down into three categories: those that operate on cpus, those that
    operate on cores, and those that operate on clusters.

    "cpu" APIs expect a cpufreq CPU id, which could be either an integer or or a string of the
    form "cpu0".

    "cluster" APIs expect a cluster ID. This is an integer as defined by the
    ``device.core_clusters`` list.

    "core" APIs expect a core name, as defined by ``device.core_names`` list.

    """
    capabilities = ['cpufreq']

    def probe(self, device):  # pylint: disable=no-self-use
        path = '/sys/devices/system/cpu/cpu{}/cpufreq'.format(device.online_cpus[0])
        return device.file_exists(path)

    def initialize(self, context):
        # pylint: disable=W0201
        CpufreqModule._available_governors = {}
        CpufreqModule._available_governor_tunables = {}
        CpufreqModule.device = self.root_owner

    def list_available_cpu_governors(self, cpu):
        """Returns a list of governors supported by the cpu."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        if cpu not in self._available_governors:
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_available_governors'.format(cpu)
            output = self.device.get_sysfile_value(sysfile)
            self._available_governors[cpu] = output.strip().split()  # pylint: disable=E1103
        return self._available_governors[cpu]

    def get_cpu_governor(self, cpu):
        """Returns the governor currently set for the specified CPU."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_governor'.format(cpu)
        return self.device.get_sysfile_value(sysfile)

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
        self.device.set_sysfile_value(sysfile, governor)
        self.set_cpu_governor_tunables(cpu, governor, **kwargs)

    def list_available_cpu_governor_tunables(self, cpu):
        """Returns a list of tunables available for the governor on the specified CPU."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        governor = self.get_cpu_governor(cpu)
        if governor not in self._available_governor_tunables:
            try:
                tunables_path = '/sys/devices/system/cpu/{}/cpufreq/{}'.format(cpu, governor)
                self._available_governor_tunables[governor] = self.device.listdir(tunables_path)
            except DeviceError:  # probably an older kernel
                try:
                    tunables_path = '/sys/devices/system/cpu/cpufreq/{}'.format(governor)
                    self._available_governor_tunables[governor] = self.device.listdir(tunables_path)
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
                    tunables[tunable] = self.device.get_sysfile_value(path)
                except DeviceError:  # May be an older kernel
                    path = '/sys/devices/system/cpu/cpufreq/{}/{}'.format(governor, tunable)
                    tunables[tunable] = self.device.get_sysfile_value(path)
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
                    self.device.set_sysfile_value(path, value)
                except DeviceError:  # May be an older kernel
                    path = '/sys/devices/system/cpu/cpufreq/{}/{}'.format(governor, tunable)
                    self.device.set_sysfile_value(path, value)
            else:
                message = 'Unexpected tunable {} for governor {} on {}.\n'.format(tunable, governor, cpu)
                message += 'Available tunables are: {}'.format(valid_tunables)
                raise ConfigError(message)

    def list_available_core_frequencies(self, core):
        cpu = self.get_core_online_cpu(core)
        return self.list_available_cpu_frequencies(cpu)

    def list_available_cpu_frequencies(self, cpu):
        """Returns a list of frequencies supported by the cpu or an empty list
        if not could be found."""
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        try:
            cmd = 'cat /sys/devices/system/cpu/{}/cpufreq/scaling_available_frequencies'.format(cpu)
            output = self.device.execute(cmd)
            #available_frequencies = map(int, output.strip().split())  # pylint: disable=E1103
            available_frequencies = []
            for f in output.strip().split():
                try:
                    available_frequencies.append(int(f))
                except ValueError:
                    pass
        except DeviceError:
            # On some devices scaling_available_frequencies  is not generated.
            # http://adrynalyne-teachtofish.blogspot.co.uk/2011/11/how-to-enable-scalingavailablefrequenci.html
            # Fall back to parsing stats/time_in_state
            cmd = 'cat /sys/devices/system/cpu/{}/cpufreq/stats/time_in_state'.format(cpu)
            out_iter = iter(self.device.execute(cmd).strip().split())
            available_frequencies = map(int, reversed([f for f, _ in zip(out_iter, out_iter)]))
        return available_frequencies

    def get_cpu_min_frequency(self, cpu):
        """
        Returns the min frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the minimum frequency and the following exception will be
        raised ::

        :raises: DeviceError if for some reason the frequency could not be read.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_min_freq'.format(cpu)
        return self.device.get_sysfile_value(sysfile)

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
            self.device.set_sysfile_value(sysfile, value)
        except ValueError:
            raise ValueError('value must be an integer; got: "{}"'.format(value))

    def get_cpu_frequency(self, cpu):
        """
        Returns the current frequency currently set for the specified CPU.

        Warning, this method does not check if the cpu is online or not. It will
        try to read the current frequency and the following exception will be
        raised ::

        :raises: DeviceError if for some reason the frequency could not be read.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_cur_freq'.format(cpu)
        return self.device.get_sysfile_value(sysfile)

    def set_cpu_frequency(self, cpu, frequency, exact=True):
        """
        Set's the minimum value for CPU frequency. Actual frequency will
        depend on the Governor used and may vary during execution. The value should be
        either an int or a string representing an integer.

        If ``exact`` flag is set (the default), the Value must also be supported by
        the device. The available frequencies can be obtained by calling
        get_available_frequencies() or examining

        /sys/devices/system/cpu/cpuX/cpufreq/scaling_available_frequencies

        on the device (if it exists).

        :raises: ConfigError if the frequency is not supported by the CPU.
        :raises: DeviceError if, for some reason, frequency could not be set.

        """
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        try:
            value = int(frequency)
            if exact:
                available_frequencies = self.list_available_cpu_frequencies(cpu)
                if available_frequencies and value not in available_frequencies:
                    raise ConfigError('Can\'t set {} frequency to {}\nmust be in {}'.format(cpu,
                                                                                            value,
                                                                                            available_frequencies))
            if self.get_cpu_governor(cpu) != 'userspace':
                raise ConfigError('Can\'t set {} frequency; governor must be "userspace"'.format(cpu))
            sysfile = '/sys/devices/system/cpu/{}/cpufreq/scaling_setspeed'.format(cpu)
            self.device.set_sysfile_value(sysfile, value, verify=False)
        except ValueError:
            raise ValueError('frequency must be an integer; got: "{}"'.format(value))

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
        return self.device.get_sysfile_value(sysfile)

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
            self.device.set_sysfile_value(sysfile, value)
        except ValueError:
            raise ValueError('value must be an integer; got: "{}"'.format(value))

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
        core_indexes = [i for i, c in enumerate(self.device.core_names) if c == core]
        clusters = sorted(list(set(self.device.core_clusters[i] for i in core_indexes)))
        if strict and not clusters:
            raise ValueError('No active clusters for core {}'.format(core))
        return clusters

    def get_cluster_active_cpu(self, cluster):
        """Returns the first *active* cpu for the cluster. If the entire cluster
        has been hotplugged, this will raise a ``ValueError``."""
        cpu_indexes = set([i for i, c in enumerate(self.device.core_clusters) if c == cluster])
        active_cpus = sorted(list(cpu_indexes.intersection(self.device.online_cpus)))
        if not active_cpus:
            raise ValueError('All cpus for cluster {} are offline'.format(cluster))
        return active_cpus[0]

    def list_available_core_governors(self, core):
        cpu = self.get_core_online_cpu(core)
        return self.list_available_cpu_governors(cpu)

    def list_available_cluster_governors(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.list_available_cpu_governors(cpu)

    def get_core_governor(self, core):
        cpu = self.get_core_online_cpu(core)
        return self.get_cpu_governor(cpu)

    def set_core_governor(self, core, governor, **tunables):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_governor(cluster, governor, **tunables)

    def get_cluster_governor(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.get_cpu_governor(cpu)

    def set_cluster_governor(self, cluster, governor, **tunables):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.set_cpu_governor(cpu, governor, **tunables)

    def list_available_cluster_governor_tunables(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.list_available_cpu_governor_tunables(cpu)

    def get_cluster_governor_tunables(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.get_cpu_governor_tunables(cpu)

    def set_cluster_governor_tunables(self, cluster, governor, **tunables):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.set_cpu_governor_tunables(cpu, governor, **tunables)

    def get_cluster_min_frequency(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.get_cpu_min_frequency(cpu)

    def set_cluster_min_frequency(self, cluster, freq):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.set_cpu_min_frequency(cpu, freq)

    def get_cluster_cur_frequency(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.get_cpu_frequency(cpu)

    def set_cluster_cur_frequency(self, cluster, freq):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.set_cpu_frequency(cpu, freq)

    def get_cluster_max_frequency(self, cluster):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.get_cpu_max_frequency(cpu)

    def set_cluster_max_frequency(self, cluster, freq):
        cpu = self.get_cluster_active_cpu(cluster)
        return self.set_cpu_max_frequency(cpu, freq)

    def get_core_online_cpu(self, core):
        for cluster in self.get_core_clusters(core):
            try:
                return self.get_cluster_active_cpu(cluster)
            except ValueError:
                pass
        raise ValueError('No active CPUs found for core {}'.format(core))

    def list_available_core_governor_tunables(self, core):
        return self.list_available_cpu_governor_tunables(self.get_core_online_cpu(core))

    def get_core_governor_tunables(self, core):
        return self.get_cpu_governor_tunables(self.get_core_online_cpu(core))

    def set_core_governor_tunables(self, core, tunables):
        for cluster in self.get_core_clusters(core):
            governor = self.get_cluster_governor(cluster)
            self.set_cluster_governor_tunables(cluster, governor, **tunables)

    def get_core_min_frequency(self, core):
        return self.get_cpu_min_frequency(self.get_core_online_cpu(core))

    def set_core_min_frequency(self, core, freq):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_min_frequency(cluster, freq)

    def get_core_cur_frequency(self, core):
        return self.get_cpu_frequency(self.get_core_online_cpu(core))

    def set_core_cur_frequency(self, core, freq):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_cur_frequency(cluster, freq)

    def get_core_max_frequency(self, core):
        return self.get_cpu_max_frequency(self.get_core_online_cpu(core))

    def set_core_max_frequency(self, core, freq):
        for cluster in self.get_core_clusters(core):
            self.set_cluster_max_frequency(cluster, freq)
