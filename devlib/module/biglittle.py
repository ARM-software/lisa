from devlib.module import Module


class BigLittleModule(Module):

    name = 'bl'

    @staticmethod
    def probe(target):
        return target.big_core is not None

    @property
    def bigs(self):
        return [i for i, c in enumerate(self.target.platform.core_names)
                if c == self.target.platform.big_core]

    @property
    def littles(self):
        return [i for i, c in enumerate(self.target.platform.core_names)
                if c == self.target.platform.little_core]

    @property
    def bigs_online(self):
        return list(sorted(set(self.bigs).intersection(self.target.list_online_cpus())))

    @property
    def littles_online(self):
        return list(sorted(set(self.littles).intersection(self.target.list_online_cpus())))

    # hotplug

    def online_all_bigs(self):
        self.target.hotplug.online(*self.bigs)

    def offline_all_bigs(self):
        self.target.hotplug.offline(*self.bigs)

    def online_all_littles(self):
        self.target.hotplug.online(*self.littles)

    def offline_all_littles(self):
        self.target.hotplug.offline(*self.littles)

    # cpufreq

    def list_bigs_frequencies(self):
        return self.target.cpufreq.list_frequencies(self.bigs_online[0])

    def list_bigs_governors(self):
        return self.target.cpufreq.list_governors(self.bigs_online[0])

    def list_bigs_governor_tunables(self):
        return self.target.cpufreq.list_governor_tunables(self.bigs_online[0])

    def list_littles_frequencies(self):
        return self.target.cpufreq.list_frequencies(self.littles_online[0])

    def list_littles_governors(self):
        return self.target.cpufreq.list_governors(self.littles_online[0])

    def list_littles_governor_tunables(self):
        return self.target.cpufreq.list_governor_tunables(self.littles_online[0])

    def get_bigs_governor(self):
        return self.target.cpufreq.get_governor(self.bigs_online[0])

    def get_bigs_governor_tunables(self):
        return self.target.cpufreq.get_governor_tunables(self.bigs_online[0])

    def get_bigs_frequency(self):
        return self.target.cpufreq.get_frequency(self.bigs_online[0])

    def get_bigs_min_frequency(self):
        return self.target.cpufreq.get_min_frequency(self.bigs_online[0])

    def get_bigs_max_frequency(self):
        return self.target.cpufreq.get_max_frequency(self.bigs_online[0])

    def get_littles_governor(self):
        return self.target.cpufreq.get_governor(self.littles_online[0])

    def get_littles_governor_tunables(self):
        return self.target.cpufreq.get_governor_tunables(self.littles_online[0])

    def get_littles_frequency(self):
        return self.target.cpufreq.get_frequency(self.littles_online[0])

    def get_littles_min_frequency(self):
        return self.target.cpufreq.get_min_frequency(self.littles_online[0])

    def get_littles_max_frequency(self):
        return self.target.cpufreq.get_max_frequency(self.littles_online[0])

    def set_bigs_governor(self, governor, **kwargs):
        self.target.cpufreq.set_governor(self.bigs_online[0], governor, **kwargs)

    def set_bigs_governor_tunables(self, governor, **kwargs):
        self.target.cpufreq.set_governor_tunables(self.bigs_online[0], governor, **kwargs)

    def set_bigs_frequency(self, frequency, exact=True):
        self.target.cpufreq.set_frequency(self.bigs_online[0], frequency, exact)

    def set_bigs_min_frequency(self, frequency, exact=True):
        self.target.cpufreq.set_min_frequency(self.bigs_online[0], frequency, exact)

    def set_bigs_max_frequency(self, frequency, exact=True):
        self.target.cpufreq.set_max_frequency(self.bigs_online[0], frequency, exact)

    def set_littles_governor(self, governor, **kwargs):
        self.target.cpufreq.set_governor(self.littles_online[0], governor, **kwargs)

    def set_littles_governor_tunables(self, governor, **kwargs):
        self.target.cpufreq.set_governor_tunables(self.littles_online[0], governor, **kwargs)

    def set_littles_frequency(self, frequency, exact=True):
        self.target.cpufreq.set_frequency(self.littles_online[0], frequency, exact)

    def set_littles_min_frequency(self, frequency, exact=True):
        self.target.cpufreq.set_min_frequency(self.littles_online[0], frequency, exact)

    def set_littles_max_frequency(self, frequency, exact=True):
        self.target.cpufreq.set_max_frequency(self.littles_online[0], frequency, exact)
