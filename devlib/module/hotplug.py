from devlib.module import Module


class HotplugModule(Module):

    name = 'hotplug'
    base_path = '/sys/devices/system/cpu'

    @classmethod
    def probe(cls, target):  # pylint: disable=arguments-differ
        path = cls._cpu_path(target, 0)
        return target.file_exists(path) and target.is_rooted

    @classmethod
    def _cpu_path(cls, target, cpu):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        return target.path.join(cls.base_path, cpu, 'online')

    def online_all(self):
        self.online(*range(self.target.number_of_cpus))

    def online(self, *args):
        for cpu in args:
            self.hotplug(cpu, online=True)

    def offline(self, *args):
        for cpu in args:
            self.hotplug(cpu, online=False)

    def hotplug(self, cpu, online):
        path = self._cpu_path(self.target, cpu)
        value = 1 if online else 0
        self.target.write_value(path, value)

