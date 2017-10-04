from devlib.module import Module


class HotplugModule(Module):

    name = 'hotplug'
    base_path = '/sys/devices/system/cpu'

    @classmethod
    def probe(cls, target):  # pylint: disable=arguments-differ
        # If a system has just 1 CPU, it makes not sense to hotplug it.
        # If a system has more than 1 CPU, CPU0 could be configured to be not
        # hotpluggable. Thus, check for hotplug support by looking at CPU1
        path = cls._cpu_path(target, 1)
        return target.file_exists(path) and target.is_rooted

    @classmethod
    def _cpu_path(cls, target, cpu):
        if isinstance(cpu, int):
            cpu = 'cpu{}'.format(cpu)
        return target.path.join(cls.base_path, cpu, 'online')

    def online_all(self):
        self.target._execute_util('hotplug_online_all',
                                  as_root=self.target.is_rooted)

    def online(self, *args):
        for cpu in args:
            self.hotplug(cpu, online=True)

    def offline(self, *args):
        for cpu in args:
            self.hotplug(cpu, online=False)

    def hotplug(self, cpu, online):
        path = self._cpu_path(self.target, cpu)
        if not self.target.file_exists(path):
            return
        value = 1 if online else 0
        self.target.write_value(path, value)

