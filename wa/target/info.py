from devlib.exception import TargetError
from devlib.target import KernelConfig, KernelVersion, Cpuinfo


class TargetInfo(object):

    hmp_config_dir = '/sys/kernel/hmp'

    def __init__(self):
        self.os = None
        self.kernel_version = None
        self.kernel_cmdline = None
        self.kernel_config = {}
        self.sched_features = []
        self.cpuinfo = None
        self.os_version = {}
        self.properties = {}

    @staticmethod
    def from_pod(pod):
        kconfig_text = '\n'.join('{}={}'.format(k, v) for k, v in pod['kernel_config'].iteritems())
        sections = []
        for section in pod['cpuinfo']:
            text = '\n'.join('{} : {}'.format(k, v) for k, v in section.iteritems())
            sections.append(text)
        cpuinfo_text = '\n\n'.join(sections)

        instance = TargetInfo()
        instance.os = pod['os']
        instance.kernel_version = KernelVersion(pod['kernel_version'])
        instance.kernel_cmdline = pod['kernel_cmdline']
        instance.kernel_config = KernelConfig(kconfig_text)
        instance.sched_features = pod['sched_features']
        instance.cpuinfo = Cpuinfo(cpuinfo_text)
        instance.os_version = pod['os_version']
        instance.properties = pod['properties']
        return instance

    def to_pod(self):
        kversion = str(self.kernel_version)
        kconfig = {k: v for k, v in self.kernel_config.iteritems()}
        return dict(
            os=self.os,
            kernel_version=kversion,
            kernel_cmdline=self.kernel_cmdline,
            kernel_config=kconfig,
            sched_features=self.sched_features,
            cpuinfo=self.cpuinfo.sections,
            os_version=self.os_version,
            properties=self.properties,
        )

    def load(self, target):
        self.os = target.os
        print target.is_rooted
        self.os_version = target.os_version
        self.kernel_version = target.kernel_version
        self.kernel_cmdline = target.execute('cat /proc/cmdline',
                                             as_root=target.is_rooted).strip()
        self.kernel_config = target.config
        self.cpuinfo = target.cpuinfo
        try:
            output = target.read_value('/sys/kernel/debug/sched_features')
            self.sched_features = output.strip().split()
        except TargetError:
            pass
        self.properties = self._get_properties(target)

    def _get_properties(self, target):
        props = {}
        if target.file_exists(self.hmp_config_dir):
            props['hmp'] = self._get_hmp_configuration(target)
        if target.os == 'android':
            props.update(target.getprop().iteritems())
        return props

    def _get_hmp_configuration(self, target):
        hmp_props = {}
        for entry in target.list_directory(self.hmp_config_dir):
            path = target.path.join(self.hmp_config_dir, entry)
            try:
                hmp_props[entry] = target.read_value(path)
            except TargetError:
                pass
        return hmp_props
