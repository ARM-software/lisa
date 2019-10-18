#    Copyright 2013-2018 ARM Limited
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


# pylint: disable=W0613,no-member,attribute-defined-outside-init
"""

Some "standard" instruments to collect additional info about workload execution.

.. note:: The run() method of a Workload may perform some "boilerplate" as well as
          the actual execution of the workload (e.g. it may contain UI automation
          needed to start the workload). This "boilerplate" execution will also
          be measured by these instruments. As such, they are not suitable for collected
          precise data about specific operations.
"""
import os
import logging
import time
import tarfile
from subprocess import CalledProcessError

from devlib.exception import TargetError
from devlib.utils.android import ApkInfo

from wa import Instrument, Parameter, very_fast
from wa.framework.exception import ConfigError
from wa.framework.instrument import slow
from wa.utils.diff import diff_sysfs_dirs, diff_interrupt_files
from wa.utils.misc import as_relative
from wa.utils.misc import ensure_file_directory_exists as _f
from wa.utils.misc import ensure_directory_exists as _d
from wa.utils.types import list_of_strings


logger = logging.getLogger(__name__)


class SysfsExtractor(Instrument):

    name = 'sysfs_extractor'
    description = """
    Collects the contest of a set of directories, before and after workload execution
    and diffs the result.

    """

    mount_command = 'mount -t tmpfs -o size={} tmpfs {}'
    extract_timeout = 30
    tarname = 'sysfs.tar.gz'
    DEVICE_PATH = 0
    BEFORE_PATH = 1
    AFTER_PATH = 2
    DIFF_PATH = 3

    parameters = [
        Parameter('paths', kind=list_of_strings, mandatory=True,
                  description="""A list of paths to be pulled from the device. These could be directories
                                as well as files.""",
                  global_alias='sysfs_extract_dirs'),
        Parameter('use_tmpfs', kind=bool, default=None,
                  description="""
                  Specifies whether tmpfs should be used to cache sysfile trees and then pull them down
                  as a tarball. This is significantly faster then just copying the directory trees from
                  the device directly, but requires root and may not work on all devices. Defaults to
                  ``True`` if the device is rooted and ``False`` if it is not.
                  """),
        Parameter('tmpfs_mount_point', default=None,
                  description="""Mount point for tmpfs partition used to store snapshots of paths."""),
        Parameter('tmpfs_size', default='32m',
                  description="""Size of the tempfs partition."""),
    ]

    def initialize(self, context):
        if not self.target.is_rooted and self.use_tmpfs:  # pylint: disable=access-member-before-definition
            raise ConfigError('use_tempfs must be False for an unrooted device.')
        elif self.use_tmpfs is None:  # pylint: disable=access-member-before-definition
            self.use_tmpfs = self.target.is_rooted

        if self.use_tmpfs:
            self.on_device_before = self.target.path.join(self.tmpfs_mount_point, 'before')
            self.on_device_after = self.target.path.join(self.tmpfs_mount_point, 'after')

            if not self.target.file_exists(self.tmpfs_mount_point):
                self.target.execute('mkdir -p {}'.format(self.tmpfs_mount_point), as_root=True)
                self.target.execute(self.mount_command.format(self.tmpfs_size, self.tmpfs_mount_point),
                                    as_root=True)

    def setup(self, context):
        before_dirs = [
            _d(os.path.join(context.output_directory, 'before', self._local_dir(d)))
            for d in self.paths
        ]
        after_dirs = [
            _d(os.path.join(context.output_directory, 'after', self._local_dir(d)))
            for d in self.paths
        ]
        diff_dirs = [
            _d(os.path.join(context.output_directory, 'diff', self._local_dir(d)))
            for d in self.paths
        ]
        self.device_and_host_paths = list(zip(self.paths, before_dirs, after_dirs, diff_dirs))

        if self.use_tmpfs:
            for d in self.paths:
                before_dir = self.target.path.join(self.on_device_before,
                                                   self.target.path.dirname(as_relative(d)))
                after_dir = self.target.path.join(self.on_device_after,
                                                  self.target.path.dirname(as_relative(d)))
                if self.target.file_exists(before_dir):
                    self.target.execute('rm -rf  {}'.format(before_dir), as_root=True)
                self.target.execute('mkdir -p {}'.format(before_dir), as_root=True)
                if self.target.file_exists(after_dir):
                    self.target.execute('rm -rf  {}'.format(after_dir), as_root=True)
                self.target.execute('mkdir -p {}'.format(after_dir), as_root=True)

    @slow
    def start(self, context):
        if self.use_tmpfs:
            for d in self.paths:
                dest_dir = self.target.path.join(self.on_device_before, as_relative(d))
                if '*' in dest_dir:
                    dest_dir = self.target.path.dirname(dest_dir)
                self.target.execute('{} cp -Hr {} {}'.format(self.target.busybox, d, dest_dir),
                                    as_root=True, check_exit_code=False)
        else:  # not rooted
            for dev_dir, before_dir, _, _ in self.device_and_host_paths:
                self.target.pull(dev_dir, before_dir)

    @slow
    def stop(self, context):
        if self.use_tmpfs:
            for d in self.paths:
                dest_dir = self.target.path.join(self.on_device_after, as_relative(d))
                if '*' in dest_dir:
                    dest_dir = self.target.path.dirname(dest_dir)
                self.target.execute('{} cp -Hr {} {}'.format(self.target.busybox, d, dest_dir),
                                    as_root=True, check_exit_code=False)
        else:  # not using tmpfs
            for dev_dir, _, after_dir, _ in self.device_and_host_paths:
                self.target.pull(dev_dir, after_dir)

    def update_output(self, context):
        if self.use_tmpfs:
            on_device_tarball = self.target.path.join(self.target.working_directory, self.tarname)
            on_host_tarball = self.target.path.join(context.output_directory, self.tarname)
            self.target.execute('{} tar czf {} -C {} .'.format(self.target.busybox,
                                                               on_device_tarball,
                                                               self.tmpfs_mount_point),
                                as_root=True)
            self.target.execute('chmod 0777 {}'.format(on_device_tarball), as_root=True)
            self.target.pull(on_device_tarball, on_host_tarball)
            with tarfile.open(on_host_tarball, 'r:gz') as tf:
                tf.extractall(context.output_directory)
            self.target.remove(on_device_tarball)
            os.remove(on_host_tarball)

        for paths in self.device_and_host_paths:
            after_dir = paths[self.AFTER_PATH]
            dev_dir = paths[self.DEVICE_PATH].strip('*')  # remove potential trailing '*'
            if (not os.listdir(after_dir) and
                    self.target.file_exists(dev_dir) and
                    self.target.list_directory(dev_dir)):
                self.logger.error('sysfs files were not pulled from the device.')
                self.device_and_host_paths.remove(paths)  # Path is removed to skip diffing it
        for dev_dir, before_dir, after_dir, diff_dir in self.device_and_host_paths:
            diff_sysfs_dirs(before_dir, after_dir, diff_dir)
            context.add_artifact('{} [before]'.format(dev_dir), before_dir,
                                 kind='data', classifiers={'stage': 'before'})
            context.add_artifact('{} [after]'.format(dev_dir), after_dir,
                                 kind='data', classifiers={'stage': 'after'})
            context.add_artifact('{} [diff]'.format(dev_dir), diff_dir,
                                 kind='data', classifiers={'stage': 'diff'})

    def teardown(self, context):
        self._one_time_setup_done = []

    def finalize(self, context):
        if self.use_tmpfs:
            try:
                self.target.execute('umount {}'.format(self.tmpfs_mount_point), as_root=True)
            except (TargetError, CalledProcessError):
                # assume a directory but not mount point
                pass
            self.target.execute('rm -rf {}'.format(self.tmpfs_mount_point),
                                as_root=True, check_exit_code=False)

    def validate(self):
        if not self.tmpfs_mount_point:  # pylint: disable=access-member-before-definition
            self.tmpfs_mount_point = self.target.get_workpath('temp-fs')

    def _local_dir(self, directory):
        return os.path.dirname(as_relative(directory).replace(self.target.path.sep, os.sep))


class ExecutionTimeInstrument(Instrument):

    name = 'execution_time'
    description = """
    Measure how long it took to execute the run() methods of a Workload.

    """

    def __init__(self, target, **kwargs):
        super(ExecutionTimeInstrument, self).__init__(target, **kwargs)
        self.start_time = None
        self.end_time = None

    @very_fast
    def start(self, context):
        self.start_time = time.time()

    @very_fast
    def stop(self, context):
        self.end_time = time.time()

    def update_output(self, context):
        execution_time = self.end_time - self.start_time
        context.add_metric('execution_time', execution_time, 'seconds')


class ApkVersion(Instrument):

    name = 'apk_version'
    description = """
    Extracts APK versions for workloads that have them.

    """

    def __init__(self, device, **kwargs):
        super(ApkVersion, self).__init__(device, **kwargs)
        self.apk_info = None

    def setup(self, context):
        if hasattr(context.workload, 'apk_file'):
            self.apk_info = ApkInfo(context.workload.apk_file)
        else:
            self.apk_info = None

    def update_output(self, context):
        if self.apk_info:
            context.result.add_metric(self.name, self.apk_info.version_name)


class InterruptStatsInstrument(Instrument):

    name = 'interrupts'
    description = """
    Pulls the ``/proc/interrupts`` file before and after workload execution and diffs them
    to show what interrupts  occurred during that time.

    """

    def __init__(self, target, **kwargs):
        super(InterruptStatsInstrument, self).__init__(target, **kwargs)
        self.before_file = None
        self.after_file = None
        self.diff_file = None

    def setup(self, context):
        self.before_file = os.path.join(context.output_directory, 'before', 'proc', 'interrupts')
        self.after_file = os.path.join(context.output_directory, 'after', 'proc', 'interrupts')
        self.diff_file = os.path.join(context.output_directory, 'diff', 'proc', 'interrupts')

    def start(self, context):
        with open(_f(self.before_file), 'w') as wfh:
            wfh.write(self.target.execute('cat /proc/interrupts'))

    def stop(self, context):
        with open(_f(self.after_file), 'w') as wfh:
            wfh.write(self.target.execute('cat /proc/interrupts'))

    def update_output(self, context):
        context.add_artifact('interrupts [before]', self.before_file, kind='data',
                             classifiers={'stage': 'before'})
        # If workload execution failed, the after_file may not have been created.
        if os.path.isfile(self.after_file):
            diff_interrupt_files(self.before_file, self.after_file, _f(self.diff_file))
            context.add_artifact('interrupts [after]', self.after_file, kind='data',
                                 classifiers={'stage': 'after'})
            context.add_artifact('interrupts [diff]', self.diff_file, kind='data',
                                 classifiers={'stage': 'diff'})


class DynamicFrequencyInstrument(SysfsExtractor):

    name = 'cpufreq'
    description = """
    Collects dynamic frequency (DVFS) settings before and after workload execution.

    """

    tarname = 'cpufreq.tar.gz'

    parameters = [
        Parameter('paths', mandatory=False, override=True),
    ]

    def setup(self, context):
        self.paths = ['/sys/devices/system/cpu']
        if self.use_tmpfs:
            self.paths.append('/sys/class/devfreq/*')  # the '*' would cause problems for adb pull.
        super(DynamicFrequencyInstrument, self).setup(context)

    def validate(self):
        super(DynamicFrequencyInstrument, self).validate()
        if not self.tmpfs_mount_point.endswith('-cpufreq'):  # pylint: disable=access-member-before-definition
            self.tmpfs_mount_point += '-cpufreq'
