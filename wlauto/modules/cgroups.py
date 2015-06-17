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
# pylint: disable=attribute-defined-outside-init
import logging

import wlauto.core.signal as signal
from wlauto import Module, Parameter
from wlauto.utils.misc import list_to_ranges, isiterable


class CgroupController(object):

    kind = 'cpuset'

    def __new__(cls, arg):
        if isinstance(arg, cls):
            return arg
        else:
            return object.__new__(cls, arg)

    def __init__(self, mount_name):
        self.mount_point = None
        self.mount_name = mount_name
        self.logger = logging.getLogger(self.kind)

    def mount(self, device, mount_root):
        self.device = device
        self.mount_point = device.path.join(mount_root, self.mount_name)
        mounted = self.device.list_file_systems()
        if self.mount_point in [e.mount_point for e in mounted]:
            self.logger.debug('controller is already mounted.')
        else:
            self.device.execute('mkdir -p {} 2>/dev/null'.format(self.mount_point),
                                as_root=True)
            self.device.execute('mount -t cgroup -o {} {} {}'.format(self.kind,
                                                                     self.mount_name,
                                                                     self.mount_point),
                                as_root=True)


class CpusetGroup(object):

    def __init__(self, controller, name, cpus, mems):
        self.controller = controller
        self.device = controller.device
        self.name = name
        if name == 'root':
            self.directory = controller.mount_point
        else:
            self.directory = self.device.path.join(controller.mount_point, name)
            self.device.execute('mkdir -p {}'.format(self.directory), as_root=True)
        self.cpus_file = self.device.path.join(self.directory, 'cpuset.cpus')
        self.mems_file = self.device.path.join(self.directory, 'cpuset.mems')
        self.tasks_file = self.device.path.join(self.directory, 'tasks')
        self.set(cpus, mems)

    def set(self, cpus, mems):
        if isiterable(cpus):
            cpus = list_to_ranges(cpus)
        if isiterable(mems):
            mems = list_to_ranges(mems)
        self.device.set_sysfile_value(self.cpus_file, cpus)
        self.device.set_sysfile_value(self.mems_file, mems)

    def get(self):
        cpus = self.device.get_sysfile_value(self.cpus_file)
        mems = self.device.get_sysfile_value(self.mems_file)
        return (cpus, mems)

    def get_tasks(self):
        task_ids = self.device.get_sysfile_value(self.tasks_file).split()
        return map(int, task_ids)

    def add_tasks(self, tasks):
        for tid in tasks:
            self.add_task(tid)

    def add_task(self, tid):
        self.device.set_sysfile_value(self.tasks_file, tid, verify=False)


class CpusetController(CgroupController):

    def __init__(self, *args, **kwargs):
        super(CpusetController, self).__init__(*args, **kwargs)
        self.groups = {}

    def mount(self, device, mount_root):
        super(CpusetController, self).mount(device, mount_root)
        self.create_group('root', self.device.online_cpus, 0)

    def create_group(self, name, cpus, mems):
        if not hasattr(self, 'device'):
            raise RuntimeError('Attempting to create group for unmounted controller {}'.format(self.kind))
        if name in self.groups:
            raise ValueError('Group {} already exists'.format(name))
        self.groups[name] = CpusetGroup(self, name, cpus, mems)

    def move_tasks(self, source, dest):
        try:
            source_group = self.groups[source]
            dest_group = self.groups[dest]
            command = 'for task in $(cat {}); do echo $task>{}; done'
            self.device.execute(command.format(source_group.tasks_file, dest_group.tasks_file),
                                # this will always fail as some of the tasks
                                # are kthreads that cannot be migrated, but we
                                # don't care about those, so don't check exit
                                # code.
                                check_exit_code=False, as_root=True)
        except KeyError as e:
            raise ValueError('Unkown group: {}'.format(e))

    def move_all_tasks_to(self, target_group):
        for group in self.groups:
            if group != target_group:
                self.move_tasks(group, target_group)

    def __getattr__(self, name):
        try:
            return self.groups[name]
        except KeyError:
            raise AttributeError(name)


class Cgroups(Module):

    name = 'cgroups'
    description = """
    Adds cgroups query and manupution APIs to a Device interface.

    Currently, only cpusets controller is supported.

    """
    capabilities = ['cgroups']

    controllers = [
        CpusetController('wa_cpuset'),
    ]

    parameters = [
        Parameter('cgroup_root', default='/sys/fs/cgroup',
                  description='Location where cgroups are mounted on the device.'),
    ]

    def initialize(self, context):
        self.device = self.root_owner
        signal.connect(self._on_device_init, signal.RUN_INIT, priority=1)

    def get_cgroup_controller(self, kind):
        for controller in self.controllers:
            if controller.kind == kind:
                return controller
        raise ValueError(kind)

    def _on_device_init(self, context):  # pylint: disable=unused-argument
        mounted = self.device.list_file_systems()
        if self.cgroup_root not in [e.mount_point for e in mounted]:
            self.device.execute('mount -t tmpfs {} {}'.format('cgroup_root', self.cgroup_root),
                                as_root=True)
        else:
            self.logger.debug('cgroup_root already mounted at {}'.format(self.cgroup_root))
        for controller in self.controllers:
            if controller.kind in [e.device for e in mounted]:
                self.logger.debug('controller {} is already mounted.'.format(controller.kind))
            else:
                controller.mount(self.device, self.cgroup_root)
