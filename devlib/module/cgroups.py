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
from collections import namedtuple

from devlib.module import Module
from devlib.exception import TargetError
from devlib.utils.misc import list_to_ranges, isiterable
from devlib.utils.types import boolean


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

    def probe(self, target):
        raise NotImplementedError()

    def mount(self, device, mount_root):
        self.target = device
        self.mount_point = device.path.join(mount_root, self.mount_name)
        mounted = self.target.list_file_systems()
        if self.mount_point in [e.mount_point for e in mounted]:
            self.logger.debug('controller is already mounted.')
        else:
            self.target.execute('mkdir -p {} 2>/dev/null'.format(self.mount_point),
                                as_root=True)
            self.target.execute('mount -t cgroup -o {} {} {}'.format(self.kind,
                                                                     self.mount_name,
                                                                     self.mount_point),
                                as_root=True)


class CpusetGroup(object):

    def __init__(self, controller, name, cpus, mems):
        self.controller = controller
        self.target = controller.target
        self.name = name
        if name == 'root':
            self.directory = controller.mount_point
        else:
            self.directory = self.target.path.join(controller.mount_point, name)
            self.target.execute('mkdir -p {}'.format(self.directory), as_root=True)
        self.cpus_file = self.target.path.join(self.directory, 'cpuset.cpus')
        self.mems_file = self.target.path.join(self.directory, 'cpuset.mems')
        self.tasks_file = self.target.path.join(self.directory, 'tasks')
        self.set(cpus, mems)

    def set(self, cpus, mems):
        if isiterable(cpus):
            cpus = list_to_ranges(cpus)
        if isiterable(mems):
            mems = list_to_ranges(mems)
        self.target.write_value(self.cpus_file, cpus)
        self.target.write_value(self.mems_file, mems)

    def get(self):
        cpus = self.target.read_value(self.cpus_file)
        mems = self.target.read_value(self.mems_file)
        return (cpus, mems)

    def get_tasks(self):
        task_ids = self.target.read_value(self.tasks_file).split()
        return map(int, task_ids)

    def add_tasks(self, tasks):
        for tid in tasks:
            self.add_task(tid)

    def add_task(self, tid):
        self.target.write_value(self.tasks_file, tid, verify=False)


class CpusetController(CgroupController):

    name = 'cpuset'

    def __init__(self, *args, **kwargs):
        super(CpusetController, self).__init__(*args, **kwargs)
        self.groups = {}

    def probe(self, target):
        return target.config.is_enabled('cpusets')

    def mount(self, device, mount_root):
        super(CpusetController, self).mount(device, mount_root)
        self.create_group('root', self.target.list_online_cpus(), 0)

    def create_group(self, name, cpus, mems):
        if not hasattr(self, 'target'):
            raise RuntimeError('Attempting to create group for unmounted controller {}'.format(self.kind))
        if name in self.groups:
            raise ValueError('Group {} already exists'.format(name))
        self.groups[name] = CpusetGroup(self, name, cpus, mems)

    def move_tasks(self, source, dest):
        try:
            source_group = self.groups[source]
            dest_group = self.groups[dest]
            command = 'for task in $(cat {}); do echo $task>{}; done'
            self.target.execute(command.format(source_group.tasks_file, dest_group.tasks_file),
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



CgroupSubsystemEntry = namedtuple('CgroupSubsystemEntry', 'name hierarchy num_cgroups enabled')


class CgroupsModule(Module):

    name = 'cgroups'
    controller_cls = [
        CpusetController,
    ]

    cgroup_root = '/sys/fs/cgroup'

    @staticmethod
    def probe(target):
        return target.config.has('cgroups') and target.is_rooted



    def __init__(self, target):
        super(CgroupsModule, self).__init__(target)
        mounted = self.target.list_file_systems()
        if self.cgroup_root not in [e.mount_point for e in mounted]:
            self.target.execute('mount -t tmpfs {} {}'.format('cgroup_root', self.cgroup_root),
                                as_root=True)
        else:
            self.logger.debug('cgroup_root already mounted at {}'.format(self.cgroup_root))
        self.controllers = []
        for cls in self.controller_cls:
            controller = cls('devlib_{}'.format(cls.name))
            if controller.probe(self.target):
                if controller.mount_name in [e.device for e in mounted]:
                    self.logger.debug('controller {} is already mounted.'.format(controller.kind))
                else:
                    try:
                        controller.mount(self.target, self.cgroup_root)
                    except TargetError:
                        message = 'cgroups {} controller is not supported by the target'
                        raise TargetError(message.format(controller.kind))

    def list_subsystems(self):
        subsystems = []
        for line in self.target.execute('cat /proc/cgroups').split('\n')[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, hierarchy, num_cgroups, enabled = line.split()
            subsystems.append(CgroupSubsystemEntry(name,
                                                   int(hierarchy),
                                                   int(num_cgroups),
                                                   boolean(enabled)))
        return subsystems


    def get_cgroup_controller(self, kind):
        for controller in self.controllers:
            if controller.kind == kind:
                return controller
        raise ValueError(kind)

