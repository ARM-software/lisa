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


class Controller(object):

    def __new__(cls, arg):
        if isinstance(arg, cls):
            return arg
        else:
            return object.__new__(cls, arg)

    def __init__(self, kind):
        self.mount_name = 'devlib_'+kind
        self.kind = kind
        self.target = None

        self.logger = logging.getLogger('cgroups.'+self.kind)
        self.mount_point = None
        self._cgroups = {}

    def probe(self, target):
        try:
            exists = target.execute('{} grep {} /proc/cgroups'\
                    .format(target.busybox, self.kind))
        except TargetError:
            return False
        return True

    def mount(self, target, mount_root):

        mounted = target.list_file_systems()
        if self.mount_name in [e.device for e in mounted]:
            # Identify mount point if controller is already in use
            self.mount_point = [
                    fs.mount_point
                    for fs in mounted
                    if fs.device == self.mount_name
                ][0]
        else:
            # Mount the controller if not already in use
            self.mount_point = target.path.join(mount_root, self.mount_name)
            target.execute('mkdir -p {} 2>/dev/null'\
                    .format(self.mount_point), as_root=True)
            target.execute('mount -t cgroup -o {} {} {}'\
                    .format(self.kind,
                            self.mount_name,
                            self.mount_point),
                            as_root=True)

        self.logger.info('Controller %s mounted under: %s',
            self.kind, self.mount_point)

        # Mark this contoller as available
        self.target = target

        # Create root control group
        self.cgroup('/')

    def cgroup(self, name):
        if not self.target:
            raise RuntimeError('CGroup creation failed: {} controller not mounted'\
                    .format(self.kind))
        if name not in self._cgroups:
            self._cgroups[name] = CGroup(self, name)
        return self._cgroups[name]

    def exists(self, name):
        if not self.target:
            raise RuntimeError('CGroup creation failed: {} controller not mounted'\
                    .format(self.kind))
        if name not in self._cgroups:
            self._cgroups[name] = CGroup(self, name, create=False)
        return self._cgroups[name].existe()

    def list_all(self):
        self.logger.debug('Listing groups for %s controller', self.kind)
        output = self.target.execute('{} find {} -type d'\
                .format(self.target.busybox, self.mount_point))
        cgroups = []
        for cg in output.split('\n'):
            cg = cg.replace(self.mount_point + '/', '/')
            cg = cg.replace(self.mount_point, '/')
            cg = cg.strip()
            if cg == '':
                continue
            self.logger.debug('Populate %s cgroup: %s', self.kind, cg)
            cgroups.append(cg)
        return cgroups

    def move_tasks(self, source, dest):
        try:
            srcg = self._cgroups[source]
            dstg = self._cgroups[dest]
            command = 'for task in $(cat {}); do echo $task>{}; done'
            self.target.execute(command.format(srcg.tasks_file, dstg.tasks_file),
                                # this will always fail as some of the tasks
                                # are kthreads that cannot be migrated, but we
                                # don't care about those, so don't check exit
                                # code.
                                check_exit_code=False, as_root=True)
        except KeyError as e:
            raise ValueError('Unkown group: {}'.format(e))

    def move_all_tasks_to(self, dest):
        for cgroup in self._cgroups:
            if cgroup != dest:
                self.move_tasks(cgroup, dest)

class CGroup(object):

    def __init__(self, controller, name, create=True):
        self.logger = logging.getLogger('cgroups.' + controller.kind)
        self.target = controller.target
        self.controller = controller
        self.name = name

        # Control cgroup path
        self.directory = controller.mount_point
        if name != '/':
            self.directory = self.target.path.join(controller.mount_point, name[1:])

        # Setup path for tasks file
        self.tasks_file = self.target.path.join(self.directory, 'tasks')
        self.procs_file = self.target.path.join(self.directory, 'cgroup.procs')

        if not create:
            return

        self.logger.info('Creating cgroup %s', self.directory)
        self.target.execute('[ -d {0} ] || mkdir -p {0}'\
                .format(self.directory), as_root=True)

    def exists(self):
        try:
            self.target.execute('[ -d {0} ]'\
                .format(self.directory))
            return True
        except TargetError:
            return False

    def get(self):
        conf = {}

        logging.debug('Reading %s attributes from:',
                self.controller.kind)
        logging.debug('  %s',
                self.directory)
        output = self.target.execute('{} grep \'\' {}/{}.*'.format(
                    self.target.busybox,
                    self.directory,
                    self.controller.kind))
        for res in output.split('\n'):
            if res.find(self.controller.kind) < 0:
                continue
            res = res.split('.')[1]
            attr = res.split(':')[0]
            value = res.split(':')[1]
            conf[attr] = value

        return conf

    def set(self, **attrs):
        for idx in attrs:
            if isiterable(attrs[idx]):
                attrs[idx] = list_to_ranges(attrs[idx])
            # Build attribute path
            path = '{}.{}'.format(self.controller.kind, idx)
            path = self.target.path.join(self.directory, path)

            self.logger.debug('Set attribute [%s] to: %s"',
                    path, attrs[idx])

            # Set the attribute value
            self.target.write_value(path, attrs[idx])

    def get_tasks(self):
        task_ids = self.target.read_value(self.tasks_file).split()
        logging.debug('Tasks: %s', task_ids)
        return map(int, task_ids)

    def add_task(self, tid):
        self.target.write_value(self.tasks_file, tid, verify=False)

    def add_tasks(self, tasks):
        for tid in tasks:
            self.add_task(tid)

    def add_proc(self, pid):
        self.target.write_value(self.procs_file, pid, verify=False)

CgroupSubsystemEntry = namedtuple('CgroupSubsystemEntry', 'name hierarchy num_cgroups enabled')

class CgroupsModule(Module):

    name = 'cgroups'
    cgroup_root = '/sys/fs/cgroup'

    @staticmethod
    def probe(target):
        return target.config.has('cgroups') and target.is_rooted

    def __init__(self, target):
        super(CgroupsModule, self).__init__(target)

        self.logger = logging.getLogger('CGroups')

        # Initialize controllers mount point
        mounted = self.target.list_file_systems()
        if self.cgroup_root not in [e.mount_point for e in mounted]:
            self.target.execute('mount -t tmpfs {} {}'\
                    .format('cgroup_root',
                            self.cgroup_root),
                            as_root=True)
        else:
            self.logger.debug('cgroup_root already mounted at %s',
                    self.cgroup_root)

        # Load list of available controllers
        controllers = []
        subsys = self.list_subsystems()
        for (n, h, c, e) in subsys:
            controllers.append(n)
        self.logger.info('Available controllers: %s', controllers)

        # Initialize controllers
        self.controllers = {}
        for idx in controllers:
            controller = Controller(idx)
            self.logger.debug('Init %s controller...', controller.kind)
            if not controller.probe(self.target):
                continue
            try:
                controller.mount(self.target, self.cgroup_root)
            except TargetError:
                message = 'cgroups {} controller is not supported by the target'
                raise TargetError(message.format(controller.kind))
            self.logger.debug('Controller %s enabled', controller.kind)
            self.controllers[idx] = controller

    def list_subsystems(self):
        subsystems = []
        for line in self.target.execute('{} cat /proc/cgroups'\
                .format(self.target.busybox)).split('\n')[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, hierarchy, num_cgroups, enabled = line.split()
            subsystems.append(CgroupSubsystemEntry(name,
                                                   int(hierarchy),
                                                   int(num_cgroups),
                                                   boolean(enabled)))
        return subsystems


    def controller(self, kind):
        if kind not in self.controllers:
            self.logger.warning('Controller %s not available', kind)
            return None
        return self.controllers[kind]

