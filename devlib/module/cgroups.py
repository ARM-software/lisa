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

    def __init__(self, kind, hid, clist):
        """
        Initialize a controller given the hierarchy it belongs to.

        :param kind: the name of the controller
        :type kind: str

        :param hid: the Hierarchy ID this controller is mounted on
        :type hid: int

        :param clist: the list of controller mounted in the same hierarchy
        :type clist: list(str)
        """
        self.mount_name = 'devlib_cgh{}'.format(hid)
        self.kind = kind
        self.hid = hid
        self.clist = clist
        self.target = None
        self._noprefix = False

        self.logger = logging.getLogger('CGroup.'+self.kind)
        self.logger.debug('Initialized [%s, %d, %s]',
                          self.kind, self.hid, self.clist)

        self.mount_point = None
        self._cgroups = {}

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
                    .format(','.join(self.clist),
                            self.mount_name,
                            self.mount_point),
                            as_root=True)

        # Check if this controller uses "noprefix" option
        output = target.execute('mount | grep "{} "'.format(self.mount_name))
        if 'noprefix' in output:
            self._noprefix = True
            # self.logger.debug('Controller %s using "noprefix" option',
            #                   self.kind)

        self.logger.debug('Controller %s mounted under: %s (noprefix=%s)',
            self.kind, self.mount_point, self._noprefix)

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
                .format(self.target.busybox, self.mount_point),
                as_root=True)
        cgroups = []
        for cg in output.splitlines():
            cg = cg.replace(self.mount_point + '/', '/')
            cg = cg.replace(self.mount_point, '/')
            cg = cg.strip()
            if cg == '':
                continue
            self.logger.debug('Populate %s cgroup: %s', self.kind, cg)
            cgroups.append(cg)
        return cgroups

    def move_tasks(self, source, dest, exclude=[]):
        try:
            srcg = self._cgroups[source]
            dstg = self._cgroups[dest]
        except KeyError as e:
            raise ValueError('Unkown group: {}'.format(e))
        output = self.target._execute_util(
                    'cgroups_tasks_move {} {} \'{}\''.format(
                    srcg.directory, dstg.directory, exclude),
                    as_root=True)

    def move_all_tasks_to(self, dest, exclude=[]):
        """
        Move all the tasks to the specified CGroup

        Tasks are moved from all their original CGroup the the specified on.
        The tasks which name matches one of the string in exclude are moved
        instead in the root CGroup for the controller.
        The name of a tasks to exclude must be a substring of the task named as
        reported by the "ps" command. Indeed, this list will be translated into
        a: "ps | grep -e name1 -e name2..." in order to obtain the PID of these
        tasks.

        :param exclude: list of commands to keep in the root CGroup
        :type exlude: list(str)
        """

        if isinstance(exclude, str):
            exclude = [exclude]
        if not isinstance(exclude, list):
            raise ValueError('wrong type for "exclude" parameter, '
                             'it must be a str or a list')

        logging.debug('Moving all tasks into %s', dest)

        # Build list of tasks to exclude
        grep_filters = ''
        for comm in exclude:
            grep_filters += '-e {} '.format(comm)
        logging.debug('   using grep filter: %s', grep_filters)
        if grep_filters != '':
            logging.debug('   excluding tasks which name matches:')
            logging.debug('   %s', ', '.join(exclude))

        for cgroup in self._cgroups:
            if cgroup != dest:
                self.move_tasks(cgroup, dest, grep_filters)

    def tasks(self, cgroup):
        try:
            cg = self._cgroups[cgroup]
        except KeyError as e:
            raise ValueError('Unkown group: {}'.format(e))
        output = self.target._execute_util(
                    'cgroups_tasks_in {}'.format(cg.directory),
                    as_root=True)
        entries = output.splitlines()
        tasks = {}
        for task in entries:
            tid = task.split(',')[0]
            try:
                tname = task.split(',')[1]
            except: continue
            try:
                tcmdline = task.split(',')[2]
            except:
                tcmdline = ''
            tasks[int(tid)] = (tname, tcmdline)
        return tasks

    def tasks_count(self, cgroup):
        try:
            cg = self._cgroups[cgroup]
        except KeyError as e:
            raise ValueError('Unkown group: {}'.format(e))
        output = self.target.execute(
                    '{} wc -l {}/tasks'.format(
                    self.target.busybox, cg.directory),
                    as_root=True)
        return int(output.split()[0])

    def tasks_per_group(self):
        tasks = {}
        for cg in self.list_all():
            tasks[cg] = self.tasks_count(cg)
        return tasks

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

        self.logger.debug('Creating cgroup %s', self.directory)
        self.target.execute('[ -d {0} ] || mkdir -p {0}'\
                .format(self.directory), as_root=True)

    def exists(self):
        try:
            self.target.execute('[ -d {0} ]'\
                .format(self.directory), as_root=True)
            return True
        except TargetError:
            return False

    def get(self):
        conf = {}

        logging.debug('Reading %s attributes from:',
                self.controller.kind)
        logging.debug('  %s',
                self.directory)
        output = self.target._execute_util(
                    'cgroups_get_attributes {} {}'.format(
                    self.directory, self.controller.kind),
                    as_root=True)
        for res in output.splitlines():
            attr = res.split(':')[0]
            value = res.split(':')[1]
            conf[attr] = value

        return conf

    def set(self, **attrs):
        for idx in attrs:
            if isiterable(attrs[idx]):
                attrs[idx] = list_to_ranges(attrs[idx])
            # Build attribute path
            if self.controller._noprefix:
                attr_name = '{}'.format(idx)
            else:
                attr_name = '{}.{}'.format(self.controller.kind, idx)
            path = self.target.path.join(self.directory, attr_name)

            self.logger.debug('Set attribute [%s] to: %s"',
                    path, attrs[idx])

            # Set the attribute value
            try:
                self.target.write_value(path, attrs[idx])
            except TargetError:
                # Check if the error is due to a non-existing attribute
                attrs = self.get()
                if idx not in attrs:
                    raise ValueError('Controller [{}] does not provide attribute [{}]'\
                                     .format(self.controller.kind, attr_name))
                raise

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
    stage = 'setup'

    @staticmethod
    def probe(target):
        if not target.is_rooted:
            return False
        if target.file_exists('/proc/cgroups'):
            return True
        return target.config.has('cgroups')

    def __init__(self, target):
        super(CgroupsModule, self).__init__(target)

        self.logger = logging.getLogger('CGroups')

        # Set Devlib's CGroups mount point
        self.cgroup_root = target.path.join(
            target.working_directory, 'cgroups')

        # Get the list of the available controllers
        subsys = self.list_subsystems()
        if len(subsys) == 0:
            self.logger.warning('No CGroups controller available')
            return

        # Map hierarchy IDs into a list of controllers
        hierarchy = {}
        for ss in subsys:
            try:
                hierarchy[ss.hierarchy].append(ss.name)
            except KeyError:
                hierarchy[ss.hierarchy] = [ss.name]
        self.logger.debug('Available hierarchies: %s', hierarchy)

        # Initialize controllers
        self.logger.info('Available controllers:')
        self.controllers = {}
        for ss in subsys:
            hid = ss.hierarchy
            controller = Controller(ss.name, hid, hierarchy[hid])
            try:
                controller.mount(self.target, self.cgroup_root)
            except TargetError:
                message = 'Failed to mount "{}" controller'
                raise TargetError(message.format(controller.kind))
            self.logger.info('  %-12s : %s', controller.kind,
                             controller.mount_point)
            self.controllers[ss.name] = controller

    def list_subsystems(self):
        subsystems = []
        for line in self.target.execute('{} cat /proc/cgroups'\
                .format(self.target.busybox)).splitlines()[1:]:
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

    def run_into_cmd(self, cgroup, cmdline):
        return 'CGMOUNT={} {} cgroups_run_into {} {}'\
                .format(self.cgroup_root, self.target.shutils,
                        cgroup, cmdline)

    def run_into(self, cgroup, cmdline):
        """
        Run the specified command into the specified CGroup
        """
        return self.target._execute_util(
            'CGMOUNT={} cgroups_run_into {} {}'\
                .format(self.cgroup_root, cgroup, cmdline),
            as_root=True)


    def cgroups_tasks_move(self, srcg, dstg, exclude=''):
        """
        Move all the tasks from the srcg CGroup to the dstg one.
        A regexps of tasks names can be used to defined tasks which should not
        be moved.
        """
        return self.target._execute_util(
            'cgroups_tasks_move {} {} {}'.format(srcg, dstg, exclude),
            as_root=True)

    def isolate(self, cpus, exclude=[]):
        """
        Remove all userspace tasks from specified CPUs.

        A list of CPUs can be specified where we do not want userspace tasks
        running. This functions creates a sandbox cpuset CGroup where all
        user-space tasks and not-pinned kernel-space tasks are moved into.
        This should allows to isolate the specified CPUs which will not get
        tasks running unless explicitely moved into the isolated group.

        :param cpus: the list of CPUs to isolate
        :type cpus: list(int)

        :return: the (sandbox, isolated) tuple, where:
                 sandbox is the CGroup of sandboxed CPUs
                 isolated is the CGroup of isolated CPUs
        """
        all_cpus = set(range(self.target.number_of_cpus))
        sbox_cpus = list(all_cpus - set(cpus))
        isol_cpus = list(all_cpus - set(sbox_cpus))

        # Create Sandbox and Isolated cpuset CGroups
        cpuset = self.controller('cpuset')
        sbox_cg = cpuset.cgroup('/DEVLIB_SBOX')
        isol_cg = cpuset.cgroup('/DEVLIB_ISOL')

        # Set CPUs for Sandbox and Isolated CGroups
        sbox_cg.set(cpus=sbox_cpus, mems=0)
        isol_cg.set(cpus=isol_cpus, mems=0)

        # Move all currently running tasks to the Sandbox CGroup
        cpuset.move_all_tasks_to('/DEVLIB_SBOX', exclude)

        return sbox_cg, isol_cg

    def freeze(self, exclude=[], thaw=False):
        """
        Freeze all user-space tasks but the specified ones

        A freezer cgroup is used to stop all the tasks in the target system but
        the ones which name match one of the path specified by the exclude
        paramater. The name of a tasks to exclude must be a substring of the
        task named as reported by the "ps" command. Indeed, this list will be
        translated into a: "ps | grep -e name1 -e name2..." in order to obtain
        the PID of these tasks.

        :param exclude: list of commands paths to exclude from freezer
        :type exclude: list(str)

        :param thaw: if true thaw tasks instead
        :type thaw: bool
        """

        # Create Freezer CGroup
        freezer = self.controller('freezer')
        if freezer is None:
            raise RuntimeError('freezer cgroup controller not present')
        freezer_cg = freezer.cgroup('/DEVLIB_FREEZER')
        thawed_cg = freezer.cgroup('/')

        if thaw:
            # Restart froozen tasks
            freezer_cg.set(state='THAWED')
            # Remove all tasks from freezer
            freezer.move_all_tasks_to('/')
            return

        # Move all tasks into the freezer group
        freezer.move_all_tasks_to('/DEVLIB_FREEZER', exclude)

        # Get list of not frozen tasks, which is reported as output
        tasks = freezer.tasks('/')

        # Freeze all tasks
        freezer_cg.set(state='FROZEN')

        return tasks

