# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from wlgen import RTA, Periodic

class SchedEntity(object):
    """
    A sched entity can be either a task or a taskgroup. If it is a task, then
    it has no children. A taskgroup, instead can have several children nodes
    being tasks or task groups.

    :param se_type: Sched entity type, either "task" or "tg" (taskgroup)
    :type se_type: str
    """

    def __init__(self, se_type):
        allowed_se_types = ['task', 'tg']
        if se_type not in allowed_se_types:
            raise ValueError('Invalid sched entity type. Allowed values are {}'
                             .format(allowed_se_types))

        self._se_type = se_type
        self.parent = None
        self.children = set()

    def add_children(self, ses):
        raise NotImplementedError("add_children() must be implemented")

    def change_taskgroup(self, new_taskgroup):
        raise NotImplementedError("change_taskgroup() must be implemented")

    def _iter(self):
        yield self
        for child in self.children:
            for child_i in child._iter():
                yield child_i

    def iter_nodes(self):
        """Pre-order traversal of all nodes"""
        return self._iter()

    def get_child(self, child_name):
        raise NotImplementedError("get_child() must be implemented")

    @property
    def is_task(self):
        return self._se_type == 'task'

    def get_expected_util(self):
        raise NotImplementedError("get_expected_util() must be implemented")

    def print_hieararchy(self, level=0):
        """In-order visualization of the tree"""
        if level > 0:
            print " " * (level) + "|--" + self.name
        else:
            print self.name

        for child in self.children:
            child.visit(level + 1)

class Task(SchedEntity):
    """
    Task Entity class

    :param name: Name of the task.
    :type name: str

    :param test_env: Test environment.
    :type test_env: env.TestEnv

    :param cpus: List of CPUs the workload can run on.
    :type cpus: list(int)

    :param period_ms: Period of each task in milliseconds.
    :type period_ms: int

    :param duty_cycle_pct: Dduty cycle of the periodic workload.
        Default 50%
    :type duty_cycle_pct: int

    :param duration_s: Total duration of the workload. Default 1 second.
    :type duration_s: int

    :param kind: Type of RTA workload. Can be 'profile' or 'custom'.
        Default value is 'profile'.
    :type kind: str

    :param num_tasks: Number of tasks to spawn.
    :type num_tasks: int
    """

    def __init__(self, name, test_env, cpus, period_ms=100, duty_cycle_pct=50,
                 duration_s=1, kind='profile', num_tasks=1):
        super(Task, self).__init__("task")

        self.name = name
        self.period_ms = period_ms
        self.duty_cycle_pct = duty_cycle_pct
        self.duration_s = duration_s
        self.cpus = cpus
        allowed_kinds = ['profile', 'custom']
        if kind not in allowed_kinds:
            raise ValueError('{} not allowed, kind can be one of {}'
                             .format(kind, allowed_kinds))
        self.kind = kind
        self.num_tasks = num_tasks

        # Create rt-app workload
        t = Periodic(period_ms=period_ms,
                     duty_cycle_pct=duty_cycle_pct,
                     duration_s=duration_s).get()
        self.wload = RTA(test_env.target, name, test_env.calibration())
        if num_tasks > 1:
            conf_params = {name + "_{}".format(i): t for i in xrange(num_tasks)}
        else:
            conf_params = {name: t}
        self.wload.conf(kind=kind,
                        params=conf_params,
                        run_dir=test_env.target.working_directory)

    def __repr__(self):
        return "Task: " + self.name

    def add_children(self, ses):
        raise TypeError('Cannot add children entities to a task entity.')

    def get_child(self, child_name):
        raise TypeError('Cannot find child entity from a task entity')

    def change_taskgroup(self, new_taskgroup):
        """
        Change the taskgroup to which the task is assigned
        :param new_taskgroup: New taskgroup assigned to the task
        :type new_taskgroup: Taskgroup
        """
        # Remove the task from its old taskgroup
        if self.parent:
            self.parent.children.remove(self)
        # Add task in the new taskgroup
        new_taskgroup.add_children([self])

    def get_expected_util(self):
        """
        Get expected utilization value. For tasks this corresponds to
        corresponds to the duty cycle of the task.

        :returns: int - expected utilization of the task
        """
        return 1024 * (self.duty_cycle_pct  * self.num_tasks / 100.0)

class Taskgroup(SchedEntity):
    """
    Taskgroup class

    Create the taskgroup object and instantiate the relative Cgroup on the
    target platform. Notice that in order to set attributes of a Cgroup,
    the attributes of its parents have to be set in the first place.
    Therefore, those should be instantieted first.

    :param name: Absolute path to the taskgroup starting from root '/'
    :type name: str

    :param cpus: List of CPUs associated to this taskgroup
    :type cpus: list(int)

    :param mems: Set cgroup mems attribute to the specified value
    :type mems: int

    :param test_env: Test environment
    :type test_env: env.TestEnv
    """

    def __init__(self, name, cpus, mems, test_env):
        super(Taskgroup, self).__init__("tg")

        self.name = name
        self.cpus = cpus
        self.mems = mems

        # Create Cgroup
        cnt = test_env.target.cgroups.controller('cpuset')
        cgp = cnt.cgroup(name)
        cgp.set(cpus=cpus, mems=mems)

        cnt = test_env.target.cgroups.controller('cpu')
        cgp = cnt.cgroup(name)

    def __repr__(self):
        return "Taskgroup: " + self.name

    def add_children(self, ses):
        """
        Add the specified sched entities as children of the current one.

        :param ses: sched entity to be added, either Task or Taskgroup
        :type ses: list(SchedEntity)
        """
        self.children.update(ses)
        for entity in ses:
            entity.parent = self

    def get_child(self, child_name):
        """
        Get the SchedEntity object associated with a name from the children
        list of the Taskgroup
        :params child_name: the name of the child that needs to be retrieved
        :type child_name: str

        :returns: a Task or a Taskgroup associated with the given name or raise
                  an error if the name does not exist in the children list
        """
        for se in self.iter_nodes():
            if se.name == child_name:
                return se
        raise ValueError("{} not in group {}".format(child_name, self.name))

    def change_taskgroup(self, new_taskgroup):
        raise TypeError('Cannot change group assignment of a Taskgroup Entity.')

    def get_expected_util(self):
        """
        Get expected utilization value. If the sched entity is a task, then
        this corresponds to corresponds to the duty cycle of the task. In case
        of task group, this is the sum of the utilizations of its children.
        """
        util = 0.0
        for child in self.children:
            util += child.get_expected_util()
        return util

