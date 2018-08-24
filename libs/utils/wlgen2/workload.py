# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, Arm Limited and contributors.
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

import logging

from devlib.utils.misc import list_to_mask

class Workload(object):
    """
    """

    def __init__(self, te, res_dir):
        self.te = te
        self.res_dir = res_dir

        self.command = None
        self.run_dir = self.te.target.working_directory

        self._log = logging.getLogger('Workload')
        # self._log.info('Setup new workload %s', self.name)

    def run(self, cpus=None, cgroup=None, background=False, as_root=False):

        if not self.command:
            raise RuntimeError("Workload does not specify any command to execute")

        _command = self.command

        if cpus:
            taskset_bin = self.te.target.which('taskset')
            if not taskset_bin:
                raise RuntimeError("Could not find 'taskset' executable on the target")

            cpumask = list_to_mask(cpus)
            taskset_cmd = '{} 0x{}'.format(taskset_bin, cpumask)
            _command = '{} {}'.format(taskset_cmd, _command)

        if cgroup:
            _command = self.te.target.cgroups.run_into_cmd(cgroup, _command)

        if background:
            self.te.target.background(_command, as_root=as_root)
        else:
            self.te.target.execute(_command, as_root=as_root)
