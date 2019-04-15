# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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
import os

from devlib.utils.misc import list_to_mask

from lisa.utils import Loggable

class Workload(Loggable):
    """
    This is pretty much a wrapper around a command to execute on a target.

    :param target: The Target on which to execute this workload
    :type target: Target

    :param name: Name of the workload. Useful for naming related artefacts.
    :type name: str

    :param res_dir: Directory into which artefacts will be stored
    :type res_dir: str

    :ivar command: Will be called in :meth:`run`. Daughter classes should
      specify its value as needed.

    **Design notes**

    ``__init__`` is there to initialize a given workload, and :meth:`run`
    can be called on it several times, with varying arguments.
    As much work as possible should be delegated to :meth:`run`, so that
    different flavours of the same workload can be run without the hassle of
    creating a superfluous amount of new instances. However, when persistent
    data is involved (e.g. the workload depends on a file), then this data
    should be exposed as an ``__init__`` parameter.

    **Implementation example**::

        class Printer(Workload):
            def __init__(self, target, name, res_dir=None):
                super().__init__(target, name, res_dir)
                self.command = "echo"

            def run(self, cpus=None, cgroup=None, background=False, as_root=False, value=42):
                self.command = "{} {}".format(self.command, value)
                super().run(cpus, cgroup, background, as_root)

    **Usage example**::

        >>> printer = Printer(target, "test")
        >>> printer.run()
        INFO    : Printer      : Execution start: echo 42
        INFO    : Printer      : Execution complete
        >>> print printer.output
        42\r\n
    """

    required_tools = ['taskset']
    """
    The tools required to execute the workload. See
    :meth:`lisa.target.Target.install_tools`.
    """

    def __init__(self, target, name, res_dir=None):
        self.target = target
        self.name = name
        self.command = None
        self.output = ""
        self.run_dir = self.target.working_directory

        res_dir = res_dir if res_dir else target.get_res_dir(
            name='{}-{}'.format(self.__class__.__qualname__, name)
        )
        self.res_dir = res_dir

        self.target.install_tools(self.required_tools)

    def run(self, cpus=None, cgroup=None, background=False, as_root=False):
        """
        Execute the workload on the configured target.

        :param cpus: CPUs on which to restrict the workload execution (taskset)
        :type cpus: list(int)

        :param cgroup: cgroup in which to run the workload
        :type cgroup: str

        :param background: Whether to run the workload in background or not
        :type background: bool

        :param as_root: Whether to run the workload as root or not
        :type as_root: bool

        The standard output will be saved into a file in ``self.res_dir``
        """
        logger = self.get_logger()
        if not self.command:
            raise RuntimeError("Workload does not specify any command to execute")

        _command = self.command
        target = self.target

        if cpus:
            taskset_bin = target.which('taskset')
            if not taskset_bin:
                raise RuntimeError("Could not find 'taskset' executable on the target")

            cpumask = list_to_mask(cpus)
            taskset_cmd = '{} 0x{:x}'.format(taskset_bin, cpumask)
            _command = '{} {}'.format(taskset_cmd, _command)

        if cgroup:
            _command = target.cgroups.run_into_cmd(cgroup, _command)

        logger.info("Execution start: %s", _command)

        if background:
            target.background(_command, as_root=as_root)
        else:
            self.output = target.execute(_command, as_root=as_root)
            logger.info("Execution complete")

            logfile = os.path.join(self.res_dir, 'output.log')
            logger.debug('Saving stdout to %s...', logfile)

            with open(logfile, 'w') as ofile:
                ofile.write(self.output)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
