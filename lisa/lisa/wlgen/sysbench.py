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

import re

from devlib.utils.misc import memoized

from workload import Workload

class SysbenchOutput(str):
    """
    A wrapper around sysbench's stdout to more easily access some results
    """

    @property
    @memoized
    def nr_events(self):
        """
        Number of events as reported on the "total number of events" line
        """
        match = re.search(r'total number of events:\s*(?P<events>[0-9]+)', self)
        return int(match.group('events'))

    @property
    @memoized
    def duration_s(self):
        """
        Execution duration as reported on the "total time" line
        """
        match = re.search(r'total time:\s*(?P<time>[0-9]+\.[0-9]+)s', self)
        return float(match.group('time'))

class Sysbench(Workload):
    """
    A sysbench workload
    """

    required_tools = Workload.required_tools + ['sysbench']

    def __init__(self, te, name, res_dir=None):
        super(Sysbench, self).__init__(te, name, res_dir)

        sysbench_bin = self.te.target.which('sysbench')
        if not sysbench_bin:
            raise RuntimeError("No sysbench executable found on the target")

        self.sysbench_bin = sysbench_bin


    def run(self, cpus=None, cgroup=None, background=False, as_root=False,
            test="cpu", max_duration_s=None, max_requests=None, **kwargs):
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

        :param test: The sysbench test to run (run ``sysbench --help``)
        :type test: str

        :param max_duration_s: Maximum duration in seconds
        :type max_duration_s: int

        :param: max_requests: Maximum number of event requests
        :type max_requests: int

        :Keyword arguments: Additionnal arguments required by the specific
          test (run ``sysbench --test=<test> help``). Due to Python limitations,
          use ``_`` instead of ``-`` for argument naming.

        The standard output will be saved into a file in :attr:`self.res_dir`
        """
        command = "{} --test={}".format(self.sysbench_bin, test)

        if max_duration_s is not None:
            command = "{} --max-time={}".format(command, max_duration_s)

        if max_requests is not None:
            command = "{} --max-requests={}".format(command, max_requests)

        extra_args = " ".join(["--{}={}".format(arg.replace("_", "-"), value)
                               for arg, value in kwargs.items()])
        if extra_args:
            command = "{} {}".format(command, extra_args)

        self.command = "{} run".format(command)

        super(Sysbench, self).run(cpus, cgroup, background, as_root)

        self.output = SysbenchOutput(self.output)
