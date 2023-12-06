# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2023, Arm Limited and contributors.
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
"""
Fio is a widely used I/O benchmarking tool that is highly configurable.
We use it as in_iowait tasks are treated differently than just blocked tasks,
so anything involving that cannot be generated just with rt-app
"""

import re
from shlex import quote

from lisa.utils import memoized, kwargs_forwarded_to
from lisa.wlgen.workload import Workload


class FioOutput(str):
    """
    A wrapper around fio's terse format output to access some results
    XXX: Could also be parsed as json, but will we ever need more than iops?
    terse output format is guaranteed to be at stable offsets.
    """

    @property
    @memoized
    def read_iops(self):
        """
        Number of read IOPS as reported by fio
        """
        return int(self.split(';')[7])


class Fio(Workload):
    """
    A fio workload

    :param target: The Target on which to execute this workload
    :type target: Target

    :param filename: Filename for fio to operate on (can be block device)
    :type filename: str

    :param testname: The test name passed on to fio
    :type filename: str

    :param rw: The mode of operation for fio, e.g. randread, read, rw
    :type rw: str

    :param bs: The blocksize as a fio-accepted string (e.g. 512B, 4K, 1M)
    :type bs: str

    :param runtime: The time the test should be running for (defaults to seconds)
    :type runtime: str

    :param ioengine: The ioengine for fio to use (e.g. psync, libaio, io_uring)
    :type ioengine: str

    :param iodepth: The number of requests in-flight at once
    :type iodepth: int

    :param numjobs: The number of processes to run this as, will be reported together
    :type numjobs: int

    :param cpus_allowed: The set of cpus the fio tasks are allowed on
    :type cpus_allowed: str

    :param cli_options: Dictionary of cli_options passed to fio command line. Run
        ``fio --help`` for available parameters. Character
        ``_`` in option names is replaced by ``-``.
    :type cli_options: dict(str, object)

    :Variable keyword arguments: Forwarded to :class:`lisa.wlgen.workload.Workload`
    """

    @kwargs_forwarded_to(
        Workload.__init__,
        ignore=['command'],
    )
    def __init__(self, target, *,
        filename,
	testname='fiotest',
	rw='randread',
	bs='4k',
	runtime=None,
	ioengine="psync",
	iodepth=1,
	numjobs=1,
	cpus_allowed=None,
        cli_options=None,
        **kwargs
    ):

        arg_list = ['fio'] + ['--minimal'] # minimal is terse output format
        cli_options = cli_options.copy() if cli_options else {}
        cli_options['name'] = testname
        cli_options['filename'] = filename

        if runtime:
            arg_list += ['--time_based']
            cli_options['runtime'] = runtime

        if numjobs > 1:
            arg_list += ['--group_reporting'] # We always want just one report
            cli_options['numjobs'] = f"{numjobs}"

        cli_options['rw'] = rw
        cli_options['bs'] = bs
        cli_options['ioengine'] = ioengine
        cli_options['iodepth'] = iodepth

        if cpus_allowed:
            cli_options['cpus_allowed'] = cpus_allowed

        arg_list = arg_list + [
            quote(f"--{arg}={value}")
            for arg, value in cli_options.items()
        ]

        command = ' '.join(arg_list)

        super().__init__(target=target, command=command, **kwargs)

    def _run(self):
        out = yield self._basic_run()
        return FioOutput(out['stdout'])

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
