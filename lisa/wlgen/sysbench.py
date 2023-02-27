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
"""
Sysbench is a useful workload to get some performance numbers, e.g. to assert
that higher frequencies lead to more work done (as done in
:class:`~lisa.tests.cpufreq.sanity.UserspaceSanity`).
"""

import re
from shlex import quote

from lisa.utils import memoized, kwargs_forwarded_to
from lisa.wlgen.workload import Workload


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

    :param target: The Target on which to execute this workload
    :type target: Target

    :param max_duration_s: Maximum duration in seconds
    :type max_duration_s: int

    :param max_requests: Maximum number of event requests
    :type max_requests: int

    :param cli_options: Dictionary of cli_options passed to sysbench command line. Run
        ``sysbench --test=<test> help`` for available parameters. Character
        ``_`` in option names is replaced by ``-``.
    :type cli_options: dict(str, object)

    :Variable keyword arguments: Forwarded to :class:`lisa.wlgen.workload.Workload`
    """

    REQUIRED_TOOLS = ['sysbench']

    @kwargs_forwarded_to(
        Workload.__init__,
        ignore=['command'],
    )
    def __init__(self, target, *,
        test='cpu',
        max_duration_s=None,
        max_requests=None,
        cli_options=None,
        **kwargs
    ):
        cli_options = cli_options.copy() if cli_options else {}
        cli_options['test'] = test

        if max_duration_s is not None:
            cli_options['max-time'] = max_duration_s

        if max_requests is not None:
            cli_options['max-requests'] = max_requests

        arg_list = ['sysbench'] + [
            quote(f"--{arg.replace('_', '-')}={value}")
            for arg, value in cli_options.items()
        ] + ['run']

        command = ' '.join(arg_list)

        # deprecated, only for backward compat
        self._output = SysbenchOutput()

        super().__init__(target=target, command=command, **kwargs)

    def _run(self):
        out = yield self._basic_run()
        return SysbenchOutput(out['stdout'])

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
