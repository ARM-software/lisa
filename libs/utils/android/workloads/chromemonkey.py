# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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
import os
import logging

from android import Screen, System, Workload

class ChromeMonkey(Workload):
    """
    Android Chrome monkey tests workload
    """

    # Package required by this workload
    package = 'com.android.chrome'

    def __init__(self, test_env):
        super(ChromeMonkey, self).__init__(test_env)
        self._log = logging.getLogger('ChromeMonkey')
        self._log.debug('Workload created')

    def run(self, out_dir, duration_s, collect='', bufsize=10000):
        """
        Run Chrome with random events from monkey. Useful to diagnose/trace
        UI related issues.

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :param collect: Specifies what to collect. Possible values:
            - 'sched'
            - 'systrace'
            - 'ftrace'
            - any combination of the above
        :type collect: list(str)
        """

        # Keep track of mandatory parameters
        self.out_dir = out_dir
        self.collect = collect

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        self.tracingStart(bufsize)

        # Use the monkey tool
        self._log.info('Running monkey for %d [s]', duration_s)
        System.monkey(self._target, self.package, event_count=100000, timeout=duration_s,
                      check_exit_code=False)

        self.tracingStop()
