# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited, Google and contributors.
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

from time import sleep

from target_script import TargetScript
from android import Screen, System
from android.workload import Workload


class Manual(Workload):
    """
    Android Manual workload

    This workload will set up a device, start tracing, start collection,
    disconnect USB, and pause. User can then conduct a manual experiment,
    and press ENTER to reconnect USB, collect traces, collect power.
    """

    # Package is optional for this test
    package = 'optional'

    def __init__(self, test_env):
        super(Manual, self).__init__(test_env)
        self._log = logging.getLogger('Manual')
        self._log.debug('Workload created')

        # Set of output data reported by Manual
        self.db_file = None

    def run(self, out_dir, collect=''):
        """
        Run a manual workload.

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :type force: str

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above
        :type collect: list(str)
        """
        # Keep track of mandatory parameters
        self.out_dir = out_dir
        self.collect = collect

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Set timeout to min value
        Screen.set_timeout(self._target, seconds=0)

        # Prevent screen from dozing
        Screen.set_doze_always_on(self._target, on=False)

        # Turn on airplane mode
        System.set_airplane_mode(self._target, on=True)

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        self.tracingStart(screen_always_on=False)
        raw_input("Tracing and energy collection started.\n\
                   Press ENTER when the manual experiment is concluded.")
        print("collecting logs...")
        self.tracingStop(screen_always_on=False)

        Screen.set_defaults(self._target)
        System.set_airplane_mode(self._target, on=False)

# vim :set tabstop=4 shiftwidth=4 expandtab
