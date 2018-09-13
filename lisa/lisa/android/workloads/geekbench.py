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
from time import sleep

from subprocess import Popen, PIPE

from lisa.android import Screen, Workload, System

# Regexps for benchmark synchronization
GEEKBENCH_BENCHMARK_START_RE = re.compile(
    r'ActivityManager: Start.* com.primatelabs.geekbench'
)
GEEKBENCH_BENCHMARK_END_RE = re.compile(
    r'GEEKBENCH_RESULT: (?P<results_file>.+)'
)

class Geekbench(Workload):
    """
    Android Geekbench workload
    """

    # Package required by this workload
    package = 'com.primatelabs.geekbench'
    activity = '.HomeActivity'

    def __init__(self, test_env):
        super(Geekbench, self).__init__(test_env)
        self._log = logging.getLogger('Geekbench')
        self._log.debug('Workload created')

    def run(self, out_dir, test_name, collect=''):
        """
        Run single Geekbench workload.

        :param out_dir: Path on host to experiment directory where
                        to store results.
        :type out_dir: str

        :param test_name: Name of the test to run
        :type test_name: str

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above in a space-separated string.
        :type collect: list(str)
        """

        # Initialize energy meter results
        self.out_dir = out_dir
        self.collect = collect

        # Clear the stored data for the application, so we always start with
        # an EULA to clear
        System.force_stop(self._target, self.package, clear=True)

        # Clear logcat from any previous runs
        # do this on the target as then we don't need to build a string
        self._target.clear_logcat()

        # Unlock device screen (assume no password required)
        System.menu(self._target)
        # Press Back button to be sure we run the benchmark from the start
        System.back(self._target)

        # Force screen in PORTRAIT mode
        Screen.set_orientation(self._target, portrait=True)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Start app on the target device
        System.start_activity(self._target, self.package, self.activity)
        # Allow the activity to start
        sleep(2)

        # Parse logcat output lines to find beginning and end
        logcat_cmd = self._adb(
                'logcat ActivityManager:* System.out:I *:S GEEKBENCH_RESULT:*'\
                .format(self._target.adb_name))
        self._log.info("%s", logcat_cmd)
        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)

        # Click to accept the EULA
        System.tap(self._target, 73, 55)
        sleep(1)

        # The main window opened will have the CPU benchmark
        # Swipe to get the COMPUTE one
        if test_name.upper() == 'COMPUTE':
            System.hswipe(self._target, 10, 80, duration=100, swipe_right=False)

        # Press the 'RUN <test_name> BENCHMARK' button
        System.tap(self._target, 73, 72)

        while True:

            # read next logcat line (up to max 1024 chars)
            message = logcat.stdout.readline(1024)

            # Benchmark start trigger
            match = GEEKBENCH_BENCHMARK_START_RE.search(message)
            if match:
                # Start tracing
                self.tracingStart()
                self._log.debug("Benchmark started!")

            # Benchmark end trigger
            match = GEEKBENCH_BENCHMARK_END_RE.search(message)
            if match:
                # Stop tracing
                self.tracingStop()
                remote_result_file = match.group('results_file')
                self._log.debug("Benchmark finished! Results are in {}".format(remote_result_file))
                break

        # Get Geekbench Results file
        target_result_file = self._target.path.basename(remote_result_file)
        result_file = os.path.join(self.out_dir, target_result_file)
        self._log.debug("result_file={}".format(result_file))
        self._target.pull(remote_result_file, result_file)

        # Close the app
        System.force_stop(self._target, self.package, clear=False)

        # Go back to home screen
        System.home(self._target)

        # Switch back to screen auto rotation
        Screen.set_orientation(self._target, auto=True)

        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
