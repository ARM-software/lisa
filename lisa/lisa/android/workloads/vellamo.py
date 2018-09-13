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
import collections
import copy
import logging
from time import sleep

from subprocess import Popen, PIPE

from lisa.android import Screen, System, Workload

# screen density of the device with which this workload has been tested
DEFAULT_DENSITY = '420'

VELLAMO_DB_PATH = '/data/data/com.quicinc.vellamo/files'
VELLAMO_SCORE_NAME = 'chapterscores.json'
VELLAMO_TESTS = ['BROWSER', 'METAL', 'MULTI']

class Vellamo(Workload):
    """
    Android Vellamo workload
    """

    # Package required by this workload
    package = 'com.quicinc.vellamo'
    activity = 'com.quicinc.vellamo.main.MainActivity'

    def __init__(self, test_env):
        super(Vellamo, self).__init__(test_env)
        self._log = logging.getLogger('Vellamo')
        self._log.debug('Workload created')

    def run(self, out_dir, test_name, collect=''):
        """
        Run single Vellamo workload. Returns a collection of results.

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :param test_name: Name of the test to run
        :type test_name: str

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above
        :type collect: list(str)
        """

        self.out_dir = out_dir
        self.collect = collect

        # Check if the density of the target device screen is different from
        # the one used to get the values below
        density = Screen.get_screen_density(self._target)
        if DEFAULT_DENSITY not in density:
            msg = 'Screen density of target device differs from {}.\n'\
                  'Please set it to {}'
            raise RuntimeError(msg.format(DEFAULT_DENSITY, DEFAULT_DENSITY))

        if test_name.upper() not in VELLAMO_TESTS:
            raise ValueError('Vellamo workload [%s] not supported', test_name)

        # Set parameter depending on test
        self._log.debug('Start Vellamo Benchmark [%s]', test_name)
        test_x, test_y = (0, 0)
        sleep_time = 0
        if test_name.upper() == 'BROWSER':
            test_x, test_y = (91, 33)
            sleep_time = 3.5
        elif test_name.upper() == 'METAL':
            test_x, test_y  = (91, 59)
            sleep_time = 1
        elif test_name.upper() == 'MULTI':
            test_x, test_y = (91, 71)
            sleep_time = 3.5

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        System.force_stop(self._target, self.package, clear=True)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Clear logcat
        os.system(self._adb('logcat -c'));

        # Regexps for benchmark synchronization
        start_logline = r'ActivityManager: Start.*'\
                         ':com.quicinc.vellamo:benchmarking'
        VELLAMO_BENCHMARK_START_RE = re.compile(start_logline)
        self._log.debug("START string [%s]", start_logline)

        # End of benchmark is marked by displaying results
        end_logline = r'ActivityManager: START.*'\
                       'act=com.quicinc.vellamo.*_RESULTS'
        VELLAMO_BENCHMARK_END_RE = re.compile(end_logline)
        self._log.debug("END string [%s]", end_logline)

        # Parse logcat output lines
        logcat_cmd = self._adb(
                'logcat ActivityManager:* System.out:I *:S BENCH:*'\
                .format(self._target.adb_name))
        self._log.info("%s", logcat_cmd)

        # Start the activity
        System.start_activity(self._target, self.package, self.activity)
        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)
        sleep(2)
        # Accept EULA
        System.tap(self._target, 80, 86)
        sleep(1)
        # Click Let's Roll
        System.tap(self._target, 50, 67)
        sleep(1)
        # Skip Arrow
        System.tap(self._target, 46, 78)
        # Run Workload
        System.tap(self._target, test_x, test_y)
        # Skip instructions
        System.hswipe(self._target, 10, 80, duration=100, swipe_right=False)
        System.hswipe(self._target, 10, 80, duration=100, swipe_right=False)
        System.hswipe(self._target, 10, 80, duration=100, swipe_right=False)
        self._log.info("Vellamo - {} started!".format(test_name.upper()))

        while True:

            # read next logcat line (up to max 1024 chars)
            message = logcat.stdout.readline(1024)

            # Benchmark start trigger
            if VELLAMO_BENCHMARK_START_RE.search(message):
                # Start tracing
                self.tracingStart()
                self._log.debug("Benchmark started!")

            elif VELLAMO_BENCHMARK_END_RE.search(message):
                # Stop tracing
                self.tracingStop()
                break

            else:
                continue

        # Gather scores file from the device
        db_file = os.path.join(out_dir, VELLAMO_SCORE_NAME)
        self._target.pull('{}/{}'.format(VELLAMO_DB_PATH, VELLAMO_SCORE_NAME),
                         db_file)

        System.force_stop(self._target, self.package, clear=True)

        # Go back to home screen
        System.home(self._target)

        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
