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
import time

from subprocess import Popen, PIPE

from android import Screen, System
from android.workload import Workload

PCMARK_DB_FILE = 'pcmarkscores.txt'
PCMARK_LOG_FILE = 'pcmarklog.txt'
PCMARK_TESTS = ['PERFORMANCE', 'BATTERY_LIFE']

# Regexps for benchmark synchronization
PCMARK_BENCHMARK_START_RE = re.compile(
    r'com.futuremark.pcmandroid.AbstractMainActivity: '\
    'running benchmark'
)

PCMARK_BENCHMARK_END_RE = re.compile(
    r'com.futuremark.pcmandroid.AbstractMainActivity: '\
    'onWebViewReady.*view_scoredetails.html'
)

class PCMark(Workload):
    """
    Android PCMark workload
    """

    # Package required by this workload
    package = 'com.futuremark.pcmark.android.benchmark'

    def __init__(self, test_env):
        super(PCMark, self).__init__(test_env)
        self._log = logging.getLogger('PCMark')
        self._log.debug('Workload created')

    def run(self, out_dir, test_name, collect=''):
        """
        Run single PCMark workload. Returns a collection of results.

        :param out_dir: Path to experiment directory on the host
                        where to store results.
        :type out_dir: str

        :param test_name: Name of the test to run
        :type test_name: str

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above as a single space-separated string.
        :type collect: list(str)
        """

        self.out_dir = out_dir
        self.collect = collect

        # Unlock device screen (assume no password required)
        System.menu(self._target)
        # Make sure we exit the app if already open
        System.back(self._target)

        # Close and NOT clear application (benchmark tests are downloaded from web)
        System.force_stop(self._target, self.package, clear=False)

        # Set airplane mode
        System.set_airplane_mode(self._target, on=True)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Force screen in PORTRAIT mode
        Screen.set_orientation(self._target, portrait=True)

        # Clear logcat
        self._target.clear_logcat()

        # Start benchmark
        System.monkey(self._target, self.package)
        # Wait a few seconds while application is being loaded
        time.sleep(8)

        # Parse logcat ActivityManager output lines to detect start
        logcat_cmd = self._adb(
                'logcat com.futuremark.pcmandroid.AbstractMainActivity:I *:S'\
                .format(self._target.adb_name))
        self._log.info("%s", logcat_cmd)

        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)

        # Run performance workload
        if test_name.upper() == 'PERFORMANCE':
            self._target.execute('input tap 750 1450')
            self._log.info('PCmark started!')
        else:
            raise ValueError('PCMark test [%s] not supported', test_name)

        # The PCMark performance test is composed of multiple benchmark
        # that have the same marker for starting
        # Use 'trace_started' to only start tracing once
        trace_started = False
        while True:

            # read next logcat line (up to max 1024 chars)
            message = logcat.stdout.readline(1024)

            match = PCMARK_BENCHMARK_START_RE.search(message)
            # Benchmark start trigger
            if match:
                # Start tracing
                if not trace_started:
                    self.tracingStart()
                    self._log.debug("Benchmark started!")
                    trace_started = True

            match = PCMARK_BENCHMARK_END_RE.search(message)
            if match:
                # Stop tracing
                if trace_started:
                    self.tracingStop()
                    self._log.debug("Benchmark done!")
                    break

        # Kill the app but not clean the downloaded tests
        System.force_stop(self._target, self.package, clear=False)
        logcat.kill()

        db_file = os.path.join(out_dir, PCMARK_DB_FILE)
        log_file = os.path.join(out_dir, PCMARK_LOG_FILE)
        self._log.info("Results can be found in {}.".format(db_file))

        with open(log_file, 'w') as log:
            logcat = Popen(self._adb(
                     'logcat com.futuremark.pcmandroid.VirtualMachineState:* *:S'\
                     .format(self._target.adb_name)),
                     stdout=log,
                     shell=True)
            time.sleep(2)
            logcat.kill()

        # Parse logcat file and create db with scores
        os.popen('grep -o "PCMA_.*_SCORE .*" {} | sed "s/ = / /g" | sort -u > {}'
                 .format(log_file, db_file))

        self._log.debug("Score file: {}, size {}B".format(db_file, os.path.getsize(db_file)))

        # Restore default configuration
        System.home(self._target)

        # Set orientation back to auto
        Screen.set_orientation(self._target, auto=True)

        # Go back to home screen
        System.home(self._target)

        # Turn off airplane mode
        System.set_airplane_mode(self._target, on=False)

        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

# vim :set tabstop=4 shiftwidth=4 expandtab
