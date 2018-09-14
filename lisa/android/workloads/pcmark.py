# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, Arm Limited and contributors.
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
import json

from subprocess import Popen, PIPE
from zipfile import ZipFile

from lisa.android import Screen, System
from lisa.android.workload import Workload

PCMARK_TESTS = ['work']

INPUT_RESULT_FILE  = 'Result.xml'
OUTPUT_RESULT_FILE = 'scores.json'

# Regexps for benchmark synchronization
REGEXPS = {
    'start'  : '.*START.*com.futuremark.pcmark.android.benchmark',
    'end'    : '.*onWebViewReady.*view_scoredetails.html',
    'result' : '.*received result for correct code, result file in (?P<path>.*\.zip)',
    'score'  : '\s*<result_Pcma(?P<name>.*)Score>(?P<score>[0-9]*)<'
}

class PCMark(Workload):
    """
    Android PCMark workload

    APK origin: http://www.futuremark.com/downloads/pcmark-android.apk

    Version 2.0.3716 is known to work

    .. note::
        Each PCMark workload needs to be installed. This requires user
        interaction and an internet connection - this is only a one time
        requirement however.
    """

    # Package required by this workload
    package  = 'com.futuremark.pcmark.android.benchmark'
    # Launch activity
    activity = 'com.futuremark.gypsum.activity.SplashPageActivity'

    def __init__(self, test_env):
        super(PCMark, self).__init__(test_env)
        self._log = logging.getLogger('PCMark')

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

        monitor = self._target.get_logcat_monitor(list(REGEXPS.values()))

        # Start PCMark on the target device
        System.start_activity(self._target, self.package, self.activity)
        # Wait a few seconds while application is being loaded
        time.sleep(5)

        if test_name.lower() == 'work':
            # Move to benchmark run page
            System.tab(self._target)
            System.enter(self._target)
            # Wait for page animations to end
            time.sleep(10)
        else:
            raise ValueError('PCMark test [%s] not supported', test_name)

        # Start benchmark
        monitor.start()
        System.enter(self._target)

        monitor.wait_for(REGEXPS['start'])
        self.tracingStart()
        self._log.info('PCmark started!')

        monitor.wait_for(REGEXPS['end'], timeout=600)
        self.tracingStop()
        self._log.info('PCmark ended!')

        # That should only match one line, but use the most recent one
        # in case logcat wasn't cleared properly
        result = monitor.wait_for(REGEXPS['result'])[-1]
        monitor.stop()

        remote_archive = re.match(REGEXPS['result'], result).group('path')
        local_archive = os.path.join(out_dir, self._target.path.basename(remote_archive))

        self._target.pull(remote_archive, local_archive, as_root=True)

        # Several files in the archive
        # Only "Result.xml" matters to us
        with ZipFile(local_archive, 'r') as archive:
            archive.extractall(out_dir)

        # Fetch workloads names and scores
        scores = {}
        input_result_filepath = os.path.join(out_dir, INPUT_RESULT_FILE)
        with open(input_result_filepath, 'r') as fd:
            for line in fd:
                match = re.match(REGEXPS['score'], line)
                if match:
                    scores[match.group('name')] = match.group('score')

        # Dump scores to json
        output_result_filepath = os.path.join(out_dir, OUTPUT_RESULT_FILE)
        with open(output_result_filepath, 'w') as fd:
            json.dump(scores, fd)

        self._log.info('Scores available in {}'.format(output_result_filepath))

        System.force_stop(self._target, self.package, clear=False)

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

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
