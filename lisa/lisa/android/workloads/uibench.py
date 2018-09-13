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

from subprocess import Popen, PIPE
from time import time, sleep

from lisa.android import Screen, System, Workload
from lisa.target_script import TargetScript

class UiBench(Workload):
    """
    Android UiBench workload
    """

    # Package required by this workload
    package = 'com.android.test.uibench'

    # Supported activities list, obtained via:
    # adb shell dumpsys package | grep -i uibench | grep Activity
    #
    # test_actions is a dictonary of standard actions to perform for each activity in uibench
    # Possible actions are as follows, or None if no action is to be performed:
    #  - 'vswipe'
    #  - 'tap'
    #  - A list with any combination of the above, in execution order
    test_actions = {
        # General
        'DialogListActivity':           'vswipe',
        'FullscreenOverdrawActivity':   None,
        'GlTextureViewActivity':        None,
        'InvalidateActivity':           None,
        'TrivialAnimationActivity':     None,
        'TrivialListActivity':          'vswipe',
        'TrivialRecyclerViewActivity':  'vswipe',
        # Inflation
        'InflatingListActivity':        'vswipe',
        # Rendering
        'BitmapUploadActivity':         None,
        'ShadowGridActivity':           ['tap', 'vswipe'],
        # Text
        'EditTextTypeActivity':         None,
        'TextCacheHighHitrateActivity': 'vswipe',
        'TextCacheLowHitrateActivity':  'vswipe',
        # Transitions
        'ActivityTransition':           ['tap', 'tap'],
    }

    def __init__(self, test_env):
        super(UiBench, self).__init__(test_env)
        self._log = logging.getLogger('UiBench')
        self._log.debug('Workload created')

        # Set of output data reported by UiBench
        self.db_file = None

    def run(self, out_dir, test_name, duration_s, collect='', actions='default'):
        """
        Run single UiBench workload.

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :param test_name: Name of the test to run
        :type test_name: str

        :param duration_s: Run benchmak for this required number of seconds
        :type duration_s: int

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above
        :type collect: list(str)

        :param actions: Specifies what actions to perform. Possible values:
            - None      : Perform no action
            - 'default' : Use the predefined default actions from the `test_actions` dict
            - 'vswipe'  : Perform a vertical swipe
            - 'tap'     : Perform a centre tap
            - A list with any combination of vswipe and tap, in execution order
        :type actions: list(str)
        """

        activity = '.' + test_name

        # If default, get the default actions from test_actions
        # If this is an undefined test, no action will be the default
        if actions == 'default':
            actions = None
            if test_name in self.test_actions:
                actions = self.test_actions[test_name]
        # Format actions as a list if it is just a single string item,
        # or an empty list if it is None
        actions = [actions] if isinstance(actions, basestring) else actions if actions else []

        # Keep track of mandatory parameters
        self.out_dir = out_dir
        self.collect = collect

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        # Close and clear application
        System.force_stop(self._target, self.package, clear=True)

        # Set airplane mode
        System.set_airplane_mode(self._target, on=True)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Start the main view of the app which must be running
        # to reset the frame statistics.
        System.monkey(self._target, self.package)

        # Force screen in PORTRAIT mode
        Screen.set_orientation(self._target, portrait=True)

        # Reset frame statistics
        System.gfxinfo_reset(self._target, self.package)
        sleep(1)

        # Clear logcat
        os.system(self._adb('logcat -c'));

        # Regexps for benchmark synchronization
        start_logline = r'ActivityManager: START.*'\
                         'cmp=com.android.test.uibench/{}'.format(activity)
        UIBENCH_BENCHMARK_START_RE = re.compile(start_logline)
        self._log.debug("START string [%s]", start_logline)

        # Parse logcat output lines
        logcat_cmd = self._adb(
                'logcat ActivityManager:* System.out:I *:S BENCH:*'\
                .format(self._target.adb_name))
        self._log.info("%s", logcat_cmd)

        # Prepare user-input commands on the device
        if actions:
            script = TargetScript(self._te, "uibench.sh")
            for action in actions:
                self._perform_action(script, action)
            script.push()

        # Start the activity
        System.start_activity(self._target, self.package, activity)
        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)
        while True:

            # read next logcat line (up to max 1024 chars)
            message = logcat.stdout.readline(1024)

            # Benchmark start trigger
            match = UIBENCH_BENCHMARK_START_RE.search(message)
            if match:
                self.tracingStart()
                self._log.debug("Benchmark started!")
                break

        # Run the workload for the required time
        self._log.info('Benchmark [%s] started, waiting %d [s]',
                     activity, duration_s)

        start = time()
        if actions:
            script.run()

        while (time() - start) < duration_s:
            sleep(1)

        self._log.debug("Benchmark done!")
        self.tracingStop()

        # Get frame stats
        self.db_file = os.path.join(out_dir, "framestats.txt")
        System.gfxinfo_get(self._target, self.package, self.db_file)

        # Close and clear application
        System.force_stop(self._target, self.package, clear=True)

        # Go back to home screen
        System.home(self._target)

        # Switch back to original settings
        Screen.set_orientation(self._target, auto=True)
        System.set_airplane_mode(self._target, on=False)
        Screen.set_brightness(self._target, auto=True)

    @staticmethod
    def _perform_action(target, action, delay_s=1.0):
        # Delay before performing action
        target.execute('sleep {}'.format(delay_s))

        if action == 'vswipe':
            # Action: A fast Swipe Up/Scroll Down
            System.vswipe(target, 20, 80, 50)

        if action == 'tap':
            # Action: Tap in the centre of the screen
            System.tap(target, 50, 50)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
