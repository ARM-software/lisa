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
from time import sleep

from android import Screen, System
from android.workload import Workload


class UiBench(Workload):
    """
    Android UiBench workload
    """

    # Package required by this workload
    package = 'com.android.test.uibench'

    # Supported activities list, obtained via:
    # adb shell dumpsys package | grep -i uibench | grep Activity
    test_BitmapUpload = 'BitmapUpload'
    test_DialogList = 'DialogList'
    test_EditTextType = 'EditTextType'
    test_FullscreenOverdraw = 'FullscreenOverdraw'
    test_GlTextureView = 'GlTextureView'
    test_InflatingList = 'InflatingList'
    test_Invalidate = 'Invalidate'
    test_ShadowGrid = 'ShadowGrid'
    test_TextCacheHighHitrate = 'TextCacheHighHitrate'
    test_TextCacheLowHitrate = 'TextCacheLowHitrate'
    test_Transition = 'Transition'
    test_TransitionDetails = 'TransitionDetails'
    test_TrivialAnimation = 'TrivialAnimation'
    test_TrivialList = 'TrivialList'
    test_TrivialRecyclerView = 'TrivialRecyclerView'

    def __init__(self, test_env):
        super(UiBench, self).__init__(test_env)
        self._log = logging.getLogger('UiBench')
        self._log.debug('Workload created')

    def run(self, exp_dir, test_name, duration_s, collect=''):
        activity = '.' + test_name + 'Activity'

        # Initialize energy meter results
        nrg_report = None

        # Press Back button to be sure we run the video from the start
        System.menu(self.target)
        System.back(self.target)

        # Close and clear application
        System.force_stop(self.target, self.package, clear=True)

        # Set airplane mode
        System.set_airplane_mode(self.target, on=True)

        # Start the main view of the app which must be running
        # to reset the frame statistics.
        System.monkey(self.target, self.package)

        # Force screen in PORTRAIT mode
        Screen.set_orientation(self.target, portrait=True)

        # Reset frame statistics
        System.gfxinfo_reset(self.target, self.package)
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
                .format(self.target.adb_name))
        self._log.info("%s", logcat_cmd)

        # Start the activity
        System.start_activity(self.target, self.package, activity)
        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)
        while True:

            # read next logcat line (up to max 1024 chars)
            message = logcat.stdout.readline(1024)

            # Benchmark start trigger
            match = UIBENCH_BENCHMARK_START_RE.search(message)
            if match:
                if 'energy' in collect and self.te.emeter:
                    self.te.emeter.reset()
                self._log.debug("Benchmark started!")
                break

        # Run the workload for the required time
        self._log.info('Benchmark [%s] started, waiting %d [s]',
                     activity, duration_s)
        sleep(duration_s)
        self._log.debug("Benchmark done!")

        if 'energy' in collect and self.te.emeter:
            nrg_report = self.te.emeter.report(exp_dir)

        # Get frame stats
        db_file = os.path.join(exp_dir, "framestats.txt")
        System.gfxinfo_get(self.target, self.package, db_file)

        # Close and clear application
        System.force_stop(self.target, self.package, clear=True)

        # Go back to home screen
        System.home(self.target)

        # Switch back to original settings
        Screen.set_orientation(self.target, auto=True)
        System.set_airplane_mode(self.target, on=False)

        return db_file, nrg_report

# vim :set tabstop=4 shiftwidth=4 expandtab
