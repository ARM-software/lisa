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

from subprocess import Popen, PIPE
from android import Screen, Workload
from time import sleep

import logging

YOUTUBE_CMD = 'shell dumpsys gfxinfo com.google.android.youtube > {}'

class YouTube(Workload):
    """
    Android YouTube workload
    """

    # Package required by this workload
    package = 'com.google.android.youtube'

    # Setup logger
    logger = logging.getLogger('YouTube')
    logger.setLevel(logging.INFO)


    def __init__(self, test_env):
        super(YouTube, self).__init__(test_env)
        logging.debug('%14s - Workload created', 'YouTube')

    def run(self, exp_dir, video_url, video_duration_s, collect=''):

        # Initialize energy meter results
        nrg_report = None

        # Unlock device screen (assume no password required)
        self.target.execute('input keyevent 82')
        # Press Back button to be sure we run the video from the start
        self.target.execute('input keyevent KEYCODE_BACK')

        # Force screen in LANDSCAPE mode
        Screen.set_orientation(self.target, portrait=False)

        # Start YouTube video on the target device
        youtube_cmd = 'am start -a android.intent.action.VIEW "{}"'\
                      .format(video_url)
        logging.info(youtube_cmd)
        self.target.execute(youtube_cmd)
        # Allow the activity to start
        sleep(3)

        # Reset framestats collection
        self.target.execute('dumpsys gfxinfo --reset')

        # Start energy collection
        if 'energy' in collect and self.te.emeter:
            self.te.emeter.reset()

        # Wait until the end of the video
        logging.info("Play video for %d [s]", video_duration_s)
        sleep(video_duration_s)

        # Stop energy collection
        if 'energy' in collect and self.te.emeter:
            nrg_report = self.te.emeter.report(exp_dir)
            logging.info("Estimated energy: %7.3f",
                         float(nrg_report.channels['BAT']))

        # Get frame stats
        db_file = os.path.join(exp_dir, "framestats.txt")
        self._adb(YOUTUBE_CMD.format(db_file))

        # Close and clear application
        self.target.execute('am force-stop com.google.android.youtube')
        self.target.execute('pm clear com.google.android.youtube')

        # Go back to home screen
        self.target.execute('input keyevent KEYCODE_HOME')

        # Switch back to screen auto rotation
        Screen.set_orientation(self.target, auto=True)

        return db_file, nrg_report

# vim :set tabstop=4 shiftwidth=4 expandtab
