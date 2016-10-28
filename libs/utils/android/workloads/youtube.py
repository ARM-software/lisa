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

from android import Screen, Workload, System
from time import sleep

import logging

class YouTube(Workload):
    """
    Android YouTube workload
    """

    # Package required by this workload
    package = 'com.google.android.youtube'
    action = 'android.intent.action.VIEW'

    # Setup logger
    logger = logging.getLogger('YouTube')
    logger.setLevel(logging.INFO)


    def __init__(self, test_env):
        super(YouTube, self).__init__(test_env)
        self.logger.debug('%14s - Workload created', 'YouTube')

    def run(self, exp_dir, video_url, video_duration_s, collect=''):

        # Initialize energy meter results
        nrg_report = None

        # Unlock device screen (assume no password required)
        System.menu(self.target)
        # Press Back button to be sure we run the video from the start
        System.back(self.target)

        # Use the monkey tool to start YouTube without playing any video.
        # This allows to subsequently set the screen orientation to LANDSCAPE
        # and to reset the frame statistics.
        System.monkey(self.target, self.package)

        # Force screen in LANDSCAPE mode
        Screen.set_orientation(self.target, portrait=False)

        System.gfxinfo_reset(self.target, self.package)
        sleep(1)

        # Start YouTube video on the target device
        System.start_action(self.target, self.action, video_url)
        # Allow the activity to start
        sleep(1)

        # Start energy collection
        if 'energy' in collect and self.te.emeter:
            self.te.emeter.reset()

        # Wait until the end of the video
        self.logger.info("Play video for %d [s]", video_duration_s)
        sleep(video_duration_s)

        # Stop energy collection
        if 'energy' in collect and self.te.emeter:
            nrg_report = self.te.emeter.report(exp_dir)

        # Get frame stats
        db_file = os.path.join(exp_dir, "framestats.txt")
        System.gfxinfo_get(self.target, self.package, db_file)

        # Close the app without clearing the local data to
        # avoid the dialog to select the account at next start
        System.force_stop(self.target, self.package, clear=False)

        # Go back to home screen
        System.home(self.target)

        # Switch back to screen auto rotation
        Screen.set_orientation(self.target, auto=True)

        return db_file, nrg_report

# vim :set tabstop=4 shiftwidth=4 expandtab
