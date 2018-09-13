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

from lisa.android import Screen, System, Workload

class YouTube(Workload):
    """
    Android YouTube workload
    """

    # Package required by this workload
    package = 'com.google.android.youtube'
    action = 'android.intent.action.VIEW'

    def __init__(self, test_env):
        super(YouTube, self).__init__(test_env)
        self._log = logging.getLogger('YouTube')
        self._log.debug('Workload created')

        # Set of output data reported by Jankbench
        self.db_file = None

    def run(self, out_dir, video_url, video_duration_s, collect=''):
        """
        Run single YouTube workload.

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :param video_url: Video URL to be played
        :type video_url: str

        :param video_duration_s: Play video for this required number of seconds
        :type video_duration_s: int

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

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        # Stop youtube if already running
        System.force_stop(self._target, self.package)

        # Use the monkey tool to start YouTube without playing any video.
        # This allows to subsequently set the screen orientation to LANDSCAPE
        # and to reset the frame statistics.
        System.monkey(self._target, self.package)

        # Force screen in LANDSCAPE mode
        Screen.set_orientation(self._target, portrait=False)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        System.gfxinfo_reset(self._target, self.package)
        sleep(1)

        # Start YouTube video on the target device
        System.start_action(self._target, self.action, video_url)
        # Allow the activity to start
        sleep(1)

        # Wait until the end of the video
        self.tracingStart()
        self._log.info('Play video for %d [s]', video_duration_s)
        sleep(video_duration_s)
        self.tracingStop()

        # Get frame stats
        self.db_file = os.path.join(out_dir, "framestats.txt")
        System.gfxinfo_get(self._target, self.package, self.db_file)

        # Close the app without clearing the local data to
        # avoid the dialog to select the account at next start
        System.force_stop(self._target, self.package, clear=False)

        # Go back to home screen
        System.home(self._target)

        # Switch back to screen auto rotation
        Screen.set_orientation(self._target, auto=True)
        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
