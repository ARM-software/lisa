# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited, Google and contributors.
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

from lisa.target_script import TargetScript
from lisa.android import Screen, System
from lisa.android.workload import Workload


class CameraPreview(Workload):
    """
    Android CameraPreview workload
    """

    # Package required by this workload
    package = 'com.google.android.GoogleCamera'
    action = 'android.intent.action.MAIN'

    def __init__(self, test_env):
        super(CameraPreview, self).__init__(test_env)
        self._log = logging.getLogger('CameraPreview')
        self._log.debug('Workload created')

    def run(self, out_dir, duration_s=30, collect='systrace'):
        """
        Run a camera preview workload

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :param duration_s: Duration of test
        :type duration_s: int

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above
        :type collect: list(str)
        """
        self._log.info("Running CameraPreview for {}s and collecting {}".format(duration_s, collect))

        # Keep track of mandatory parameters
        self.out_dir = out_dir
        self.collect = collect

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        # Set airplane mode
        System.set_airplane_mode(self._target, on=True)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Use the monkey tool to start CameraPreview
        # This allows to subsequently set the screen orientation to LANDSCAPE
        # and to reset the frame statistics.
        System.monkey(self._target, self.package)

        # Force screen in PORTRAIT  mode
        Screen.set_orientation(self._target, portrait=True)

        sleep(2)

        self.tracingStart()

        sleep(duration_s)

        self.tracingStop()

        # Close the app without clearing the local data to
        # avoid the dialog to select the account at next start
        System.force_stop(self._target, self.package, clear=False)

        # Go back to home screen
        System.home(self._target)

        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

        # Switch back to screen auto rotation
        Screen.set_orientation(self._target, auto=True)

        # Switch off airplane mode
        System.set_airplane_mode(self._target, on=False)
