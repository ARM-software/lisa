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

from subprocess import Popen, PIPE

from time import sleep

from lisa.android import Screen, System, Workload
from devlib.utils.android import grant_app_permissions

# Regexps for benchmark synchronization

REGEXPS = {
    'start'    : '.*Displayed com.google.android.exoplayer2.demo/.PlayerActivity',
    'duration' : '.*period \[(?P<duration>[0-9]+.*)\]',
    'end'      : '.*state \[.+, .+, E\]'
}

class ExoPlayer(Workload):
    """
    Android ExoPlayer workload

    Exoplayer sources: https://github.com/google/ExoPlayer

    The 'demo' application is used by this workload.
    It can easily be built by loading the ExoPlayer sources
    into Android Studio

    Expected apk is 'demo-noExtensions-debug.apk'

    Version r2.4.0 (d979469) is known to work
    """

    # Package required by this workload
    package = 'com.google.android.exoplayer2.demo'
    action = 'com.google.android.exoplayer.demo.action.VIEW'

    def __init__(self, test_env):
        super(ExoPlayer, self).__init__(test_env)
        self._log = logging.getLogger('ExoPlayer')

    def _play(self):

        # Grant app all permissions
        grant_app_permissions(self._target, self.package)

        # Handle media file location
        if not self.from_device:
            remote_file = self._target.path.join(
                self._target.working_directory,
                os.path.basename(self.media_file)
            )

            self._log.info('Pushing media file to device...')
            self._target.push(
                self.media_file,
                remote_file,
                timeout = 60
            )
            self._log.info('Media file transfer complete')
        else:
            remote_file = self.media_file

        # Prepare logcat monitor
        monitor = self._target.get_logcat_monitor(REGEXPS.values())
        monitor.start()

        # Play media file
        play_cmd = 'am start -a "{}" -d "file://{}"'\
                   .format(self.action, remote_file)
        self._log.info(play_cmd)
        self._target.execute(play_cmd)

        monitor.wait_for(REGEXPS['start'])
        self.tracingStart()
        self._log.info('Playing media file')

        line = monitor.wait_for(REGEXPS['duration'])[0]
        media_duration_s = int(round(float(re.search(REGEXPS['duration'], line)
                                   .group('duration'))))

        self._log.info('Media duration is {}'.format(media_duration_s))

        if self.play_duration_s and self.play_duration_s < media_duration_s:
            self._log.info('Waiting {} seconds before ending playback'
                           .format(self.play_duration_s))
            sleep(self.play_duration_s)
        else:
            self._log.info('Waiting for playback completion ({} seconds)'
                           .format(media_duration_s))
            monitor.wait_for(REGEXPS['end'], timeout = media_duration_s + 30)

        self.tracingStop()
        monitor.stop()
        self._log.info('Media file playback completed')

        # Remove file if it was pushed
        if not self.from_device:
            self._target.remove(remote_file)

    def run(self, out_dir, collect, media_file, from_device=False, play_duration_s=None):
        """
        Run Exoplayer workload

        :param out_dir: Path to experiment directory on the host
                        where to store results.
        :type out_dir: str

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
            - 'systrace'
            - 'ftrace'
            - any combination of the above as a single space-separated string.
        :type collect: list(str)

        :param media_file: Filepath of the media to play
            Path on device if 'from_device' is True
            Path on host   if 'from_device' is False (default)
        :type media_file: str

        :param from_device: Whether file to play is already on the device
        :type from_device: bool

        :param play_duration_s: If set, maximum duration (seconds) of the media playback
                                If not set, media will play to completion
        :type play_duration_s: int
        """

        # Keep track of mandatory parameters
        self.out_dir = out_dir
        self.collect = collect
        self.media_file = media_file
        self.from_device = from_device
        self.play_duration_s = play_duration_s

        # Check media file exists
        if from_device and not self._target.file_exists(self.media_file):
            raise RuntimeError('Cannot find "{}" on target'.format(self.media_file))
        elif not from_device and not os.path.isfile(self.media_file):
            raise RuntimeError('Cannot find "{}" on host'.format(self.media_file))

        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        # Close and clear application
        System.force_stop(self._target, self.package, clear=True)

        # Enable airplane mode
        System.set_airplane_mode(self._target, on=True)

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)

        # Force screen in PORTRAIT mode
        Screen.set_orientation(self._target, portrait=True)

        # Launch Exoplayer benchmark
        self._play()

        # Go back to home screen
        System.home(self._target)

        # Set orientation back to auto
        Screen.set_orientation(self._target, auto=True)

        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

        # Turn off airplane mode
        System.set_airplane_mode(self._target, on=False)

        # Close and clear application
        System.force_stop(self._target, self.package, clear=True)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
