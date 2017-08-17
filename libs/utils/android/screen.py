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

import logging

from system import System
from time import sleep
import os

class Screen(object):
    """
    Set of utility functions to control an Android Screen
    """

    @staticmethod
    def set_orientation(target, auto=True, portrait=None):
        """
        Set screen orientation mode
        """
        log = logging.getLogger('Screen')
        acc_mode = 1 if auto else 0
        # Force manual orientation of portrait specified
        if portrait is not None:
            acc_mode = 0
            log.info('Force manual orientation')
        if acc_mode == 0:
            usr_mode = 0 if portrait else 1
            usr_mode_str = 'PORTRAIT' if portrait else 'LANDSCAPE'
            log.info('Set orientation: %s', usr_mode_str)
        else:
            usr_mode = 0
            log.info('Set orientation: AUTO')

        if acc_mode == 0:
            target.execute('content insert '\
                           '--uri content://settings/system '\
                           '--bind name:s:accelerometer_rotation '\
                           '--bind value:i:{}'.format(acc_mode))
            target.execute('content insert '\
                           '--uri content://settings/system '\
                           '--bind name:s:user_rotation '\
                           '--bind value:i:{}'.format(usr_mode))
        else:
            # Force PORTRAIT mode when activation AUTO rotation
            target.execute('content insert '\
                           '--uri content://settings/system '\
                           '--bind name:s:user_rotation '\
                           '--bind value:i:{}'.format(usr_mode))
            target.execute('content insert '\
                           '--uri content://settings/system '\
                           '--bind name:s:accelerometer_rotation '\
                           '--bind value:i:{}'.format(acc_mode))

    @staticmethod
    def set_brightness(target, auto=True, percent=None):
        """
        Set screen brightness percentage
        """
        log = logging.getLogger('Screen')
        bri_mode = 1 if auto else 0
        # Force manual brightness if a percent specified
        if percent:
            bri_mode = 0
        target.execute('content insert '\
                       '--uri content://settings/system '\
                       '--bind name:s:screen_brightness_mode '\
                       '--bind value:i:{}'.format(bri_mode))
        if bri_mode == 0:
            if percent<0 or percent>100:
                msg = "Screen brightness {} out of range (0,100)"\
                      .format(percent)
                raise ValueError(msg)
            value = 255 * percent / 100
            target.execute('content insert '\
                           '--uri content://settings/system '\
                           '--bind name:s:screen_brightness '\
                           '--bind value:i:{}'.format(value))
            log.info('Set brightness: %d%%', percent)
        else:
            log.info('Set brightness: AUTO')

    @staticmethod
    def set_dim(target, auto=True):
        """
        Set screen dimming mode
        """
        log = logging.getLogger('Screen')
        dim_mode = 1 if auto else 0
        dim_mode_str = 'ON' if auto else 'OFF'
        target.execute('content insert '\
                       '--uri content://settings/system '\
                       '--bind name:s:dim_screen '\
                       '--bind value:i:{}'.format(dim_mode))
        log.info('Dim screen mode: %s', dim_mode_str)

    @staticmethod
    def set_timeout(target, seconds=30):
        """
        Set screen off timeout in seconds
        """
        log = logging.getLogger('Screen')
        if seconds<0:
            msg = "Screen timeout {}: cannot be negative".format(seconds)
            raise ValueError(msg)
        value = seconds * 1000
        target.execute('content insert '\
                       '--uri content://settings/system '\
                       '--bind name:s:screen_off_timeout '\
                       '--bind value:i:{}'.format(value))
        log.info('Screen timeout: %d [s]', seconds)

    @staticmethod
    def set_defaults(target):
        """
        Reset screen settings to a reasonable default
        """
        Screen.set_orientation(target)
        Screen.set_brightness(target)
        Screen.set_dim(target)
        Screen.set_timeout(target)

    @staticmethod
    def get_screen_density(target):
        """
        Get screen density of the device.
        """
        return target.execute('getprop ro.sf.lcd_density')

    @staticmethod
    def set_screen(target, on=True):
        log = logging.getLogger('Screen')
        if not on:
            log.info('Setting screen OFF')
            System.sleep(target)
            return
        log.info('Setting screen ON')
        System.wakeup(target)

    @staticmethod
    def unlock(target):
       Screen.set_screen(target, on=True)
       sleep(1)
       System.menu(target)
       System.home(target)

    class Recorder(object):
        """
        Android screenrecord utility class
        """

        def __init__(self, target):
            self._target = target
            self._log = logging.getLogger('Recorder')

            self._recording = False

        def start(self, out_file, timestamp=True, max_time=None):
            """
            Start recording the screen of the device
            """
            if self._recording:
                raise RuntimeError('Recording already started')

            self._local_file = out_file
            self._remote_file = self._target.path.join(
                self._target.working_directory,
                os.path.basename(out_file)
            )

            self._max_time = max_time

            options = []
            if timestamp:
                options.append('--bugreport')
            if max_time:
                options.append('--time-limit {}'.format(max_time))

            record_cmd = 'screenrecord {} {}'.format(
                ' '.join(options), self._remote_file
            )

            self._log.debug('screenrecord command = \"{}\"'.format(record_cmd))
            self._log.info('Starting screen recording')

            self._recording = True
            self._process = self._target.background(record_cmd, as_root=True)

        def _finalize(self):
            self._process.wait()

            self._target.pull(self._remote_file,
                              self._local_file,
                              as_root=True)

            self._recording = False
            self._log.info('Recording available at {}'.format(self._local_file))

        def wait(self):
            """
            Wait for the recording to end
            """
            if not self._recording:
                raise RuntimeError('Recording not started')

            if not self._max_time:
                self.stop()
                raise RuntimeError('max_time not set, stopped Recorder')

            self._log.info('Waiting for recording to end ({}s total)'.format(self._max_time))
            self._finalize()

        def stop(self):
            """
            Force stop recording the screen of the device
            """
            if not self._recording:
                raise RuntimeError('Recording not started')

            # Recording must be ended by Ctrl-C on the device
            self._log.debug('Sending Ctrl-C to screenrecord')
            self._target.execute('kill -INT $(pgrep screenrecord)', as_root=True)

            self._finalize()


# vim :set tabstop=4 shiftwidth=4 expandtab
