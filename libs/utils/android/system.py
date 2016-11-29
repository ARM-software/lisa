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

from devlib.utils.android import adb_command
from devlib import TargetError
import os
import pexpect as pe

GET_FRAMESTATS_CMD = 'shell dumpsys gfxinfo {} > {}'

class System(object):
    """
    Collection of Android related services
    """

    @staticmethod
    def systrace_start(target, trace_file, time,
                       events=['gfx', 'view', 'sched', 'freq', 'idle']):

        log = logging.getLogger('System')

        # Check which systrace binary is available under CATAPULT_HOME
        for systrace in ['systrace.py', 'run_systrace.py']:
                systrace_path = os.path.join(target.CATAPULT_HOME, 'systrace',
                                             'systrace', systrace)
                if os.path.isfile(systrace_path):
                        break
        else:
                log.warning("Systrace binary not available under CATAPULT_HOME: %s!",
                            target.CATAPULT_HOME)
                return None

        #  Format the command according to the specified time and events
        systrace_pattern = "{} -e {} -o {} {} -t {}"
        trace_cmd = systrace_pattern.format(systrace_path, target.conf['device'],
                                            trace_file, " ".join(events), time)
        log.info('SysTrace: %s', trace_cmd)

        # Actually spawn systrace
        return pe.spawn(trace_cmd)

    @staticmethod
    def systrace_wait(target, systrace_output):
        systrace_output.wait()

    @staticmethod
    def set_airplane_mode(target, on=True):
        """
        Set airplane mode
        """
        ap_mode = 1 if on else 0
        ap_state = 'true' if on else 'false'

        try:
            target.execute('settings put global airplane_mode_on {}'\
                           .format(ap_mode), as_root=True)
            target.execute('am broadcast '\
                           '-a android.intent.action.AIRPLANE_MODE '\
                           '--ez state {}'\
                           .format(ap_state), as_root=True)
        except TargetError:
            log = logging.getLogger('System')
            log.warning('Failed to toggle airplane mode, permission denied.')

    @staticmethod
    def start_app(target, apk_name):
        """
        Start the main activity of the specified application

        :param apk_name: name of the apk
        :type apk_name: str
        """
        target.execute('monkey -p {} -c android.intent.category.LAUNCHER 1'\
                      .format(apk_name))

    @staticmethod
    def start_activity(target, apk_name, activity_name):
        """
        Start an application by specifying package and activity name.

        :param apk_name: name of the apk
        :type apk_name: str

        :param activity_name: name of the activity to launch
        :type activity_name: str
        """
        target.execute('am start -n {}/{}'.format(apk_name, activity_name))

    @staticmethod
    def start_action(target, action, action_args=''):
        """
        Start an activity by specifying an action.

        :param action: action to be executed
        :type action: str

        :param action_args: arguments for the activity
        :type action_args: str
        """
        target.execute('am start -a {} {}'.format(action, action_args))

    @staticmethod
    def force_stop(target, apk_name, clear=False):
        """
        Stop the application and clear its data if necessary.

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param apk_name: name of the apk
        :type apk_name: str

        :param clear: clear application data
        :type clear: bool
        """
        target.execute('am force-stop {}'.format(apk_name))
        if clear:
            target.execute('pm clear {}'.format(apk_name))

    @staticmethod
    def tap(target, x, y, absolute=False):
        """
        Tap a given point on the screen.

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param x: horizontal coordinate
        :type x: int

        :param y: vertical coordinate
        :type y: int

        :param absolute: use absolute coordinates or percentage of screen
            resolution
        :type absolute: bool
        """
        if not absolute:
            w, h = target.screen_resolution
            x = w * x / 100
            y = h * y / 100

        target.execute('input tap {} {}'.format(x, y))

    @staticmethod
    def vswipe(target, y_low_pct, y_top_pct, duration='', swipe_up=True):
        """
        Vertical swipe

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param y_low_pct: vertical lower coordinate percentage
        :type y_low_pct: int

        :param y_top_pct: vertical upper coordinate percentage
        :type y_top_pct: int

        :param duration: duration of the swipe in milliseconds
        :type duration: int

        :param swipe_up: swipe up or down
        :type swipe_up: bool
        """
        w, h = target.screen_resolution
        x = w / 2
        if swipe_up:
            y1 = h * y_top_pct / 100
            y2 = h * y_low_pct / 100
        else:
            y1 = h * y_low_pct / 100
            y2 = h * y_top_pct / 100

        target.execute('input swipe {} {} {} {} {}'\
                       .format(x, y1, x, y2, duration))

    @staticmethod
    def hswipe(target, x_left_pct, x_right_pct, duration='', swipe_right=True):
        """
        Horizontal swipe

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param x_left_pct: horizontal left coordinate percentage
        :type x_left_pct: int

        :param x_right_pct: horizontal right coordinate percentage
        :type x_right_pct: int

        :param duration: duration of the swipe in milliseconds
        :type duration: int

        :param swipe_right: swipe right or left
        :type swipe_right: bool
        """
        w, h = target.screen_resolution
        y = h / 2
        if swipe_right:
            x1 = w * x_left_pct / 100
            x2 = w * x_right_pct / 100
        else:
            x1 = w * x_right_pct / 100
            x2 = w * x_left_pct / 100
        target.execute('input swipe {} {} {} {} {}'\
                       .format(x1, y, x2, y, duration))

    @staticmethod
    def menu(target):
        """
        Press MENU button

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget
        """
        target.execute('input keyevent KEYCODE_MENU')

    @staticmethod
    def home(target):
        """
        Press HOME button

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget
        """
        target.execute('input keyevent KEYCODE_HOME')

    @staticmethod
    def back(target):
        """
        Press BACK button

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget
        """
        target.execute('input keyevent KEYCODE_BACK')

    @staticmethod
    def gfxinfo_reset(target, apk_name):
        """
        Reset gfxinfo frame statistics for a given app.

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param apk_name: name of the apk
        :type apk_name: str
        """
        target.execute('dumpsys gfxinfo {} reset'.format(apk_name))

    @staticmethod
    def gfxinfo_get(target, apk_name, out_file):
        """
        Collect frame statistics for the given app.

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param apk_name: name of the apk
        :type apk_name: str

        :param out_file: output file name
        :type out_file: str
        """
        adb_command(target.adb_name,
                    GET_FRAMESTATS_CMD.format(apk_name, out_file))

    @staticmethod
    def monkey(target, apk_name, event_count=1):
        """
        Wrapper for adb monkey tool.

        The Monkey is a program that runs on your emulator or device and
        generates pseudo-random streams of user events such as clicks, touches,
        or gestures, as well as a number of system-level events. You can use
        the Monkey to stress-test applications that you are developing, in a
        random yet repeatable manner.

        Full documentation is available at:

        https://developer.android.com/studio/test/monkey.html

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param apk_name: name of the apk
        :type apk_name: str

        :param event_count: number of events to generate
        :type event_count: int
        """
        target.execute('monkey -p {} {}'.format(apk_name, event_count))

    @staticmethod
    def list_packages(target, apk_filter=''):
        """
        List the packages matching the specified filter

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param apk_filter: a substring which must be part of the package name
        :type apk_filter: str
        """
        packages = []

        pkgs = target.execute('cmd package list packages {}'\
                              .format(apk_filter.lower()))
        for pkg in pkgs.splitlines():
            packages.append(pkg.replace('package:', ''))
        packages.sort()

        if len(packages):
            return packages
        return None

    @staticmethod
    def packages_info(target, apk_filter=''):
        """
        Get a dictionary of installed APKs and related information

        :param target: instance of devlib Android target
        :type target: devlib.target.AndroidTarget

        :param apk_filter: a substring which must be part of the package name
        :type apk_filter: str
        """
        packages = {}

        pkgs = target.execute('cmd package list packages {}'\
                              .format(apk_filter.lower()))
        for pkg in pkgs.splitlines():
            pkg = pkg.replace('package:', '')
            # Lookup for additional APK information
            apk = target.execute('pm path {}'.format(pkg))
            apk = apk.replace('package:', '')
            packages[pkg] = {
                'apk' : apk.strip()
            }

        if len(packages):
            return packages
        return None

# vim :set tabstop=4 shiftwidth=4 expandtab
