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

class System(object):
    """
    Collection of Android related services
    """

    @staticmethod
    def set_airplane_mode(target, on=True):
        """
        Set airplane mode
        """
        ap_mode = 1 if on else 0
        ap_state = 'true' if on else 'false'

        target.execute('settings put global airplane_mode_on {}'\
                       .format(ap_mode))
        target.execute('am broadcast '\
                       '-a android.intent.action.AIRPLANE_MODE '\
                       '--ez state {}'\
                       .format(ap_state))

    @staticmethod
    def start(target, apk_name, activity_name):
        """
        Start an application.

        :param apk_name: name of the apk
        :type apk_name: str

        :param activity_name: name of the activity to launch
        :type activity_name: str
        """
        target.execute('am start -n {}/{}'.format(apk_name, activity_name))

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
    def tap(target, x, y):
        """
        Tap a given point on the screen.

        :param x: horizontal coordinate
        :type x: int

        :param y: vertical coordinate
        :type y: int
        """
        target.execute('tap {} {}'.format(x, y))

# vim :set tabstop=4 shiftwidth=4 expandtab
