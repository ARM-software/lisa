# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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

class ViewerWorkload(Workload):
    """
    Android generic Viewer workload

    This workload will use a given URI and will let Android pick the best
    application for viewing the item designated by that URI. For instance,
    a Youtube video URL would lead to opening the Youtube App if Google
    Services are available, if not the default web browser will be used
    to load the Youtube page.

    Three methods are available for customizing the workload in a subclass,
    see their respective docstring for more details. At least interact() must
    be implemented.

    Here's a minimalist example use case of this class, that loads a gif
    and keeps it on display for 10 seconds:

    definition
    -----------
    class Example(ViewerWorkload):
        def interact(self):
            sleep(10)

    execution
    ----------
    wload = Workload.getInstance(te, 'Example')
    wload.run(out_dir=te.res_dir,
        uri="https://media.giphy.com/media/XIqCQx02E1U9W/giphy.gif")
    """

    # Let the system pick the best package
    package = ''

    def __init__(self, test_env):
        super(ViewerWorkload, self).__init__(test_env)

        # Set of output data reported by the viewer
        self.db_file = None

    def pre_interact(self):
        """
        This method will be called right before tracing starts, but after the
        item-viewing app has been launched. This can be useful to configure
        some app-specific settings, to press buttons, start a video, etc.
        """
        pass

    def interact(self):
        """
        This method will be called right after the tracing starts. Tracing will
        continue as long as this method is running, so it can be tailored to
        your workload requirements. It could simply be sleeping for x seconds,
        or monitoring logcat to wait for a certain event, or issuing input
        commands to swipe around a gallery/web page/app, etc.
        """
        raise NotImplemented("interact() must be implemented")

    def post_interact(self):
        """
        This method will be called right after tracing stops, but before the
        item-viewing app has been closed. This can be useful to dump some
        app-specific statistics.
        """
        pass

    def run(self, out_dir, uri, portrait=True, collect=''):
        """
        Run viewer workload

        :param out_dir: Path to experiment directory where to store results.
        :type out_dir: str

        :param uri: The URI of the item to display
        :type location_search: str

        :param portrait: If True, display mode will be set to 'portrait' prior
            to viewing the item. If False, 'landscape' mode will be set.

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

        # Set min brightness
        Screen.set_brightness(self._target, auto=False, percent=0)
        # Unlock device screen (assume no password required)
        Screen.unlock(self._target)

        # Force screen in requested orientation
        Screen.set_orientation(self._target, portrait=portrait)

        System.gfxinfo_reset(self._target, self.package)
        # Wait for gfxinfo reset to be completed
        sleep(1)

        # Open the requested uri
        System.view_uri(self._target, uri)

        self.pre_interact()
        self.tracingStart()

        self.interact()

        self.tracingStop()
        self.post_interact()

        # Get frame stats
        self.db_file = os.path.join(out_dir, "framestats.txt")
        System.gfxinfo_get(self._target, self.package, self.db_file)

        # Go back to home screen
        System.home(self._target)

        # Set brightness back to auto
        Screen.set_brightness(self._target, auto=True)

        # Switch back to screen auto rotation
        Screen.set_orientation(self._target, auto=True)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
