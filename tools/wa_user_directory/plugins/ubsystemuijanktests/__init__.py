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

from devlib.utils.android import grant_app_permissions

from wa import ApkWorkload, Parameter, WorkloadError

class UbSystemUiJankTests(ApkWorkload):
    """
    AOSP UbSystemUiJankTests tests

    Performs actions on the System UI (launcher, settings, etc) so that UI
    responsiveness can be evaluated.

    The .apk can be built with `make UbSystemUiJankTests` in the AOSP tree.

    Reports metrics the metrics reported by instrumentation system - these will
    likely overlap with those reported by the 'fps' instrument, but should be
    more accurately recorded.
    """

    name = 'ubsystemuijanktests'

    package_names = ['android.platform.systemui.tests.jank']

    tests = [
        'LauncherJankTests#testOpenAllAppsContainer',
        'LauncherJankTests#testAllAppsContainerSwipe',
        'LauncherJankTests#testHomeScreenSwipe',
        'LauncherJankTests#testWidgetsContainerFling',
        'SettingsJankTests#testSettingsFling',
        'SystemUiJankTests#testRecentAppsFling',
        'SystemUiJankTests#testRecentAppsDismiss',
        'SystemUiJankTests#testNotificationListPull',
        'SystemUiJankTests#testNotificationListPull_manyNotifications',
        'SystemUiJankTests#testQuickSettingsPull',
        'SystemUiJankTests#testUnlock',
        'SystemUiJankTests#testExpandGroup',
        'SystemUiJankTests#testClearAll',
        'SystemUiJankTests#testChangeBrightness',
        'SystemUiJankTests#testNotificationAppear',
        'SystemUiJankTests#testCameraFromLockscreen',
        'SystemUiJankTests#testAmbientWakeUp',
        'SystemUiJankTests#testGoToFullShade',
        'SystemUiJankTests#testInlineReply',
        'SystemUiJankTests#testPinAppearance',
        'SystemUiJankTests#testLaunchSettings',
    ]

    parameters = [
        Parameter('test', default=tests[0], allowed_values=tests,
                  description='Which of the System UI jank tests to run')
    ]

    def setup(self, context):
        # Override the default setup method, as it calls
        # self.apk.start_activity. We dont want to do that.

        self.apk.initialize_package(context)
        self.target.execute('am kill-all')  # kill all *background* activities
        grant_app_permissions(self.target, self.package)

        self.target.clear_logcat()

        jclass = '{}.{}'.format(self.package, self.test)
        self.command = 'am instrument -e iterations 1 -e class {} -w {}'.format(
            jclass, self.package)

    def run(self, context):
        self.output = self.target.execute(self.command)

        # You see 'FAILURES' if an exception is thrown.
        # You see 'Process crashed' if it doesn't recognise the class for some
        # reason.
        # But neither reports an error in the exit code, so check explicitly.
        if 'FAILURES' in self.output or 'Process crashed' in self.output:
            raise WorkloadError('Failed to run workload: {}'.format(self.output))

    def update_output(self, context):
        # The 'am instrument' command dumps the instrumentation results into
        # stdout. It also gets written by the autotester to a storage file - on
        # my devices that is /storage/emulated/0/results.log, but I dont know if
        # that's the same for every device.
        #
        # AOSP probably provides standard tooling for parsing this, but I don't
        # know how to use it. Anyway, for this use-case just parsing stdout
        # works fine.

        regex = re.compile('INSTRUMENTATION_STATUS: (?P<key>[\w-]+)=(?P<value>[0-9\.]+)')

        for line in self.output.splitlines():
            match = regex.match(line)
            if match:
                key = match.group('key')
                value = float(match.group('value'))

                name = 'instrumentation_{}'.format(key)
                context.add_metric(name, value, lower_is_better=True)
