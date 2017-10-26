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

import os
import re
import time
from zipfile import ZipFile

from wa import Parameter, Workload
from wa.framework.exception import WorkloadError

INSTALL_INSTRUCTIONS="""
This workload has incomplete automation support. Please download the APK from
http://www.futuremark.com/downloads/pcmark-android.apk
and install it on the device. Connect the device to the internet, then open the
app on the device, and hit the 'install' button to set up the 'Work v2'
benchmark.
"""

class PcMark(Workload):
    """
    Android PCMark workload

    TODO: This isn't a proper WA workload! It requires that the app is already
    installed set up like so:

    - Install the APK from http://www.futuremark.com/downloads/pcmark-android.apk
    - Open the app and hit "install"

    """
    name = 'pcmark'

    package  = 'com.futuremark.pcmark.android.benchmark'
    activity = 'com.futuremark.gypsum.activity.SplashPageActivity'

    package_names = ['com.google.android.youtube']
    action = 'android.intent.action.VIEW'

    parameters = [
        Parameter('test', default='work', allowed_values=['work'],
                  description='PCMark sub-benchmark to run'),
    ]


    regexps = {
        'start'  : '.*START.*com.futuremark.pcmark.android.benchmark',
        'result': '.*received result for correct code, result file in (?P<path>.*\.zip)'
    }

    def initialize(self, context):
        super(PcMark, self).initialize(context)

        # Need root to get results
        if not self.target.is_rooted:
            raise WorkloadError('PCMark workload requires device to be rooted')

        if not self.target.is_installed(self.package):
            raise WorkloadError('Package not installed. ' + INSTALL_INSTRUCTIONS)

        path = ('/storage/emulated/0/Android/data/{}/files/dlc/pcma-workv2-data'
                .format(self.package))
        if not self.target.file_exists(path):
            raise WorkloadError('"Work v2" benchmark not installed through app. '
                                + INSTALL_INSTRUCTIONS)

    def setup(self, context):
        super(PcMark, self).setup(context)

        self.target.execute('am kill-all')  # kill all *background* activities
        self.target.execute('am start -n {}/{}'.format(self.package, self.activity))
        time.sleep(5)

        # TODO: we clobber the old auto-rotation setting here.
        self.target.set_auto_rotation(False)
        self._saved_screen_rotation = self.target.get_rotation()
        # Move to benchmark run page
        self.target.set_left_rotation() # Needed to make TAB work
        self.target.execute('input keyevent KEYCODE_TAB')
        self.target.execute('input keyevent KEYCODE_TAB')

        self.monitor = self.target.get_logcat_monitor(self.regexps.values())
        self.monitor.start()

    def run(self, context):
        self.target.execute('input keyevent KEYCODE_ENTER')

        self.monitor.wait_for('.*START.*com.futuremark.pcmark.android.benchmark',
                              timeout=20)
        self.logger.info('Detected PCMark start')

        [self.output] = self.monitor.wait_for(self.regexps['result'], timeout=600)

    def extract_results(self, context):
        remote_zip_path = re.match(self.regexps['result'], self.output).group('path')
        local_zip_path = os.path.join(context.output_directory,
                                      self.target.path.basename(remote_zip_path))
        self.logger.info('pulling {} -> {}'.format(remote_zip_path, local_zip_path))
        self.target.pull(remote_zip_path, local_zip_path, as_root=True)

        with ZipFile(local_zip_path, 'r') as archive:
            archive.extractall(context.output_directory)

        xml_path = os.path.join(context.output_directory, 'Result.xml')
        if not os.path.exists(xml_path):
            raise WorkloadError("PCMark results .zip didn't contain Result.xml")
        context.add_artifact('pcmark_result_xml', xml_path, 'data')

        # Fetch workloads names and scores
        score_regex = re.compile('\s*<result_Pcma(?P<name>.*)Score>(?P<score>[0-9]*)<')
        with open(xml_path) as f:
            for line in f:
                match = score_regex.match(line)
                if match:
                    metric_name = 'pcmark_{}'.format(match.group('name'))
                    context.add_metric(metric_name, match.group('score'))


    def teardown(self, context):
        super(PcMark, self).teardown(context)

        self.target.execute('am force-stop {}'.format(self.package))

        self.monitor.stop()
        self.target.set_rotation(int(self._saved_screen_rotation))

