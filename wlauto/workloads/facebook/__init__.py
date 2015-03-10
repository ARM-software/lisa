#    Copyright 2013-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os
import time
import sys

from wlauto import AndroidUiAutoBenchmark
from wlauto import UiAutomatorWorkload
from wlauto import AndroidBenchmark


class Facebook(AndroidUiAutoBenchmark):

    name = 'facebook'
    description = """
    Uses com.facebook.patana apk for facebook workload.
    This workload does the following activities in facebook

        Login to facebook account.
        Send a message.
        Check latest notification.
        Search particular user account and visit his/her facebook account.
        Find friends.
        Update the facebook status

    [NOTE:  This workload starts disableUpdate workload as a part of setup to
    disable online updates, which helps to tackle problem of uncertain
    behavier during facebook workload run.]

    """
    package = 'com.facebook.katana'
    activity = '.LoginActivity'

    #'du' specify 'disable update'
    du_activity = 'com.android.vending/.AssetBrowserActivity'
    du_method_string = 'com.arm.wlauto.uiauto.facebook.UiAutomation#disableUpdate'
    du_jar_file = '/data/local/wa_usecases/com.arm.wlauto.uiauto.facebook.jar'
    du_run_timeout = 4 * 60
    du_working_dir = '/data/local/wa_usecases'
    du_apk_file = '/disableupdateapk/com.android.vending-4.3.10.apk'
    DELAY = 5

    def setup(self, context):
        UiAutomatorWorkload.setup(self, context)

        #Start the play store activity
        self.device.execute('am start {}'.format(self.du_activity))

        #Creating command
        command = 'uiautomator runtest {} -e workdir {} -c {}'.format(self.du_jar_file,
                                                                      self.du_working_dir,
                                                                      self.du_method_string)

        #Start the disable update workload
        self.device.execute(command, self.du_run_timeout)
        time.sleep(self.DELAY)

        #Stop the play store activity
        self.device.execute('am force-stop com.android.vending')

        AndroidBenchmark.setup(self, context)

    def update_result(self, context):
        super(Facebook, self).update_result(context)

    def teardown(self, context):
        pass

