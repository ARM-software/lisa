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

# pylint: disable=E1101

import time

from wlauto import Workload, Parameter


class HomeScreen(Workload):

    name = 'homescreen'
    description = """
    A workload that goes to the home screen and idles for the the
    specified duration.

    """
    supported_platforms = ['android']

    parameters = [
        Parameter('duration', kind=int, default=20,
                  description='Specifies the duration, in seconds, of this workload.'),
    ]

    def setup(self, context):
        self.device.clear_logcat()
        self.device.execute('input keyevent 3')  # press the home key

    def run(self, context):
        time.sleep(self.duration)
