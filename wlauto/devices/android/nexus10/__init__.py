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


import time

from wlauto import AndroidDevice, Parameter


class Nexus10Device(AndroidDevice):

    name = 'Nexus10'
    description = """
    Nexus10 is a 10 inch tablet device, which has dual-core A15.

    To be able to use Nexus10 in WA, the following must be true:

        - USB Debugging Mode is enabled.
        - Generate USB debugging authorisation for the host machine

    """

    default_working_directory = '/sdcard/working'
    has_gpu = True
    max_cores = 2

    parameters = [
        Parameter('core_names', default=['A15', 'A15'], override=True),
        Parameter('core_clusters', default=[0, 0], override=True),
    ]

    def initialize(self, context):
        time.sleep(self.long_delay)
        self.execute('svc power stayon true', check_exit_code=False)
        time.sleep(self.long_delay)
        self.execute('input keyevent 82')
