#    Copyright 2015 ARM Limited
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
# pylint: disable=unused-argument
from wlauto import Instrument
from wlauto.exceptions import InstrumentError


class ScreenOnInstrument(Instrument):

    name = 'screenon'

    description = """
    Ensure screen is on before each iteration on Android devices.

    A very basic instrument that checks that the screen is on on android devices.

    """

    def initialize(self, context):
        if self.device.platform != 'android':
            raise InstrumentError('screenon instrument currently only supports Android devices.')

    def slow_setup(self, context):   # slow to run before most other setups
        self.device.ensure_screen_is_on()
