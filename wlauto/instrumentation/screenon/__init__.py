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
import time
import threading

from wlauto import Instrument, Parameter
from wlauto.exceptions import InstrumentError


class ScreenMonitor(threading.Thread):

    def __init__(self, device, polling_period):
        super(ScreenMonitor, self).__init__()
        self.device = device
        self.polling_period = polling_period
        self.stop_event = threading.Event()

    def run(self):
        last_poll = time.time()
        while not self.stop_event.is_set():
            time.sleep(1)
            if (time.time() - last_poll) >= self.polling_period:
                self.device.ensure_screen_is_on()
                last_poll = time.time()

    def stop(self):
        self.stop_event.set()
        self.join()


class ScreenOnInstrument(Instrument):
    # pylint: disable=attribute-defined-outside-init

    name = 'screenon'

    description = """
    Ensure screen is on before each iteration on Android devices.

    A very basic instrument that checks that the screen is on on android devices. Optionally,
    it call poll the device periodically to ensure that the screen is still on.

    """

    parameters = [
        Parameter('polling_period', kind=int,
                  description="""
                  Set this to a non-zero value to enable periodic (every
                  ``polling_period`` seconds) polling of the screen on
                  the device to ensure it is on during a run.
                  """),
    ]

    def initialize(self, context):
        self.monitor = None
        if self.device.platform != 'android':
            raise InstrumentError('screenon instrument currently only supports Android devices.')

    def slow_setup(self, context):   # slow to run before most other setups
        self.device.ensure_screen_is_on()
        if self.polling_period:
            self.monitor = ScreenMonitor(self.device, self.polling_period)
            self.monitor.start()

    def teardown(self, context):
        if self.polling_period:
            self.monitor.stop()

