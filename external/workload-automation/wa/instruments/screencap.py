#    Copyright 2018 ARM Limited
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

from devlib.collector.screencapture import ScreenCaptureCollector

from wa import Instrument, Parameter


class ScreenCaptureInstrument(Instrument):

    name = 'screen_capture'
    description = """
    A simple instrument which captures the screen on the target devices with a user-specified period.

    Please note that if a too short period is specified, then this
    instrument will capture the screen as fast as possible, rather
    than at the specified periodicity.
    """

    parameters = [
        Parameter('period', kind=int, default=10,
                  description="""
                  Period (in seconds) at which to capture the screen on the target.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(ScreenCaptureInstrument, self).__init__(target, **kwargs)
        self.collector = None

    def setup(self, context):
        # We need to create a directory for the captured screenshots
        output_path = os.path.join(context.output_directory, "screen-capture")
        os.mkdir(output_path)
        self.collector = ScreenCaptureCollector(self.target,
                                                self.period)
        self.collector.set_output(output_path)
        self.collector.reset()

    def start(self, context):  # pylint: disable=unused-argument
        self.collector.start()

    def stop(self, context):  # pylint: disable=unused-argument
        self.collector.stop()
