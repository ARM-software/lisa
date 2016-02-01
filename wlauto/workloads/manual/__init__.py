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
# pylint: disable=E1101,W0201,E0203
import os
import time

from wlauto import Workload, Parameter
from wlauto.exceptions import ConfigError
from wlauto.utils.misc import getch
from wlauto.utils.types import boolean


class ManualWorkloadConfig(object):

    default_duration = 30

    def __init__(self,
                 duration=None,  # Seconds
                 user_triggered=None,
                 view=None,
                 enable_logcat=True
                 ):
        self.user_triggered = user_triggered if user_triggered is not None else (False if duration else True)
        self.duration = duration or (None if self.user_triggered else self.default_duration)
        self.view = view
        self.enable_logcat = enable_logcat


class ManualWorkload(Workload):

    name = 'manual'
    description = """
    Yields control to the user, either for a fixed period or based on user input, to perform
    custom operations on the device, about which workload automation does not know of.

    """

    parameters = [
        Parameter('duration', kind=int, default=None,
                  description=('Control of the devices is yielded for the duration (in seconds) specified. '
                               'If not specified, ``user_triggered`` is assumed.')),
        Parameter('user_triggered', kind=boolean, default=None,
                  description="""If ``True``, WA will wait for user input after starting the workload;
                                otherwise fixed duration is expected. Defaults to ``True`` if ``duration``
                                is not specified, and ``False`` otherwise.
                  """),
        Parameter('view', default='SurfaceView',
                  description="""Specifies the View of the workload. This enables instruments that require a
                                 View to be specified, such as the ``fps`` instrument."""),
        Parameter('enable_logcat', kind=boolean,
                  description='If ``True``, ``manual`` workload will collect logcat as part of the results.'),
    ]

    def setup(self, context):
        self.logger.info('Any setup required by your workload should be done now.')
        self.logger.info('As soon as you are done hit any key and wait for the message')
        self.logger.info('"START NOW!" to begin your manual workload.')
        self.logger.info('')
        self.logger.info('hit any key to finalize your setup...')
        getch()

    def run(self, context):
        self.logger.info('START NOW!')
        if self.duration:
            time.sleep(self.duration)
        elif self.user_triggered:
            self.logger.info('')
            self.logger.info('hit any key to end your workload execution...')
            getch()
        else:
            raise ConfigError('Illegal parameters for manual workload')
        self.logger.info('DONE! your results are now being collected!')

    def update_result(self, context):
        if self.enable_logcat:
            logcat_dir = os.path.join(context.output_directory, 'logcat')
            self.device.dump_logcat(logcat_dir)

    def teardown(self, context):
        pass

    def validate(self):
        if self.duration is None:
            if self.user_triggered is None:
                self.user_triggered = True
            elif self.user_triggered is False:
                self.duration = self.default_duration
        if self.user_triggered and self.duration:
            message = 'Manual Workload can either specify duration or be user triggered, but not both'
            raise ConfigError(message)
        if not self.user_triggered and not self.duration:
            raise ConfigError('Either user_triggered must be ``True`` or duration must be > 0.')

        if self.enable_logcat is None:
            self.enable_logcat = self.device.platform == "android"
        elif self.enable_logcat and self.device.platform != "android":
            raise ConfigError("The `enable_logcat` parameter can only be used on Android devices")
