#    Copyright 2014-2015 ARM Limited
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


# pylint: disable=W0613,W0201
import os
import csv
import time
import threading
import logging
from operator import itemgetter

from wlauto import Instrument, File, Parameter
from wlauto.exceptions import InstrumentError


class JunoEnergy(Instrument):

    name = 'juno_energy'
    description = """
    Collects internal energy meter measurements from Juno development board.

    This instrument was created because (at the time of creation) Juno's energy
    meter measurements aren't exposed through HWMON or similar standardized mechanism,
    necessitating  a dedicated instrument to access them.

    This instrument, and the ``readenergy`` executable it relies on are very much tied
    to the Juno platform and are not expected to work on other boards.

    """

    parameters = [
        Parameter('period', kind=float, default=0.1,
                  description='Specifies the time, in Seconds, between polling energy counters.'),
    ]

    def on_run_init(self, context):
        local_file = context.resolver.get(File(self, 'readenergy'))
        self.device.killall('readenergy', as_root=True)
        self.readenergy = self.device.install(local_file)

    def setup(self, context):
        self.host_output_file = os.path.join(context.output_directory, 'energy.csv')
        self.device_output_file = self.device.path.join(self.device.working_directory, 'energy.csv')
        self.command = '{} -o {}'.format(self.readenergy, self.device_output_file)
        self.device.killall('readenergy', as_root=True)

    def start(self, context):
        self.device.kick_off(self.command)

    def stop(self, context):
        self.device.killall('readenergy', signal='TERM', as_root=True)

    def update_result(self, context):
        self.device.pull_file(self.device_output_file, self.host_output_file)
        context.add_artifact('junoenergy', self.host_output_file, 'data')

    def teardown(self, conetext):
        self.device.delete_file(self.device_output_file)

    def validate(self):
        if self.device.name.lower() != 'juno':
            message = 'juno_energy instrument is only supported on juno devices; found {}'
            raise InstrumentError(message.format(self.device.name))

