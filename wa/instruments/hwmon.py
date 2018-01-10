#    Copyright 2017 ARM Limited
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

from devlib import HwmonInstrument as _Instrument

from wa import Instrument
from wa.framework.instruments import fast

MOMENTARY_QUANTITIES = ['temperature', 'power', 'voltage', 'current', 'fps']
CUMULATIVE_QUANTITIES = ['energy', 'tx', 'tx/rx', 'frames']

class HwmonInstrument(Instrument):
    name = 'hwmon'

    description = """
    Hardware Monitor (hwmon) is a generic Linux kernel subsystem,
    providing access to hardware monitoring components like temperature or
    voltage/current sensors.

    Data from hwmon that are a snapshot of a fluctuating value, such as
    temperature and voltage, are reported once at the beginning and once at the
    end of the workload run. Data that are a cumulative total of a quantity,
    such as energy (which is the cumulative total of power consumption), are
    reported as the difference between the values at the beginning and at the
    end of the workload run.

    There is currently no functionality to filter sensors: all of the available
    hwmon data will be reported.
    """

    def initialize(self, context):
        self.instrument = _Instrument(self.target)

    def setup(self, context):
        self.instrument.reset()

    @fast
    def start(self, context):
        self.before = self.instrument.take_measurement()

    @fast
    def stop(self, context):
        self.after = self.instrument.take_measurement()

    def update_output(self, context):
        measurements_before = {m.channel.label: m for m in self.before}
        measurements_after = {m.channel.label: m for m in self.after}

        if measurements_before.keys() != measurements_after.keys():
            self.logger.warning(
                'hwmon before/after measurements returned different entries!')

        for label, measurement_after in measurements_after.iteritems():
            if label not in measurements_before:
                continue # We've already warned about this
            measurement_before = measurements_before[label]

            if measurement_after.channel.kind in MOMENTARY_QUANTITIES:
                context.add_metric('{}_before'.format(label),
                                   measurement_before.value,
                                   measurement_before.channel.units)
                context.add_metric('{}_after'.format(label),
                                   measurement_after.value,
                                   measurement_after.channel.units)

            elif measurement_after.channel.kind in CUMULATIVE_QUANTITIES:
                diff = measurement_after.value - measurement_before.value
                context.add_metric(label, diff, measurement_after.channel.units)

            else:
                self.logger.warning(
                    "Don't know what to do with hwmon channel '{}'"
                    .format(measurement_after.channel))

    def teardown(self, context):
        self.instrument.teardown()
