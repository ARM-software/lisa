#    Copyright 2015-2017 ARM Limited
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

from devlib.instrument import MeasurementType, MEASUREMENT_TYPES


class DerivedMetric(object):

    __slots__ = ['name', 'value', 'measurement_type']

    @property
    def units(self):
        return self.measurement_type.units

    def __init__(self, name, value, measurement_type):
        self.name = name
        self.value = value
        if isinstance(measurement_type, MeasurementType):
            self.measurement_type = measurement_type
        else:
            try:
                self.measurement_type = MEASUREMENT_TYPES[measurement_type]
            except KeyError:
                msg = 'Unknown measurement type:  {}'
                raise ValueError(msg.format(measurement_type))

    def __str__(self):
        if self.units:
            return '{}: {} {}'.format(self.name, self.value, self.units)
        else:
            return '{}: {}'.format(self.name, self.value)

    # pylint: disable=undefined-variable
    def __cmp__(self, other):
        if hasattr(other, 'value'):
            return cmp(self.value, other.value)
        else:
            return cmp(self.value, other)

    __repr__ = __str__


class DerivedMeasurements(object):

    # pylint: disable=no-self-use,unused-argument
    def process(self, measurements_csv):
        return []

    # pylint: disable=no-self-use
    def process_raw(self, *args):
        return []
