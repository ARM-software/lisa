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
from __future__ import division
import logging
import collections

from past.builtins import basestring

from devlib.utils.csvutil import csvreader
from devlib.utils.types import numeric
from devlib.utils.types import identifier


# Channel modes describe what sort of measurement the instrument supports.
# Values must be powers of 2
INSTANTANEOUS = 1
CONTINUOUS = 2

MEASUREMENT_TYPES = {}  # populated further down


class MeasurementType(object):

    def __init__(self, name, units, category=None, conversions=None):
        self.name = name
        self.units = units
        self.category = category
        self.conversions = {}
        if conversions is not None:
            for key, value in conversions.items():
                if not callable(value):
                    msg = 'Converter must be callable; got {} "{}"'
                    raise ValueError(msg.format(type(value), value))
                self.conversions[key] = value

    def convert(self, value, to):
        if isinstance(to, basestring) and to in MEASUREMENT_TYPES:
            to = MEASUREMENT_TYPES[to]
        if not isinstance(to, MeasurementType):
            msg = 'Unexpected conversion target: "{}"'
            raise ValueError(msg.format(to))
        if to.name == self.name:
            return value
        if not to.name in self.conversions:
            msg = 'No conversion from {} to {} available'
            raise ValueError(msg.format(self.name, to.name))
        return self.conversions[to.name](value)

    # pylint: disable=undefined-variable
    def __cmp__(self, other):
        if isinstance(other, MeasurementType):
            other = other.name
        return cmp(self.name, other)

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.category:
            text = 'MeasurementType({}, {}, {})'
            return text.format(self.name, self.units, self.category)
        else:
            text = 'MeasurementType({}, {})'
            return text.format(self.name, self.units)


# Standard measures. In order to make sure that downstream data processing is not tied
# to particular insturments (e.g. a particular method of mearuing power), instruments
# must, where possible, resport their measurments formatted as on of the standard types
# defined here.
_measurement_types = [
    # For whatever reason, the type of measurement could not be established.
    MeasurementType('unknown', None),

    # Generic measurements
    MeasurementType('count', 'count'),
    MeasurementType('percent', 'percent'),

    # Time measurement. While there is typically a single "canonical" unit
    # used for each type of measurmenent, time may be measured to a wide variety
    # of events occuring at a wide range of scales. Forcing everying into a
    # single scale will lead to inefficient and awkward to work with result tables.
    # Coversion functions between the formats are specified, so that downstream
    # processors that expect all times time be at a particular scale can automatically
    # covert without being familar with individual instruments.
    MeasurementType('time', 'seconds', 'time',
        conversions={
            'time_us': lambda x: x * 1e6,
            'time_ms': lambda x: x * 1e3,
            'time_ns': lambda x: x * 1e9,
        }
    ),
    MeasurementType('time_us', 'microseconds', 'time',
        conversions={
            'time': lambda x: x / 1e6,
            'time_ms': lambda x: x / 1e3,
            'time_ns': lambda x: x * 1e3,
        }
    ),
    MeasurementType('time_ms', 'milliseconds', 'time',
        conversions={
            'time': lambda x: x / 1e3,
            'time_us': lambda x: x * 1e3,
            'time_ns': lambda x: x * 1e6,
        }
    ),
    MeasurementType('time_ns', 'nanoseconds', 'time',
    conversions={
        'time': lambda x: x / 1e9,
        'time_ms': lambda x: x / 1e6,
        'time_us': lambda x: x / 1e3,
        }
    ),

    # Measurements related to thermals.
    MeasurementType('temperature', 'degrees', 'thermal'),

    # Measurements related to power end energy consumption.
    MeasurementType('power', 'watts', 'power/energy'),
    MeasurementType('voltage', 'volts', 'power/energy'),
    MeasurementType('current', 'amps', 'power/energy'),
    MeasurementType('energy', 'joules', 'power/energy'),

    # Measurments realted to data transfer, e.g. neworking,
    # memory, or backing storage.
    MeasurementType('tx', 'bytes', 'data transfer'),
    MeasurementType('rx', 'bytes', 'data transfer'),
    MeasurementType('tx/rx', 'bytes', 'data transfer'),

    MeasurementType('fps', 'fps', 'ui render'),
    MeasurementType('frames', 'frames', 'ui render'),
]
for m in _measurement_types:
    MEASUREMENT_TYPES[m.name] = m


class Measurement(object):

    __slots__ = ['value', 'channel']

    @property
    def name(self):
        return '{}_{}'.format(self.channel.site, self.channel.kind)

    @property
    def units(self):
        return self.channel.units

    def __init__(self, value, channel):
        self.value = value
        self.channel = channel

    # pylint: disable=undefined-variable
    def __cmp__(self, other):
        if hasattr(other, 'value'):
            return cmp(self.value, other.value)
        else:
            return cmp(self.value, other)

    def __str__(self):
        if self.units:
            return '{}: {} {}'.format(self.name, self.value, self.units)
        else:
            return '{}: {}'.format(self.name, self.value)

    __repr__ = __str__


class MeasurementsCsv(object):

    def __init__(self, path, channels=None, sample_rate_hz=None):
        self.path = path
        self.channels = channels
        self.sample_rate_hz = sample_rate_hz
        if self.channels is None:
            self._load_channels()
        headings = [chan.label for chan in self.channels]
        self.data_tuple = collections.namedtuple('csv_entry',
                                                 map(identifier, headings))

    def measurements(self):
        return list(self.iter_measurements())

    def iter_measurements(self):
        for row in self._iter_rows():
            values = map(numeric, row)
            yield [Measurement(v, c) for (v, c) in zip(values, self.channels)]

    def values(self):
        return list(self.iter_values())

    def iter_values(self):
        for row in self._iter_rows():
            values = list(map(numeric, row))
            yield self.data_tuple(*values)

    def _load_channels(self):
        header = []
        with csvreader(self.path) as reader:
            header = next(reader)

        self.channels = []
        for entry in header:
            for mt in MEASUREMENT_TYPES:
                suffix = '_{}'.format(mt)
                if entry.endswith(suffix):
                    site = entry[:-len(suffix)]
                    measure = mt
                    break
            else:
                if entry in MEASUREMENT_TYPES:
                    site = None
                    measure = entry
                else:
                    site = entry
                    measure = 'unknown'

            chan = InstrumentChannel(site, measure)
            self.channels.append(chan)

    # pylint: disable=stop-iteration-return
    def _iter_rows(self):
        with csvreader(self.path) as reader:
            next(reader)  # headings
            for row in reader:
                yield row


class InstrumentChannel(object):

    @property
    def label(self):
        if self.site is not None:
            return '{}_{}'.format(self.site, self.kind)
        return self.kind

    name = label

    @property
    def kind(self):
        return self.measurement_type.name

    @property
    def units(self):
        return self.measurement_type.units

    def __init__(self, site, measurement_type, **attrs):
        self.site = site
        if isinstance(measurement_type, MeasurementType):
            self.measurement_type = measurement_type
        else:
            try:
                self.measurement_type = MEASUREMENT_TYPES[measurement_type]
            except KeyError:
                raise ValueError('Unknown measurement type:  {}'.format(measurement_type))
        for atname, atvalue in attrs.items():
            setattr(self, atname, atvalue)

    def __str__(self):
        if self.name == self.label:
            return 'CHAN({})'.format(self.label)
        else:
            return 'CHAN({}, {})'.format(self.name, self.label)

    __repr__ = __str__


class Instrument(object):

    mode = 0

    def __init__(self, target):
        self.target = target
        self.logger = logging.getLogger(self.__class__.__name__)
        self.channels = collections.OrderedDict()
        self.active_channels = []
        self.sample_rate_hz = None

    # channel management

    def list_channels(self):
        return list(self.channels.values())

    def get_channels(self, measure):
        if hasattr(measure, 'name'):
            measure = measure.name
        return [c for c in self.list_channels() if c.kind == measure]

    def add_channel(self, site, measure, **attrs):
        chan = InstrumentChannel(site, measure, **attrs)
        self.channels[chan.label] = chan

    # initialization and teardown

    def setup(self, *args, **kwargs):
        pass

    def teardown(self):
        pass

    def reset(self, sites=None, kinds=None, channels=None):
        if channels is not None:
            if sites is not None or kinds is not None:
                raise ValueError('sites and kinds should not be set if channels is set')

            try:
                self.active_channels = [self.channels[ch] for ch in channels]
            except KeyError as e:
                msg = 'Unexpected channel "{}"; must be in {}'
                raise ValueError(msg.format(e, self.channels.keys()))
        elif sites is None and kinds is None:
            self.active_channels = sorted(self.channels.values(), key=lambda x: x.label)
        else:
            if isinstance(sites, basestring):
                sites = [sites]
            if isinstance(kinds, basestring):
                kinds = [kinds]

            wanted = lambda ch: ((kinds is None or ch.kind in kinds) and
                                  (sites is None or ch.site in sites))
            self.active_channels = list(filter(wanted, self.channels.values()))

    # instantaneous

    def take_measurement(self):
        pass

    # continuous

    def start(self):
        pass

    def stop(self):
        pass

    # pylint: disable=no-self-use
    def get_data(self, outfile):
        pass

    def get_raw(self):
        return []
