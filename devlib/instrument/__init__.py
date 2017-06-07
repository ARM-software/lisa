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
from __future__ import division
import csv
import logging
import collections

from devlib.utils.types import numeric


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
            for key, value in conversions.iteritems():
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
        if not to.name in self.conversions:
            msg = 'No conversion from {} to {} available'
            raise ValueError(msg.format(self.name, to.name))
        return self.conversions[to.name](value)

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


# Standard measures
_measurement_types = [
    MeasurementType('unknown', None),
    MeasurementType('time', 'seconds',
        conversions={
            'time_us': lambda x: x * 1000,
        }
    ),
    MeasurementType('time_us', 'microseconds',
        conversions={
            'time': lambda x: x / 1000,
        }
    ),
    MeasurementType('temperature', 'degrees'),

    MeasurementType('power', 'watts', 'power/energy'),
    MeasurementType('voltage', 'volts', 'power/energy'),
    MeasurementType('current', 'amps', 'power/energy'),
    MeasurementType('energy', 'joules', 'power/energy'),

    MeasurementType('tx', 'bytes', 'data transfer'),
    MeasurementType('rx', 'bytes', 'data transfer'),
    MeasurementType('tx/rx', 'bytes', 'data transfer'),

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

    def __cmp__(self, other):
        if isinstance(other, Measurement):
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

    def __init__(self, path, channels=None):
        self.path = path
        self.channels = channels
        self._fh = open(path, 'rb')
        if self.channels is None:
            self._load_channels()

    def measurements(self):
        return list(self.itermeasurements())

    def itermeasurements(self):
        self._fh.seek(0)
        reader = csv.reader(self._fh)
        reader.next()  # headings
        for row in reader:
            values = map(numeric, row)
            yield [Measurement(v, c) for (v, c) in zip(values, self.channels)]

    def _load_channels(self):
        self._fh.seek(0)
        reader = csv.reader(self._fh)
        header = reader.next()
        self._fh.seek(0)

        self.channels = []
        for entry in header:
            for mt in MEASUREMENT_TYPES:
                suffix = '_{}'.format(mt)
                if entry.endswith(suffix):
                    site =  entry[:-len(suffix)]
                    measure = mt
                    name = '{}_{}'.format(site, measure)
                    break
            else:
                site = entry
                measure = 'unknown'
                name = entry

            chan = InstrumentChannel(name, site, measure)
            self.channels.append(chan)


class InstrumentChannel(object):

    @property
    def label(self):
        return '{}_{}'.format(self.site, self.kind)

    @property
    def kind(self):
        return self.measurement_type.name

    @property
    def units(self):
        return self.measurement_type.units

    def __init__(self, name, site, measurement_type, **attrs):
        self.name = name
        self.site = site
        if isinstance(measurement_type, MeasurementType):
            self.measurement_type = measurement_type
        else:
            try:
                self.measurement_type = MEASUREMENT_TYPES[measurement_type]
            except KeyError:
                raise ValueError('Unknown measurement type:  {}'.format(measurement_type))
        for atname, atvalue in attrs.iteritems():
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
        return self.channels.values()

    def get_channels(self, measure):
        if hasattr(measure, 'name'):
            measure = measure.name
        return [c for c in self.list_channels() if c.kind == measure]

    def add_channel(self, site, measure, name=None, **attrs):
        if name is None:
            name = '{}_{}'.format(site, measure)
        chan = InstrumentChannel(name, site, measure, **attrs)
        self.channels[chan.label] = chan

    # initialization and teardown

    def setup(self, *args, **kwargs):
        pass

    def teardown(self):
        pass

    def reset(self, sites=None, kinds=None, channels=None):
        if kinds is None and sites is None and channels is None:
            self.active_channels = sorted(self.channels.values(), key=lambda x: x.label)
        else:
            if isinstance(sites, basestring):
                sites = [sites]
            if isinstance(kinds, basestring):
                kinds = [kinds]
            self.active_channels = []
            for chan_name in (channels or []):
                try:
                    self.active_channels.append(self.channels[chan_name])
                except KeyError:
                    msg = 'Unexpected channel "{}"; must be in {}'
                    raise ValueError(msg.format(chan_name, self.channels.keys()))
            for chan in self.channels.values():
                if (kinds is None or chan.kind in kinds) and \
                   (sites is None or chan.site in sites):
                    self.active_channels.append(chan)

    # instantaneous

    def take_measurement(self):
        pass

    # continuous

    def start(self):
        pass

    def stop(self):
        pass

    def get_data(self, outfile):
        pass
