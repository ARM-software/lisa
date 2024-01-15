#pylint: disable=attribute-defined-outside-init

import collections
import functools
import re
import threading

from past.builtins import basestring

try:
    import iio
except ImportError as e:
    iio_import_failed = True
    iio_import_error  = e
else:
    iio_import_failed = False
import numpy as np
import pandas as pd

from devlib import CONTINUOUS, Instrument, HostError, MeasurementsCsv, TargetError
from devlib.utils.ssh import SshConnection

class IIOINA226Channel(object):

    def __init__(self, iio_channel):

        channel_id   = iio_channel.id
        channel_type = iio_channel.attrs['type'].value

        re_measure = r'(?P<measure>\w+)(?P<index>\d*)$'
        re_dtype = r'le:(?P<sign>\w)(?P<width>\d+)/(?P<size>\d+)>>(?P<align>\d+)'

        match_measure = re.search(re_measure, channel_id)
        match_dtype = re.search(re_dtype, channel_type)

        if not match_measure:
            msg = "IIO channel ID '{}' does not match expected RE '{}'"
            raise ValueError(msg.format(channel_id, re_measure))

        if not match_dtype:
            msg = "'IIO channel type '{}' does not match expected RE '{}'"
            raise ValueError(msg.format(channel_type, re_dtype))

        self.measure = match_measure.group('measure')
        self.iio_dtype = 'int{}'.format(match_dtype.group('width'))
        self.iio_channel = iio_channel
        # Data is reported in amps, volts, watts and microseconds:
        self.iio_scale   = (1. if 'scale' not in iio_channel.attrs
                            else float(iio_channel.attrs['scale'].value))
        self.iio_scale /= 1000
        # As calls to iio_store_buffer will be blocking and probably coming
        # from a loop retrieving samples from the ACME, we want to provide
        # consistency in processing timing between iterations i.e. we want
        # iio_store_buffer to be o(1) for every call (can't have that with []):
        self.sample_buffers = collections.deque()

    def iio_store_buffer_samples(self, iio_buffer):
        # IIO buffers receive and store their data as an interlaced array of
        # samples from all the IIO channels of the IIO device. The IIO library
        # provides a reliable function to extract the samples (bytes, actually)
        # corresponding to a channel from the received buffer; in Python, it is
        # iio.Channel.read(iio.Buffer).
        #
        # NB: As this is called in a potentially tightly timed loop, we do as
        #     little work as possible:
        self.sample_buffers.append(self.iio_channel.read(iio_buffer))

    def iio_get_samples(self, absolute_timestamps=False):
        # Up to this point, the data is not interpreted yet i.e. these are
        # bytearrays. Hence the use of np.dtypes.
        buffers = [np.frombuffer(b, dtype=self.iio_dtype)
                   for b in self.sample_buffers]

        must_shift = (self.measure == 'timestamp' and not absolute_timestamps)
        samples = np.concatenate(buffers)
        return (samples - samples[0] if must_shift else samples) * self.iio_scale

    def iio_forget_samples(self):
        self.sample_buffers.clear()


# Decorators for the attributes of IIOINA226Instrument:

def only_set_to(valid_values, dynamic=False):
    def validating_wrapper(func):
        @functools.wraps(func)
        def wrapper(self, value):
            values = (valid_values if not dynamic
                      else getattr(self, valid_values))
            if value not in values: 
                msg = '{} is invalid; expected values are {}'
                raise ValueError(msg.format(value, valid_values))
            return func(self, value)
        return wrapper
    return validating_wrapper


def with_input_as(wanted_type):
    def typecasting_wrapper(func):
        @functools.wraps(func)
        def wrapper(self, value):
            return func(self, wanted_type(value))
        return wrapper
    return typecasting_wrapper


def _IIODeviceAttr(attr_name, attr_type, writable=False, dyn_vals=None, stat_vals=None):

    def getter(self):
        return attr_type(self.iio_device.attrs[attr_name].value)

    def setter(self, value):
        self.iio_device.attrs[attr_name].value = str(attr_type(value))

    if writable and (dyn_vals or stat_vals):
        vals, dyn = dyn_vals or stat_vals, dyn_vals is not None
        setter = with_input_as(attr_type)(only_set_to(vals, dyn)(setter))

    return property(getter, setter if writable else None)


def _IIOChannelIntTime(chan_name):

    attr_name, attr_type = 'integration_time', float

    def getter(self):
        ch = self.iio_device.find_channel(chan_name)
        return attr_type(ch.attrs[attr_name].value)

    @only_set_to('INTEGRATION_TIMES_AVAILABLE', dynamic=True)
    @with_input_as(attr_type)
    def setter(self, value):
        ch = self.iio_device.find_channel(chan_name)
        ch.attrs[attr_name].value = str(value)

    return property(getter, setter)


def _setify(x):
    return {x} if isinstance(x, basestring) else set(x) #Py3: basestring->str


class IIOINA226Instrument(object):

    IIO_DEVICE_NAME = 'ina226'

    def __init__(self, iio_device):

        if iio_device.name != self.IIO_DEVICE_NAME:
            msg = 'IIO device is {}; expected {}'
            raise TargetError(msg.format(iio_device.name, self.IIO_DEVICE_NAME))

        self.iio_device           = iio_device
        self.absolute_timestamps  = False
        self.high_resolution      = True
        self.buffer_samples_count = None
        self.buffer_is_circular   = False

        self.collector = None
        self.work_done = threading.Event()
        self.collector_exception = None

        self.data = collections.OrderedDict()

        channels = {
                'timestamp': 'timestamp',
                'shunt'    : 'voltage0',
                'voltage'  : 'voltage1', # bus
                'power'    : 'power2',
                'current'  : 'current3',
        }
        self.computable_channels = {'current' : {'shunt'},
                                    'power'   : {'shunt', 'voltage'}}
        self.uncomputable_channels = set(channels) - set(self.computable_channels)
        self.channels = {k: IIOINA226Channel(self.iio_device.find_channel(v))
                         for k, v in channels.items()}
        # We distinguish between "output" channels (as seen by the user of this
        # class) and "hardware" channels (as requested from the INA226).
        # This is necessary because of the 'high_resolution' feature which
        # requires outputting computed channels:
        self.active_channels = set()  # "hardware" channels
        self.wanted_channels = set()  # "output" channels


    # Properties

    OVERSAMPLING_RATIOS_AVAILABLE = (1, 4, 16, 64, 128, 256, 512, 1024)
    INTEGRATION_TIMES_AVAILABLE = _IIODeviceAttr('integration_time_available',
                                                 lambda x: tuple(map(float, x.split())))

    sample_rate_hz     = _IIODeviceAttr('in_sampling_frequency', int)
    shunt_resistor     = _IIODeviceAttr('in_shunt_resistor'    , int, True)
    oversampling_ratio = _IIODeviceAttr('in_oversampling_ratio', int, True,
                                        dyn_vals='OVERSAMPLING_RATIOS_AVAILABLE')

    integration_time_shunt = _IIOChannelIntTime('voltage0')
    integration_time_bus   = _IIOChannelIntTime('voltage1')

    def list_channels(self):
        return self.channels.keys()

    def activate(self, channels=None):
        all_channels = set(self.channels)
        requested_channels = (all_channels if channels is None
                              else _setify(channels))

        unknown = ', '.join(requested_channels - all_channels)
        if unknown:
            raise ValueError('Unknown channel(s): {}'.format(unknown))

        self.wanted_channels |= requested_channels

    def deactivate(self, channels=None):
        unwanted_channels = (self.wanted_channels if channels is None
                             else _setify(channels))

        unknown = ', '.join(unwanted_channels - set(self.channels))
        if unknown:
            raise ValueError('Unknown channel(s): {}'.format(unknown))

        unactive = ', '.join(unwanted_channels - self.wanted_channels)
        if unactive:
            raise ValueError('Already unactive channel(s): {}'.format(unactive))

        self.wanted_channels -= unwanted_channels

    def sample_collector(self):
        class Collector(threading.Thread):
            def run(collector_self):
                for name, ch in self.channels.items():
                    ch.iio_channel.enabled = (name in self.active_channels)

                samples_count = self.buffer_samples_count or self.sample_rate_hz

                iio_buffer = iio.Buffer(self.iio_device, samples_count,
                                        self.buffer_is_circular)
                # NB: This buffer creates a communication pipe to the
                #     BeagleBone (or is it between the BBB and the ACME?)
                #     that locks down any configuration. The IIO drivers
                #     do not limit access when a buffer exists so that
                #     configuring the INA226 (i.e. accessing iio.Device.attrs
                #     or iio.Channel.attrs from iio.Device.channels i.e.
                #     assigning to or reading from any property of this class
                #     or calling its setup or reset methods) will screw up the
                #     whole system and will require rebooting the BBB-ACME board!

                self.collector_exception = None
                try:
                    refilled_once = False
                    while not (refilled_once and self.work_done.is_set()):
                        refilled_once = True
                        iio_buffer.refill()
                        for name in self.active_channels:
                            self.channels[name].iio_store_buffer_samples(iio_buffer)
                except Exception as e:
                    self.collector_exception = e
                finally:
                    del iio_buffer
                    for ch in self.channels.values():
                        ch.enabled = False

        return Collector()

    def start_capturing(self):
        if not self.wanted_channels:
            raise TargetError('No active channel: aborting.')

        self.active_channels = self.wanted_channels.copy()
        if self.high_resolution:
            self.active_channels &= self.uncomputable_channels
            for channel, dependencies in self.computable_channels.items():
                if channel in self.wanted_channels:
                    self.active_channels |= dependencies

        self.work_done.clear()
        self.collector = self.sample_collector()
        self.collector.daemon = True
        self.collector.start()

    def stop_capturing(self):
        self.work_done.set()
        self.collector.join()

        if self.collector_exception:
            raise self.collector_exception

        self.data.clear()
        for channel in self.active_channels:
            ch = self.channels[channel]
            self.data[channel] = ch.iio_get_samples(self.absolute_timestamps)
            ch.iio_forget_samples()

        if self.high_resolution:
            res_ohm = 1e-6 * self.shunt_resistor
            current = self.data['shunt'] / res_ohm
            if 'current' in self.wanted_channels:
                self.data['current'] = current
            if 'power' in self.wanted_channels:
                self.data['power'] = current * self.data['voltage']
            for channel in set(self.data) - self.wanted_channels:
                del self.data[channel]

        self.active_channels.clear()

    def get_data(self):
        return self.data


class BaylibreAcmeInstrument(Instrument):

    mode = CONTINUOUS

    MINIMAL_ACME_SD_IMAGE_VERSION    = (2, 1, 3)
    MINIMAL_ACME_IIO_DRIVERS_VERSION = (0, 6)
    MINIMAL_HOST_IIO_DRIVERS_VERSION = (0, 15)

    def __init__(self, target=None, iio_context=None,
                 use_base_iio_context=False, probe_names=None):

        if iio_import_failed:
            raise HostError('Could not import "iio": {}'.format(iio_import_error))

        super(BaylibreAcmeInstrument, self).__init__(target)

        if isinstance(probe_names, basestring):
            probe_names = [probe_names]

        self.iio_context = (iio_context if not use_base_iio_context
                            else iio.Context(iio_context))

        self.check_version()

        if probe_names is not None:
            if len(probe_names) != len(set(probe_names)):
                msg = 'Probe names should be unique: {}'
                raise ValueError(msg.format(probe_names))

            if len(probe_names) != len(self.iio_context.devices):
                msg = ('There should be as many probe_names ({}) '
                       'as detected probes ({}).')
                raise ValueError(msg.format(len(probe_names),
                                            len(self.iio_context.devices)))

        probes = [IIOINA226Instrument(d) for d in self.iio_context.devices]

        self.probes = (dict(zip(probe_names, probes)) if probe_names
                       else {p.iio_device.id : p for p in probes})
        self.active_probes = set()

        for probe in self.probes:
            for measure in ['voltage', 'power', 'current']:
                self.add_channel(site=probe, measure=measure)
        self.add_channel('timestamp', 'time_us')

        self.data = pd.DataFrame()

    def check_version(self):
        msg = ('The IIO drivers running on {} ({}) are out-of-date; '
               'devlib requires {} or later.')

        if iio.version[:2] < self.MINIMAL_HOST_IIO_DRIVERS_VERSION:
            ver_str = '.'.join(map(str, iio.version[:2]))
            min_str = '.'.join(map(str, self.MINIMAL_HOST_IIO_DRIVERS_VERSION))
            raise HostError(msg.format('this host', ver_str, min_str))

        if self.version[:2] < self.MINIMAL_ACME_IIO_DRIVERS_VERSION:
            ver_str = '.'.join(map(str, self.version[:2]))
            min_str = '.'.join(map(str, self.MINIMAL_ACME_IIO_DRIVERS_VERSION))
            raise TargetError(msg.format('the BBB', ver_str, min_str))

    # properties

    def probes_unique_property(self, property_name):
        probes = self.active_probes or self.probes
        try:
            # This will fail if there is not exactly one single value:
            (value,) = {getattr(self.probes[p], property_name) for p in probes}
        except ValueError:
            msg = 'Probes have different values for {}.'
            raise ValueError(msg.format(property_name) if probes else 'No probe')
        return value

    @property
    def version(self):
        return self.iio_context.version

    @property
    def OVERSAMPLING_RATIOS_AVAILABLE(self):
        return self.probes_unique_property('OVERSAMPLING_RATIOS_AVAILABLE')

    @property
    def INTEGRATION_TIMES_AVAILABLE(self):
        return self.probes_unique_property('INTEGRATION_TIMES_AVAILABLE')

    @property
    def sample_rate_hz(self):
        return self.probes_unique_property('sample_rate_hz')

    @sample_rate_hz.setter
    # This setter is required for compliance with the inherited methods
    def sample_rate_hz(self, value):
        if value is not None:
            raise AttributeError("can't set attribute")

    # initialization and teardown

    def setup(self, shunt_resistor,
              integration_time_bus,
              integration_time_shunt,
              oversampling_ratio,
              buffer_samples_count=None,
              buffer_is_circular=False,
              absolute_timestamps=False,
              high_resolution=True):

        def pseudo_list(v, i):
            try:
                return v[i]
            except TypeError:
                return v

        for i, p in enumerate(self.probes.values()):
            for attr, val in locals().items():
                if attr != 'self':
                    setattr(p, attr, pseudo_list(val, i))

        self.absolute_timestamps = all(pseudo_list(absolute_timestamps, i)
                                       for i in range(len(self.probes)))

    def reset(self, sites=None, kinds=None, channels=None):

        # populate self.active_channels:
        super(BaylibreAcmeInstrument, self).reset(sites, kinds, channels)

        for ch in self.active_channels:
            if ch.site != 'timestamp':
                self.probes[ch.site].activate(['timestamp', ch.kind])
                self.active_probes.add(ch.site)

    def teardown(self):
        del self.active_channels[:]
        self.active_probes.clear()

    def start(self):
        for p in self.active_probes:
            self.probes[p].start_capturing()

    def stop(self):
        for p in self.active_probes:
            self.probes[p].stop_capturing()

        max_rate_probe = max(self.active_probes,
                             key=lambda p: self.probes[p].sample_rate_hz)

        probes_dataframes = {
                probe: pd.DataFrame.from_dict(self.probes[probe].get_data())
                                   .set_index('timestamp')
                for probe in self.active_probes
        }

        for df in probes_dataframes.values():
            df.set_index(pd.to_datetime(df.index, unit='us'), inplace=True)

        final_index = probes_dataframes[max_rate_probe].index

        df = pd.concat(probes_dataframes, axis=1).sort_index()
        df.columns = ['_'.join(c).strip() for c in df.columns.values]

        self.data = df.interpolate('time').reindex(final_index)

        if not self.absolute_timestamps:
            epoch_index = self.data.index.astype(np.int64) // 1000
            self.data.set_index(epoch_index, inplace=True)
            # self.data.index is in [us]
            # columns are in volts, amps and watts

    def get_data(self, outfile=None, **to_csv_kwargs):
        if outfile is None:
            return self.data

        self.data.to_csv(outfile, **to_csv_kwargs)
        return MeasurementsCsv(outfile, self.active_channels)

class BaylibreAcmeLocalInstrument(BaylibreAcmeInstrument):

    def __init__(self, target=None, probe_names=None):

        if iio_import_failed:
            raise HostError('Could not import "iio": {}'.format(iio_import_error))

        super(BaylibreAcmeLocalInstrument, self).__init__(
                        target=target,
                        iio_context=iio.LocalContext(),
                        probe_names=probe_names
        )

class BaylibreAcmeXMLInstrument(BaylibreAcmeInstrument):

    def __init__(self, target=None, xmlfile=None, probe_names=None):

        if iio_import_failed:
            raise HostError('Could not import "iio": {}'.format(iio_import_error))

        super(BaylibreAcmeXMLInstrument, self).__init__(
                        target=target,
                        iio_context=iio.XMLContext(xmlfile),
                        probe_names=probe_names
        )

class BaylibreAcmeNetworkInstrument(BaylibreAcmeInstrument):

    def __init__(self, target=None, hostname=None, probe_names=None):

        if iio_import_failed:
            raise HostError('Could not import "iio": {}'.format(iio_import_error))

        super(BaylibreAcmeNetworkInstrument, self).__init__(
                        target=target,
                        iio_context=iio.NetworkContext(hostname),
                        probe_names=probe_names
        )

        try:
            self.ssh_connection = SshConnection(hostname, username='root', password=None)
        except TargetError as e:
            msg = 'No SSH connexion could be established to {}: {}'
            self.logger.debug(msg.format(hostname, e))
            self.ssh_connection = None

    def check_version(self):
        super(BaylibreAcmeNetworkInstrument, self).check_version()

        cmd = r"""sed -nr 's/^VERSION_ID="(.+)"$/\1/p' < /etc/os-release"""
        try:
            ver_str = self._ssh(cmd).rstrip()
            ver = tuple(map(int, ver_str.split('.')))
        except Exception as e:
            self.logger.debug('Unable to verify ACME SD image version through SSH: {}'.format(e))
        else:
            if ver < self.MINIMAL_ACME_SD_IMAGE_VERSION:
                min_str = '.'.join(map(str, self.MINIMAL_ACME_SD_IMAGE_VERSION))
                msg = ('The ACME SD image for the BBB (ver. {}) is out-of-date; '
                       'devlib requires {} or later.')
                raise TargetError(msg.format(ver_str, min_str))

    def _ssh(self, cmd=''):
        """Connections are assumed to be rare."""
        if self.ssh_connection is None:
            raise TargetError('No SSH connection; see log.')
        return self.ssh_connection.execute(cmd)

    def _reboot(self):
        """Always delete the object after calling its _reboot method"""
        try:
            self._ssh('reboot')
        except:
            pass
