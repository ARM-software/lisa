#    Copyright 2013-2017 ARM Limited
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


# pylint: disable=W0613,E1101
from __future__ import division
from collections import defaultdict
import os

from devlib import DerivedEnergyMeasurements
from devlib.instrument import CONTINUOUS
from devlib.instrument.energy_probe import EnergyProbeInstrument
from devlib.instrument.daq import DaqInstrument
from devlib.instrument.acmecape import AcmeCapeInstrument
from devlib.instrument.monsoon import MonsoonInstrument
from devlib.platform.arm import JunoEnergyInstrument
from devlib.utils.misc import which

from wa import Instrument, Parameter
from wa.framework import pluginloader
from wa.framework.plugin import Plugin
from wa.framework.exception import ConfigError, InstrumentError
from wa.utils.types import list_of_strings, list_of_ints, list_or_string, obj_dict, identifier


class EnergyInstrumentBackend(Plugin):

    name = None
    kind = 'energy_instrument_backend'
    parameters = []

    instrument = None

    def get_parameters(self):
        return {p.name : p for p in self.parameters}

    def validate_parameters(self, params):
        pass

    def get_instruments(self, target, **kwargs):
        """
        Get a dict mapping device keys to an Instruments

        Typically there is just a single device/instrument, in which case the
        device key is arbitrary.
        """
        return {None: self.instrument(target, **kwargs)}

class DAQBackend(EnergyInstrumentBackend):

    name = 'daq'

    parameters = [
        Parameter('resistor_values', kind=list_of_ints,
                  description="""
                  The values of resistors (in Ohms) across which the voltages
                  are measured on.
                  """),
        Parameter('labels', kind=list_of_strings,
                  description="""
                  'List of port labels. If specified, the length of the list
                  must match the length of ``resistor_values``.
                  """),
        Parameter('host', kind=str, default='localhost',
                  description="""
                  The host address of the machine that runs the daq Server which
                  the instrument communicates with.
                  """),
        Parameter('port', kind=int, default=45677,
                  description="""
                  The port number for daq Server in which daq instrument
                  communicates with.
                  """),
        Parameter('device_id', kind=str, default='Dev1',
                  description="""
                  The ID under which the DAQ is registered with the driver.
                  """),
        Parameter('v_range', kind=str, default=2.5,
                  description="""
                  Specifies the voltage range for the SOC voltage channel on the
                  DAQ (please refer to :ref:`daq_setup` for details).
                  """),
        Parameter('dv_range', kind=str, default=0.2,
                  description="""
                  Specifies the voltage range for the resistor voltage channel
                  on the DAQ (please refer to :ref:`daq_setup` for details).
                  """),
        Parameter('sample_rate_hz', kind=str, default=10000,
                  description="""
                  Specify the sample rate in Hz.
                  """),
        Parameter('channel_map', kind=list_of_ints,
                  default=(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
                  description="""
                  Represents mapping from  logical AI channel number to physical
                  connector on the DAQ (varies between DAQ models). The default
                  assumes DAQ 6363 and similar with AI channels on connectors
                  0-7 and 16-23.
                  """)
        ]

    instrument = DaqInstrument

    def validate_parameters(self, params):
        if not params.get('resistor_values'):
            raise ConfigError('Mandatory parameter "resistor_values" is not set.')
        if params.get('labels'):
            if len(params.get('labels')) != len(params.get('resistor_values')):
                msg = 'Number of DAQ port labels does not match the number of resistor values.'
                raise ConfigError(msg)


class EnergyProbeBackend(EnergyInstrumentBackend):

    name = 'energy_probe'

    parameters = [
        Parameter('resistor_values', kind=list_of_ints,
                  description="""
                  The values of resistors (in Ohms) across which the voltages
                  are measured on.
                  """),
        Parameter('labels', kind=list_of_strings,
                  description="""
                  'List of port labels. If specified, the length of the list
                  must match the length of ``resistor_values``.
                  """),
        Parameter('device_entry', kind=str, default='/dev/ttyACM0',
                  description="""
                  Path to /dev entry for the energy probe (it should be /dev/ttyACMx)
                  """),
    ]

    instrument = EnergyProbeInstrument

    def validate_parameters(self, params):
        if not params.get('resistor_values'):
            raise ConfigError('Mandatory parameter "resistor_values" is not set.')
        if params.get('labels'):
            if len(params.get('labels')) != len(params.get('resistor_values')):
                msg = 'Number of Energy Probe port labels does not match the number of resistor values.'
                raise ConfigError(msg)


class AcmeCapeBackend(EnergyInstrumentBackend):

    name = 'acme_cape'

    parameters = [
        Parameter('iio-capture', default=which('iio-capture'),
                  description="""
                  Path to the iio-capture binary will be taken from the
                  environment, if not specfied.
                  """),
        Parameter('host', default='baylibre-acme.local',
                  description="""
                  Host name (or IP address) of the ACME cape board.
                  """),
        Parameter('iio-devices', default='iio:device0',
                  kind=list_or_string,
                  description="""
                  """),
        Parameter('buffer-size', kind=int, default=256,
                  description="""
                  Size of the capture buffer (in KB).
                  """),
    ]

    def get_instruments(self, target,
                        iio_capture, host, iio_devices, buffer_size):

        #
        # Devlib's ACME instrument uses iio-capture under the hood, which can
        # only capture data from one IIO device at a time. Devlib's instrument
        # API expects to produce a single CSV file for the Instrument, with a
        # single axis of sample timestamps. These two things cannot be correctly
        # reconciled without changing the devlib Instrument API - get_data would
        # need to be able to return two distinct sets of data.
        #
        # Instead, where required WA will instantiate the ACME instrument
        # multiple times (once for each IIO device), producing two separate CSV
        # files. Aggregated energy info _can_ be meaningfully combined from
        # multiple IIO devices, so we will later sum the derived stats across
        # each of the channels reported by the instruments.
        #

        ret = {}
        for iio_device in iio_devices:
            ret[iio_device] = AcmeCapeInstrument(
                target, iio_capture=iio_capture, host=host,
                iio_device=iio_device, buffer_size=buffer_size)
        return ret

class MonsoonBackend(EnergyInstrumentBackend):

    name = 'monsoon'

    parameters = [
        Parameter('monsoon_bin', default=which('monsoon.py'),
                  description="""
                  Path to monsoon.py executable. If not provided,
                  ``$PATH`` is searched.
                  """),
        Parameter('tty_device', default='/dev/ttyACM0',
                  description="""
                  TTY device to use to communicate with the Power
                  Monitor. If not provided, /dev/ttyACM0 is used.
                  """)
    ]

    instrument = MonsoonInstrument


class JunoEnergyBackend(EnergyInstrumentBackend):

    name = 'juno_readenergy'
    instrument = JunoEnergyInstrument


class EnergyMeasurement(Instrument):

    name = 'energy_measurement'

    description = """
    This instrument is designed to be used as an interface to the various
    energy measurement instruments located in devlib.
    """

    parameters = [
        Parameter('instrument', kind=str, mandatory=True,
                  allowed_values=['daq', 'energy_probe', 'acme_cape', 'monsoon', 'juno_readenergy'],
                  description="""
                  Specify the energy instruments to be enabled.
                  """),
        Parameter('instrument_parameters', kind=dict, default={},
                   description="""
                   Specify the parameters used to initialize the desired
                   instruments.
                   """),
        Parameter('sites', kind=list_or_string,
                  description="""
                  Specify which sites measurements should be collected
                  from, if not specified the measurements will be
                  collected for all available sites.
                  """),
        Parameter('kinds', kind=list_or_string,
                  description="""
                  Specify the kinds of measurements should be collected,
                  if not specified measurements will be
                  collected for all available kinds.
                  """),
        Parameter('channels', kind=list_or_string,
                  description="""
                  Specify the channels to be collected,
                  if not specified the measurements will be
                  collected for all available channels.
                  """),
    ]

    def __init__(self, target, loader=pluginloader, **kwargs):
        super(EnergyMeasurement, self).__init__(target, **kwargs)
        self.instruments = None
        self.measurement_csvs = {}
        self.loader = loader
        self.backend = self.loader.get_plugin(self.instrument)
        self.params = obj_dict()

        instrument_parameters = {identifier(k): v
                                 for k, v in self.instrument_parameters.iteritems()}
        supported_params = self.backend.get_parameters()
        for name, param in supported_params.iteritems():
            value = instrument_parameters.pop(name, None)
            param.set_value(self.params, value)
        if instrument_parameters:
            msg = 'Unexpected parameters for backend "{}": {}'
            raise ConfigError(msg.format(self.instrument, instrument_parameters))
        self.backend.validate_parameters(self.params)

    def initialize(self, context):
        self.instruments = self.backend.get_instruments(self.target, **self.params)

        for instrument in self.instruments.itervalues():
            if not (instrument.mode & CONTINUOUS):
                msg = '{} instrument does not support continuous measurement collection'
                raise ConfigError(msg.format(self.instrument))
            instrument.setup()

        for channel in self.channels or []:
            # Check that the expeccted channels exist.
            # If there are multiple Instruments, they were all constructed with
            # the same channels param, so check them all.
            for instrument in self.instruments.itervalues():
                if not instrument.get_channels(channel):
                    raise ConfigError('No channels found for "{}"'.format(channel))

    def setup(self, context):
        for instrument in self.instruments.itervalues():
            instrument.reset(sites=self.sites,
                             kinds=self.kinds,
                             channels=self.channels)

    def start(self, context):
        for instrument in self.instruments.itervalues():
            instrument.start()

    def stop(self, context):
        for instrument in self.instruments.itervalues():
            instrument.stop()

    def update_output(self, context):
        for device, instrument in self.instruments.iteritems():
            # Append the device key to the filename and artifact name, unless
            # it's None (as it will be for backends with only 1
            # devce/instrument)
            if len(self.instruments) > 1:
                name = 'energy_instrument_output_{}'.format(device)
            else:
                name = 'energy_instrument_output'

            outfile = os.path.join(context.output_directory, '{}.csv'.format(name))
            measurements = instrument.get_data(outfile)
            if not measurements:
                raise InstrumentError("Failed to collect energy data from {}"
                                      .format(self.backend.name))

            self.measurement_csvs[device] = measurements
            context.add_artifact(name, measurements.path, 'data',
                                 classifiers={'device': device})
        self.extract_metrics(context)

    def extract_metrics(self, context):
        metrics_by_name = defaultdict(list)

        for device in self.instruments:
            csv = self.measurement_csvs[device]
            derived_measurements = DerivedEnergyMeasurements.process(csv)
            for meas in derived_measurements:
                # Append the device key to the metric name, unless it's None (as
                # it will be for backends with only 1 devce/instrument)
                if len(self.instruments) > 1:
                    metric_name = '{}_{}'.format(meas.name, device)
                else:
                    metric_name = meas.name
                context.add_metric(metric_name, meas.value, meas.units,
                                   classifiers={'device': device})

                metrics_by_name[meas.name].append(meas)

        # Where we have multiple instruments, add up all the metrics with the
        # same name. For instance with ACME we may have multiple IIO devices
        # each reporting 'device_energy' and 'device_power', so sum them up to
        # produce aggregated energy and power metrics.
        # (Note that metrics_by_name uses the metric name originally reported by
        #  the devlib instrument, before we potentially appended a device key to
        #  it)
        if len(self.instruments) > 1:
            for name, metrics in metrics_by_name.iteritems():
                units = metrics[0].units
                value = sum(m.value for m in metrics)
                context.add_metric(name, value, units)

