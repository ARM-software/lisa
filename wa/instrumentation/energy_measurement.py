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
import os

from devlib import DerivedEnergyMeasurements
from devlib.instrument import CONTINUOUS
from devlib.instrument.energy_probe import EnergyProbeInstrument
from devlib.instrument.daq import DaqInstrument
from devlib.instrument.acmecape import AcmeCapeInstrument
from devlib.utils.misc import which

from wa import Instrument, Parameter
from wa.framework import pluginloader
from wa.framework.plugin import Plugin
from wa.framework.exception import ConfigError
from wa.utils.types import list_of_strings, list_of_ints, list_or_string


class EnergyInstrumentBackend(Plugin):

    name = None
    kind = 'energy_instrument_backend'
    parameters = []

    instrument = None

    def get_parameters(self):
        return {p.name : p for p in self.parameters}

    def validate_parameters(self, params):
        pass


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
        Parameter('iio-device', default='iio:device0',
                  description="""
                  """),
        Parameter('buffer-size', kind=int, default=256,
                  description="""
                  Size of the capture buffer (in KB).
                  """),
    ]

    instrument = AcmeCapeInstrument


class EnergyMeasurement(Instrument):

    name = 'energy_measurement'

    description = """
    This instrument is designed to be used as an interface to the various
    energy measurement instruments located in devlib.
    """

    parameters = [
        Parameter('instrument', kind=str, mandatory=True,
                  allowed_values=['daq', 'energy_probe', 'acme_cape'],
                  description="""
                  Specify the energy instrumentation to be enabled.
                  """),
        Parameter('instrument_parameters', kind=dict, default={},
                   description="""
                   Specify the parameters used to initialize the desired
                   instrumentation.
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
        self.instrumentation = None
        self.measurement_csv = None
        self.loader = loader
        self.backend = self.loader.get_plugin(self.instrument)
        self.params = {}

        if self.backend.instrument.mode != CONTINUOUS:
            msg = '{} instrument does not support continuous measurement collection'
            raise ConfigError(msg.format(self.instrument))

        supported_params = self.backend.get_parameters()
        for name, value in supported_params.iteritems():
            if name in self.instrument_parameters:
                self.params[name] = self.instrument_parameters[name]
            elif value.default:
                self.params[name] = value.default
        self.backend.validate_parameters(self.params)

    def initialize(self, context):
        self.instrumentation = self.backend.instrument(self.target, **self.params)

        for channel in self.channels or []:
            if not self.instrumentation.get_channels(channel):
                raise ConfigError('No channels found for "{}"'.format(channel))

    def setup(self, context):
        self.instrumentation.reset(sites=self.sites,
                                   kinds=self.kinds,
                                   channels=self.channels)

    def start(self, context):
        self.instrumentation.start()

    def stop(self, context):
        self.instrumentation.stop()

    def update_result(self, context):
        outfile = os.path.join(context.output_directory, 'energy_instrument_output.csv')
        self.measurement_csv = self.instrumentation.get_data(outfile)
        context.add_artifact('energy_instrument_output', outfile, 'data')
        self.extract_metrics(context)

    def extract_metrics(self, context):
        derived_measurements = DerivedEnergyMeasurements.process(self.measurement_csv)
        for meas in derived_measurements:
            name = '{}_{}'.format(meas.channel.site, meas.channel.name)
            context.add_metric(name, meas.value, meas.units)
