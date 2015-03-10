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


# pylint: disable=W0613,E1101,access-member-before-definition,attribute-defined-outside-init
from __future__ import division
import os
import sys
import csv
from collections import OrderedDict

from wlauto import Instrument, Parameter
from wlauto.exceptions import ConfigError, InstrumentError
from wlauto.utils.misc import ensure_directory_exists as _d
from wlauto.utils.types import list_of_ints, list_of_strs

daqpower_path = os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'daq_server', 'src')
sys.path.insert(0, daqpower_path)
try:
    import daqpower.client as daq  # pylint: disable=F0401
    from daqpower.config import DeviceConfiguration, ServerConfiguration, ConfigurationError  # pylint: disable=F0401
except ImportError, e:
    daq, DeviceConfiguration, ServerConfiguration, ConfigurationError = None, None, None, None
    import_error_mesg = e.message
sys.path.pop(0)


UNITS = {
    'power': 'Watts',
    'voltage': 'Volts',
}


class Daq(Instrument):

    name = 'daq'
    description = """
    DAQ instrument obtains the power consumption of the target device's core
    measured by National Instruments Data Acquisition(DAQ) device.

    WA communicates with a DAQ device server running on a Windows machine
    (Please refer to :ref:`daq_setup`) over a network. You must specify the IP
    address and port the server is listening on in the config file as follows ::

        daq_server_host = '10.1.197.176'
        daq_server_port = 45677

    These values will be output by the server when you run it on Windows.

    You must also specify the values of resistors (in Ohms) across which the
    voltages are measured (Please refer to :ref:`daq_setup`). The values should be
    specified as a list with an entry for each resistor, e.g.::

        daq_resistor_values = [0.005, 0.005]

    In addition to this mandatory configuration, you can also optionally specify the
    following::

        :daq_labels: Labels to be used for ports. Defaults to ``'PORT_<pnum>'``, where
                     'pnum' is the number of the port.
        :daq_device_id: The ID under which the DAQ is registered with the driver.
                        Defaults to ``'Dev1'``.
        :daq_v_range: Specifies the voltage range for the SOC voltage channel on the DAQ
                      (please refer to :ref:`daq_setup` for details). Defaults to ``2.5``.
        :daq_dv_range: Specifies the voltage range for the resistor voltage channel on
                       the DAQ (please refer to :ref:`daq_setup` for details).
                       Defaults to ``0.2``.
        :daq_sampling_rate: DAQ sampling rate. DAQ will take this many samples each
                            second. Please note that this maybe limitted by your DAQ model
                            and then number of ports you're measuring (again, see
                            :ref:`daq_setup`). Defaults to ``10000``.
        :daq_channel_map: Represents mapping from  logical AI channel number to physical
                          connector on the DAQ (varies between DAQ models). The default
                          assumes DAQ 6363 and similar with AI channels on connectors
                          0-7 and 16-23.

    """

    parameters = [
        Parameter('server_host', kind=str, default='localhost',
                  description='The host address of the machine that runs the daq Server which the '
                              'insturment communicates with.'),
        Parameter('server_port', kind=int, default=56788,
                  description='The port number for daq Server in which daq insturment communicates '
                              'with.'),
        Parameter('device_id', kind=str, default='Dev1',
                  description='The ID under which the DAQ is registered with the driver.'),
        Parameter('v_range', kind=float, default=2.5,
                  description='Specifies the voltage range for the SOC voltage channel on the DAQ '
                              '(please refer to :ref:`daq_setup` for details).'),
        Parameter('dv_range', kind=float, default=0.2,
                  description='Specifies the voltage range for the resistor voltage channel on '
                              'the DAQ (please refer to :ref:`daq_setup` for details).'),
        Parameter('sampling_rate', kind=int, default=10000,
                  description='DAQ sampling rate. DAQ will take this many samples each '
                              'second. Please note that this maybe limitted by your DAQ model '
                              'and then number of ports you\'re measuring (again, see '
                              ':ref:`daq_setup`)'),
        Parameter('resistor_values', kind=list, mandatory=True,
                  description='The values of resistors (in Ohms) across which the voltages are measured on '
                              'each port.'),
        Parameter('channel_map', kind=list_of_ints, default=(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
                  description='Represents mapping from  logical AI channel number to physical '
                              'connector on the DAQ (varies between DAQ models). The default '
                              'assumes DAQ 6363 and similar with AI channels on connectors '
                              '0-7 and 16-23.'),
        Parameter('labels', kind=list_of_strs,
                  description='List of port labels. If specified, the lenght of the list must match '
                              'the length of ``resistor_values``. Defaults to "PORT_<pnum>", where '
                              '"pnum" is the number of the port.')
    ]

    def initialize(self, context):
        devices = self._execute_command('list_devices')
        if not devices:
            raise InstrumentError('DAQ: server did not report any devices registered with the driver.')
        self._results = OrderedDict()

    def setup(self, context):
        self.logger.debug('Initialising session.')
        self._execute_command('configure', config=self.device_config)

    def slow_start(self, context):
        self.logger.debug('Starting collecting measurements.')
        self._execute_command('start')

    def slow_stop(self, context):
        self.logger.debug('Stopping collecting measurements.')
        self._execute_command('stop')

    def update_result(self, context):  # pylint: disable=R0914
        self.logger.debug('Downloading data files.')
        output_directory = _d(os.path.join(context.output_directory, 'daq'))
        self._execute_command('get_data', output_directory=output_directory)
        for entry in os.listdir(output_directory):
            context.add_iteration_artifact('DAQ_{}'.format(os.path.splitext(entry)[0]),
                                           path=os.path.join('daq', entry),
                                           kind='data',
                                           description='DAQ power measurments.')
            port = os.path.splitext(entry)[0]
            path = os.path.join(output_directory, entry)
            key = (context.spec.id, context.workload.name, context.current_iteration)
            if key not in self._results:
                self._results[key] = {}
            with open(path) as fh:
                reader = csv.reader(fh)
                metrics = reader.next()
                data = [map(float, d) for d in zip(*list(reader))]
                n = len(data[0])
                means = [s / n for s in map(sum, data)]
                for metric, value in zip(metrics, means):
                    metric_name = '{}_{}'.format(port, metric)
                    context.result.add_metric(metric_name, round(value, 3), UNITS[metric])
                    self._results[key][metric_name] = round(value, 3)

    def teardown(self, context):
        self.logger.debug('Terminating session.')
        self._execute_command('close')

    def validate(self):
        if not daq:
            raise ImportError(import_error_mesg)
        self._results = None
        if self.labels:
            if not (len(self.labels) == len(self.resistor_values)):  # pylint: disable=superfluous-parens
                raise ConfigError('Number of DAQ port labels does not match the number of resistor values.')
        else:
            self.labels = ['PORT_{}'.format(i) for i, _ in enumerate(self.resistor_values)]
        self.server_config = ServerConfiguration(host=self.server_host,
                                                 port=self.server_port)
        self.device_config = DeviceConfiguration(device_id=self.device_id,
                                                 v_range=self.v_range,
                                                 dv_range=self.dv_range,
                                                 sampling_rate=self.sampling_rate,
                                                 resistor_values=self.resistor_values,
                                                 channel_map=self.channel_map,
                                                 labels=self.labels)
        try:
            self.server_config.validate()
            self.device_config.validate()
        except ConfigurationError, ex:
            raise ConfigError('DAQ configuration: ' + ex.message)  # Re-raise as a WA error

    def before_overall_results_processing(self, context):
        if self._results:
            headers = ['id', 'workload', 'iteration']
            metrics = sorted(self._results.iteritems().next()[1].keys())
            headers += metrics
            rows = [headers]
            for key, value in self._results.iteritems():
                rows.append(list(key) + [value[m] for m in metrics])

            outfile = os.path.join(context.output_directory, 'daq_power.csv')
            with open(outfile, 'wb') as fh:
                writer = csv.writer(fh)
                writer.writerows(rows)

    def _execute_command(self, command, **kwargs):
        # pylint: disable=E1101
        result = daq.execute_command(self.server_config, command, **kwargs)
        if result.status == daq.Status.OK:
            pass  # all good
        elif result.status == daq.Status.OKISH:
            self.logger.debug(result.message)
        elif result.status == daq.Status.ERROR:
            raise InstrumentError('DAQ: {}'.format(result.message))
        else:
            raise InstrumentError('DAQ: Unexpected result: {} - {}'.format(result.status, result.message))
        return result.data
