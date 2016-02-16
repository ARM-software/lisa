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
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from string import ascii_lowercase

from multiprocessing import Process, Queue

from wlauto import Instrument, Parameter
from wlauto.core import signal
from wlauto.exceptions import ConfigError, InstrumentError, DeviceError
from wlauto.utils.misc import ensure_directory_exists as _d
from wlauto.utils.types import list_of_ints, list_of_strs, boolean

# pylint: disable=wrong-import-position,wrong-import-order
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
    'energy': 'Joules',
    'power': 'Watts',
    'voltage': 'Volts',
}


GPIO_ROOT = '/sys/class/gpio'
TRACE_MARKER_PATH = '/sys/kernel/debug/tracing/trace_marker'


def dict_or_bool(value):
    """
    Ensures that either a dictionary or a boolean is used as a parameter.
    """
    if isinstance(value, dict):
        return value
    return boolean(value)


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
                  global_alias='daq_server_host',
                  description='The host address of the machine that runs the daq Server which the '
                              'insturment communicates with.'),
        Parameter('server_port', kind=int, default=45677,
                  global_alias='daq_server_port',
                  description='The port number for daq Server in which daq insturment communicates '
                              'with.'),
        Parameter('device_id', kind=str, default='Dev1',
                  global_alias='daq_device_id',
                  description='The ID under which the DAQ is registered with the driver.'),
        Parameter('v_range', kind=float, default=2.5,
                  global_alias='daq_v_range',
                  description='Specifies the voltage range for the SOC voltage channel on the DAQ '
                              '(please refer to :ref:`daq_setup` for details).'),
        Parameter('dv_range', kind=float, default=0.2,
                  global_alias='daq_dv_range',
                  description='Specifies the voltage range for the resistor voltage channel on '
                              'the DAQ (please refer to :ref:`daq_setup` for details).'),
        Parameter('sampling_rate', kind=int, default=10000,
                  global_alias='daq_sampling_rate',
                  description='DAQ sampling rate. DAQ will take this many samples each '
                              'second. Please note that this maybe limitted by your DAQ model '
                              'and then number of ports you\'re measuring (again, see '
                              ':ref:`daq_setup`)'),
        Parameter('resistor_values', kind=list, mandatory=True,
                  global_alias='daq_resistor_values',
                  description='The values of resistors (in Ohms) across which the voltages are measured on '
                              'each port.'),
        Parameter('channel_map', kind=list_of_ints, default=(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
                  global_alias='daq_channel_map',
                  description='Represents mapping from  logical AI channel number to physical '
                              'connector on the DAQ (varies between DAQ models). The default '
                              'assumes DAQ 6363 and similar with AI channels on connectors '
                              '0-7 and 16-23.'),
        Parameter('labels', kind=list_of_strs,
                  global_alias='daq_labels',
                  description='List of port labels. If specified, the lenght of the list must match '
                              'the length of ``resistor_values``. Defaults to "PORT_<pnum>", where '
                              '"pnum" is the number of the port.'),
        Parameter('negative_samples', default='keep', allowed_values=['keep', 'zero', 'drop', 'abs'],
                  global_alias='daq_negative_samples',
                  description="""
                  Specifies how negative power samples should be handled. The following
                  methods are possible:

                    :keep: keep them as they are
                    :zero: turn negative values to zero
                    :drop: drop samples if they contain negative values. *warning:* this may result in
                           port files containing different numbers of samples
                    :abs: take the absoulte value of negave samples

                  """),
        Parameter('gpio_sync', kind=int, constraint=lambda x: x > 0,
                  description="""
                  If specified, the instrument will simultaneously set the
                  specified GPIO pin high and put a marker into ftrace. This is
                  to facillitate syncing kernel trace events to DAQ power
                  trace.
                  """),
        Parameter('merge_channels', kind=dict_or_bool, default=False,
                  description="""
                  If set to ``True``, channels with consecutive letter suffixes will be summed.
                  e.g. If you have channels A7a, A7b, A7c, A15a, A15b they will be summed to A7, A15

                  You can also manually specify the name of channels to be merged and the name of the
                  result like so:

                  merge_channels:
                       A15: [A15dvfs, A15ram]
                       NonCPU: [GPU, RoS, Mem]

                  In the above exaples the DAQ channels labeled A15a and A15b will be summed together
                  with the results being saved as 'channel' ''a''. A7, GPU and RoS will be summed to 'c'
                  """)
    ]

    def initialize(self, context):
        status, devices = self._execute_command('list_devices')
        if status == daq.Status.OK and not devices:
            raise InstrumentError('DAQ: server did not report any devices registered with the driver.')
        self._results = OrderedDict()
        self.gpio_path = None
        if self.gpio_sync:
            if not self.device.file_exists(GPIO_ROOT):
                raise InstrumentError('GPIO sysfs not enabled on the device.')
            try:
                export_path = self.device.path.join(GPIO_ROOT, 'export')
                self.device.write_value(export_path, self.gpio_sync, verify=False)
                pin_root = self.device.path.join(GPIO_ROOT, 'gpio{}'.format(self.gpio_sync))
                direction_path = self.device.path.join(pin_root, 'direction')
                self.device.write_value(direction_path, 'out')
                self.gpio_path = self.device.path.join(pin_root, 'value')
                self.device.write_value(self.gpio_path, 0, verify=False)
                signal.connect(self.insert_start_marker, signal.BEFORE_WORKLOAD_EXECUTION, priority=11)
                signal.connect(self.insert_stop_marker, signal.AFTER_WORKLOAD_EXECUTION, priority=11)
            except DeviceError as e:
                raise InstrumentError('Could not configure GPIO on device: {}'.format(e))

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

        if self.merge_channels:
            self._merge_channels(context)

        for entry in os.listdir(output_directory):
            context.add_iteration_artifact('DAQ_{}'.format(os.path.splitext(entry)[0]),
                                           path=os.path.join('daq', entry),
                                           kind='data',
                                           description='DAQ power measurments.')
            port = os.path.splitext(entry)[0]
            path = os.path.join(output_directory, entry)
            key = (context.spec.id, context.spec.label, context.current_iteration)
            if key not in self._results:
                self._results[key] = {}

            temp_file = os.path.join(tempfile.gettempdir(), entry)
            writer, wfh = None, None

            with open(path) as fh:
                if self.negative_samples != 'keep':
                    wfh = open(temp_file, 'wb')
                    writer = csv.writer(wfh)

                reader = csv.reader(fh)
                metrics = reader.next()
                if writer:
                    writer.writerow(metrics)
                self._metrics |= set(metrics)

                rows = _get_rows(reader, writer, self.negative_samples)
                data = zip(*rows)

                if writer:
                    wfh.close()
                    shutil.move(temp_file, os.path.join(output_directory, entry))

                n = len(data[0])
                means = [s / n for s in map(sum, data)]
                for metric, value in zip(metrics, means):
                    metric_name = '{}_{}'.format(port, metric)
                    context.result.add_metric(metric_name, round(value, 3), UNITS[metric])
                    self._results[key][metric_name] = round(value, 3)
                energy = sum(data[metrics.index('power')]) * (self.sampling_rate / 1000000)
                context.result.add_metric('{}_energy'.format(port), round(energy, 3), UNITS['energy'])

    def teardown(self, context):
        self.logger.debug('Terminating session.')
        self._execute_command('close')

    def finalize(self, context):
        if self.gpio_path:
            unexport_path = self.device.path.join(GPIO_ROOT, 'unexport')
            self.device.write_value(unexport_path, self.gpio_sync, verify=False)

    def validate(self):  # pylint: disable=too-many-branches
        if not daq:
            raise ImportError(import_error_mesg)
        self._results = None
        self._metrics = set()
        if self.labels:
            if len(self.labels) != len(self.resistor_values):
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
        self.grouped_suffixes = defaultdict(str)
        if isinstance(self.merge_channels, bool):
            if self.merge_channels:
                # Create a dict of potential prefixes and a list of their suffixes
                grouped_suffixes = {label[:-1]: label for label in sorted(self.labels) if len(label) > 1}
                # Only merge channels if more than one channel has the same prefix and the prefixes
                # are consecutive letters starting with 'a'.
                self.label_map = {}
                for channel, suffixes in grouped_suffixes.iteritems():
                    if len(suffixes) > 1:
                        if "".join([s[-1] for s in suffixes]) in ascii_lowercase[:len(suffixes)]:
                            self.label_map[channel] = suffixes

        elif isinstance(self.merge_channels, dict):
            # Check if given channel names match labels
            for old_names in self.merge_channels.values():
                for name in old_names:
                    if name not in self.labels:
                        raise ConfigError("No channel with label {} specified".format(name))
            self.label_map = self.merge_channels  # pylint: disable=redefined-variable-type
            self.merge_channels = True
        else:  # Should never reach here
            raise AssertionError("``merge_channels`` is of invalid type")

    def before_overall_results_processing(self, context):
        if self._results:
            headers = ['id', 'workload', 'iteration']
            metrics = ['{}_{}'.format(p, m) for p in self.labels for m in sorted(self._metrics)]
            headers += metrics
            rows = [headers]
            for key, value in self._results.iteritems():
                rows.append(list(key) + [value[m] for m in metrics])

            outfile = os.path.join(context.output_directory, 'daq_power.csv')
            with open(outfile, 'wb') as fh:
                writer = csv.writer(fh)
                writer.writerows(rows)

    def insert_start_marker(self, context):
        if self.gpio_path:
            command = 'echo DAQ_START_MARKER > {}; echo 1 > {}'.format(TRACE_MARKER_PATH, self.gpio_path)
            self.device.execute(command, as_root=self.device.is_rooted)

    def insert_stop_marker(self, context):
        if self.gpio_path:
            command = 'echo DAQ_STOP_MARKER > {}; echo 0 > {}'.format(TRACE_MARKER_PATH, self.gpio_path)
            self.device.execute(command, as_root=self.device.is_rooted)

    def _execute_command(self, command, **kwargs):
        # pylint: disable=E1101
        q = Queue()
        p = Process(target=_send_daq_command, args=(q, self.server_config, command), kwargs=kwargs)
        p.start()
        result = q.get()
        p.join()
        if result.status == daq.Status.OK:
            pass  # all good
        elif result.status == daq.Status.OKISH:
            self.logger.debug(result.message)
        elif result.status == daq.Status.ERROR:
            raise InstrumentError('DAQ: {}'.format(result.message))
        else:
            raise InstrumentError('DAQ: Unexpected result: {} - {}'.format(result.status, result.message))
        return (result.status, result.data)

    def _merge_channels(self, context):  # pylint: disable=r0914
        output_directory = _d(os.path.join(context.output_directory, 'daq'))
        for name, labels in self.label_map.iteritems():
            summed = None
            for label in labels:
                path = os.path.join(output_directory, "{}.csv".format(label))
                with open(path) as fh:
                    reader = csv.reader(fh)
                    metrics = reader.next()
                    rows = _get_rows(reader, None, self.negative_samples)
                    if summed:
                        summed = [[x + y for x, y in zip(a, b)] for a, b in zip(rows, summed)]
                    else:
                        summed = rows
            output_path = os.path.join(output_directory, "{}.csv".format(name))
            with open(output_path, 'wb') as wfh:
                writer = csv.writer(wfh)
                writer.writerow(metrics)
                for row in summed:
                    writer.writerow(row)


def _send_daq_command(q, *args, **kwargs):
    result = daq.execute_command(*args, **kwargs)
    q.put(result)


def _get_rows(reader, writer, negative_samples):
    rows = []
    for row in reader:
        row = map(float, row)
        if negative_samples == 'keep':
            rows.append(row)
        elif negative_samples == 'zero':
            def nonneg(v):
                return v if v >= 0 else 0
            rows.append([nonneg(v) for v in row])
        elif negative_samples == 'drop':
            if all(v >= 0 for v in row):
                rows.append(row)
        elif negative_samples == 'abs':
            rows.append([abs(v) for v in row])
        else:
            raise AssertionError(negative_samples)  # should never get here
        if writer:
            writer.writerow(row)
    return rows
