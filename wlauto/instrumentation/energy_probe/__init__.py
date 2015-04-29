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
import os
import subprocess
import signal
import struct
import csv
try:
    import pandas
except ImportError:
    pandas = None

from wlauto import Instrument, Parameter, Executable
from wlauto.exceptions import InstrumentError, ConfigError
from wlauto.utils.types import list_of_numbers


class EnergyProbe(Instrument):

    name = 'energy_probe'
    description = """Collects power traces using the ARM energy probe.

                     This instrument requires ``caiman`` utility to be installed in the workload automation
                     host and be in the PATH. Caiman is part of DS-5 and should be in ``/path/to/DS-5/bin/`` .
                     Energy probe can simultaneously collect energy from up to 3 power rails.

                     To connect the energy probe on a rail, connect the white wire to the pin that is closer to the
                     Voltage source and the black wire to the pin that is closer to the load (the SoC or the device
                     you are probing). Between the pins there should be a shunt resistor of known resistance in the
                     range of 5 to 20 mOhm. The resistance of the shunt resistors is a mandatory parameter
                     ``resistor_values``.

                    .. note:: This instrument can process results a lot faster if python pandas is installed.
                    """

    parameters = [
        Parameter('resistor_values', kind=list_of_numbers, default=[],
                  description="""The value of shunt resistors. This is a mandatory parameter."""),
        Parameter('labels', kind=list, default=[],
                  description="""Meaningful labels for each of the monitored rails."""),
        Parameter('device_entry', kind=str, default='/dev/ttyACM0',
                  description="""Path to /dev entry for the energy probe (it should be /dev/ttyACMx)"""),
    ]

    MAX_CHANNELS = 3

    def __init__(self, device, **kwargs):
        super(EnergyProbe, self).__init__(device, **kwargs)
        self.attributes_per_sample = 3
        self.bytes_per_sample = self.attributes_per_sample * 4
        self.attributes = ['power', 'voltage', 'current']
        for i, val in enumerate(self.resistor_values):
            self.resistor_values[i] = int(1000 * float(val))

    def validate(self):
        if subprocess.call('which caiman', stdout=subprocess.PIPE, shell=True):
            raise InstrumentError('caiman not in PATH. Cannot enable energy probe')
        if not self.resistor_values:
            raise ConfigError('At least one resistor value must be specified')
        if len(self.resistor_values) > self.MAX_CHANNELS:
            raise ConfigError('{} Channels where specified when Energy Probe supports up to {}'
                              .format(len(self.resistor_values), self.MAX_CHANNELS))
        if pandas is None:
            self.logger.warning("pandas package will significantly speed up this instrument")
            self.logger.warning("to install it try: pip install pandas")

    def setup(self, context):
        if not self.labels:
            self.labels = ["PORT_{}".format(channel) for channel, _ in enumerate(self.resistor_values)]
        self.output_directory = os.path.join(context.output_directory, 'energy_probe')
        rstring = ""
        for i, rval in enumerate(self.resistor_values):
            rstring += '-r {}:{} '.format(i, rval)
        self.command = 'caiman -d {} -l {} {}'.format(self.device_entry, rstring, self.output_directory)
        os.makedirs(self.output_directory)

    def start(self, context):
        self.logger.debug(self.command)
        self.caiman = subprocess.Popen(self.command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       preexec_fn=os.setpgrp,
                                       shell=True)

    def stop(self, context):
        os.killpg(self.caiman.pid, signal.SIGTERM)

    def update_result(self, context):  # pylint: disable=too-many-locals
        num_of_channels = len(self.resistor_values)
        processed_data = [[] for _ in xrange(num_of_channels)]
        filenames = [os.path.join(self.output_directory, '{}.csv'.format(label)) for label in self.labels]
        struct_format = '{}I'.format(num_of_channels * self.attributes_per_sample)
        not_a_full_row_seen = False
        with open(os.path.join(self.output_directory, "0000000000"), "rb") as bfile:
            while True:
                data = bfile.read(num_of_channels * self.bytes_per_sample)
                if data == '':
                    break
                try:
                    unpacked_data = struct.unpack(struct_format, data)
                except struct.error:
                    if not_a_full_row_seen:
                        self.logger.warn('possibly missaligned caiman raw data, row contained {} bytes'.format(len(data)))
                        continue
                    else:
                        not_a_full_row_seen = True
                for i in xrange(num_of_channels):
                    index = i * self.attributes_per_sample
                    processed_data[i].append({attr: val for attr, val in
                                              zip(self.attributes, unpacked_data[index:index + self.attributes_per_sample])})
        for i, path in enumerate(filenames):
            with open(path, 'w') as f:
                if pandas is not None:
                    self._pandas_produce_csv(processed_data[i], f)
                else:
                    self._slow_produce_csv(processed_data[i], f)

    # pylint: disable=R0201
    def _pandas_produce_csv(self, data, f):
        dframe = pandas.DataFrame(data)
        dframe = dframe / 1000.0
        dframe.to_csv(f)

    def _slow_produce_csv(self, data, f):
        new_data = []
        for entry in data:
            new_data.append({key: val / 1000.0 for key, val in entry.items()})
        writer = csv.DictWriter(f, self.attributes)
        writer.writeheader()
        writer.writerows(new_data)

