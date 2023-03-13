#    Copyright 2015-2018 ARM Limited
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
import os
import signal
import tempfile
import struct
import subprocess
import sys
from pipes import quote

from devlib.instrument import Instrument, CONTINUOUS, MeasurementsCsv
from devlib.exception import HostError
from devlib.utils.csvutil import csvwriter
from devlib.utils.misc import which


class EnergyProbeInstrument(Instrument):

    mode = CONTINUOUS

    def __init__(self, target, resistor_values,
                 labels=None,
                 device_entry='/dev/ttyACM0',
                 keep_raw=False
                 ):
        super(EnergyProbeInstrument, self).__init__(target)
        self.resistor_values = resistor_values
        self.keep_raw = keep_raw
        if labels is not None:
            self.labels = labels
        else:
            self.labels = ['PORT_{}'.format(i)
                           for i in range(len(resistor_values))]
        self.device_entry = device_entry
        self.caiman = which('caiman')
        if self.caiman is None:
            raise HostError('caiman must be installed on the host '
                            '(see https://github.com/ARM-software/caiman)')
        self.attributes_per_sample = 3
        self.bytes_per_sample = self.attributes_per_sample * 4
        self.attributes = ['power', 'voltage', 'current']
        self.command = None
        self.raw_output_directory = None
        self.process = None
        self.sample_rate_hz = 10000 # Determined empirically
        self.raw_data_file = None

        for label in self.labels:
            for kind in self.attributes:
                self.add_channel(label, kind)

    def reset(self, sites=None, kinds=None, channels=None):
        super(EnergyProbeInstrument, self).reset(sites, kinds, channels)
        self.raw_output_directory = tempfile.mkdtemp(prefix='eprobe-caiman-')
        parts = ['-r {}:{} '.format(i, int(1000 * rval))
                 for i, rval in enumerate(self.resistor_values)]
        rstring = ''.join(parts)
        self.command = '{} -d {} -l {} {}'.format(
            quote(self.caiman), quote(self.device_entry),
            rstring, quote(self.raw_output_directory)
        )
        self.raw_data_file = None

    def start(self):
        self.logger.debug(self.command)
        self.process = subprocess.Popen(self.command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        stdin=subprocess.PIPE,
                                        preexec_fn=os.setpgrp,
                                        shell=True)

    def stop(self):
        self.process.poll()
        if self.process.returncode is not None:
            stdout, stderr = self.process.communicate()
            if sys.version_info[0] == 3:
                stdout = stdout.decode(sys.stdout.encoding or 'utf-8', 'replace')
                stderr = stderr.decode(sys.stdout.encoding or 'utf-8', 'replace')
            raise HostError(
                'Energy Probe: Caiman exited unexpectedly with exit code {}.\n'
                'stdout:\n{}\nstderr:\n{}'.format(self.process.returncode,
                                                  stdout, stderr))
        os.killpg(self.process.pid, signal.SIGINT)

    def get_data(self, outfile):  # pylint: disable=R0914
        all_channels = [c.label for c in self.list_channels()]
        active_channels = [c.label for c in self.active_channels]
        active_indexes = [all_channels.index(ac) for ac in active_channels]

        num_of_ports = len(self.resistor_values)
        struct_format = '{}I'.format(num_of_ports * self.attributes_per_sample)
        not_a_full_row_seen = False
        self.raw_data_file = os.path.join(self.raw_output_directory, '0000000000')

        self.logger.debug('Parsing raw data file: {}'.format(self.raw_data_file))
        with open(self.raw_data_file, 'rb') as bfile:
            with csvwriter(outfile) as writer:
                writer.writerow(active_channels)
                while True:
                    data = bfile.read(num_of_ports * self.bytes_per_sample)
                    if data == '':
                        break
                    try:
                        unpacked_data = struct.unpack(struct_format, data)
                        row = [unpacked_data[i] / 1000 for i in active_indexes]
                        writer.writerow(row)
                    except struct.error:
                        if not_a_full_row_seen:
                            self.logger.warning('possibly missaligned caiman raw data, row contained {} bytes'.format(len(data)))
                            continue
                        else:
                            not_a_full_row_seen = True
        return MeasurementsCsv(outfile, self.active_channels, self.sample_rate_hz)

    def get_raw(self):
        return [self.raw_data_file]

    def teardown(self):
        if self.keep_raw:
            if os.path.isfile(self.raw_data_file):
                os.remove(self.raw_data_file)
