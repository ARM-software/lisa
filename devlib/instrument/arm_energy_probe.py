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

#    Copyright 2018 Linaro Limited
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
import subprocess
import signal
from pipes import quote

import tempfile
import shutil

from devlib.instrument import Instrument, CONTINUOUS, MeasurementsCsv
from devlib.exception import HostError
from devlib.utils.csvutil import csvreader, csvwriter
from devlib.utils.misc import which

from devlib.utils.parse_aep import AepParser

class ArmEnergyProbeInstrument(Instrument):
    """
    Collects power traces using the ARM Energy Probe.

    This instrument requires ``arm-probe`` utility to be installed on the host and be in the PATH.
    arm-probe is available here:
    ``https://git.linaro.org/tools/arm-probe.git``.

    Details about how to build and use it is available here:
    ``https://git.linaro.org/tools/arm-probe.git/tree/README``

    ARM energy probe (AEP) device can simultaneously collect power from up to 3 power rails and
    arm-probe utility can record data from several AEP devices simultaneously.

    To connect the energy probe on a rail, connect the white wire to the pin that is closer to the
    Voltage source and the black wire to the pin that is closer to the load (the SoC or the device
    you are probing). Between the pins there should be a shunt resistor of known resistance in the
    range of 5 to 500 mOhm but the voltage on the shunt resistor must stay smaller than 165mV.
    The resistance of the shunt resistors is a mandatory parameter to be set in the ``config`` file.
    """

    mode = CONTINUOUS

    MAX_CHANNELS = 12 # 4 Arm Energy Probes

    def __init__(self, target, config_file='./config-aep', keep_raw=False):
        super(ArmEnergyProbeInstrument, self).__init__(target)
        self.arm_probe = which('arm-probe')
        if self.arm_probe is None:
            raise HostError('arm-probe must be installed on the host')
        #todo detect is config file exist
        self.attributes = ['power', 'voltage', 'current']
        self.sample_rate_hz = 10000
        self.config_file = config_file
        self.keep_raw = keep_raw

        self.parser = AepParser()
        #TODO make it generic
        topo = self.parser.topology_from_config(self.config_file)
        for item in topo:
            if item == 'time':
                self.add_channel('timestamp', 'time')
            else:
                self.add_channel(item, 'power')

    def reset(self, sites=None, kinds=None, channels=None):
        super(ArmEnergyProbeInstrument, self).reset(sites, kinds, channels)
        self.output_directory = tempfile.mkdtemp(prefix='energy_probe')
        self.output_file_raw = os.path.join(self.output_directory, 'data_raw')
        self.output_file = os.path.join(self.output_directory, 'data')
        self.output_file_figure = os.path.join(self.output_directory, 'summary.txt')
        self.output_file_error = os.path.join(self.output_directory, 'error.log')
        self.output_fd_error = open(self.output_file_error, 'w')
        self.command = 'arm-probe --config {} > {}'.format(quote(self.config_file), quote(self.output_file_raw))

    def start(self):
        self.logger.debug(self.command)
        self.armprobe = subprocess.Popen(self.command,
                                       stderr=self.output_fd_error,
                                       preexec_fn=os.setpgrp,
                                       shell=True)

    def stop(self):
        self.logger.debug("kill running arm-probe")
        os.killpg(self.armprobe.pid, signal.SIGTERM)

    def get_data(self, outfile):  # pylint: disable=R0914
        self.logger.debug("Parse data and compute consumed energy")
        self.parser.prepare(self.output_file_raw, self.output_file, self.output_file_figure)
        self.parser.parse_aep()
        self.parser.unprepare()
        skip_header = 1

        all_channels = [c.label for c in self.list_channels()]
        active_channels = [c.label for c in self.active_channels]
        active_indexes = [all_channels.index(ac) for ac in active_channels]

        with csvreader(self.output_file, delimiter=' ') as reader:
            with csvwriter(outfile) as writer:
                for row in reader:
                    if skip_header == 1:
                        writer.writerow(active_channels)
                        skip_header = 0
                        continue
                    if len(row) < len(active_channels):
                        continue
                    # all data are in micro (seconds/watt)
                    new = [float(row[i])/1000000 for i in active_indexes]
                    writer.writerow(new)

        self.output_fd_error.close()
        shutil.rmtree(self.output_directory)

        return MeasurementsCsv(outfile, self.active_channels, self.sample_rate_hz)

    def get_raw(self):
        return [self.output_file_raw]

    def teardown(self):
        if not self.keep_raw:
            if os.path.isfile(self.output_file_raw):
                os.remove(self.output_file_raw)
