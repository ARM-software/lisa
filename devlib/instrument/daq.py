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

import os
import shutil
import tempfile
import time
from itertools import chain, zip_longest

from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.instrument import Instrument, MeasurementsCsv, CONTINUOUS
from devlib.exception import HostError
from devlib.utils.csvutil import csvwriter, create_reader
from devlib.utils.misc import unique

try:
    from daqpower.client import DaqClient
    from daqpower.config import DeviceConfiguration
except ImportError as e:
    DaqClient = None
    DeviceConfiguration = None
    import_error_mesg = e.args[0] if e.args else str(e)


class DaqInstrument(Instrument):

    mode = CONTINUOUS

    def __init__(self, target, resistor_values,  # pylint: disable=R0914
                 labels=None,
                 host='localhost',
                 port=45677,
                 device_id='Dev1',
                 v_range=2.5,
                 dv_range=0.2,
                 sample_rate_hz=10000,
                 channel_map=(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
                 keep_raw=False,
                 time_as_clock_boottime=True
                 ):
        # pylint: disable=no-member
        super(DaqInstrument, self).__init__(target)
        self.keep_raw = keep_raw
        self._need_reset = True
        self._raw_files = []
        self.tempdir = None
        self.target_boottime_clock_at_start = 0.0
        if DaqClient is None:
            raise HostError('Could not import "daqpower": {}'.format(import_error_mesg))
        if labels is None:
            labels = ['PORT_{}'.format(i) for i in range(len(resistor_values))]
        if len(labels) != len(resistor_values):
            raise ValueError('"labels" and "resistor_values" must be of the same length')
        self.daq_client = DaqClient(host, port)
        try:
            devices = self.daq_client.list_devices()
            if device_id not in devices:
                msg = 'Device "{}" is not found on the DAQ server. Available devices are: "{}"'
                raise ValueError(msg.format(device_id, ', '.join(devices)))
        except Exception as e:
            raise HostError('Problem querying DAQ server: {}'.format(e))

        self.device_config = DeviceConfiguration(device_id=device_id,
                                                 v_range=v_range,
                                                 dv_range=dv_range,
                                                 sampling_rate=sample_rate_hz,
                                                 resistor_values=resistor_values,
                                                 channel_map=channel_map,
                                                 labels=labels)
        self.sample_rate_hz = sample_rate_hz
        self.time_as_clock_boottime = time_as_clock_boottime

        self.add_channel('Time', 'time')
        for label in labels:
            for kind in ['power', 'voltage']:
                self.add_channel(label, kind)

        if time_as_clock_boottime:
            host_path = os.path.join(PACKAGE_BIN_DIRECTORY, self.target.abi,
                                     'get_clock_boottime')
            self.clock_boottime_cmd = self.target.install_if_needed(host_path,
                                                                    search_system_binaries=False)

    def calculate_boottime_offset(self):
        time_before = time.time()
        out = self.target.execute(self.clock_boottime_cmd)
        time_after = time.time()

        remote_clock_boottime = float(out)
        propagation_delay = (time_after - time_before) / 2
        boottime_at_end = remote_clock_boottime + propagation_delay

        return time_after - boottime_at_end

    def reset(self, sites=None, kinds=None, channels=None):
        super(DaqInstrument, self).reset(sites, kinds, channels)
        self.daq_client.close()
        self.daq_client.configure(self.device_config)
        self._need_reset = False
        self._raw_files = []

    def start(self):
        if self._need_reset:
            # Preserve channel order
            self.reset(channels=self.channels.keys())

        if self.time_as_clock_boottime:
            target_boottime_offset = self.calculate_boottime_offset()
            time_start = time.time()

        self.daq_client.start()

        if self.time_as_clock_boottime:
            time_end = time.time()
            self.target_boottime_clock_at_start = (time_start + time_end) / 2 - target_boottime_offset

    def stop(self):
        self.daq_client.stop()
        self._need_reset = True

    def get_data(self, outfile):  # pylint: disable=R0914
        self.tempdir = tempfile.mkdtemp(prefix='daq-raw-')
        self.daq_client.get_data(self.tempdir)
        raw_file_map = {}
        for entry in os.listdir(self.tempdir):
            site = os.path.splitext(entry)[0]
            path = os.path.join(self.tempdir, entry)
            raw_file_map[site] = path
            self._raw_files.append(path)

        active_sites = unique([c.site for c in self.active_channels])
        file_handles = []
        try:
            site_readers = {}
            for site in active_sites:
                try:
                    site_file = raw_file_map[site]
                    reader, fh = create_reader(site_file)
                    site_readers[site] = reader
                    file_handles.append(fh)
                except KeyError:
                    if not site.startswith("Time"):
                        message = 'Could not get DAQ trace for {}; Obtained traces are in {}'
                        raise HostError(message.format(site, self.tempdir))

            # The first row is the headers
            channel_order = ['Time_time']
            for site, reader in site_readers.items():
                channel_order.extend(['{}_{}'.format(site, kind)
                                      for kind in next(reader)])

            def _read_rows():
                row_iter = zip_longest(*site_readers.values(), fillvalue=(None, None))
                for raw_row in row_iter:
                    raw_row = list(chain.from_iterable(raw_row))
                    raw_row.insert(0, _read_rows.row_time_s)
                    yield raw_row
                    _read_rows.row_time_s += 1.0 / self.sample_rate_hz

            _read_rows.row_time_s = self.target_boottime_clock_at_start

            with csvwriter(outfile) as writer:
                field_names = [c.label for c in self.active_channels]
                writer.writerow(field_names)
                for raw_row in _read_rows():
                    row = [raw_row[channel_order.index(f)] for f in field_names]
                    writer.writerow(row)

            return MeasurementsCsv(outfile, self.active_channels, self.sample_rate_hz)
        finally:
            for fh in file_handles:
                fh.close()

    def get_raw(self):
        return self._raw_files

    def teardown(self):
        self.daq_client.close()
        if not self.keep_raw:
            if self.tempdir and os.path.isdir(self.tempdir):
                shutil.rmtree(self.tempdir)
