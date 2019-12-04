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
from itertools import chain

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
                 keep_raw=False
                 ):
        # pylint: disable=no-member
        super(DaqInstrument, self).__init__(target)
        self.keep_raw = keep_raw
        self._need_reset = True
        self._raw_files = []
        self.tempdir = None
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
                raise ValueError(msg.format(device_id, ', '.join(result.data)))
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

        for label in labels:
            for kind in ['power', 'voltage']:
                self.add_channel(label, kind)

    def reset(self, sites=None, kinds=None, channels=None):
        super(DaqInstrument, self).reset(sites, kinds, channels)
        self.daq_client.close()
        self.daq_client.configure(self.device_config)
        self._need_reset = False
        self._raw_files = []

    def start(self):
        if self._need_reset:
            self.reset()
        self.daq_client.start()

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
                    message = 'Could not get DAQ trace for {}; Obtained traces are in {}'
                    raise HostError(message.format(site, self.tempdir))

            # The first row is the headers
            channel_order = []
            for site, reader in site_readers.items():
                channel_order.extend(['{}_{}'.format(site, kind)
                                      for kind in next(reader)])

            def _read_next_rows():
                parts = []
                for reader in site_readers.values():
                    try:
                        parts.extend(next(reader))
                    except StopIteration:
                        parts.extend([None, None])
                return list(chain(parts))

            with csvwriter(outfile) as writer:
                field_names = [c.label for c in self.active_channels]
                writer.writerow(field_names)
                raw_row = _read_next_rows()
                while any(raw_row):
                    row = [raw_row[channel_order.index(f)] for f in field_names]
                    writer.writerow(row)
                    raw_row = _read_next_rows()

            return MeasurementsCsv(outfile, self.active_channels, self.sample_rate_hz)
        finally:
            for fh in file_handles:
                fh.close()

    def get_raw(self):
        return self._raw_files

    def teardown(self):
        self.daq_client.close()
        if not self.keep_raw:
            if os.path.isdir(self.tempdir):
                shutil.rmtree(self.tempdir)
