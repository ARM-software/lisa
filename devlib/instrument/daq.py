import os
import csv
import tempfile
from itertools import chain

from devlib.instrument import Instrument, MeasurementsCsv, CONTINUOUS
from devlib.exception import HostError
from devlib.utils.misc import unique

try:
    from daqpower.client import execute_command, Status
    from daqpower.config import DeviceConfiguration, ServerConfiguration
except ImportError, e:
    execute_command, Status = None, None
    DeviceConfiguration, ServerConfiguration, ConfigurationError = None, None, None
    import_error_mesg = e.message


class DaqInstrument(Instrument):

    mode = CONTINUOUS

    def __init__(self, target, resistor_values,  # pylint: disable=R0914
                 labels=None,
                 host='localhost',
                 port=45677,
                 device_id='Dev1',
                 v_range=2.5,
                 dv_range=0.2,
                 sampling_rate=10000,
                 channel_map=(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23),
                 ):
        # pylint: disable=no-member
        super(DaqInstrument, self).__init__(target)
        self._need_reset = True
        if execute_command is None:
            raise HostError('Could not import "daqpower": {}'.format(import_error_mesg))
        if labels is None:
            labels = ['PORT_{}'.format(i) for i in xrange(len(resistor_values))]
        if len(labels) != len(resistor_values):
            raise ValueError('"labels" and "resistor_values" must be of the same length')
        self.server_config = ServerConfiguration(host=host,
                                                 port=port)
        result = self.execute('list_devices')
        if result.status == Status.OK:
            if device_id not in result.data:
                raise ValueError('Device "{}" is not found on the DAQ server.'.format(device_id))
        elif result.status != Status.OKISH:
            raise HostError('Problem querying DAQ server: {}'.format(result.message))

        self.device_config = DeviceConfiguration(device_id=device_id,
                                                 v_range=v_range,
                                                 dv_range=dv_range,
                                                 sampling_rate=sampling_rate,
                                                 resistor_values=resistor_values,
                                                 channel_map=channel_map,
                                                 labels=labels)

        for label in labels:
            for kind in ['power', 'voltage']:
                self.add_channel(label, kind)

    def reset(self, sites=None, kinds=None):
        super(DaqInstrument, self).reset(sites, kinds)
        self.execute('close')
        result = self.execute('configure', config=self.device_config)
        if not result.status == Status.OK:  # pylint: disable=no-member
            raise HostError(result.message)
        self._need_reset = False

    def start(self):
        if self._need_reset:
            self.reset()
        self.execute('start')

    def stop(self):
        self.execute('stop')
        self._need_reset = True

    def get_data(self, outfile):  # pylint: disable=R0914
        tempdir = tempfile.mkdtemp(prefix='daq-raw-')
        self.execute('get_data', output_directory=tempdir)
        raw_file_map = {}
        for entry in os.listdir(tempdir):
            site = os.path.splitext(entry)[0]
            path = os.path.join(tempdir, entry)
            raw_file_map[site] = path

        active_sites = unique([c.site for c in self.active_channels])
        file_handles = []
        try:
            site_readers = {}
            for site in active_sites:
                try:
                    site_file = raw_file_map[site]
                    fh = open(site_file, 'rb')
                    site_readers[site] = csv.reader(fh)
                    file_handles.append(fh)
                except KeyError:
                    message = 'Could not get DAQ trace for {}; Obtained traces are in {}'
                    raise HostError(message.format(site, tempdir))

            # The first row is the headers
            channel_order = []
            for site, reader in site_readers.iteritems():
                channel_order.extend(['{}_{}'.format(site, kind)
                                      for kind in reader.next()])

            def _read_next_rows():
                parts = []
                for reader in site_readers.itervalues():
                    try:
                        parts.extend(reader.next())
                    except StopIteration:
                        parts.extend([None, None])
                return list(chain(parts))

            with open(outfile, 'wb') as wfh:
                field_names = [c.label for c in self.active_channels]
                writer = csv.writer(wfh)
                writer.writerow(field_names)
                raw_row = _read_next_rows()
                while any(raw_row):
                    row = [raw_row[channel_order.index(f)] for f in field_names]
                    writer.writerow(row)
                    raw_row = _read_next_rows()

            return MeasurementsCsv(outfile, self.active_channels)
        finally:
            for fh in file_handles:
                fh.close()

    def teardown(self):
        self.execute('close')

    def execute(self, command, **kwargs):
        return execute_command(self.server_config, command, **kwargs)

