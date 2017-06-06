from devlib.instrument import (Instrument, CONTINUOUS,
                               MeasurementsCsv, MeasurementType)
from devlib.utils.rendering import (GfxinfoFrameCollector,
                                    read_gfxinfo_columns)


class GfxInfoFramesInstrument(Instrument):

    mode = CONTINUOUS

    def __init__(self, target, package, period=2, keep_raw=True):
        super(GfxInfoFramesInstrument, self).__init__(target)
        self.package = package
        self.period = period
        self.keep_raw = keep_raw
        self.collector = None
        self.header = None
        self._need_reset = True
        self._init_channels()

    def reset(self, sites=None, kinds=None, channels=None):
        super(GfxInfoFramesInstrument, self).reset(sites, kinds, channels)
        self.collector = GfxinfoFrameCollector(self.target, self.period,
                                               self.package, self.header)
        self._need_reset = False

    def start(self):
        if self._need_reset:
            self.reset()
        self.collector.start()

    def stop(self):
        self.collector.stop()
        self._need_reset = True

    def get_data(self, outfile):
        raw_outfile = None
        if self.keep_raw:
            raw_outfile = outfile + '.raw'
        self.collector.process_frames(raw_outfile)
        active_sites = [chan.label for chan in self.active_channels]
        self.collector.write_frames(outfile, columns=active_sites)
        return MeasurementsCsv(outfile, self.active_channels)

    def _init_channels(self):
        columns = read_gfxinfo_columns(self.target)
        for entry in columns:
            if entry == 'Flags':
                self.add_channel('Flags', MeasurementType('flags', 'flags'))
            else:
                self.add_channel(entry, 'time_us')
        self.header = [chan.label for chan in self.channels.values()]

