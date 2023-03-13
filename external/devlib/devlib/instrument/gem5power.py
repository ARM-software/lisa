#    Copyright 2017-2018 ARM Limited
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

from __future__ import division

from devlib.platform.gem5 import Gem5SimulationPlatform
from devlib.instrument import Instrument, CONTINUOUS, MeasurementsCsv
from devlib.exception import TargetStableError
from devlib.utils.csvutil import csvwriter


class Gem5PowerInstrument(Instrument):
    '''
    Instrument enabling power monitoring in gem5
    '''

    mode = CONTINUOUS
    roi_label = 'power_instrument'
    site_mapping = {'timestamp': 'sim_seconds'}

    def __init__(self, target, power_sites):
        '''
        Parameter power_sites is a list of gem5 identifiers for power values.
        One example of such a field:
            system.cluster0.cores0.power_model.static_power
        '''
        if not isinstance(target.platform, Gem5SimulationPlatform):
            raise TargetStableError('Gem5PowerInstrument requires a gem5 platform')
        if not target.has('gem5stats'):
            raise TargetStableError('Gem5StatsModule is not loaded')
        super(Gem5PowerInstrument, self).__init__(target)

        # power_sites is assumed to be a list later
        if isinstance(power_sites, list):
            self.power_sites = power_sites
        else:
            self.power_sites = [power_sites]
        self.add_channel('timestamp', 'time')
        for field in self.power_sites:
            self.add_channel(field, 'power')
        self.target.gem5stats.book_roi(self.roi_label)
        self.sample_period_ns = 10000000
        # Sample rate must remain unset as gem5 does not provide samples
        # at regular intervals therefore the reported timestamp should be used.
        self.sample_rate_hz = None
        self.target.gem5stats.start_periodic_dump(0, self.sample_period_ns)
        self._base_stats_dump = 0

    def start(self):
        self.target.gem5stats.roi_start(self.roi_label)

    def stop(self):
        self.target.gem5stats.roi_end(self.roi_label)

    def get_data(self, outfile):
        active_sites = [c.site for c in self.active_channels]
        with csvwriter(outfile) as writer:
            writer.writerow([c.label for c in self.active_channels]) # headers
            sites_to_match = [self.site_mapping.get(s, s) for s in active_sites]
            for rec, _ in self.target.gem5stats.match_iter(sites_to_match,
                    [self.roi_label], self._base_stats_dump):
                writer.writerow([rec[s] for s in sites_to_match])
        return MeasurementsCsv(outfile, self.active_channels, self.sample_rate_hz)

    def reset(self, sites=None, kinds=None, channels=None):
        super(Gem5PowerInstrument, self).reset(sites, kinds, channels)
        self._base_stats_dump = self.target.gem5stats.next_dump_no()
