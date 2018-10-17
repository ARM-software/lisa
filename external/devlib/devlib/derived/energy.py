#    Copyright 2013-2018 ARM Limited
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
from collections import defaultdict

from devlib.derived import DerivedMeasurements, DerivedMetric
from devlib.instrument import  MEASUREMENT_TYPES


class DerivedEnergyMeasurements(DerivedMeasurements):

    # pylint: disable=too-many-locals,too-many-branches
    @staticmethod
    def process(measurements_csv):

        should_calculate_energy = []
        use_timestamp = False

        # Determine sites to calculate energy for
        channel_map = defaultdict(list)
        for channel in measurements_csv.channels:
            channel_map[channel.site].append(channel.kind)
            if channel.site == 'timestamp':
                use_timestamp = True
                time_measurment = channel.measurement_type
        for site, kinds in channel_map.items():
            if 'power' in kinds and not 'energy' in kinds:
                should_calculate_energy.append(site)

        if measurements_csv.sample_rate_hz is None and not use_timestamp:
            msg = 'Timestamp data is unavailable, please provide a sample rate'
            raise ValueError(msg)

        if use_timestamp:
            # Find index of timestamp column
            ts_index = [i for i, chan in enumerate(measurements_csv.channels)
                        if chan.site == 'timestamp']
            if len(ts_index) > 1:
                raise ValueError('Multiple timestamps detected')
            ts_index = ts_index[0]

        row_ts = 0
        last_ts = 0
        energy_results = defaultdict(dict)
        power_results = defaultdict(float)

        # Process data
        for count, row in enumerate(measurements_csv.iter_measurements()):
            if use_timestamp:
                last_ts = row_ts
                row_ts = time_measurment.convert(float(row[ts_index].value), 'time')
            for entry in row:
                channel = entry.channel
                site = channel.site
                if channel.kind == 'energy':
                    if count == 0:
                        energy_results[site]['start'] = entry.value
                    else:
                        energy_results[site]['end'] = entry.value

                if channel.kind == 'power':
                    power_results[site] += entry.value

                    if site in should_calculate_energy:
                        if count == 0:
                            energy_results[site]['start'] = 0
                            energy_results[site]['end'] = 0
                        elif use_timestamp:
                            energy_results[site]['end'] += entry.value * (row_ts - last_ts)
                        else:
                            energy_results[site]['end'] += entry.value * (1 /
                                                           measurements_csv.sample_rate_hz)

        # Calculate final measurements
        derived_measurements = []
        for site in energy_results:
            total_energy = energy_results[site]['end'] - energy_results[site]['start']
            name = '{}_total_energy'.format(site)
            derived_measurements.append(DerivedMetric(name, total_energy, MEASUREMENT_TYPES['energy']))

        for site in power_results:
            power = power_results[site] / (count + 1)  #pylint: disable=undefined-loop-variable
            name = '{}_average_power'.format(site)
            derived_measurements.append(DerivedMetric(name, power, MEASUREMENT_TYPES['power']))

        return derived_measurements
