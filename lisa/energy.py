# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import devlib
import json
import os
import psutil
import time
import logging

from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT
from time import sleep

import numpy as np
import pandas as pd

from bart.common.Utils import area_under_curve

# Default energy measurements for each board
DEFAULT_ENERGY_METER = {

    # ARM TC2: by default use HWMON
    'tc2' : {
        'instrument' : 'hwmon',
        'channel_map' : {
            'LITTLE' : 'A7 Jcore',
            'big' : 'A15 Jcore',
        }
    },

    # ARM Juno: by default use HWMON
    'juno' : {
        'instrument' : 'hwmon',
        # if the channels do not contain a core name we can match to the
        # little/big cores on the board, use a channel_map section to
        # indicate which channel is which
        'channel_map' : {
            'LITTLE' : 'BOARDLITTLE',
            'big' : 'BOARDBIG',
        }
    },

}

EnergyReport = namedtuple('EnergyReport',
                          ['channels', 'report_file', 'data_frame'])

class EnergyMeter(object):

    _meter = None

    def __init__(self, target, res_dir=None):
        self._target = target
        res_dir = res_dir if res_dir else tempfile.gettempdir()
        self._res_dir = Path(res_dir)

        # Setup logging
        self._log = logging.getLogger('EnergyMeter')

    @staticmethod
    def getInstance(target, conf, force=False, res_dir=None):

        if not force and EnergyMeter._meter:
            return EnergyMeter._meter

        log = logging.getLogger('EnergyMeter')

        # Initialize energy meter based on configuration
        if 'emeter' in conf:
            emeter = conf['emeter']
            log.debug('using user-defined configuration')

        # Initialize energy probe to board default
        elif 'board' in conf and \
            conf['board'] in DEFAULT_ENERGY_METER:
                emeter = DEFAULT_ENERGY_METER[conf['board']]
                log.debug('using default energy meter for [%s]',
                          conf['board'])
        else:
            return None

        if emeter['instrument'] == 'hwmon':
            EnergyMeter._meter = HWMon(target, emeter, res_dir)
        elif emeter['instrument'] == 'aep':
            EnergyMeter._meter = AEP(target, emeter, res_dir)
        elif emeter['instrument'] == 'monsoon':
            EnergyMeter._meter = Monsoon(target, emeter, res_dir)
        elif emeter['instrument'] == 'acme':
            EnergyMeter._meter = ACME(target, emeter, res_dir)
        elif emeter['instrument'] == 'gem5':
            EnergyMeter._meter = Gem5EnergyMeter(target, emeter, res_dir)

        log.debug('Results dir: %s', res_dir)
        return EnergyMeter._meter

    def sample(self):
        raise NotImplementedError('Missing implementation')

    def reset(self):
        raise NotImplementedError('Missing implementation')

    def report(self, out_dir):
        raise NotImplementedError('Missing implementation')

class HWMon(EnergyMeter):

    def __init__(self, target, conf=None, res_dir=None):
        super(HWMon, self).__init__(target, res_dir)

        # The HWMon energy meter
        self._hwmon = None

        # Energy readings
        self.readings = {}

        if 'hwmon' not in self._target.modules:
            self._log.info('HWMON module not enabled')
            self._log.warning('Energy sampling disabled by configuration')
            return

        # Initialize HWMON instrument
        self._log.info('Scanning for HWMON channels, may take some time...')
        self._hwmon = devlib.HwmonInstrument(self._target)

        # Decide which channels we'll collect data from.
        # If the caller provided a channel_map, require that all the named
        # channels exist.
        # Otherwise, try using the big.LITTLE core names as channel names.
        # If they don't match, just collect all available channels.

        available_sites = [c.site for c in self._hwmon.get_channels('energy')]

        self._channels = conf.get('channel_map')
        if self._channels:
            # If the user provides a channel_map then require it to be correct.
            if not all (s in available_sites for s in list(self._channels.values())):
                raise RuntimeError(
                    "Found sites {} but channel_map contains {}".format(
                        sorted(available_sites), sorted(self._channels.values())))
        elif self._target.big_core:
            bl_sites = [self._target.big_core.upper(),
                        self._target.little_core.upper()]
            if all(s in available_sites for s in bl_sites):
                self._log.info('Using default big.LITTLE hwmon channels')
                self._channels = dict(list(zip(['big', 'LITTLE'], bl_sites)))

        if not self._channels:
            self._log.info('Using all hwmon energy channels')
            self._channels = {site: site for site in available_sites}

        # Configure channels for energy measurements
        self._log.debug('Enabling channels %s', list(self._channels.values()))
        self._hwmon.reset(kinds=['energy'], sites=list(self._channels.values()))

        # Logging enabled channels
        self._log.info('Channels selected for energy sampling:')
        for channel in self._hwmon.active_channels:
            self._log.info('   %s', channel.label)


    def sample(self):
        if self._hwmon is None:
            return None
        samples = self._hwmon.take_measurement()
        for s in samples:
            site = s.channel.site
            value = s.value

            if site not in self.readings:
                self.readings[site] = {
                        'last'  : value,
                        'delta' : 0,
                        'total' : 0
                        }
                continue

            self.readings[site]['delta'] = value - self.readings[site]['last']
            self.readings[site]['last']  = value
            self.readings[site]['total'] += self.readings[site]['delta']

        self._log.debug('SAMPLE: %s', self.readings)
        return self.readings

    def reset(self):
        if self._hwmon is None:
            return
        self.sample()
        for site in self.readings:
            self.readings[site]['delta'] = 0
            self.readings[site]['total'] = 0
        self._log.debug('RESET: %s', self.readings)

    def report(self, out_dir, out_file='energy.json'):
        if self._hwmon is None:
            return (None, None)
        # Retrive energy consumption data
        nrg = self.sample()
        # Reformat data for output generation
        clusters_nrg = {}
        for channel, site in self._channels.items():
            if site not in nrg:
                raise RuntimeError('hwmon channel "{}" not available. '
                                   'Selected channels: {}'.format(
                                       channel, list(nrg.keys())))
            nrg_total = nrg[site]['total']
            self._log.debug('Energy [%16s]: %.6f', site, nrg_total)
            clusters_nrg[channel] = nrg_total

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_file)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

        return EnergyReport(clusters_nrg, nrg_file, None)

class _DevlibContinuousEnergyMeter(EnergyMeter):
    """Common functionality for devlib Instruments in CONTINUOUS mode"""

    def reset(self):
        self._instrument.start()

    def report(self, out_dir, out_energy='energy.json', out_samples='samples.csv'):
        self._instrument.stop()

        df = self._read_csv(out_dir, out_samples)
        df = self._build_timeline(df)
        if df.empty:
            raise RuntimeError('No energy data collected')
        channels_nrg = self._compute_energy(df)
        # Dump data as JSON file
        nrg_file = os.path.join(out_dir, out_energy)
        with open(nrg_file, 'w') as ofile:
            json.dump(channels_nrg, ofile, sort_keys=True, indent=4)

        return EnergyReport(channels_nrg, nrg_file, df)

    def _read_csv(self, out_dir, out_samples):
        csv_path = os.path.join(out_dir, out_samples)
        csv_data = self._instrument.get_data(csv_path)
        with open(csv_path) as f:
            # Each column in the CSV will be headed with 'SITE_measure'
            # (e.g. 'BAT_power'). Convert that to a list of ('SITE', 'measure')
            # tuples, then pass that as the `names` parameter to read_csv to get
            # a nested column index. None of devlib's standard measurement types
            # have '_' in the name so this use of rsplit should be fine.
            exp_headers = [c.label for c in csv_data.channels]
            headers = f.readline().strip().split(',')
            if set(headers) != set(exp_headers):
                raise ValueError(
                    'Unexpected headers in CSV from devlib instrument. '
                    'Expected {}, found {}'.format(sorted(headers),
                                                   sorted(exp_headers)))
            columns = [tuple(h.rsplit('_', 1)) for h in headers]
            # Passing `names` means read_csv doesn't expect to find headers in
            # the CSV (i.e. expects every line to hold data). This works because
            # we have already consumed the first line of `f`.
            df = pd.read_csv(f, names=columns)
        return df

    def _build_timeline(self, df):
        sample_period = 1. / self._instrument.sample_rate_hz
        df.index = np.linspace(0, sample_period * len(df), num=len(df))
        return df

    def _compute_energy(self, df):
        channels_nrg = {}
        for site, measure in df:
            if measure == 'power':
                channels_nrg[site] = area_under_curve(df[site]['power'])
        return channels_nrg

class AEP(_DevlibContinuousEnergyMeter):

    def __init__(self, target, conf, res_dir):
        super(AEP, self).__init__(target, res_dir)

        # Configure channels for energy measurements
        self._log.info('AEP configuration')
        self._log.info('    %s', conf)
        self._instrument = devlib.EnergyProbeInstrument(
            self._target, labels=conf.get('channel_map'), **conf['conf'])

        # Configure channels for energy measurements
        self._log.debug('Enabling channels')
        self._instrument.reset()

        # Logging enabled channels
        self._log.info('Channels selected for energy sampling:')
        self._log.info('   %s', str(self._instrument.active_channels))
        self._log.debug('Results dir: %s', self._res_dir)

class Monsoon(_DevlibContinuousEnergyMeter):
    """
    Monsoon Solutions energy monitor
    """

    def __init__(self, target, conf, res_dir):
        super(Monsoon, self).__init__(target, res_dir)

        self._instrument = devlib.MonsoonInstrument(self._target, **conf['conf'])
        self._instrument.reset()

_acme_install_instructions = '''

  If you need to measure energy using an ACME EnergyProbe,
  please do follow installation instructions available here:
     https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#iiocapture---baylibre-acme-cape

  Othwerwise, please select a different energy meter in your
  configuration file.

'''

class ACME(EnergyMeter):
    """
    BayLibre's ACME board based EnergyMeter
    """

    def __init__(self, target, conf, res_dir):
        super(ACME, self).__init__(target, res_dir)

        # Assume iio-capture is available in PATH
        iioc = conf.get('conf', {
            'iio-capture' : 'iio-capture',
            'ip_address'  : 'baylibre-acme.local',
        })
        self._iiocapturebin = iioc.get('iio-capture', 'iio-capture')
        self._hostname = iioc.get('ip_address', 'baylibre-acme.local')

        self._channels = conf.get('channel_map', {
            'CH0': '0'
        })
        self._iio = {}

        self._log.info('ACME configuration:')
        self._log.info('    binary: %s', self._iiocapturebin)
        self._log.info('    device: %s', self._hostname)
        self._log.info('  channels:')
        for channel in self._channels:
            self._log.info('     %s', self._str(channel))

        # Check if iio-capture binary is available
        try:
            p = Popen([self._iiocapturebin, '-h'], stdout=PIPE, stderr=STDOUT)
        except:
            self._log.error('iio-capture binary [%s] not available',
                            self._iiocapturebin)
            self._log.warning(_acme_install_instructions)
            raise RuntimeError('Missing iio-capture binary')

    def sample(self):
        raise NotImplementedError('Not available for ACME')

    def _iio_device(self, channel):
        return 'iio:device{}'.format(self._channels[channel])

    def _str(self, channel):
        return '{} ({})'.format(channel, self._iio_device(channel))

    def reset(self):
        """
        Reset energy meter and start sampling from channels specified in the
        target configuration.
        """
        # Terminate already running iio-capture instance (if any)
        wait_for_termination = 0
        for proc in psutil.process_iter():
            if self._iiocapturebin not in proc.cmdline():
                continue
            for channel in self._channels:
                if self._iio_device(channel) in proc.cmdline():
                    self._log.debug('Killing previous iio-capture for [%s]',
                                     self._iio_device(channel))
                    self._log.debug(proc.cmdline())
                    proc.kill()
                    wait_for_termination = 2

        # Wait for previous instances to be killed
        sleep(wait_for_termination)

        # Start iio-capture for all channels required
        for channel in self._channels:
            ch_id = self._channels[channel]

            # Setup CSV file to collect samples for this channel
            csv_file = self._res_dir.joinpath('samples_{}.csv'.format(channel))

            # Start a dedicated iio-capture instance for this channel
            self._iio[ch_id] = Popen([self._iiocapturebin, '-n',
                                       self._hostname, '-o',
                                       '-c', '-f',
                                       str(csv_file),
                                       self._iio_device(channel)],
                                       stdout=PIPE, stderr=STDOUT)

        # Wait few milliseconds before to check if there is any output
        sleep(1)

        # Check that all required channels have been started
        for channel in self._channels:
            ch_id = self._channels[channel]

            self._iio[ch_id].poll()
            if self._iio[ch_id].returncode:
                self._log.error('Failed to run %s for %s',
                                 self._iiocapturebin, self._str(channel))
                self._log.warning('\n\n'\
                    '  Make sure there are no iio-capture processes\n'\
                    '  connected to %s and device %s\n',
                    self._hostname, self._str(channel))
                out, _ = self._iio[ch_id].communicate()
                self._log.error('Output: [%s]', out.strip())
                self._iio[ch_id] = None
                raise RuntimeError('iio-capture connection error')

        self._log.debug('Started %s on %s...',
                        self._iiocapturebin, self._str(channel))

    def report(self, out_dir, out_energy='energy.json'):
        """
        Stop iio-capture and collect sampled data.

        :param out_dir: Output directory where to store results
        :type out_dir: str

        :param out_file: File name where to save energy data
        :type out_file: str
        """
        channels_nrg = {}
        channels_stats = {}
        for channel in self._channels:
            ch_id = self._channels[channel]

            if self._iio[ch_id] is None:
                continue

            self._iio[ch_id].poll()
            if self._iio[ch_id].returncode:
                # returncode not None means that iio-capture has terminated
                # already, so there must have been an error
                self._log.error('%s terminated for %s',
                                self._iiocapturebin, self._str(channel))
                out, _ = self._iio[ch_id].communicate()
                self._log.error('[%s]', out)
                self._iio[ch_id] = None
                continue

            # kill process and get return
            self._iio[ch_id].terminate()
            out, _ = self._iio[ch_id].communicate()
            self._iio[ch_id].wait()
            self._iio[ch_id] = None

            # iio-capture return "energy=value", add a simple format check
            if '=' not in out:
                self._log.error('Bad output format for %s:',
                                self._str(channel))
                self._log.error('[%s]', out)
                continue
            else:
                self._log.debug('%s: %s', self._str(channel), out)

            # Build energy counter object
            nrg = {}
            for kv_pair in out.split():
                key, val = kv_pair.partition('=')[::2]
                nrg[key] = float(val)
            channels_stats[channel] = nrg

            self._log.debug(self._str(channel))
            self._log.debug(nrg)

            # Save CSV samples file to out_dir
            os.system('mv {} {}'.format(
                self._res_dir.joinpath('samples_{}.csv'.format(channel)),
                out_dir))

            # Add channel's energy to return results
            channels_nrg['{}'.format(channel)] = nrg['energy']

        # Dump energy data
        nrg_file = '{}/{}'.format(out_dir, out_energy)
        with open(nrg_file, 'w') as ofile:
            json.dump(channels_nrg, ofile, sort_keys=True, indent=4)

        # Dump energy stats
        nrg_stats_file = os.path.splitext(out_energy)[0] + \
                        '_stats' + os.path.splitext(out_energy)[1]
        nrg_stats_file = '{}/{}'.format(out_dir, nrg_stats_file)
        with open(nrg_stats_file, 'w') as ofile:
            json.dump(channels_stats, ofile, sort_keys=True, indent=4)

        return EnergyReport(channels_nrg, nrg_file, None)

class Gem5EnergyMeter(_DevlibContinuousEnergyMeter):
    def __init__(self, target, conf, res_dir):
        super(Gem5EnergyMeter, self).__init__(target, res_dir)

        power_sites = list(conf['channel_map'].values())
        self._instrument = devlib.Gem5PowerInstrument(self._target, power_sites)

    def reset(self):
        self._instrument.reset()
        self._instrument.start()

    def _build_timeline(self, df):
        # Power measurements on gem5 are performed not only periodically but also
        # spuriously on OPP changes. Let's use the time channel provided by the
        # gem5 power instrument to build the timeline accordingly.
        for site, measure in df:
            if measure == 'time':
                meas_dur = df[site]['time']
                break
        timeline = np.zeros(len(meas_dur))
        # The time channel gives the elapsed time since previous measurement
        for i in range(1, len(meas_dur)):
            timeline[i] = meas_dur[i] + timeline[i - 1]
        df.index = timeline
        return df

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
