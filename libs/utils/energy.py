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


# Default energy measurements for each board
DEFAULT_ENERGY_METER = {

    # ARM TC2: by default use HWMON
    'tc2' : {
        'instrument' : 'hwmon',
        'conf' : {
            'sites' : [ 'A7 Jcore', 'A15 Jcore' ],
            'kinds' : [ 'energy']
        },
        'channel_map' : {
            'LITTLE' : 'A7 Jcore',
            'big' : 'A15 Jcore',
        }
    },

    # ARM Juno: by default use HWMON
    'juno' : {
        'instrument' : 'hwmon',
        'conf' : {
            'sites' : [ 'BOARDLITTLE', 'BOARDBIG' ],
            'kinds' : [ 'energy' ]
        },
        # if the channels do not contain a core name we can match to the
        # little/big cores on the board, use a channel_map section to
        # indicate which channel is which
        'channel_map' : {
            'LITTLE' : 'BOARDLITTLE',
            'big' : 'BOARDBIG',
        }
    },

}

EnergyCounter = namedtuple('EnergyCounter', ['site', 'pwr_total' , 'pwr_samples', 'pwr_avg', 'time', 'nrg'])
EnergyReport = namedtuple('EnergyReport', ['channels', 'report_file'])

class EnergyMeter(object):

    _meter = None

    def __init__(self, target, res_dir=None):
        self._target = target
        self._res_dir = res_dir
        if not self._res_dir:
            self._res_dir = '/tmp'

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
        elif emeter['instrument'] == 'acme':
            EnergyMeter._meter = ACME(target, emeter, res_dir)

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

        # Configure channels for energy measurements
        self._log.debug('Enabling channels %s', conf['conf'])
        self._hwmon.reset(**conf['conf'])

        # Logging enabled channels
        self._log.info('Channels selected for energy sampling:')
        for channel in self._hwmon.active_channels:
            self._log.info('   %s', channel.label)

        # record the HWMon channels
        self._channels = conf.get('channel_map', {
            'LITTLE': self._target.little_core.upper(),
            'big': self._target.big_core.upper()
        })

    def sample(self):
        if self._hwmon is None:
            return None
        samples = self._hwmon.take_measurement()
        for s in samples:
            label = s.channel.label\
                    .replace('_energy', '')\
                    .replace(" ", "_")
            value = s.value

            if label not in self.readings:
                self.readings[label] = {
                        'last'  : value,
                        'delta' : 0,
                        'total' : 0
                        }
                continue

            self.readings[label]['delta'] = value - self.readings[label]['last']
            self.readings[label]['last']  = value
            self.readings[label]['total'] += self.readings[label]['delta']

        self._log.debug('SAMPLE: %s', self.readings)
        return self.readings

    def reset(self):
        if self._hwmon is None:
            return
        self.sample()
        for label in self.readings:
            self.readings[label]['delta'] = 0
            self.readings[label]['total'] = 0
        self._log.debug('RESET: %s', self.readings)

    def report(self, out_dir, out_file='energy.json'):
        if self._hwmon is None:
            return (None, None)
        # Retrive energy consumption data
        nrg = self.sample()
        # Reformat data for output generation
        clusters_nrg = {}
        for channel in self._channels:
            label = self._channels[channel]
            nrg_total = nrg[label]['total']
            self._log.debug('Energy [%16s]: %.6f', label, nrg_total)
            clusters_nrg[channel] = nrg_total

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_file)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

        return EnergyReport(clusters_nrg, nrg_file)

class AEP(EnergyMeter):

    def __init__(self, target, conf, res_dir):
        super(AEP, self).__init__(target, res_dir)

        # Energy channels
        self.channels = []

        # Time (start and diff) for power measurment
        self.time = {}

        # Configure channels for energy measurements
        self._log.info('AEP configuration')
        self._log.info('    %s', conf)
        self._aep = devlib.EnergyProbeInstrument(
            self._target, labels=conf['channel_map'], **conf['conf'])

        # Configure channels for energy measurements
        self._log.debug('Enabling channels')
        self._aep.reset()

        # Logging enabled channels
        self._log.info('Channels selected for energy sampling:')
        self._log.info('   %s', str(self._aep.active_channels))
        self._log.debug('Results dir: %s', self._res_dir)

    def _get_energy(self, samples, time, idx, site):
        pwr_total = 0
        pwr_samples = 0
        for sample in samples:
            pwr_total += sample[idx].value
            pwr_samples += 1

        pwr_avg = pwr_total / pwr_samples
        nrg = pwr_avg * time

        ec = EnergyCounter(site, pwr_total, pwr_samples, pwr_avg, time, nrg)
        return ec

    def _sample(self, csv_file):
        if self._aep is None:
            return

        self.time['diff'] = time.time() - self.time['start']
        self._aep.stop()

        csv_data = self._aep.get_data(csv_file)
        samples = csv_data.measurements()

        # Calculate energy for each channel
        for idx,channel in enumerate(csv_data.channels):
            if channel.kind is not 'power':
                continue
            ec = self._get_energy(samples, self.time['diff'], idx, channel.site)
            self.channels.append(ec)

    def reset(self):
        if self._aep is None:
            return

        self.channels = []
        self._log.debug('RESET: %s', self.channels)

        self._aep.start()
        self.time['start'] = time.time()

    def report(self, out_dir, out_energy='energy.json', out_samples='samples.csv'):
        if self._aep is None:
            return

        # Retrieve energy consumption data
        csv_file = '{}/{}'.format(out_dir, out_samples)
        self._sample(csv_file)

        # Reformat data for output generation
        channels_nrg = {}
        for channel in self.channels:
            self._log.debug('Energy [%16s]: %.6f',
                            channel.site, channel.nrg)
            channels_nrg[channel.site] = channel.nrg

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_energy)
        with open(nrg_file, 'w') as ofile:
            json.dump(channels_nrg, ofile, sort_keys=True, indent=4)

        return EnergyReport(channels_nrg, nrg_file)


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
        super(ACME, self).__init__(target)

        # Assume iio-capture is available in PATH
        self._iiocapturebin = conf['conf'].get('iio-capture', 'iio-capture')
        self._hostname = conf['conf'].get('ip_address', 'baylibre-acme.local')
        self._channels = conf.get('channel_map', {
            'CH0': '0'
        })
        self._iio = [None] * len(self._channels)

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
            csv_file = '{}/{}'.format(
                self._res_dir,
                'samples_{}.csv'.format(channel)
            )

            # Start a dedicated iio-capture instance for this channel
            self._iio[ch_id] = Popen([self._iiocapturebin, '-n',
                                       self._hostname, '-o',
                                       '-c', '-f',
                                       csv_file,
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

            self._log.debug('Completed IIOCapture for %s...',
                            self._str(channel))

            # iio-capture return "energy=value", add a simple format check
            if '=' not in out:
                self._log.error('Bad output format for %s:',
                                self._str(channel))
                self._log.error('[%s]', out)
                continue

            # Build energy counter object
            nrg = {}
            for kv_pair in out.split():
                key, val = kv_pair.partition('=')[::2]
                nrg[key] = float(val)
            channels_stats[channel] = nrg

            self._log.debug(self._str(channel))
            self._log.debug(nrg)

            # Save CSV samples file to out_dir
            os.system('mv {}/samples_{}.csv {}'
                      .format(self._res_dir, channel, out_dir))

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

        return EnergyReport(channels_nrg, nrg_file)

# vim :set tabstop=4 shiftwidth=4 expandtab
