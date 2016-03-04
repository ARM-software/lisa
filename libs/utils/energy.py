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
import logging
import time
import os

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
        }
    },

    # ARM Juno: by default use HWMON
    'juno' : {
        'instrument' : 'hwmon',
        'conf' : {
            'sites' : [ 'a53', 'a57' ],
            'kinds' : [ 'energy' ],
        }
    },
    'juno2' : {
        'instrument' : 'hwmon',
        'conf' : {
            'sites' : [ 'BOARDLITTLE', 'BOARDBIG' ],
            'kinds' : [ 'energy' ]
        },
        # if the channels do not contain a core name we can match to the
        # little/big cores on the board, use a channel_map section to
        # indicate which channel is which
        'channel_map' : {
            'little' : 'BOARDLITTLE',
            'big' : 'BOARDBIG',
        }
    },

    # Hikey: by default use AEP
    'hikey' : {
        'instrument' : 'aep',
        'conf' : {
            'labels'          : ['LITTLE'],
            'resistor_values' : [0.033],
            'device_entry'    : '/dev/ttyACM0',
        }
    },

    # ACME: by default uses iiocapture
    'acme' : {
        'instrument' : 'iiocapture',
        'conf' : {
            # Assume it is available in PATH
            'iiocapturebin' : 'iio-capture',
            # By default, Use the first connected probe only, i.e. the one
            # connected to probe n, with n being the lowest probe number with a
            # device attached
            'iiodevice' : ['iio:device0'],
            # BBB default IP
            'device_ip' : '192.168.7.2'
        }
    }
}

class EnergyMeter(object):

    _meter = None

    def __init__(self, target, res_dir=None):
        self._target = target
        self._res_dir = res_dir
        if not self._res_dir:
            self._res_dir = '/tmp'

    @staticmethod
    def getInstance(target, conf, force=False, res_dir=None):

        if not force and EnergyMeter._meter:
            return EnergyMeter._meter

        # Initialize energy meter based on configuration
        if 'emeter' in conf:
            emeter = conf['emeter']
            logging.debug('%14s - using user-defined configuration',
                          'EnergyMeter')

        # Initialize energy probe to board default
        elif 'board' in conf and \
            conf['board'] in DEFAULT_ENERGY_METER:
                emeter = DEFAULT_ENERGY_METER[conf['board']]
                logging.debug('%14s - using default energy meter for [%s]',
                        'EnergyMeter', conf['board'])
        else:
            return None

        if emeter['instrument'] == 'hwmon':
            EnergyMeter._meter = HWMon(target, emeter, res_dir)
        elif emeter['instrument'] == 'aep':
            EnergyMeter._meter = AEP(target, emeter['conf'], res_dir)
        elif emeter['instrument'] == 'iiocapture':
            EnergyMeter._meter = IIOCapture(target, emeter['conf'], res_dir)

        logging.debug('%14s - Results dir: %s', 'EnergyMeter', res_dir)
        return EnergyMeter._meter

    def sample(self):
        raise NotImplementedError('Missing implementation')

    def reset(self):
        raise NotImplementedError('Missing implementation')

    def report(self, out_dir):
        raise NotImplementedError('Missing implementation')

class HWMon(EnergyMeter):

    def __init__(self, target, hwmon_conf=None, res_dir=None):
        super(HWMon, self).__init__(target, res_dir)

        # The HWMon energy meter
        self._hwmon = None

        # Energy readings
        self.readings = {}

        if 'hwmon' not in self._target.modules:
            logging.info('%14s - HWMON module not enabled',
                    'EnergyMeter')
            logging.warning('%14s - Energy sampling disabled by configuration',
                    'EnergyMeter')
            return

        # Initialize HWMON instrument
        logging.info('%14s - Scanning for HWMON channels, may take some time...', 'EnergyMeter')
        self._hwmon = devlib.HwmonInstrument(self._target)

        # Configure channels for energy measurements
        logging.debug('%14s - Enabling channels %s', 'EnergyMeter', hwmon_conf['conf'])
        self._hwmon.reset(**hwmon_conf['conf'])

        # Logging enabled channels
        logging.info('%14s - Channels selected for energy sampling:',
                     'EnergyMeter')
        for channel in self._hwmon.active_channels:
            logging.info('%14s -    %s', 'EnergyMeter', channel.label)

        # record the hwmon channel mapping
        self.little_channel = self._target.little_core.upper()
        self.big_channel = self._target.big_core.upper()
        if hwmon_conf and 'channel_map' in hwmon_conf:
            self.little_channel = hwmon_conf['channel_map']['little']
            self.big_channel = hwmon_conf['channel_map']['big']
        logging.info('%14s - Using channel %s as little channel',
                     'EnergyMeter', self.little_channel)
        logging.info('%14s - Using channel %s as big channel',
                     'EnergyMeter', self.big_channel)


    def sample(self):
        if self._hwmon is None:
            return
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

        logging.debug('SAMPLE: %s', self.readings)
        return self.readings

    def reset(self):
        if self._hwmon is None:
            return
        self.sample()
        for label in self.readings:
            self.readings[label]['delta'] = 0
            self.readings[label]['total'] = 0
        logging.debug('RESET: %s', self.readings)


    def report(self, out_dir, out_file='energy.json'):
        if self._hwmon is None:
            return
        # Retrive energy consumption data
        nrg = self.sample()
        # Reformat data for output generation
        clusters_nrg = {}
        for ch in nrg:
            nrg_total = nrg[ch]['total']
            logging.info('%14s - Energy [%16s]: %.6f',
                    'EnergyReport', ch, nrg_total)
            if ch.upper() == self.little_channel:
                clusters_nrg['LITTLE'] = '{:.6f}'.format(nrg_total)
            elif ch.upper() == self.big_channel:
                clusters_nrg['big'] = '{:.6f}'.format(nrg_total)
            else:
                logging.warning('%14s - Unable to bind hwmon channel [%s]'\
                        ' to a big.LITTLE cluster',
                        'EnergyReport', ch)
                clusters_nrg[ch] = '{:.6f}'.format(nrg_total)
        if 'LITTLE' not in clusters_nrg:
                logging.warning('%14s - No energy data for LITTLE cluster',
                        'EnergyMeter')
        if 'big' not in clusters_nrg:
                logging.warning('%14s - No energy data for big cluster',
                        'EnergyMeter')

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_file)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

        return (clusters_nrg, nrg_file)

EnergyCounter = namedtuple('EnergyCounter', ['site', 'pwr_total' , 'pwr_samples', 'pwr_avg', 'time', 'nrg'])

class AEP(EnergyMeter):

    def __init__(self, target, aep_conf, res_dir):
        super(AEP, self).__init__(target, res_dir)

        # Energy channels
        self.channels = []

        # Time (start and diff) for power measurment
        self.time = {}

        # Configure channels for energy measurements
        logging.info('%14s - AEP configuration', 'EnergyMeter')
        logging.info('%14s -     %s', 'EnergyMeter', aep_conf)
        self._aep = devlib.EnergyProbeInstrument(self._target, **aep_conf)

        # Configure channels for energy measurements
        logging.debug('EnergyMeter - Enabling channels')
        self._aep.reset()

        # Logging enabled channels
        logging.info('%14s - Channels selected for energy sampling:\n%s',
                'EnergyMeter', str(self._aep.active_channels))
        logging.debug('%14s - Results dir: %s', 'EnergyMeter', self._res_dir)

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
            logging.debug('%14s - CH[%s] Power: %.6f (samples: %d, time: %.6f), avg: %.f6',
                          'EnergyMeter', channel.site, ec.pwr_total,
                          ec.pwr_samples, ec.time, ec.pwr_avg)
            logging.debug('%14s - CH[%s] Estimated energy:  %.6f',
                          'EnergyMeter', channel.site, ec.nrg)
            self.channels.append(ec)

        logging.debug('%14s - SAMPLE: %s', 'EnergyMeter', self.channels)

    def reset(self):
        if self._aep is None:
            return

        self.channels = []
        logging.debug('RESET: %s', self.channels)

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
            channels_nrg[channel.site] = '{:.6f}'.format(channel.nrg)

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_energy)
        with open(nrg_file, 'w') as ofile:
            json.dump(channels_nrg, ofile, sort_keys=True, indent=4)

        return (channels_nrg, nrg_file)


class IIOCapture(EnergyMeter):
    "iio-capture Energy Meter for BayLibre ACME board"

    def __init__(self, target, conf, res_dir):
        super(IIOCapture, self).__init__(target)

        self._iiocapturebin = conf['iiocapturebin']
        self._hostname = conf['device_ip']
        # Channels represent the connected probes
        # iio:device<n>
        self._channels = conf['iiodevice']
        self._iio = [None] * len(self._channels)

    def sample(self):
        raise NotImplementedError('Not available for IIOCapture')

    def reset(self):
        """
        Reset energy meter and start sampling from channels specified in the
        target configuration.
        """
        for i, iiodevice in enumerate(self._channels):
            # Terminate instance of iio-capture for this (channel, iiodevice)
            if self._iio[i]:
                self._iio[i].poll()
                if self._iio[i].returncode is None:
                    self._iio[i].terminate()
                    self._iio[i].wait()
                    self._iio[i] = None

            csv_file = '{}/{}'.format(
                self._res_dir,
                'samples_{}.csv'.format(iiodevice)
            )
            self._iio[i] = Popen([self._iiocapturebin, "-n",
                                  self._hostname, "-o",
                                  "-c", "-f",
                                  csv_file, iiodevice],
                                 stdout=PIPE, stderr=STDOUT)

            if self._iio[i] is None:
                logging.error("%14s - Failed to run %s for %s", "IIOCapture",
                              self._iiocapturebin, iiodevice)
                logging.error("%14s - Make sure there are no iio-capture "
                              "processes connected to %s and device %s",
                              "IIOCapture", self._hostname, iiodevice)
                continue

            # Wait few milliseconds to check if there is any output.
            # If this is the case, there must have been an error, so the user
            # must be informed
            sleep(0.02)
            self._iio[i].poll()
            if self._iio[i].returncode:
                logging.error("%14s - Failed to run %s for %s", "IIOCapture",
                              self._iiocapturebin, iiodevice)
                logging.error("%14s - Make sure there are no iio-capture "
                              "processes connected to %s and device %s",
                              "IIOCapture", self._hostname, iiodevice)
                out, _ = self._iio[i].communicate()
                logging.error("%14s - Output: '%s'", "IIOCapture", out)
                self._iio[i] = None
            else:
                logging.debug("%14s - Started %s on %s...", "IIOCapture",
                              self._iiocapturebin, iiodevice)

    def report(self, out_dir, out_file='energy.json'):
        """
        Stop iio-capture and collect sampled data.

        :param out_dir: Output directory where to store results
        :type out_dir: str

        :param out_file: File name where to save energy data
        :type out_file: str
        """
        channels_nrg = {}
        nrg_files = {}
        for i, iiodevice in enumerate(self._channels):
            if self._iio[i] is None:
                continue

            self._iio[i].poll()
            if self._iio[i].returncode:
                # returncode not None means that iio-capture has terminated
                # already, so there must have been an error
                logging.error("%14s - %s terminated for %s", "IIOCapture",
                              self._iiocapturebin, iiodevice)
                out, _ = self._iio[i].communicate()
                logging.error("%14s - '%s'", "IIOCapture", out)
                self._iio[i] = None
                continue

            # kill process and get return
            self._iio[i].terminate()
            out, _ = self._iio[i].communicate()
            logging.debug("%14s - Completed IIOCapture for %s...",
                          "IIOCapture", iiodevice)
            logging.info("%14s - Got energy %s", "IIOCapture", out)
            self._iio[i].wait()
            self._iio[i] = None

            # iio-capture return "energy=value", add a simple format check
            if '=' not in out:
                logging.error("%14s - Bad output format from %s:",
                              "IIOCapture",
                              iiodevice)
                logging.error("%14s - '%s'", "IIOCapture", out)
                continue

            nrg = {}
            for kv_pair in out.split():
                key, val = kv_pair.partition('=')[::2]
                nrg[key] = val
            channels_nrg['{}_nrg'.format(iiodevice)] = nrg

            # Dump data as JSON file
            nrg_file = '{}/{}_{}'.format(out_dir, iiodevice, out_file)
            with open(nrg_file, 'w') as ofile:
                json.dump(nrg, ofile, sort_keys=True, indent=4)
            nrg_files['{}_nrg_file'.format(iiodevice)] = nrg_file

            # Copy CSV samples file to out_dir
            os.system('cp {}/samples_{}.csv {}'
                      .format(self._res_dir, iiodevice, out_dir))

        return (channels_nrg, nrg_files)

# vim :set tabstop=4 shiftwidth=4 expandtab
