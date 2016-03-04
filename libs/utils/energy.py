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
import subprocess

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
            'kinds' : [ 'energy' ]
        }
    },

    # Hikey: by default use AEP
    'hikey' : {
        'instrument' : 'aep',
    },

    # Salvator-X: by default uses iiocapture
    'salvatorx' : {
	'instrument' : 'iiocapture',
    }

}

class EnergyMeter(object):

    _meter = None

    def __init__(self, target):
        self._target = target

    @staticmethod
    def getInstance(target, conf, force=False):

        if not force and EnergyMeter._meter:
            return EnergyMeter._meter

        # Initialize energy probe to board default
        if 'board' in conf and \
            conf['board'] in DEFAULT_ENERGY_METER:
                emeter = DEFAULT_ENERGY_METER[conf['board']]
                logging.debug('%14s - using default energy meter for [%s]',
                        'EnergyMeter', conf['board'])
        else:
            return None

        if emeter['instrument'] == 'hwmon':
            EnergyMeter._meter = HWMon(target, emeter['conf'])
        elif emeter['instrument'] == 'aep':
            EnergyMeter._meter = Aep(target)
        elif emeter['instrument'] == 'iiocapture':
            EnergyMeter._meter = IIOCapture(target, conf['iiocapturebin'], conf['hostname'], conf['iiodevice'])
        return EnergyMeter._meter

    def sample(self):
        raise NotImplementedError('Missing implementation')

    def reset(self):
        raise NotImplementedError('Missing implementation')

    def report(self, out_dir):
        raise NotImplementedError('Missing implementation')

class HWMon(EnergyMeter):

    def __init__(self, target, hwmon_conf=None):
        super(HWMon, self).__init__(target)

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
        logging.debug('%14s - Enabling channels %s', 'EnergyMeter', hwmon_conf)
        self._hwmon.reset(**hwmon_conf)

        # Logging enabled channels
        logging.info('%14s - Channels selected for energy sampling:',
                     'EnergyMeter')
        for channel in self._hwmon.active_channels:
            logging.info('%14s -    %s', 'EnergyMeter', channel.label)

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
            if self._target.little_core.upper() in ch.upper():
                clusters_nrg['LITTLE'] = '{:.6f}'.format(nrg_total)
            elif self._target.big_core.upper() in ch.upper():
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

class Aep(EnergyMeter):

    def __init__(self, target):
        super(Aep, self).__init__(target)

        # Energy readings
        self.readings = {}

        # Time (start and diff) for power measurment
        self.time = {}

        # Initialize instrument
        # Only one channel (first AEP channel: pc1 ... probe channel 1) is used
        self._aep = devlib.EnergyProbeInstrument(self._target, labels=["pc1"], resistor_values=[0.033])

        # Configure channels for energy measurements
        logging.debug('EnergyMeter - Enabling channels')
        self._aep.reset()

        # Logging enabled channels
        logging.info('%14s - Channels selected for energy sampling:\n%s',
                'EnergyMeter', str(self._aep.active_channels))

    def __calc_nrg(self, samples):

        power = {'sum' : 0, 'count' : 0, 'avg' : 0}

        for s in samples:
            power['sum'] += s[1].value # s[1] ... power value of channel 1
            power['count'] += 1

        power['avg'] =  power['sum'] / power['count']

        nrg = power['avg'] * self.time['diff']

        logging.debug('avg power: %.6f count: %s time: %.6f nrg: %.6f',
	              power['avg'], power['count'], self.time['diff'] , nrg)
        return nrg

    def sample(self):
        if self._aep is None:
            return

        self.time['diff'] = time.time() - self.time['start']
        self._aep.stop()

        csv_data = self._aep.get_data("/tmp/aep.csv")
        samples = csv_data.measurements()

        value = self.__calc_nrg(samples)

        self.readings['last'] = value
        self.readings['delta'] = value
        self.readings['total'] = value

        logging.debug('SAMPLE: %s', self.readings)
        return self.readings

    def reset(self):
        if self._aep is None:
            return

        logging.debug('RESET: %s', self.readings)

        self._aep.start()
        self.time['start'] = time.time()

    def report(self, out_dir, out_file='energy.json'):
        if self._aep is None:
            return

        # Retrieve energy consumption data
        nrg = self.sample()

        # Reformat data for output generation
        clusters_nrg = {}
        clusters_nrg['LITTLE'] = '{:.6f}'.format(self.readings['total'])

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_file)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

        return (clusters_nrg, nrg_file)

class IIOCapture(EnergyMeter):

    def __init__(self, target, iiocapturebin, hostname, iiodevice):
        super(IIOCapture, self).__init__(target)

        self._iiocapturebin = iiocapturebin
        self._hostname = hostname
        self._iiodevice = iiodevice

        self._p = None

    def report(self, out_dir, out_file='energy.json'):
        if self._p is None:
            return
        if self._p.returncode is not None:
            out, err = self._p.communicate()
            logging.error("%14s - Failed to run bin return %d : %s", "IIOCapture", self._p.returncode, out)
            return

        # kill process and get return
        self._p.send_signal(2)
        out, err = self._p.communicate()
        logging.info("%14s - Killed IIOCapture...", "IIOCapture")
        logging.info("%14s - Got energy %s", "IIOCapture", out)
        self._p.wait()
        self._p = None

        # iio-capture return "energy=value", add a simple format check
        if '=' not in out:
            logging.error("%14s - Bad output format '%s'", "IIOCapture", out)
            return

        clusters_nrg = {}
        clusters_nrg['big'] = 0
        clusters_nrg['LITTLE'] = out.split('=')[1].strip()

        # Dump data as JSON file
        nrg_file = '{}/{}'.format(out_dir, out_file)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

        return (clusters_nrg, nrg_file)

    def reset(self):
        if self._p is not None:
            if self._p.returncode is not None:
                self._p.kill()

        self._p = subprocess.Popen([self._iiocapturebin, "-n", self._hostname, "-o", "-e", self._iiodevice],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        logging.info("%14s - Started %s...", "IIOCapture", self._iiocapturebin)
        if self._p.returncode is not None and self._p.returncode != 0:
            logging.error("%14s - Failed to run bin", "IIOCapture")

    def sample(self):
        logging.debug('SAMPLE')

# vim :set tabstop=4 shiftwidth=4 expandtab
