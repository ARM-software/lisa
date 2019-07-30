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

import abc
import json
import os
import os.path
import psutil
import time
import logging
import inspect
import abc

from collections import namedtuple
from collections.abc import Mapping
from subprocess import Popen, PIPE, STDOUT
import subprocess
from time import sleep

import numpy as np
import pandas as pd

import devlib

from lisa.utils import Loggable, get_subclasses, ArtifactPath, HideExekallID
from lisa.datautils import series_integrate
from lisa.conf import (
    SimpleMultiSrcConf, KeyDesc, TopLevelKeyDesc, Configurable,
    StrList, FloatList
)
from lisa.target import Target

# Default energy measurements for each board
EnergyReport = namedtuple('EnergyReport',
                          ['channels', 'report_file', 'data_frame'])


class EnergyMeter(Loggable, Configurable):
    """
    Abstract Base Class of energy meters.
    """
    def __init__(self, target, res_dir=None):
        self._target = target
        res_dir = res_dir if res_dir else target.get_res_dir(
            name='EnergyMeter-{}'.format(self.name),
            symlink=False,
        )
        self._res_dir = res_dir


    @classmethod
    def from_conf(cls, target, conf, res_dir=None):
        """
        Build an instance of :class:`EnergyMeter` from a
        configuration object.

        :param target: Target to use
        :type target: lisa.target.Target

        :param conf: Configuration object to use

        :param res_dir: Result directory to use
        :type res_dir: str or None
        """
        # Select the right subclass according to the type of the configuration
        # object we are given
        for subcls in get_subclasses(cls) | {cls}:
            try:
                conf_cls = subcls.CONF_CLASS
            except AttributeError:
                continue
            if isinstance(conf, conf_cls):
                cls = subcls
                break

        cls.get_logger('{} energy meter configuration:\n{}'.format(cls.name, conf))
        kwargs = cls.conf_to_init_kwargs(conf)
        kwargs.update({
            'target': target,
            'res_dir': res_dir,
        })
        cls.check_init_param(**kwargs)
        return cls(**kwargs)

    @abc.abstractmethod
    def name():
        pass

    @abc.abstractmethod
    def sample(self):
        """
        Get a sample from the energy meter
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the energy meter
        """
        pass

    @abc.abstractmethod
    def report(self):
        """
        Get total energy consumption since last :meth:`reset`
        """
        pass

class HWMonConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class for :class:`HWMon`.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('hwmon-conf', 'HWMon Energy Meter configuration', (
        #TODO: find a better help and maybe a better type
        KeyDesc('channel-map', 'Channels to use', [Mapping]),
    ))

class HWMon(EnergyMeter):
    """
    HWMon energy meter

    {configurable_params}
    """

    CONF_CLASS = HWMonConf
    name = 'hwmon'

    def __init__(self, target, channel_map, res_dir=None):
        super().__init__(target, res_dir)
        logger = self.get_logger()

        # Energy readings
        self.readings = {}

        if 'hwmon' not in self._target.modules:
            raise RuntimeError('HWMON devlib module not enabled')

        # Initialize HWMON instrument
        logger.info('Scanning for HWMON channels, may take some time...')
        self._hwmon = devlib.HwmonInstrument(self._target)

        # Decide which channels we'll collect data from.
        # If the caller provided a channel_map, require that all the named
        # channels exist.
        # Otherwise, try using the big.LITTLE core names as channel names.
        # If they don't match, just collect all available channels.

        available_sites = [c.site for c in self._hwmon.get_channels('energy')]

        self._channels = channel_map
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
                logger.info('Using default big.LITTLE hwmon channels')
                self._channels = dict(zip(['big', 'LITTLE'], bl_sites))

        if not self._channels:
            logger.info('Using all hwmon energy channels')
            self._channels = {site: site for site in available_sites}

        # Configure channels for energy measurements
        logger.debug('Enabling channels %s', list(self._channels.values()))
        self._hwmon.reset(kinds=['energy'], sites=list(self._channels.values()))

        # Logging enabled channels
        logger.info('Channels selected for energy sampling:')
        for channel in self._hwmon.active_channels:
            logger.info('   %s', channel.label)


    def sample(self):
        logger = self.get_logger()
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

        logger.debug('SAMPLE: %s', self.readings)
        return self.readings

    def reset(self):
        self.sample()
        for site in self.readings:
            self.readings[site]['delta'] = 0
            self.readings[site]['total'] = 0
        self.get_logger().debug('RESET: %s', self.readings)

    def report(self, out_dir, out_file='energy.json'):
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
            self.get_logger().debug('Energy [%16s]: %.6f', site, nrg_total)
            clusters_nrg[channel] = nrg_total

        # Dump data as JSON file
        nrg_file = os.path.join(out_dir, out_file)
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
                channels_nrg[site] = series_integrate(df[site]['power'], method='trapz')
        return channels_nrg

class AEPConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class for :class:`AEP`.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('aep-conf', 'AEP Energy Meter configuration', (
        KeyDesc('channel-map', 'Channels to use', [Mapping]),
        KeyDesc('resistor-values', 'Resistor values', [FloatList]),
        KeyDesc('labels', 'List of labels', [StrList]),
        KeyDesc('device-entry', 'TTY device', [StrList]),
    ))

class AEP(_DevlibContinuousEnergyMeter):
    """
    Arm Energy Probe energy meter

    {configurable_params}
    """
    name = 'aep'
    CONF_CLASS = AEPConf

    def __init__(self, target, resistor_values, labels=None, device_entry='/dev/ttyACM0', res_dir=None):
        super().__init__(target, res_dir)
        logger = self.get_logger()

        # Configure channels for energy measurements
        self._instrument = devlib.EnergyProbeInstrument(
            self._target,
            resistor_values=resistor_values,
            labels=labels,
            device_entry=device_entry,
        )

        # Configure channels for energy measurements
        logger.debug('Enabling channels')
        self._instrument.reset()

        # Logging enabled channels
        logger.info('Channels selected for energy sampling:')
        logger.info('   %s', str(self._instrument.active_channels))
        logger.debug('Results dir: %s', self._res_dir)

class MonsoonConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class for :class:`Monsoon`.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('monsoon-conf', 'Monsoon Energy Meter configuration', (
        KeyDesc('channel-map', 'Channels to use', [Mapping]),
        KeyDesc('monsoon-bin', 'monsoon binary path', [str]),
        KeyDesc('tty-device', 'TTY device to use', [str]),
    ))

class Monsoon(_DevlibContinuousEnergyMeter):
    """
    Monsoon Solutions energy meter

    {configurable_params}
    """
    name = 'monsoon'
    CONF_CLASS = MonsoonConf

    def __init__(self, target, monsoon_bin=None, tty_device=None, res_dir=None):
        super().__init__(target, res_dir)

        self._instrument = devlib.MonsoonInstrument(self._target,
            monsoon_bin=monsoon_bin, tty_device=tty_device)
        self._instrument.reset()

_acme_install_instructions = '''

  If you need to measure energy using an ACME EnergyProbe,
  please do follow installation instructions available here:
     https://github.com/ARM-software/lisa/wiki/Energy-Meters-Requirements#iiocapture---baylibre-acme-cape

  Othwerwise, please select a different energy meter in your
  configuration file.

'''

class ACMEConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class for :class:`ACME`.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('acme-conf', 'ACME Energy Meter configuration', (
        KeyDesc('channel-map', 'Channels to use', [Mapping]),
        KeyDesc('host', 'Hostname or IP address of the ACME board', [str]),
        KeyDesc('iio-capture-bin', 'path to iio-capture binary', [str]),
    ))

class ACME(EnergyMeter):
    """
    BayLibre's ACME board based EnergyMeter

    {configurable_params}
    """
    name = 'acme'

    CONF_CLASS = ACMEConf

    REPORT_DELAY_S = 2.0
    """
    iio-capture returns an empty string if killed right after its invocation,
    so we have to enforce a delay between reset() and report()
    """

    def __init__(self, target,
          channel_map={'CH0': 0},
          host='baylibre-acme.local', iio_capture_bin='iio-capture',
          res_dir=None):
        super().__init__(target, res_dir)
        logger = self.get_logger()

        self._iiocapturebin = iio_capture_bin
        self._hostname = host

        # Make a copy to be sure to never modify the default value
        self._channels = dict(channel_map)
        self._iio = {}

        logger.info('ACME configuration:')
        logger.info('    binary: %s', self._iiocapturebin)
        logger.info('    device: %s', self._hostname)
        logger.info('  channels:')
        for channel in self._channels:
            logger.info('     %s', self._str(channel))

        # Check if iio-capture binary is available
        try:
            p = subprocess.call([self._iiocapturebin, '-h'], stdout=PIPE, stderr=STDOUT)
        except FileNotFoundError as e:
            logger.error('iio-capture binary [%s] not available',
                            self._iiocapturebin)
            logger.warning(_acme_install_instructions)
            raise FileNotFoundError('Missing iio-capture binary') from e

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
        logger = self.get_logger()
        # Terminate already running iio-capture instance (if any)
        wait_for_termination = 0
        for proc in psutil.process_iter():
            if self._iiocapturebin not in proc.cmdline():
                continue
            for channel in self._channels:
                if self._iio_device(channel) in proc.cmdline():
                    logger.debug('Killing previous iio-capture for [%s]',
                                     self._iio_device(channel))
                    logger.debug(proc.cmdline())
                    proc.kill()
                    wait_for_termination = 2

        # Wait for previous instances to be killed
        sleep(wait_for_termination)

        # Start iio-capture for all channels required
        for channel in self._channels:
            ch_id = self._channels[channel]

            # Setup CSV file to collect samples for this channel
            csv_file = os.path.join(self._res_dir, 'samples_{}.csv'.format(channel))

            # Start a dedicated iio-capture instance for this channel
            self._iio[ch_id] = Popen(['stdbuf', '-i0', '-o0', '-e0',
                                       self._iiocapturebin, '-n',
                                       self._hostname, '-o',
                                       '-c', '-f',
                                       str(csv_file),
                                       self._iio_device(channel)],
                                       stdout=PIPE, stderr=STDOUT,
                                       universal_newlines=True)

        # Wait some time before to check if there is any output
        sleep(1)

        # Check that all required channels have been started
        for channel in self._channels:
            ch_id = self._channels[channel]

            self._iio[ch_id].poll()
            if self._iio[ch_id].returncode:
                logger.error('Failed to run %s for %s',
                                 self._iiocapturebin, self._str(channel))
                logger.warning('\n\n'\
                    '  Make sure there are no iio-capture processes\n'\
                    '  connected to %s and device %s\n',
                    self._hostname, self._str(channel))
                out, _ = self._iio[ch_id].communicate()
                logger.error('Output: [%s]', out.strip())
                self._iio[ch_id] = None
                raise RuntimeError('iio-capture connection error')

        logger.debug('Started %s on %s...',
                        self._iiocapturebin, self._str(channel))
        self.reset_time = time.monotonic()

    def report(self, out_dir, out_energy='energy.json'):
        """
        Stop iio-capture and collect sampled data.

        :param out_dir: Output directory where to store results
        :type out_dir: str

        :param out_file: File name where to save energy data
        :type out_file: str
        """

        delta = time.monotonic() - self.reset_time
        if delta < self.REPORT_DELAY_S:
            sleep(self.REPORT_DELAY_S - delta)

        logger = self.get_logger()
        channels_nrg = {}
        channels_stats = {}
        for channel, ch_id in self._channels.items():

            if self._iio[ch_id] is None:
                continue

            self._iio[ch_id].poll()
            if self._iio[ch_id].returncode:
                # returncode not None means that iio-capture has terminated
                # already, so there must have been an error
                logger.error('%s terminated for %s',
                                self._iiocapturebin, self._str(channel))
                out, _ = self._iio[ch_id].communicate()
                logger.error('[%s]', out)
                self._iio[ch_id] = None
                continue

            # kill process and get return
            self._iio[ch_id].terminate()
            out, _ = self._iio[ch_id].communicate()
            self._iio[ch_id].wait()
            self._iio[ch_id] = None

            # iio-capture return "energy=value", add a simple format check
            if '=' not in out:
                logger.error('Bad output format for %s:',
                                self._str(channel))
                logger.error('[%s]', out)
                continue
            else:
                logger.debug('%s: %s', self._str(channel), out)

            # Build energy counter object
            nrg = {}
            for kv_pair in out.split():
                key, val = kv_pair.partition('=')[::2]
                nrg[key] = float(val)
            channels_stats[channel] = nrg

            logger.debug(self._str(channel))
            logger.debug(nrg)

            # Save CSV samples file to out_dir
            os.system('mv {} {}'.format(
                os.path.join(self._res_dir, 'samples_{}.csv'.format(channel)),
                out_dir))

            # Add channel's energy to return results
            channels_nrg['{}'.format(channel)] = nrg['energy']

        # Dump energy data
        nrg_file = os.path.join(out_dir, out_energy)
        with open(nrg_file, 'w') as ofile:
            json.dump(channels_nrg, ofile, sort_keys=True, indent=4)

        # Dump energy stats
        nrg_stats_file = os.path.splitext(out_energy)[0] + \
                        '_stats' + os.path.splitext(out_energy)[1]
        nrg_stats_file = os.path.join(out_dir, nrg_stats_file)
        with open(nrg_stats_file, 'w') as ofile:
            json.dump(channels_stats, ofile, sort_keys=True, indent=4)

        return EnergyReport(channels_nrg, nrg_file, None)

class Gem5EnergyMeterConf(SimpleMultiSrcConf, HideExekallID):
    """
    Configuration class for :class:`Gem5EnergyMeter`.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('gem5-energy-meter-conf', 'Gem5 Energy Meter configuration', (
        KeyDesc('channel-map', 'Channels to use', [Mapping]),
    ))

class Gem5EnergyMeter(_DevlibContinuousEnergyMeter):
    name = 'gem5'
    CONF_CLASS = Gem5EnergyMeterConf

    def __init__(self, target, channel_map, res_dir=None):
        super().__init__(target, res_dir)

        power_sites = list(channel_map.values())
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
