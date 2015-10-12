
import datetime
import json
import logging
import os
import shutil
import sys
import unittest

import libs
import devlib
import wlgen
from devlib.utils.misc import memoized
from trappy.stats.Topology import Topology

USERNAME_DEFAULT = 'root'
PASSWORD_DEFAULT = ''
WORKING_DIR_DEFAULT = '/data/local/schedtest'
FTRACE_EVENTS_DEFAULT = ['sched:*']
FTRACE_BUFSIZE_DEFAULT = 10240
OUT_PREFIX = './results'
LATEST_LINK = './results_latest'

class ShareState(object):
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

class TestEnv(ShareState):

    _init = False

    def __init__(self, conf=None):
        super(TestEnv, self).__init__()

        self.target = None
        self.ftrace = None
        self.workdir = WORKING_DIR_DEFAULT
        self.__tools = None
        self.__modules = None
        self.__connection_settings = None
        self.calib = None

        # Default energy measurements for each board
        self.energy_probe = {
            'tc2' : {
                'instrument' : 'hwmon',
                'conf' : {
                    'sites' : [ 'A7 Jcore', 'A15 Jcore' ],
                    'kinds' : [ 'energy']
                }
            },
            'juno' : {
                'instrument' : 'hwmon',
                'conf' : {
                    'sites' : [ 'a53', 'a57' ],
                    'kinds' : [ 'energy' ]
                }
            }
        }
        self.eprobe = None
        self.eprobe_readings = {}

        if self._init:
            return

        if 'workdir' in conf.keys():
            self.workdir = conf['workdir']

        # Initialize binary tools to deploy
        if 'tools' in conf.keys():
            self.__tools = conf['tools']

        # Initialize modules to use on the target
        if 'modules' in conf.keys():
            self.__modules = conf['modules']

        # Initialize target
        self.init_target(conf)

        # Initialize FTrace events collection
        self.init_ftrace(conf)

        # Initialize energy probe instrument
        self.init_energy(conf)

        # # Initialize RT-App calibration values
        self.calibrate(conf)

        # Initialize target Topology for behavior analysis
        CLUSTERS = []
        if self.target.abi == 'arm64' or self.target.abi == 'armeabi':
            CLUSTERS.append(
                [i for i,t in enumerate(self.target.core_names)
                            if t == self.target.little_core])
            CLUSTERS.append(
                [i for i,t in enumerate(self.target.core_names)
                            if t == self.target.big_core])
        elif self.target.abi == 'x86_64':
            for c in set(self.target.core_cluster):
                CLUSTERS.append(
                    [i for i,v in enumerate(self.target.core_clusters)
                                if v == c])
        self.topology = Topology(clusters=CLUSTERS)
        logging.info('Target topology: %s', CLUSTERS)

        self.res_dir = datetime.datetime.now().strftime(
                           OUT_PREFIX + '/%Y%m%d_%H%M%S')
        os.mkdir(self.res_dir)

        if os.path.islink(LATEST_LINK):
            os.remove(LATEST_LINK)
        os.symlink(self.res_dir, LATEST_LINK)

        self._init = True

    def init_target(self, conf):

        if self.target is not None:
            return self.target

        self.__connection_settings = {
            'username' : USERNAME_DEFAULT,
        }

        if 'usename' in conf.keys():
            self.__connection_settings['username'] = conf['username']
        if 'keyfile' in conf.keys():
            self.__connection_settings['keyfile'] = conf['keyfile']
        elif 'password' in conf.keys():
            self.__connection_settings['password'] = conf['password']
        else:
            self.__connection_settings['password'] = PASSWORD_DEFAULT

        try:
            self.__connection_settings['host'] = conf['host']
        except KeyError:
            raise ValueError('Config error: missing [host] parameter')

        try:
            platform_type = conf['platform']
        except KeyError:
            raise ValueError('Config error: missing [platform] parameter')

        logging.info(r'Connecing %s target with: %s',
                platform_type, self.__connection_settings)

        if platform_type.lower() == 'linux':
            logging.debug('Setup LINUX target...')
            self.target = devlib.LinuxTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'android':
            logging.debug('Setup ANDROID target...')
            self.target = devlib.AndroidTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'host':
            logging.debug('Setup HOST target...')
            self.target = devlib.LocalLinuxTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        else:
            raise ValueError('Config error: not supported [platform] type {}'\
                    .format(platform_type))

        logging.debug('Checking target connection...')
        logging.debug('Target info:')
        logging.debug('      ABI: %s', self.target.abi)
        logging.debug('     CPUs: %s', self.target.cpuinfo)
        logging.debug(' Clusters: %s', self.target.core_clusters)

        logging.info('Initializing target workdir [%s]',
                        self.target.working_directory)
        tools_to_install = []
        for tool in self.__tools:
            binary = './tools/scripts/{}'.format(tool)
            if not os.path.isfile(binary):
                binary = './tools/{}/{}'.format(self.target.abi, tool)
            tools_to_install.append(binary)
        self.target.setup(tools_to_install)

    def init_ftrace(self, conf):

        if self.ftrace is not None:
            return self.ftrace

        if 'ftrace' not in conf.keys():
            return None

        ftrace = conf['ftrace']

        events = FTRACE_EVENTS_DEFAULT
        if 'events' in ftrace.keys():
            events = ftrace['events']

        buffsize = FTRACE_BUFSIZE_DEFAULT
        if 'buffsize' in ftrace.keys():
            buffsize = ftrace['buffsize']

        self.ftrace = devlib.FtraceCollector(
            self.target,
            events      = events,
            buffer_size = buffsize,
            autoreport  = False,
            autoview    = False
        )

        return self.ftrace

    def init_energy(self, conf):

        # Initialize energy probe to board default
        if 'board' in conf.keys():
            if conf['board'] in self.energy_probe.keys():
                eprobe = self.energy_probe[conf['board']]
                logging.debug('%14s - using default instrument for [%s]',
                        'EnergyProbe', conf['board'])

        if eprobe['instrument'] == 'hwmon':
           self.hwmon_init(conf)


    def calibrate(self, conf):

        if self.calib is not None:
            return self.calib

        if 'rtapp-calib' in conf.keys():
            logging.info('Loading RTApp calibration from configuration file...')
            self.calib = {
                    int(key): int(value)
                    for key, value in conf['rtapp-calib'].items()
                }
        else:
            logging.info('Calibrating RTApp...')
            self.calib = wlgen.RTA.calibrate(self.target)

        logging.info('Using RT-App calibration values: %s', self.calib)


    def hwmon_init(self, conf):
        # Initialize HWMON instrument
        self.eprobe = devlib.HwmonInstrument(self.target)

        # Configure channels for energy measurements
        probes_conf = self.energy_probe[conf['board']]['conf']
        logging.debug('%14s - Enabling channels %s', 'EnergyProbe', probes_conf)
        self.eprobe.reset(**probes_conf)

        # Logging enabled channels
        logging.info('%14s - Channels selected for energy sampling:\n%s',
                'EnergyProbe', str(self.eprobe.active_channels))

    def energy_sample(self):
        if self.eprobe is None:
            return
        samples = self.eprobe.take_measurement()
        for s in samples:
            label = s.channel.label\
                    .replace('_energy', '')\
                    .replace(" ", "_")
            value = s.value

            if label not in self.eprobe_readings:
                self.eprobe_readings[label] = {
                        'last'  : value,
                        'delta' : 0,
                        'total' : 0
                        }
                continue

            last  = self.eprobe_readings[label]['last']
            delta = value - last
            total = self.eprobe_readings[label]['total']

            self.eprobe_readings[label]['last']  = value
            self.eprobe_readings[label]['delta'] = delta
            self.eprobe_readings[label]['total'] = total + delta

        # logging.debug('SAMPLE: %s', self.eprobe_readings)
        return self.eprobe_readings

    def energy_reset(self):
        if self.eprobe is None:
            return
        self.energy_sample()
        for label in self.eprobe_readings:
            self.eprobe_readings[label]['delta'] = 0
            self.eprobe_readings[label]['total'] = 0
        # logging.debug('RESET: %s', self.eprobe_readings)


    def energy_report(self, out_dir):
        # Retrive energy consumption data
        nrg = self.energy_sample()
        # Reformat data for output generation
        clusters_nrg = {}
        for ch in nrg:
            nrg_total = nrg[ch]['total']
            logging.info('%14s - Energy [%16s]: %.6f',
                    'EnergyReport', ch, nrg_total)
            if self.target.little_core in ch:
                clusters_nrg['LITTLE'] = '{:.6f}'.format(nrg_total)
            elif self.target.big_core in ch:
                clusters_nrg['big'] = '{:.6f}'.format(nrg_total)
            else:
                logging.warning('%14s - Unable to bind hwmon channel [%s]'\
                        ' to a big.LITTLE cluster',
                        'EnergyReport', ch)
                clusters_nrg[ch] = '{:.6f}'.format(nrg_total)
        # Dump data as JSON file
        nrg_file = '{}/energy.json'.format(out_dir)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

# vim :set tabstop=4 shiftwidth=4 expandtab
