
import datetime
import json
import logging
import os
import re
import shutil
import sys
import time
import unittest

import libs
import devlib
import wlgen
from devlib.utils.misc import memoized
from trappy.stats.Topology import Topology

from devlib import Platform

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

        self.conf = conf
        self.target = None
        self.ftrace = None
        self.workdir = WORKING_DIR_DEFAULT
        self.__tools = None
        self.__modules = None
        self.__connection_settings = None
        self.calib = None

        # Keep track of target IP and MAC address
        self.ip = None
        self.mac = None

        # Keep track of last installed kernel
        self.kernel = None
        self.dtb = None

        # Default energy measurements for each board
        self.energy_meter = {
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
            },
            'oak' : {
                'instrument' : 'daq',
                'conf' : {
                    'host'              : '10.1.208.38',
                    'port'              : '8888',
                    'channel_map'       : [2, 3, 4, 5, 6, 7, 0, 1],
                    'resistor_values'   : [0.01, 0.1, 0.1, 0.1],
                    'labels'            : ['a72', 'VSRAM-CA7', 'a53', 'VSRAM-CA15'],
                    'sampling_rate'     : 10000
                }
            }
        }
        # Supported energy instruments
        self.daq = None
        self.hwmon = None

        # Energy readings
        self.energy_reading = {}

        # Energy meter configuration
        self.emeter = None

        # The platform descriptor to be saved into the results folder
        self.platform = {}

        if self._init:
            return

        if 'workdir' in self.conf:
            self.workdir = self.conf['workdir']

        # Initialize binary tools to deploy
        if 'tools' in self.conf:
            self.__tools = self.conf['tools']

        # Initialize modules to use on the target
        if 'modules' in self.conf:
            self.__modules = self.conf['modules']

        self.init()

        # Initialize RT-App calibration values
        self.calibrate()

        # Initialize local results folder
        self.res_dir = datetime.datetime.now().strftime(
                           OUT_PREFIX + '/%Y%m%d_%H%M%S')
        os.makedirs(self.res_dir)

        if os.path.islink(LATEST_LINK):
            os.remove(LATEST_LINK)
        os.symlink(self.res_dir, LATEST_LINK)

        self._init = True

    def init(self, force = False):

        if self.feature('debug'):
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize target
        self.init_target(force)

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


    def init_target(self, force = False):

        if not force and self.target is not None:
            return self.target

        self.__connection_settings = {
            'username' : USERNAME_DEFAULT,
        }

        if 'usename' in self.conf:
            self.__connection_settings['username'] = self.conf['username']
        if 'keyfile' in self.conf:
            self.__connection_settings['keyfile'] = self.conf['keyfile']
        elif 'password' in self.conf:
            self.__connection_settings['password'] = self.conf['password']
        else:
            self.__connection_settings['password'] = PASSWORD_DEFAULT

        try:
            (self.mac, self.ip) = self.resolv_host(self.conf['host'])
            self.__connection_settings['host'] = self.ip
        except KeyError:
            raise ValueError('Config error: missing [host] parameter')

        try:
            platform_type = self.conf['platform']
        except KeyError:
            raise ValueError('Config error: missing [platform] parameter')

        logging.info(r'%14s - Connecing %s target with: %s',
                'Target', platform_type, self.__connection_settings)

        if platform_type.lower() == 'linux':
            logging.debug('%14s - Setup LINUX target...', 'Target')
            self.target = devlib.LinuxTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'android':
            logging.debug('%14s - Setup ANDROID target...', 'Target')
            self.target = devlib.AndroidTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        # TO BE REMOVED: Temporary fix for dmidecode not working on Chromebook
        elif platform_type.lower() == 'oak':
            logging.debug('%14s - Setup OAK target...', 'Target')
            self.target = devlib.LinuxTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    platform = Platform(model='MT8173'),
                    modules = self.__modules)
            # Reset the target to a standard linux target
            platform_type == 'linux'
            # Ensure rootfs is RW mounted
            self.target.execute('mount -o remount,rw /', as_root=True)
        elif platform_type.lower() == 'host':
            logging.debug('%14s - Setup HOST target...', 'Target')
            self.target = devlib.LocalLinuxTarget(
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        else:
            raise ValueError('Config error: not supported [platform] type {}'\
                    .format(platform_type))

        logging.debug('%14s - Checking target connection...', 'Target')
        logging.debug('%14s - Target info:', 'Target')
        logging.debug('%14s -       ABI: %s', 'Target', self.target.abi)
        logging.debug('%14s -      CPUs: %s', 'Target', self.target.cpuinfo)
        logging.debug('%14s -  Clusters: %s', 'Target', self.target.core_clusters)

        logging.info('%14s - Initializing target workdir [%s]',
                    'Target', self.target.working_directory)
        tools_to_install = []
        for tool in self.__tools:
            binary = './tools/scripts/{}'.format(tool)
            if not os.path.isfile(binary):
                binary = './tools/{}/{}'.format(self.target.abi, tool)
            tools_to_install.append(binary)
        self.target.setup(tools_to_install)

        # Initialize the platform descriptor
        self.init_platform()

    def init_ftrace(self, force=False):

        if not force and self.ftrace is not None:
            return self.ftrace

        if 'ftrace' not in self.conf:
            return None

        ftrace = self.conf['ftrace']

        events = FTRACE_EVENTS_DEFAULT
        if 'events' in ftrace:
            events = ftrace['events']

        buffsize = FTRACE_BUFSIZE_DEFAULT
        if 'buffsize' in ftrace:
            buffsize = ftrace['buffsize']

        self.ftrace = devlib.FtraceCollector(
            self.target,
            events      = events,
            buffer_size = buffsize,
            autoreport  = False,
            autoview    = False
        )

        logging.info('%14s - Enabled events:', 'FTrace')
        logging.info('%14s -   %s', 'FTrace', events)

        return self.ftrace

    def init_energy(self, force):

        # Initialize energy probe to board default
        if 'board' in self.conf:
            if self.conf['board'] in self.energy_meter:
                self.emeter = self.energy_meter[self.conf['board']]
                logging.debug('%14s - using default energy meter for [%s]',
                        'EnergyMeter', self.conf['board'])

        if self.emeter['instrument'] == 'hwmon':
           self.hwmon_init(force)
        if self.emeter['instrument'] == 'daq':
            self.daq_init(force)

    def init_platform(self):
        self.platform = {
            'clusters' : {
                'little'    : self.target.bl.littles,
                'big'       : self.target.bl.bigs
            },
            'freqs' : {
                'little'    : self.target.bl.list_littles_frequencies(),
                'big'       : self.target.bl.list_bigs_frequencies()
            }
        }
        self.platform['cpus_count'] = \
            len(self.platform['clusters']['little']) + \
            len(self.platform['clusters']['big'])
        if 'nrg_model' in self.conf:
            self.platform['nrg_model'] = self.conf['nrg_model']

        logging.debug('%14s - Platform descriptor initialized\n%s',
            'Platform', self.platform)

    def platform_dump(self, dest_dir):
        plt_file = os.path.join(dest_dir, 'platform.json')
        logging.debug('%14s - Dump platform descriptor in [%s]',
            'Platform', plt_file)
        with open(plt_file, 'w') as ofile:
            json.dump(self.platform, ofile, sort_keys=True, indent=4)

    def calibrate(self):

        if self.calib is not None:
            return self.calib

        required = False
        wloads = self.conf['wloads']
        for wl_idx in wloads:
            if 'rt-app' in wloads[wl_idx]['type']:
                required = True
                break

        if not required:
            logging.debug('No RT-App workloads, skipping calibration')
            return

        if 'rtapp-calib' in self.conf:
            logging.info('Loading RTApp calibration from configuration file...')
            self.calib = {
                    int(key): int(value)
                    for key, value in self.conf['rtapp-calib'].items()
                }
        else:
            logging.info('Calibrating RTApp...')
            self.calib = wlgen.RTA.calibrate(self.target)

        logging.info('Using RT-App calibration values: %s', self.calib)


    def hwmon_init(self, force=False):

        if not force and self.hwmon is not None:
            return self.hwmon

        if 'hwmon' not in self.__modules:
            logging.info('%14s - HWMON module not enabled',
                    'EnergyMeter')
            logging.warning('%14s - Energy sampling disabled by configuration',
                    'EnergyMeter')
            self.hwmon = None
            return

        # Initialize HWMON instrument
        self.hwmon = devlib.HwmonInstrument(self.target)

        # Configure channels for energy measurements
        hwmon_conf = self.emeter['conf']
        logging.debug('%14s - Enabling channels %s', 'EnergyMeter', hwmon_conf)
        self.hwmon.reset(**hwmon_conf)

        # Logging enabled channels
        logging.info('%14s - Channels selected for energy sampling:\n%s',
                'EnergyMeter', str(self.hwmon.active_channels))

    def daq_init(self, force=False):

        if not force and self.daq is not None:
            return self.daq

        if 'daq' not in self.__modules:
            logging.info('%14s - DAQ module not enabled',
                    'EnergyMeter')
            logging.warning('%14s - Energy sampling disabled by configuration',
                    'EnergyMeter')
            self.daq = None
            return

        # Initialize DAQ instrument
        daq_conf = self.emeter['conf']
        logging.debug('%14s - Using configuration:', 'EnergyMeter')
        logging.debug('%14s -   %s', 'EnergyMeter', daq_conf)
        self.daq = devlib.DaqInstrument(self.target, **daq_conf)

        # Configure channels for energy measurements
        self.daq.reset()

        # # Logging enabled channels
        # logging.info('%14s - Channels selected for energy sampling:\n%s',
        #         'EnergyMeter', str(self.daq.active_channels))

    def hwmon_sample(self):
        if self.hwmon is None:
            return
        samples = self.hwmon.take_measurement()
        for s in samples:
            label = s.channel.label\
                    .replace('_energy', '')\
                    .replace(" ", "_")
            value = s.value

            if label not in self.energy_reading:
                self.energy_reading[label] = {
                        'last'  : value,
                        'delta' : 0,
                        'total' : 0
                        }
                continue

            last  = self.energy_reading[label]['last']
            delta = value - last
            total = self.energy_reading[label]['total']

            self.energy_reading[label]['last']  = value
            self.energy_reading[label]['delta'] = delta
            self.energy_reading[label]['total'] = total + delta

        # logging.debug('SAMPLE: %s', self.energy_reading)
        return self.energy_reading

    def daq_sample(self):
        if self.daq is None:
            return
        self.daq.start()

    def energy_sample(self):
        if self.hwmon:
            return self.hwmon_sample()
        if self.daq:
            return self.daq_sample()

    def hwmon_reset(self):
        if self.hwmon is None:
            return
        self.energy_sample()
        for label in self.energy_reading:
            self.energy_reading[label]['delta'] = 0
            self.energy_reading[label]['total'] = 0
        # logging.debug('RESET: %s', self.energy_reading)

    def daq_reset(self):
        if self.daq is None:
            return
        self.daq.reset()

    def energy_reset(self):
        if self.hwmon:
            self.hwmon_reset()
        if self.daq:
            self.daq_reset()

    def hwmon_report(self, out_dir):
        if self.hwmon is None:
            return
        # Retrive energy consumption data
        nrg = self.energy_sample()
        # Reformat data for output generation
        clusters_nrg = {}
        for ch in nrg:
            nrg_total = nrg[ch]['total']
            logging.info('%14s - Energy [%16s]: %.6f',
                    'EnergyReport', ch, nrg_total)
            if self.target.little_core.upper() in ch.upper():
                clusters_nrg['LITTLE'] = '{:.6f}'.format(nrg_total)
            elif self.target.big_core.upper() in ch.upper():
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
        nrg_file = '{}/energy.json'.format(out_dir)
        with open(nrg_file, 'w') as ofile:
            json.dump(clusters_nrg, ofile, sort_keys=True, indent=4)

    def daq_report(self, out_dir):
        if self.daq is None:
            return
        # Stop DAQ data acquisition
        self.daq.stop()
        # Retrive energy consumption data
        self.daq.get_data(out_dir)

        # # TODO compute and report cluster energy consumption
    def energy_report(self, out_dir):
        if self.hwmon:
            self.hwmon_report(out_dir)
        if self.daq:
            self.daq_report(out_dir)

    def resolv_host(self, host=None):
        if host is None:
            host = self.conf['host']

        # Refresh ARP for local network IPs
        logging.debug('%14s - Collecting all Bcast address', 'HostResolver')
        output = os.popen(r'ifconfig').read().split('\n')
        for line in output:
            match = IFCFG_BCAST_RE.search(line)
            if not match:
                continue
            baddr = match.group(1)
            try:
                cmd = r'nmap -T4 -sP {}/24 &>/dev/null'.format(baddr.strip())
                logging.debug('%14s - %s', 'HostResolver', cmd)
                os.popen(cmd)
            except RuntimeError:
                logging.warning('%14s - Nmap not available, try IP lookup using broadcast ping')
                cmd = r'ping -b -c1 {} &>/dev/null'.format(baddr)
                logging.debug('%14s - %s', 'HostResolver', cmd)
                os.popen(cmd)

        if ':' in host:
            # Assuming this is a MAC address
            # TODO add a suitable check on MAC address format
            # Query ARP for the specified HW address
            ARP_RE = re.compile(
                r'([^ ]*).*({}|{})'.format(host.lower(), host.upper())
            )
        else:
            # Assuming this is an IP address
            # TODO add a suitable check on IP address format
            # Query ARP for the specified IP address
            ARP_RE = re.compile(
                r'{}.*ether *([0-9a-fA-F:]*)'.format(host)
            )

        output = os.popen(r'arp -n')
        ipaddr = '0.0.0.0'
        for line in output:
            match = ARP_RE.search(line)
            if not match:
                continue
            ipaddr = match.group(1)
            break
        if ipaddr == '0.0.0.0':
            raise ValueError('Unable to lookup for target IP address')
        logging.info('%14s - Target (%s) at IP address: %s',
                'HostResolver', host, ipaddr)
        return (host, ipaddr)

    def reboot(self, reboot_time=60):
        # Send remote target a reboot command
        if self.feature('no-reboot'):
            logging.warning('%14s - Reboot disabled by conf features', 'Reboot')
        else:
            self.target.execute('sleep 2 && reboot -f &', as_root=True)

            # Wait for the target to complete the reboot
            logging.info('%14s - Waiting %s [s]for target to reboot...',
                    'Reboot', reboot_time)
            time.sleep(reboot_time)

        # Force re-initialization of all the devlib modules
        force = True

        # Reset the connection to the target
        self.init(force)

        # Initialize FTrace events collection
        self.init_ftrace(force)

        # Initialize energy probe instrument
        self.init_energy(force)

    def install_kernel(self, tc, reboot=False):

        if self.kernel == tc['kernel'] and self.dtb == tc['dtb']:
            return

        logging.info('%14s - Install kernel [%s] on target...',
                'KernelSetup', tc)

        # Install kernel/dtb via FTFP
        if self.feature('no-kernel'):
            logging.warning('%14s - Kernel deploy disabled by conf features',
                    'KernelSetup')

        elif 'tftp' in self.conf:
            logging.info('%14s - Deply kernel via FTFP...', 'KernelSetup')

            # Deply kernel in FTFP folder (madatory)
            if 'kernel' not in tc:
                raise ValueError('Missing "kernel" paramtere in conf: %s',
                        'KernelSetup', tc)
            self.tftp_deploy(tc['kernel'])

            # Deploy DTB in TFTP folder (if provided)
            if 'dtb' not in tc:
                logging.warn('%14s - DTB not provided for current conf: %s',
                        'KernelSetup', tc)
                logging.warn('%14s - Using pre-installed DTB', 'KernelSetup')
            else:
                self.tftp_deploy(tc['dtb'])

        else:
            raise ValueError('%14s - Kernel installation method not supported',
                    'KernelSetup')

        # Keep track of last installed kernel
        self.kernel = tc['kernel']
        if 'dtb' in tc:
            self.dtb = tc['dtb']

        if not reboot:
            return

        # Reboot target
        logging.info('%14s - Rebooting taget...', 'KernelSetup')
        self.reboot()


    def tftp_deploy(self, src):

        tftp = self.conf['tftp']

        dst = tftp['folder']
        if 'kernel' in src:
            dst = os.path.join(dst, tftp['kernel'])
        elif 'dtb' in src:
            dst = os.path.join(dst, tftp['dtb'])
        else:
            dst = os.path.join(dst, os.path.basename(src))

        cmd = 'cp {} {}'.format(src, dst)
        logging.info('%14s - Deploy %s into %s',
                'TFTP', src, dst)
        result = os.system(cmd)
        if result != 0:
            logging.error('%14s - Failed to deploy image: %s',
                    'FTFP', src)
            raise ValueError('copy error')

    def feature(self, feature):
        return feature in self.conf['__features__']

IFCFG_BCAST_RE = re.compile(
    r'Bcast:(.*) '
)

# vim :set tabstop=4 shiftwidth=4 expandtab
