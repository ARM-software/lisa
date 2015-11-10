
import datetime
import json
import logging
import os
import re
import shutil
import sys
import time
import unittest

import devlib

from wlgen import RTA
from energy import EnergyMeter
from conf import JsonConf

from devlib.utils.misc import memoized
from trappy.stats.Topology import Topology

from devlib import Platform

USERNAME_DEFAULT = 'root'
PASSWORD_DEFAULT = ''
WORKING_DIR_DEFAULT = '/data/local/schedtest'
FTRACE_EVENTS_DEFAULT = ['sched:*']
FTRACE_BUFSIZE_DEFAULT = 10240
OUT_PREFIX = 'results'
LATEST_LINK = 'results_latest'

basepath = os.path.dirname(os.path.realpath(__file__))
basepath = basepath.replace('/libs/utils', '')

class ShareState(object):
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

class TestEnv(ShareState):

    _initialized = False

    def __init__(self, target_conf=None, test_conf=None):
        super(TestEnv, self).__init__()

        if self._initialized:
            return

        self.conf = None
        self.target = None
        self.ftrace = None
        self.workdir = WORKING_DIR_DEFAULT
        self.__tools = []
        self.__modules = []
        self.__connection_settings = None
        self._calib = None

        # Keep track of target IP and MAC address
        self.ip = None
        self.mac = None

        # Keep track of last installed kernel
        self.kernel = None
        self.dtb = None

        # Energy meter configuration
        self.emeter = None

        # The platform descriptor to be saved into the results folder
        self.platform = {}

        # Compute base installation path
        logging.info('%14s - Using base path: %s',
                'Target', basepath)

        # Setup target configuration
        if isinstance(target_conf, dict):
            self.conf = target_conf
        elif isinstance(target_conf, str):
            self.conf = TestEnv.loadTargetConfig(target_conf)
        elif target_conf is None:
            self.conf = TestEnv.loadTargetConfig()
        else:
            raise ValueError('target_conf must be either a dictionary or a filepath')

        logging.debug('%14s - Target configuration %s', 'Target', self.conf)

        if 'workdir' in self.conf:
            self.workdir = self.conf['workdir']

        # Initialize binary tools to deploy
        if 'tools' in self.conf:
            self.__tools = self.conf['tools']
        # Merge tests specific tools
        if test_conf and 'tools' in test_conf and test_conf['tools']:
            if 'tools' not in self.conf:
                self.conf['tools'] = []
            self.__tools = list(set(
                self.conf['tools'] + test_conf['tools']
            ))

        # Initialize modules to use on the target
        if 'modules' in self.conf:
            self.__modules = self.conf['modules']
        # Merge tests specific modules
        if test_conf and 'modules' in test_conf and test_conf['modules']:
            if 'modules' not in self.conf:
                self.conf['modules'] = []
            self.__modules = list(set(
                self.conf['modules'] + test_conf['modules']
            ))

        # Initialize ftrace events
        if test_conf and 'ftrace' in test_conf:
            self.conf['ftrace'] = test_conf['ftrace']

        # Initialize features
        if '__features__' not in target_conf:
            target_conf['__features__'] = []

        self._init()

        # Initialize FTrace events collection
        self._init_ftrace(True)

        # Initialize energy probe instrument
        self.emeter = EnergyMeter.getInstance(
                self.target, self.conf, force=True)

        # Initialize RT-App calibration values
        self.calibration()

        # Initialize local results folder
        res_dir = os.path.join(basepath, OUT_PREFIX)
        self.res_dir = datetime.datetime.now()\
                .strftime(res_dir + '/%Y%m%d_%H%M%S')
        os.makedirs(self.res_dir)

        res_lnk = os.path.join(basepath, LATEST_LINK)
        if os.path.islink(res_lnk):
            os.remove(res_lnk)
        os.symlink(self.res_dir, res_lnk)

        self._initialized = True

    @staticmethod
    def loadTargetConfig(filepath='target.config'):
        """
        Load the target configuration from the specified file.

        The configuration file path must be relative to the test suite
        installation root folder.

        :param filepath: A string representing the path of the target
        configuration file. This path must be relative to the root folder of
        the test suite.
        :type filepath: str

        """

        # Loading default target configuration
        conf_file = os.path.join(basepath, filepath)
        logging.info('%14s - Loading target configuration [%s]...',
                'Target', conf_file)
        conf = JsonConf(conf_file)
        return conf.load()

    def _init(self, force = False):

        if self._feature('debug'):
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize target
        self._init_target(force)

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

        # Initialize the platform descriptor
        self._init_platform()


    def _init_target(self, force = False):

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
            if ':' in self.conf['host']:
                (self.mac, self.ip) = self.resolv_host(self.conf['host'])
            else:
                self.ip = self.conf['host']
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
        if self.__tools:
            for tool in self.__tools:
                binary = '{}/tools/scripts/{}'.format(basepath, tool)
                if not os.path.isfile(binary):
                    binary = '{}/tools/{}/{}'\
                            .format(basepath, self.target.abi, tool)
                tools_to_install.append(binary)
        self.target.setup(tools_to_install)

    def ftrace_conf(self, conf):
        self._init_ftrace(True, conf)

    def _init_ftrace(self, force=False, conf=None):

        if not force and self.ftrace is not None:
            return self.ftrace

        if conf is None and 'ftrace' not in self.conf:
            return None

        if conf is not None:
            ftrace = conf
        else:
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

    def _init_energy(self, force):

        # Initialize energy probe to board default
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force)

    def _init_platform(self):
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

        # Adding topology information
        self.platform['topology'] = self.topology.get_level("cluster")

        logging.debug('%14s - Platform descriptor initialized\n%s',
            'Platform', self.platform)
        # self.platform_dump('./')

    def platform_dump(self, dest_dir):
        plt_file = os.path.join(dest_dir, 'platform.json')
        logging.debug('%14s - Dump platform descriptor in [%s]',
            'Platform', plt_file)
        with open(plt_file, 'w') as ofile:
            json.dump(self.platform, ofile, sort_keys=True, indent=4)

    def calibration(self, force=False):

        if not force and self._calib:
            return self._calib

        required = False
        if force:
            required = True
        if 'rt-app' in self.__tools:
            required = True
        elif 'wloads' in self.conf:
            wloads = self.conf['wloads']
            for wl_idx in wloads:
                if 'rt-app' in wloads[wl_idx]['type']:
                    required = True
                    break

        if not required:
            logging.debug('No RT-App workloads, skipping calibration')
            return

        if not force and 'rtapp-calib' in self.conf:
            logging.info('Loading RTApp calibration from configuration file...')
            self._calib = {
                    int(key): int(value)
                    for key, value in self.conf['rtapp-calib'].items()
                }
        else:
            logging.info('Calibrating RTApp...')
            self._calib = RTA.calibrate(self.target)

        logging.info('Using RT-App calibration values: %s', self._calib)
        return self._calib

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
        if self._feature('no-reboot'):
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
        self._init(force)

        # Initialize FTrace events collection
        self._init_ftrace(force)

        # Initialize energy probe instrument
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force)

    def install_kernel(self, tc, reboot=False):

        if self.kernel == tc['kernel'] and self.dtb == tc['dtb']:
            return

        logging.info('%14s - Install kernel [%s] on target...',
                'KernelSetup', tc)

        # Install kernel/dtb via FTFP
        if self._feature('no-kernel'):
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

    def _feature(self, feature):
        return feature in self.conf['__features__']

IFCFG_BCAST_RE = re.compile(
    r'Bcast:(.*) '
)

# vim :set tabstop=4 shiftwidth=4 expandtab
