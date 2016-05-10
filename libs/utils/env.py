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

    def __init__(self, target_conf=None, test_conf=None, wipe=True):
        """
        Initialize the LISA test environment.

        The test environment is defined by:
        - a target configuration (target_conf) defining which HW platform we
        want to use to run the experiments
        - a test configuration (test_conf) defining which SW setups we need on
        that HW target
        - a folder to collect the experiments results, which can be specified
        using the test_conf::results_dir option and is by default wiped from
        all the previous contents (if wipe=True)

        :param target_conf: the HW target we want to use
        :type target_conf: dict

        :param test_conf: the SW setup of the HW target in use
        :type test_conf: dict

        :param wipe: set true to cleanup all previous content from the output
        folder
        :type wipe: bool
        """
        super(TestEnv, self).__init__()

        if self._initialized:
            return

        self.conf = None
        self.test_conf = None
        self.res_dir = None
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
            logging.info('%14s - Loading custom (inline) target configuration',
                    'Target')
            self.conf = target_conf
        elif isinstance(target_conf, str):
            logging.info('%14s - Loading custom (file) target configuration',
                    'Target')
            self.conf = TestEnv.loadTargetConfig(target_conf)
        elif target_conf is None:
            logging.info('%14s - Loading default (file) target configuration',
                    'Target')
            self.conf = TestEnv.loadTargetConfig()
        logging.debug('%14s - Target configuration %s', 'Target', self.conf)

        # Setup test configuration
        if test_conf:
            if isinstance(test_conf, dict):
                logging.info('%14s - Loading custom (inline) test configuration',
                        'Target')
                self.test_conf = test_conf
            elif isinstance(test_conf, str):
                logging.info('%14s - Loading custom (file) test configuration',
                        'Target')
                self.test_conf = TestEnv.loadTargetConfig(test_conf)
            else:
                raise ValueError('test_conf must be either a dictionary or a filepath')
            logging.debug('%14s - Test configuration %s', 'Target', self.conf)

        # Setup target working directory
        if 'workdir' in self.conf:
            self.workdir = self.conf['workdir']

        # Initialize binary tools to deploy
        if 'tools' in self.conf:
            self.__tools = self.conf['tools']
        # Merge tests specific tools
        if self.test_conf and 'tools' in self.test_conf and \
           self.test_conf['tools']:
            if 'tools' not in self.conf:
                self.conf['tools'] = []
            self.__tools = list(set(
                self.conf['tools'] + self.test_conf['tools']
            ))

        # Initialize ftrace events
        # test configuration override target one
        if self.test_conf and 'ftrace' in self.test_conf:
            self.conf['ftrace'] = self.test_conf['ftrace']
        if 'ftrace' in self.conf and self.conf['ftrace']:
            self.__tools.append('trace-cmd')

        # Add tools dependencies
        if 'rt-app' in self.__tools:
            self.__tools.append('taskset')
            self.__tools.append('trace-cmd')
            self.__tools.append('perf')
            self.__tools.append('cgroup_run_into.sh')
        # Sanitize list of dependencies to remove duplicates
        self.__tools = list(set(self.__tools))

        # Initialize features
        if '__features__' not in self.conf:
            self.conf['__features__'] = []

        self._init()

        # Initialize FTrace events collection
        self._init_ftrace(True)

        # Initialize energy probe instrument
        self.emeter = EnergyMeter.getInstance(
                self.target, self.conf, force=True)

        # Initialize RT-App calibration values
        self.calibration()

        # Initialize local results folder
        # test configuration override target one
        if self.test_conf and 'results_dir' in self.test_conf:
            self.res_dir = self.test_conf['results_dir']
        if not self.res_dir and 'results_dir' in self.conf:
            self.res_dir = self.conf['results_dir']
        if self.res_dir and not os.path.isabs(self.res_dir):
                self.res_dir = os.path.join(basepath, 'results', self.res_dir)
        else:
            self.res_dir = os.path.join(basepath, OUT_PREFIX)
            self.res_dir = datetime.datetime.now()\
                            .strftime(self.res_dir + '/%Y%m%d_%H%M%S')
        if wipe and os.path.exists(self.res_dir):
            logging.warning('%14s - Wipe previous contents of the results folder:', 'TestEnv')
            logging.warning('%14s -    %s', 'TestEnv', self.res_dir)
            shutil.rmtree(self.res_dir, ignore_errors=True)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        res_lnk = os.path.join(basepath, LATEST_LINK)
        if os.path.islink(res_lnk):
            os.remove(res_lnk)
        os.symlink(self.res_dir, res_lnk)

        logging.info('%14s - Set results folder to:', 'TestEnv')
        logging.info('%14s -    %s', 'TestEnv', self.res_dir)
        logging.info('%14s - Experiment results available also in:', 'TestEnv')
        logging.info('%14s -    %s', 'TestEnv', res_lnk)

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
        conf.load()
        return conf.json

    def _init(self, force = False):

        if self._feature('debug'):
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize target
        self._init_target(force)

        # Initialize target Topology for behavior analysis
        CLUSTERS = []

        # Build topology for a big.LITTLE systems
        if self.target.big_core and \
           (self.target.abi == 'arm64' or self.target.abi == 'armeabi'):
            # Populate cluster for a big.LITTLE platform
            if self.target.big_core:
                # Load cluster of LITTLE cores
                CLUSTERS.append(
                    [i for i,t in enumerate(self.target.core_names)
                                if t == self.target.little_core])
                # Load cluster of big cores
                CLUSTERS.append(
                    [i for i,t in enumerate(self.target.core_names)
                                if t == self.target.big_core])
        # Build topology for an SMP systems
        elif not self.target.big_core or \
             self.target.abi == 'x86_64':
            for c in set(self.target.core_clusters):
                CLUSTERS.append(
                    [i for i,v in enumerate(self.target.core_clusters)
                                if v == c])
        self.topology = Topology(clusters=CLUSTERS)
        logging.info(r'%14s - Topology:', 'Target')
        logging.info(r'%14s -    %s', 'Target', CLUSTERS)

        # Initialize the platform descriptor
        self._init_platform()


    def _init_target(self, force = False):

        if not force and self.target is not None:
            return self.target

        self.__connection_settings = {}

        # Configure username
        if 'username' in self.conf:
            self.__connection_settings['username'] = self.conf['username']
        else:
            self.__connection_settings['username'] = USERNAME_DEFAULT

        # Configure password or SSH keyfile
        if 'keyfile' in self.conf:
            self.__connection_settings['keyfile'] = self.conf['keyfile']
        elif 'password' in self.conf:
            self.__connection_settings['password'] = self.conf['password']
        else:
            self.__connection_settings['password'] = PASSWORD_DEFAULT

        # Configure port
        if 'port' in self.conf:
            self.__connection_settings['port'] = self.conf['port']

        # Configure the host IP/MAC address
        if 'host' in self.conf:
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


        ########################################################################
        # Board configuration
        ########################################################################

        # Setup board default if not specified by configuration
        if 'board' not in self.conf:
            self.conf['board'] = 'UNKNOWN'

        # Initialize TC2 board
        if self.conf['board'].upper() == 'TC2':
            platform = devlib.platform.arm.TC2()
            self.__modules = ['bl', 'hwmon', 'cpufreq']

        # Initialize JUNO board
        elif self.conf['board'].upper() in ('JUNO', 'JUNO2'):
            platform = devlib.platform.arm.Juno()
            self.__modules = ['bl', 'hwmon', 'cpufreq']

        # Initialize OAK board
        elif self.conf['board'].upper() == 'OAK':
            platform = Platform(model='MT8173')
            self.__modules = ['bl', 'cpufreq']

        # Initialize N5X device
        elif self.conf['board'].upper() == 'N5X':
            platform = Platform(model='bullhead',
                                core_names = ['A53', 'A53', 'A53', 'A53',
                                              'A57', 'A57'],
                                core_clusters = [ 0, 0, 0, 0, 1, 1],
                                big_core = 'A57',
                               )
            self.__modules = ['bl', 'cpufreq']

        # Initialize default UNKNOWN board
        else:
            platform = None
            self.__modules = []

        ########################################################################
        # Modules configuration
        ########################################################################

        # Rinfine modules list based on target.conf options
        if 'modules' in self.conf:
            self.__modules = list(set(
                self.__modules + self.conf['modules']
            ))
        # Merge tests specific modules
        if self.test_conf and 'modules' in self.test_conf and \
           self.test_conf['modules']:
            self.__modules = list(set(
                self.__modules + self.test_conf['modules']
            ))

        # Initialize modules to exclude on the target
        if 'exclude_modules' in self.conf:
            for module in self.conf['exclude_modules']:
                if module in self.__modules:
                    self.__modules.remove(module)
        # Remove tests specific modules
        if self.test_conf and 'exclude_modules' in self.test_conf:
            for module in self.test_conf['exclude_modules']:
                if module in self.__modules:
                    self.__modules.remove(module)

        logging.info(r'%14s - Devlib modules to load: %s',
                'Target', self.__modules)

        ########################################################################
        # Devlib target setup (based on target.config::platform)
        ########################################################################

        # If the target is Android, we need just (eventually) the device
        if platform_type.lower() == 'android':
            self.__connection_settings = None
            device = 'DEFAULT'
            if 'device' in self.conf:
                device = self.conf['device']
                self.__connection_settings = {'device' : device}
            elif 'host' in self.conf:
                host = self.conf['host']
                port = '5555'
                if 'port' in self.conf:
                    port = str(self.conf['port'])
                device = '{}:{}'.format(host, port)
                self.__connection_settings = {'device' : device}
            logging.info(r'%14s - Connecting Android target [%s]',
                         'Target', device)
        else:
            logging.info(r'%14s - Connecting %s target:', 'Target',
                         platform_type)
            for key in self.__connection_settings:
                logging.info(r'%14s - %10s : %s', 'Target',
                             key, self.__connection_settings[key])

        if platform_type.lower() == 'linux':
            logging.debug('%14s - Setup LINUX target...', 'Target')
            self.target = devlib.LinuxTarget(
                    platform = platform,
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'android':
            logging.debug('%14s - Setup ANDROID target...', 'Target')
            self.target = devlib.AndroidTarget(
                    platform = platform,
                    connection_settings = self.__connection_settings,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'host':
            logging.debug('%14s - Setup HOST target...', 'Target')
            self.target = devlib.LocalLinuxTarget(
                    platform = platform,
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

        logging.info('%14s - Initializing target workdir:', 'Target')
        logging.info('%14s -    %s', 'Target', self.target.working_directory)
        tools_to_install = []
        if self.__tools:
            for tool in self.__tools:
                binary = '{}/tools/scripts/{}'.format(basepath, tool)
                if not os.path.isfile(binary):
                    binary = '{}/tools/{}/{}'\
                            .format(basepath, self.target.abi, tool)
                tools_to_install.append(binary)
        self.target.setup(tools_to_install)

        # Verify that all the required modules have been initialized
        for module in self.__modules:
            logging.debug('%14s - Check for module [%s]...', 'Target', module)
            if not hasattr(self.target, module):
                logging.warning('%14s - Unable to initialize [%s] module',
                        'Target', module)
                logging.error('%14s - Fix your target kernel configuration or '
                        'disable module from configuration', 'Target')
                raise RuntimeError('Failed to initialized [{}] module, '
                        'update your kernel or test configurations'.format(module))

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

        functions = None
        if 'functions' in ftrace:
            functions = ftrace['functions']

        buffsize = FTRACE_BUFSIZE_DEFAULT
        if 'buffsize' in ftrace:
            buffsize = ftrace['buffsize']

        self.ftrace = devlib.FtraceCollector(
            self.target,
            events      = events,
            functions   = functions,
            buffer_size = buffsize,
            autoreport  = False,
            autoview    = False
        )

        if events:
            logging.info('%14s - Enabled tracepoints:', 'FTrace')
            for event in events:
                logging.info('%14s -   %s', 'FTrace', event)
        if functions:
            logging.info('%14s - Kernel functions profiled:', 'FTrace')
            for function in functions:
                logging.info('%14s -   %s', 'FTrace', function)

        return self.ftrace

    def _init_energy(self, force):

        # Initialize energy probe to board default
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force)

    def _init_platform_bl(self):
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

    def _init_platform_smp(self):
        self.platform = {
            'clusters' : {},
            'freqs' : {}
        }
        for cpu_id,node_id in enumerate(self.target.core_clusters):
            if node_id not in self.platform['clusters']:
                self.platform['clusters'][node_id] = []
            self.platform['clusters'][node_id].append(cpu_id)

        if 'cpufreq' in self.target.modules:
            # Try loading frequencies using the cpufreq module
            for cluster_id in self.platform['clusters']:
                core_id = self.platform['clusters'][cluster_id][0]
                self.platform['freqs'][cluster_id] = \
                    self.target.cpufreq.list_frequencies(core_id)
        else:
            logging.warn(
                    '%14s - Unable to identify cluster frequencies',
                    'Target')

        # TODO: get the performance boundaries in case of intel_pstate driver

        self.platform['cpus_count'] = len(self.target.core_clusters)

    def _load_em(self, board):
        em_path = os.path.join(basepath,
                'libs/utils/platforms', board.lower() + '.json')
        logging.debug('%14s - Trying to load default EM from %s',
                'Platform', em_path)
        if not os.path.exists(em_path):
            return None
        logging.info('%14s - Loading default EM:', 'Platform')
        logging.info('%14s -    %s', 'Platform', em_path)
        board = JsonConf(em_path)
        board.load()
        if 'nrg_model' not in board.json:
            return None
        return board.json['nrg_model']

    def _init_platform(self):
        if 'bl' in self.target.modules:
            self._init_platform_bl()
        else:
            self._init_platform_smp()

        # Adding energy model information
        if 'nrg_model' in self.conf:
            self.platform['nrg_model'] = self.conf['nrg_model']
        # Try to load the default energy model (if available)
        else:
            self.platform['nrg_model'] = self._load_em(self.conf['board'])

        # Adding topology information
        self.platform['topology'] = self.topology.get_level("cluster")

        logging.debug('%14s - Platform descriptor initialized\n%s',
            'Platform', self.platform)
        # self.platform_dump('./')

    def platform_dump(self, dest_dir, dest_file='platform.json'):
        plt_file = os.path.join(dest_dir, dest_file)
        logging.debug('%14s - Dump platform descriptor in [%s]',
            'Platform', plt_file)
        with open(plt_file, 'w') as ofile:
            json.dump(self.platform, ofile, sort_keys=True, indent=4)
        return (self.platform, plt_file)

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
            logging.warning(
                r'%14s - Using configuration provided RTApp calibration',
                'Target')
            self._calib = {
                    int(key): int(value)
                    for key, value in self.conf['rtapp-calib'].items()
                }
        else:
            logging.info(r'%14s - Calibrating RTApp...', 'Target')
            self._calib = RTA.calibrate(self.target)

        logging.info(r'%14s - Using RT-App calibration values:', 'Target')
        logging.info(r'%14s -    %s', 'Target',
                "{" + ", ".join('"%r": %r' % (key, self._calib[key])
                                for key in sorted(self._calib)) + "}")
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

        return self.parse_arp_cache(host)

    def parse_arp_cache(self, host):
        output = os.popen(r'arp -n')
        if ':' in host:
            # Assuming this is a MAC address
            # TODO add a suitable check on MAC address format
            # Query ARP for the specified HW address
            ARP_RE = re.compile(
                r'([^ ]*).*({}|{})'.format(host.lower(), host.upper())
            )
            macaddr = host
            ipaddr = None
            for line in output:
                match = ARP_RE.search(line)
                if not match:
                    continue
                ipaddr = match.group(1)
                break
        else:
            # Assuming this is an IP address
            # TODO add a suitable check on IP address format
            # Query ARP for the specified IP address
            ARP_RE = re.compile(
                r'{}.*ether *([0-9a-fA-F:]*)'.format(host)
            )
            macaddr = None
            ipaddr = host
            for line in output:
                match = ARP_RE.search(line)
                if not match:
                    continue
                macaddr = match.group(1)
                break
            else:
                # When target is accessed via WiFi, there is not MAC address
                # reported by arp. In these cases we can know only the IP
                # of the remote target.
                macaddr = 'UNKNOWN'

        if not ipaddr or not macaddr:
            raise ValueError('Unable to lookup for target IP/MAC address')
        logging.info('%14s - Target (%s) at IP address: %s',
                'HostResolver', macaddr, ipaddr)
        return (macaddr, ipaddr)

    def reboot(self, reboot_time=120, ping_time=15):
        # Send remote target a reboot command
        if self._feature('no-reboot'):
            logging.warning('%14s - Reboot disabled by conf features', 'Reboot')
        else:
            if 'reboot_time' in self.conf:
                reboot_time = int(self.conf['reboot_time'])

            if 'ping_time' in self.conf:
                ping_time = int(self.conf['ping_time'])

            # Before rebooting make sure to have IP and MAC addresses
            # of the target
            (self.mac, self.ip) = self.parse_arp_cache(self.ip)

            self.target.execute('sleep 2 && reboot -f &', as_root=True)

            # Wait for the target to complete the reboot
            logging.info('%14s - Waiting up to %s[s] for target [%s] to reboot...',
                    'Reboot', reboot_time, self.ip)

            ping_cmd = "ping -c 1 {} >/dev/null".format(self.ip)
            elapsed = 0
            start = time.time()
            while elapsed <= reboot_time:
                time.sleep(ping_time)
                logging.debug('%14s - Trying to connect to [%s] target...',
                        'Reboot', self.ip)
                if os.system(ping_cmd) == 0:
                    break
                elapsed = time.time() - start
            if elapsed > reboot_time:
                if self.mac:
                    logging.warning('%14s - target [%s] not responding to \
                            PINGs, trying to resolve MAC address...', 'Reboot', self.ip)
                    (self.mac, self.ip) = self.resolv_host(self.mac)
                else:
                    logging.warning('%14s - target [%s] not responding to PINGs, trying to continue...',
                        'Reboot', self.ip)

        # Force re-initialization of all the devlib modules
        force = True

        # Reset the connection to the target
        self._init(force)

        # Initialize FTrace events collection
        self._init_ftrace(force)

        # Initialize energy probe instrument
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force)

    def install_kernel(self, tc, reboot=False):

        # Default initialize the kernel/dtb settings
        tc.setdefault('kernel', None)
        tc.setdefault('dtb', None)

        if self.kernel == tc['kernel'] and self.dtb == tc['dtb']:
            return

        logging.info('%14s - Install kernel [%s] on target...',
                'KernelSetup', tc['kernel'])

        # Install kernel/dtb via FTFP
        if self._feature('no-kernel'):
            logging.warning('%14s - Kernel deploy disabled by conf features',
                    'KernelSetup')

        elif 'tftp' in self.conf:
            logging.info('%14s - Deploy kernel via TFTP...', 'KernelSetup')

            # Deploy kernel in TFTP folder (mandatory)
            if 'kernel' not in tc or not tc['kernel']:
                raise ValueError('Missing "kernel" parameter in conf: %s',
                        'KernelSetup', tc)
            self.tftp_deploy(tc['kernel'])

            # Deploy DTB in TFTP folder (if provided)
            if 'dtb' not in tc or not tc['dtb']:
                logging.debug('%14s - DTB not provided, using existing one',
                        'KernelSetup')
                logging.debug('%14s - Current conf:\n%s', 'KernelSetup', tc)
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

        cmd = 'cp {} {} && sync'.format(src, dst)
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
