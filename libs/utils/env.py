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

from datetime import datetime
import json
import logging
import os
import re
import shutil
import sys
import time
import unittest
import contextlib

import devlib
from devlib.utils.misc import memoized, which
from devlib import Platform, TargetError
from trappy.stats.Topology import Topology

from wlgen import RTA
from energy import EnergyMeter
from energy_model import EnergyModel
from conf import JsonConf
from platforms.juno_r0_energy import juno_r0_energy
from platforms.hikey_energy import hikey_energy
from platforms.pixel_energy import pixel_energy

USERNAME_DEFAULT = 'root'
PASSWORD_DEFAULT = ''
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
IFCFG_BCAST_RE = re.compile(
    r'Bcast:(.*) '
)

class TestEnv(ShareState):
    """
    Represents the environment configuring LISA, the target, and the test setup

    The test environment is defined by:

    a target configuration (target_conf) defining which HW platform we
      want to use to run the experiments

    :param target_conf:
        Configuration defining the target to run experiments on. May be

            - A dict defining the values directly
            - A path to a JSON file containing the configuration
            - ``None``, in which case:
                - LISA_TARGET_CONF environment variable is read to locate a
                  config file.
                - If the variable is not set, $LISA_HOME/target.config is used.

        You need to provide the information needed to connect to the
        target. For SSH targets that means "host", "username" and
        either "password" or "keyfile". All other fields are optional if
        the relevant features aren't needed. Has the following keys:

        **host**
            Target IP or MAC address for SSH access
        **username**
            For SSH access
        **keyfile**
            Path to SSH key (alternative to password)
        **password**
            SSH password (alternative to keyfile)
        **device**
            Target Android device ID if using ADB
        **port**
            Port for Android connection default port is 5555
        **ANDROID_HOME**
            Path to Android SDK. Defaults to ``$ANDROID_HOME`` from the
            environment.
        **rtapp-calib**
            Calibration values for RT-App. If unspecified, LISA will
            calibrate RT-App on the target. A message will be logged with
            a value that can be copied here to avoid having to re-run
            calibration on subsequent tests.
        **ftrace**
            Ftrace configuration.
            Currently, only additional events through "events" key is supported.

    :param force_new: Create a new TestEnv object even if there is one available
                      for this session.  By default, TestEnv only creates one
                      object per session, use this to override this behaviour.
    :type force_new: bool
    """

    critical_tasks = {
        'linux': [
            'init',
            'systemd',
            'dbus',
            'sh',
            'ssh',
            'rsyslogd',
            'jbd2'
        ],
        'android': [
            'sh', 'adbd',
            'usb', 'transport',
            # We don't actually need this task but on Google Pixel it apparently
            # cannot be frozen, so the cgroup state gets stuck in FREEZING if we
            # try to freeze it.
            'thermal-engine',
            # Similar issue with HiKey960, the board will crash if this is frozen
            # for too long.
            'watchdogd',
        ]
    }
    """
    Dictionary mapping OS name to list of task names that we can't afford to
    freeze when using freeeze_userspace.
    """

    _initialized = False

    def __init__(self, target_conf=None, force_new=False):
        super(TestEnv, self).__init__()

        if self._initialized and not force_new:
            return

        # Setup logging
        self._log = logging.getLogger('TestEnv')

        self._pre_target_init(target_conf)
        self._init_target()
        self._post_target_init()

        self._initialized = True

    def _load_em(self, board):
        em_path = os.path.join(basepath,
                'libs/utils/platforms', board.lower() + '.json')
        self._log.debug('Trying to load default EM from %s', em_path)
        if not os.path.exists(em_path):
            return None
        self._log.info('Loading default EM:')
        self._log.info('   %s', em_path)
        board = JsonConf(em_path)
        board.load()
        if 'nrg_model' not in board.json:
            return None
        return board.json['nrg_model']

    def _load_board(self, board):
        board_path = os.path.join(basepath,
                'libs/utils/platforms', board.lower() + '.json')
        self._log.debug('Trying to load board descriptor from %s', board_path)
        if not os.path.exists(board_path):
            return None
        self._log.info('Loading board:')
        self._log.info('   %s', board_path)
        board = JsonConf(board_path)
        board.load()
        if 'board' not in board.json:
            return None
        return board.json['board']

    def _build_topology(self):
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
        self._log.info('Topology:')
        self._log.info('   %s', CLUSTERS)

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
            self._log.warning('Unable to identify cluster frequencies')

        # TODO: get the performance boundaries in case of intel_pstate driver

        self.platform['cpus_count'] = len(self.target.core_clusters)

    def _get_clusters(self, core_names):
        idx = 0
        clusters = []
        ids_map = { core_names[0] : 0 }
        for name in core_names:
            idx = ids_map.get(name, idx+1)
            ids_map[name] = idx
            clusters.append(idx)
        return clusters

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
            nrg_model = self._load_em(self.conf['board'])
            # We shouldn't have an 'nrg_model' key if there is no energy model data
            if nrg_model:
                self.platform['nrg_model'] = nrg_model

        # Adding topology information
        self.platform['topology'] = self.topology.get_level("cluster")

        # Adding kernel build information
        kver = self.target.kernel_version
        self.platform['kernel'] = {t: getattr(kver, t, None)
            for t in [
                'release', 'version',
                'version_number', 'major', 'minor',
                'rc', 'sha1', 'parts'
            ]
        }
        self.platform['abi'] = self.target.abi
        self.platform['os'] = self.target.os

        self._log.debug('Platform descriptor initialized\n%s', self.platform)
        # self.platform_dump('./')

    def _init_energy(self, force):
        # Initialize energy probe to board default
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force)

    def _pre_target_init(self, target_conf):
        """
        Initialization code that doesn't need a :class:`devlib.Target` instance
        """

        self.conf = {}
        self.target = None
        self.ftrace = None
        self.workdir = None
        self.__installed_tools = set()
        self.__modules = []
        self.__connection_settings = None
        self._calib = None

        # Keep track of target IP and MAC address
        self.ip = None
        self.mac = None

        # Energy meter configuration
        self.emeter = None

        # The platform descriptor to be saved into the results folder
        self.platform = {}

        # Keep track of android support
        self.LISA_HOME = os.environ.get('LISA_HOME', '/vagrant')
        self.ANDROID_HOME = os.environ.get('ANDROID_HOME', None)
        self.CATAPULT_HOME = os.environ.get('CATAPULT_HOME',
                os.path.join(self.LISA_HOME, 'tools', 'catapult'))

        # Setup target configuration
        if isinstance(target_conf, dict):
            self._log.info('Loading custom (inline) target configuration')
            self.conf = target_conf
        elif isinstance(target_conf, str):
            self._log.info('Loading %s target configuration', target_conf)
            self.conf = self.loadTargetConfig(target_conf)
        else:
            target_conf = os.environ.get('LISA_TARGET_CONF', '')
            self._log.info('Loading [%s] target configuration',
                    target_conf or 'default')
            self.conf = self.loadTargetConfig(target_conf)

        self._log.debug('Target configuration %s', self.conf)

        # Setup target working directory
        if 'workdir' in self.conf:
            self.workdir = self.conf['workdir']

        # Initialize binary tools to deploy
        self.__tools = list(set(self.conf.get('tools', [])))

        # Initialize ftrace events
        ftrace_conf = self.conf.get('ftrace', {})
        ftrace_conf['events'] = sorted(set(ftrace_conf.get('events', [])))
        self.conf['ftrace'] = ftrace_conf

    def _init_target(self):
        """
        Create a :class:`devlib.Target` object
        """
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

        if platform_type.lower() == 'android':
            self.ANDROID_HOME = self.conf.get('ANDROID_HOME',
                                              self.ANDROID_HOME)
            if self.ANDROID_HOME:
                self._adb = os.path.join(self.ANDROID_HOME,
                                         'platform-tools', 'adb')
                self._fastboot = os.path.join(self.ANDROID_HOME,
                                              'platform-tools', 'fastboot')
                os.environ['ANDROID_HOME'] = self.ANDROID_HOME
                os.environ['CATAPULT_HOME'] = self.CATAPULT_HOME
            else:
                raise RuntimeError('Android SDK not found, ANDROID_HOME must be defined!')

            self._log.info('External tools using:')
            self._log.info('   ANDROID_HOME: %s', self.ANDROID_HOME)
            self._log.info('   CATAPULT_HOME: %s', self.CATAPULT_HOME)

            if not os.path.exists(self._adb):
                raise RuntimeError('\nADB binary not found\n\t{}\ndoes not exists!\n\n'
                                   'Please configure ANDROID_HOME to point to '
                                   'a valid Android SDK installation folder.'\
                                   .format(self._adb))

        ########################################################################
        # Board configuration
        ########################################################################

        # Setup board default if not specified by configuration
        self.nrg_model = None
        platform = None

        default_modules = ['sched']
        self.__modules = ['cpufreq', 'cpuidle']

        if 'board' not in self.conf:
            self.conf['board'] = 'UNKNOWN'

        board_name = self.conf['board'].upper()

        # Initialize TC2 board
        if board_name == 'TC2':
            platform = devlib.platform.arm.TC2()
            self.__modules = ['bl', 'hwmon', 'cpufreq']

        # Initialize JUNO board
        elif board_name in ('JUNO', 'JUNO2'):
            platform = devlib.platform.arm.Juno()
            self.__modules = ['bl', 'hwmon', 'cpufreq']

            if board_name == 'JUNO':
                self.nrg_model = juno_r0_energy

        # Initialize OAK board
        elif board_name == 'OAK':
            platform = Platform(model='MT8173')
            self.__modules = ['bl', 'cpufreq']

        # Initialized HiKey board
        elif board_name == 'HIKEY':
            self.nrg_model = hikey_energy
            self.__modules = [ "cpufreq", "cpuidle" ]
            platform = Platform(model='hikey')

        # Initialize HiKey960 board
        elif board_name == 'HIKEY960':
            self.__modules = ['bl', 'cpufreq', 'cpuidle']
            platform = Platform(model='hikey960')

        # Initialize Pixel phone
        elif board_name == 'PIXEL':
            self.nrg_model = pixel_energy
            self.__modules = ['bl', 'cpufreq']
            platform = Platform(model='pixel')

        # Initialize gem5 platform
        elif board_name == 'GEM5':
            self.__modules=['cpufreq']
            platform = self._init_target_gem5()

        elif board_name != 'UNKNOWN':
            # Initilize from platform descriptor (if available)
            board = self._load_board(self.conf['board'])
            if board:
                core_names = board['cores']
                platform = Platform(
                    model=self.conf['board'],
                    core_names=core_names,
                    core_clusters = self._get_clusters(core_names),
                    big_core=board.get('big_core', None)
                )
                if 'modules' in board:
                    self.__modules = board['modules']

        ########################################################################
        # Modules configuration
        ########################################################################

        modules = set(self.__modules + default_modules)

        # Refine modules list based on target.conf
        modules.update(self.conf.get('modules', []))

        remove_modules = set(self.conf.get('exclude_modules', []))
        modules.difference_update(remove_modules)

        self.__modules = list(modules)
        self._log.info('Devlib modules to load: %s', self.__modules)

        ########################################################################
        # Devlib target setup (based on target.config::platform)
        ########################################################################

        # If the target is Android, we need just (eventually) the device
        if platform_type.lower() == 'android':
            self.__connection_settings = None
            device = 'DEFAULT'

            # Workaround for ARM-software/devlib#225
            if not self.workdir:
                self.workdir = '/data/local/tmp/devlib-target'

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
            self._log.info('Connecting Android target [%s]', device)
        else:
            self._log.info('Connecting %s target:', platform_type)
            for key in self.__connection_settings:
                self._log.info('%10s : %s', key,
                               self.__connection_settings[key])

        self._log.info('Connection settings:')
        self._log.info('   %s', self.__connection_settings)

        if platform_type.lower() == 'linux':
            self._log.debug('Setup LINUX target...')
            if "host" not in self.__connection_settings:
                raise ValueError('Missing "host" param in Linux target conf')

            self.target = devlib.LinuxTarget(
                    platform = platform,
                    connection_settings = self.__connection_settings,
                    working_directory = self.workdir,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'android':
            self._log.debug('Setup ANDROID target...')
            self.target = devlib.AndroidTarget(
                    platform = platform,
                    connection_settings = self.__connection_settings,
                    working_directory = self.workdir,
                    load_default_modules = False,
                    modules = self.__modules)
        elif platform_type.lower() == 'host':
            self._log.debug('Setup HOST target...')
            self.target = devlib.LocalLinuxTarget(
                    platform = platform,
                    working_directory = '/tmp/devlib-target',
                    executables_directory = '/tmp/devlib-target/bin',
                    load_default_modules = False,
                    modules = self.__modules,
                    connection_settings = {'unrooted': True})
        else:
            raise ValueError('Config error: not supported [platform] type {}'\
                    .format(platform_type))

        self._log.debug('Checking target connection...')
        self._log.debug('Target info:')
        self._log.debug('      ABI: %s', self.target.abi)
        self._log.debug('     CPUs: %s', self.target.cpuinfo)
        self._log.debug(' Clusters: %s', self.target.core_clusters)

        self._log.info('Initializing target workdir:')
        self._log.info('   %s', self.target.working_directory)

        self.target.setup()
        self.install_tools(self.__tools)

        # Verify that all the required modules have been initialized
        for module in self.__modules:
            self._log.debug('Check for module [%s]...', module)
            if not hasattr(self.target, module):
                self._log.warning('Unable to initialize [%s] module', module)
                self._log.error('Fix your target kernel configuration or '
                                'disable module from configuration')
                raise RuntimeError('Failed to initialized [{}] module, '
                        'update your kernel or test configurations'.format(module))

        if not self.nrg_model:
            try:
                self._log.info('Attempting to read energy model from target')
                self.nrg_model = EnergyModel.from_target(self.target)
            except (TargetError, RuntimeError, ValueError) as e:
                self._log.error("Couldn't read target energy model: %s", e)

    def _init_target_gem5(self):
        system = self.conf['gem5']['system']
        simulator = self.conf['gem5']['simulator']

        # Get gem5 binary arguments
        args = simulator.get('args', [])
        args.append('--listener-mode=on')

        # Get platform description
        args.append(system['platform']['description'])

        # Get platform arguments
        args += system['platform'].get('args', [])
        args += ['--kernel {}'.format(system['kernel']),
                 '--dtb {}'.format(system['dtb']),
                 '--disk-image {}'.format(system['disk'])]

        # Gather all arguments
        args = ' '.join(args)

        diod_path = which('diod')
        if diod_path is None:
            raise RuntimeError('Failed to find "diod" on your host machine, '
                               'check your installation or your PATH variable')

        # Setup virtio
        # Brackets are there to let the output dir be created automatically
        virtio_args = '--which-diod={} --workload-automation-vio={{}}'.format(diod_path)

        # Change conf['board'] to include platform information
        suffix = os.path.splitext(os.path.basename(
            system['platform']['description']))[0]
        self.conf['board'] = self.conf['board'].lower() + suffix

        board = self._load_board(self.conf['board'])

        # Merge all arguments
        platform = devlib.platform.gem5.Gem5SimulationPlatform(
            name = 'gem5',
            gem5_bin = simulator['bin'],
            gem5_args = args,
            gem5_virtio = virtio_args,
            host_output_dir = self.get_res_dir('gem5'),
            core_names = board['cores'] if board else None,
            core_clusters = self._get_clusters(board['cores']) if board else None,
            big_core = board.get('big_core', None) if board else None,
        )

        return platform

    def _post_target_init(self):
        """
        Initialization code that needs a :class:`devlib.Target` instance
        """
        self._build_topology()

        # Initialize the platform descriptor
        self._init_platform()

        # Initialize energy probe instrument
        self._init_energy(True)

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
        self._log.info('Target (%s) at IP address: %s', macaddr, ipaddr)
        return (macaddr, ipaddr)

    def resolv_host(self, host=None):
        """
        Resolve a host name or IP address to a MAC address

        .. TODO Is my networking terminology correct here?

        :param host: IP address or host name to resolve. If None, use 'host'
                    value from target_config.
        :type host: str
        """
        if host is None:
            host = self.conf['host']

        # Refresh ARP for local network IPs
        self._log.debug('Collecting all Bcast address')
        output = os.popen(r'ifconfig').read().split('\n')
        for line in output:
            match = IFCFG_BCAST_RE.search(line)
            if not match:
                continue
            baddr = match.group(1)
            try:
                cmd = r'nmap -T4 -sP {}/24 &>/dev/null'.format(baddr.strip())
                self._log.debug(cmd)
                os.popen(cmd)
            except RuntimeError:
                self._log.warning('Nmap not available, try IP lookup using broadcast ping')
                cmd = r'ping -b -c1 {} &>/dev/null'.format(baddr)
                self._log.debug(cmd)
                os.popen(cmd)

        return self.parse_arp_cache(host)

    def loadTargetConfig(self, filepath=None):
        """
        Load the target configuration from the specified file.

        :param filepath: Path of the target configuration file. Relative to the
                         root folder of the test suite.
        :type filepath: str

        """

        # "" and None are replaced by the default 'target.config' value
        filepath = filepath or 'target.config'

        # Loading default target configuration
        conf_file = os.path.join(basepath, filepath)

        self._log.info('Loading target configuration [%s]...', conf_file)
        conf = JsonConf(conf_file)
        conf.load()
        return conf.json

    def get_res_dir(self, name=None):
        """
        Returns a directory managed by LISA to store results
        """
        # Initialize local results folder.
        # The test configuration overrides the target's one and the environment
        # variable overrides everything else.
        res_dir = (
            os.getenv('LISA_RESULTS_DIR') or
            self.conf.get('results_dir')
        )

        # Default result dir based on the current time
        if not res_dir:
            if not name:
                name = datetime.now().strftime('%Y%m%d_%H%M%S')

            res_dir = os.path.join(basepath, OUT_PREFIX, name)

        # Relative paths are interpreted as relative to a fixed root.
        if not os.path.isabs(res_dir):
            res_dir = os.path.join(basepath, OUT_PREFIX, res_dir)

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        res_lnk = os.path.join(basepath, LATEST_LINK)
        if os.path.islink(res_lnk):
            os.remove(res_lnk)
        os.symlink(res_dir, res_lnk)

        return res_dir

    def install_tools(self, tools):
        """
        Install tools additional to those specified in the test config 'tools'
        field

        :param tools: The list of names of tools to install
        :type tools: list(str)
        """
        tools = set(tools)

        # Remove duplicates and already-installed tools
        tools.difference_update(self.__installed_tools)

        tools_to_install = []
        for tool in tools:
            binary = '{}/tools/scripts/{}'.format(basepath, tool)
            if not os.path.isfile(binary):
                binary = '{}/tools/{}/{}'\
                         .format(basepath, self.target.abi, tool)
            tools_to_install.append(binary)

        for tool_to_install in tools_to_install:
            self.target.install(tool_to_install)

        self.__installed_tools.update(tools)

    def configure_ftrace(self, events=None, functions=None,
                         buffsize=FTRACE_BUFSIZE_DEFAULT):
        """
        Setup the environment's :class:`devlib.trace.FtraceCollector`

        :param events: The events to trace
        :type events: list(str)

        :param functions: the kernel functions to trace
        :type functions: list(str)

        :param buffsize: The size of the Ftrace buffer
        :type buffsize: int

        :raises RuntimeError: If no event nor function is to be traced
        """

        # Merge with setup from target config
        target_conf = self.conf.get('ftrace', {})

        if events is None:
            events = []
        if functions is None:
            functions = []

        def merge_conf(value, index, default):
            return sorted(set(value), target_conf.get(index, default))

        events = merge_conf(events, 'events', [])
        functions = merge_conf(functions, 'functions', [])
        buffsize = max(buffsize, target_conf.get('buffsize', 0))

        # If no events or functions have been specified:
        # do not create the FtraceCollector
        if not (events or functions):
            raise RuntimeError(
                "Tried to configure Ftrace, but no events nor functions were"
                "provided from neither method parameters nor target_config"
            )

        # Ensure we have trace-cmd on the target
        if 'trace-cmd' not in self.__installed_tools:
            self.install_tools(['trace-cmd'])

        self.ftrace = devlib.FtraceCollector(
            self.target,
            events      = events,
            functions   = functions,
            buffer_size = buffsize,
            autoreport  = False,
            autoview    = False
        )

        if events:
            self._log.info('Enabled tracepoints:')
            for event in events:
                self._log.info('   %s', event)
        if functions:
            self._log.info('Kernel functions profiled:')
            for function in functions:
                self._log.info('   %s', function)

    def platform_dump(self, dest_dir, dest_file='platform.json'):
        plt_file = os.path.join(dest_dir, dest_file)
        self._log.debug('Dump platform descriptor in [%s]', plt_file)
        with open(plt_file, 'w') as ofile:
            json.dump(self.platform, ofile, sort_keys=True, indent=4)
        return (self.platform, plt_file)

    def calibration(self, force=False):
        """
        Get rt-app calibration. Run calibration on target if necessary.

        :param force: Always run calibration on target, even if we have not
                      installed rt-app or have already run calibration.
        :returns: A dict with calibration results, which can be passed as the
                  ``calibration`` parameter to :class:`RTA`, or ``None`` if
                  force=False and we have not installed rt-app.
        """

        if not force and self._calib:
            return self._calib

        required_tools = ['rt-app', 'taskset', 'trace-cmd', 'perf', 'cgroup_run_into.sh']
        if not all([tool in self.__installed_tools for tool in required_tools]):
            self.install_tools(required_tools)

        if not force and 'rtapp-calib' in self.conf:
            self._log.info('Using configuration provided RTApp calibration')
            self._calib = {
                    int(key): int(value)
                    for key, value in self.conf['rtapp-calib'].items()
                }
        else:
            self._log.info('Calibrating RTApp...')
            self._calib = RTA.calibrate(self.target)

        self._log.info('Using RT-App calibration values:')
        self._log.info('   %s',
                       "{" + ", ".join('"%r": %r' % (key, self._calib[key])
                                       for key in sorted(self._calib)) + "}")
        return self._calib

    @contextlib.contextmanager
    def freeze_userspace(self):
        if 'cgroups' not in self.target.modules:
            raise RuntimeError(
                'Failed to freeze userspace. Ensure "cgroups" module is listed '
                'among modules in target/test configuration')

        controllers = [s.name for s in self.target.cgroups.list_subsystems()]
        if 'freezer' not in controllers:
            self._log.warning('No freezer cgroup controller on target. '
                              'Not freezing userspace')
            yield
            return

        exclude = self.critical_tasks[self.target.os]
        self._log.info('Freezing all tasks except: %s', ','.join(exclude))
        self.target.cgroups.freeze(exclude)

        try:
            yield

        finally:
            self._log.info('Un-freezing userspace tasks')
            self.target.cgroups.freeze(thaw=True)

    @contextlib.contextmanager
    def record_ftrace(self, output_file=None):
        if not output_file:
            output_file = os.path.join(self.get_res_dir(), "trace.dat")

        self.ftrace.start()

        yield

        self.ftrace.stop()
        self.ftrace.get_trace(output_file)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
