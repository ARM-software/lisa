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
from devlib.utils.misc import memoized, which
from devlib import Platform, TargetError
from trappy.stats.Topology import Topology

from wlgen import RTA
from energy import EnergyMeter
from energy_model import EnergyModel
from conf import JsonConf
from platforms.juno_energy import juno_energy
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

class TestEnv(ShareState):
    """
    Represents the environment configuring LISA, the target, and the test setup

    The test environment is defined by:

    - a target configuration (target_conf) defining which HW platform we
      want to use to run the experiments
    - a test configuration (test_conf) defining which SW setups we need on
      that HW target
    - a folder to collect the experiments results, which can be specified
      using the test_conf::results_dir option and is by default wiped from
      all the previous contents (if wipe=True)

    :param target_conf:
        Configuration defining the target to run experiments on. May be

            - A dict defining the values directly
            - A path to a JSON file containing the configuration
            - ``None``, in which case $LISA_HOME/target.config is used.

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
        **tftp**
            Directory path containing kernels and DTB images for the
            target. LISA does *not* manage this TFTP server, it must be
            provided externally. Optional.

    :param test_conf: Configuration of software for target experiments. Takes
                      the same form as target_conf. Fields are:

        **modules**
            Devlib modules to be enabled. Default is []
        **exclude_modules**
            Devlib modules to be disabled. Default is [].
        **tools**
            List of tools (available under ./tools/$ARCH/) to install on
            the target. Names, not paths (e.g. ['ftrace']). Default is [].
        **ping_time**, **reboot_time**
            Override parameters to :meth:`reboot` method
        **__features__**
            List of test environment features to enable. Options are:

            "no-kernel"
                do not deploy kernel/dtb images
            "no-reboot"
                do not force reboot the target at each configuration change
            "debug"
                enable debugging messages

        **ftrace**
            Configuration for ftrace. Dictionary with keys:

            events
                events to enable.
            functions
                functions to enable in the function tracer. Optional.
            buffsize
                Size of buffer. Default is 10240.

        **results_dir**
            location of results of the experiments

    :param wipe: set true to cleanup all previous content from the output
                 folder
    :type wipe: bool

    :param force_new: Create a new TestEnv object even if there is one available
                      for this session.  By default, TestEnv only creates one
                      object per session, use this to override this behaviour.
    :type force_new: bool
    """

    _initialized = False

    def __init__(self, target_conf=None, test_conf=None, wipe=True,
                 force_new=False):
        super(TestEnv, self).__init__()

        if self._initialized and not force_new:
            return

        self.conf = {}
        self.test_conf = {}
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

        # Keep track of last installed kernel
        self.kernel = None
        self.dtb = None

        # Energy meter configuration
        self.emeter = None

        # The platform descriptor to be saved into the results folder
        self.platform = {}

        # Keep track of android support
        self.LISA_HOME = os.environ.get('LISA_HOME', '/vagrant')
        self.ANDROID_HOME = os.environ.get('ANDROID_HOME', None)
        self.CATAPULT_HOME = os.environ.get('CATAPULT_HOME',
                os.path.join(self.LISA_HOME, 'tools', 'catapult'))

        # Setup logging
        self._log = logging.getLogger('TestEnv')

        # Compute base installation path
        self._log.info('Using base path: %s', basepath)

        # Setup target configuration
        if isinstance(target_conf, dict):
            self._log.info('Loading custom (inline) target configuration')
            self.conf = target_conf
        elif isinstance(target_conf, str):
            self._log.info('Loading custom (file) target configuration')
            self.conf = self.loadTargetConfig(target_conf)
        elif target_conf is None:
            self._log.info('Loading default (file) target configuration')
            self.conf = self.loadTargetConfig()
        self._log.debug('Target configuration %s', self.conf)

        # Setup test configuration
        if test_conf:
            if isinstance(test_conf, dict):
                self._log.info('Loading custom (inline) test configuration')
                self.test_conf = test_conf
            elif isinstance(test_conf, str):
                self._log.info('Loading custom (file) test configuration')
                self.test_conf = self.loadTargetConfig(test_conf)
            else:
                raise ValueError('test_conf must be either a dictionary or a filepath')
            self._log.debug('Test configuration %s', self.conf)

        # Setup target working directory
        if 'workdir' in self.conf:
            self.workdir = self.conf['workdir']

        # Initialize binary tools to deploy
        test_conf_tools = self.test_conf.get('tools', [])
        target_conf_tools = self.conf.get('tools', [])
        self.__tools = list(set(test_conf_tools + target_conf_tools))

        # Initialize ftrace events
        # test configuration override target one
        if 'ftrace' in self.test_conf:
            self.conf['ftrace'] = self.test_conf['ftrace']
        if self.conf.get('ftrace'):
            self.__tools.append('trace-cmd')

        # Initialize features
        if '__features__' not in self.conf:
            self.conf['__features__'] = []

        # Initialize local results folder
        # test configuration overrides target one
        self.res_dir = (self.test_conf.get('results_dir') or
                        self.conf.get('results_dir'))

        if self.res_dir and not os.path.isabs(self.res_dir):
            self.res_dir = os.path.join(basepath, 'results', self.res_dir)
        else:
            self.res_dir = os.path.join(basepath, OUT_PREFIX)
            self.res_dir = datetime.datetime.now()\
                            .strftime(self.res_dir + '/%Y%m%d_%H%M%S')

        if wipe and os.path.exists(self.res_dir):
            self._log.warning('Wipe previous contents of the results folder:')
            self._log.warning('   %s', self.res_dir)
            shutil.rmtree(self.res_dir, ignore_errors=True)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        res_lnk = os.path.join(basepath, LATEST_LINK)
        if os.path.islink(res_lnk):
            os.remove(res_lnk)
        os.symlink(self.res_dir, res_lnk)

        self._init()

        # Initialize FTrace events collection
        self._init_ftrace(True)

        # Initialize RT-App calibration values
        self.calibration()

        # Initialize energy probe instrument
        self._init_energy(True)

        self._log.info('Set results folder to:')
        self._log.info('   %s', self.res_dir)
        self._log.info('Experiment results available also in:')
        self._log.info('   %s', res_lnk)

        self._initialized = True

    def loadTargetConfig(self, filepath='target.config'):
        """
        Load the target configuration from the specified file.

        :param filepath: Path of the target configuration file. Relative to the
                         root folder of the test suite.
        :type filepath: str

        """

        # Loading default target configuration
        conf_file = os.path.join(basepath, filepath)

        self._log.info('Loading target configuration [%s]...', conf_file)
        conf = JsonConf(conf_file)
        conf.load()
        return conf.json

    def _init(self, force = False):

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
        self._log.info('Topology:')
        self._log.info('   %s', CLUSTERS)

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
        self.__modules = ['cpufreq', 'cpuidle']
        if 'board' not in self.conf:
            self.conf['board'] = 'UNKNOWN'

        # Initialize TC2 board
        if self.conf['board'].upper() == 'TC2':
            platform = devlib.platform.arm.TC2()
            self.__modules = ['bl', 'hwmon', 'cpufreq']

        # Initialize JUNO board
        elif self.conf['board'].upper() in ('JUNO', 'JUNO2'):
            platform = devlib.platform.arm.Juno()
            self.nrg_model = juno_energy
            self.__modules = ['bl', 'hwmon', 'cpufreq']

        # Initialize OAK board
        elif self.conf['board'].upper() == 'OAK':
            platform = Platform(model='MT8173')
            self.__modules = ['bl', 'cpufreq']

        # Initialized HiKey board
        elif self.conf['board'].upper() == 'HIKEY':
            self.nrg_model = hikey_energy
            self.__modules = [ "cpufreq", "cpuidle" ]
            platform = Platform(model='hikey')

        # Initialize HiKey960 board
        elif self.conf['board'].upper() == 'HIKEY960':
            self.__modules = ['bl', 'cpufreq', 'cpuidle']
            platform = Platform(model='hikey960')

        # Initialize Pixel phone
        elif self.conf['board'].upper() == 'PIXEL':
            self.nrg_model = pixel_energy
            self.__modules = ['bl', 'cpufreq']
            platform = Platform(model='pixel')

        # Initialize gem5 platform
        elif self.conf['board'].upper() == 'GEM5':
            self.nrg_model = None
            self.__modules=['cpufreq']
            platform = self._init_target_gem5()

        elif self.conf['board'] != 'UNKNOWN':
            # Initilize from platform descriptor (if available)
            board = self._load_board(self.conf['board'])
            if board:
                core_names=board['cores']
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

        modules = set(self.__modules)

        # Refine modules list based on target.conf
        modules.update(self.conf.get('modules', []))
        # Merge tests specific modules
        modules.update(self.test_conf.get('modules', []))

        remove_modules = set(self.conf.get('exclude_modules', []) +
                             self.test_conf.get('exclude_modules', []))
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
            host_output_dir = self.res_dir,
            core_names = board['cores'] if board else None,
            core_clusters = self._get_clusters(board['cores']) if board else None,
            big_core = board.get('big_core', None) if board else None,
        )

        return platform

    def install_tools(self, tools):
        """
        Install tools additional to those specified in the test config 'tools'
        field

        :param tools: The list of names of tools to install
        :type tools: list(str)
        """
        tools = set(tools)

        # Add tools dependencies
        if 'rt-app' in tools:
            tools.update(['taskset', 'trace-cmd', 'perf', 'cgroup_run_into.sh'])

        # Remove duplicates and already-instaled tools
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
            self._log.info('Enabled tracepoints:')
            for event in events:
                self._log.info('   %s', event)
        if functions:
            self._log.info('Kernel functions profiled:')
            for function in functions:
                self._log.info('   %s', function)

        return self.ftrace

    def _init_energy(self, force):

        # Initialize energy probe to board default
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force,
                                              self.res_dir)

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

    def _load_peripherals(self, board):
        peripherals_path = os.path.join(basepath,
                'libs/utils/platforms', board.lower() + '.json')
        self._log.debug('Trying to load peripheral config from %s', peripherals_path)
        if not os.path.exists(peripherals_path):
            return None
        board = JsonConf(peripherals_path)
        board.load()
        if 'peripherals' not in board.json:
            return None
        return board.json['peripherals']

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

        if 'peripherals' in self.conf:
            self.platform['peripherals'] = self.conf['peripherals']
        else:
            peripheral_conf = self._load_peripherals(self.conf['board'])
            if peripheral_conf is not None:
                self.platform['peripherals'] = peripheral_conf

        self._log.debug('Platform descriptor initialized\n%s', self.platform)
        # self.platform_dump('./')

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

        required = force or 'rt-app' in self.__installed_tools

        if not required:
            self._log.debug('No RT-App workloads, skipping calibration')
            return

        if not force and 'rtapp-calib' in self.conf:
            self._log.warning('Using configuration provided RTApp calibration')
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

    def reboot(self, reboot_time=120, ping_time=15):
        """
        Reboot target.

        :param boot_time: Time to wait for the target to become available after
                          reboot before declaring failure.
        :param ping_time: Period between attempts to ping the target while
                          waiting for reboot.
        """
        # Send remote target a reboot command
        if self._feature('no-reboot'):
            self._log.warning('Reboot disabled by conf features')
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
            self._log.info('Waiting up to %s[s] for target [%s] to reboot...',
                           reboot_time, self.ip)

            ping_cmd = "ping -c 1 {} >/dev/null".format(self.ip)
            elapsed = 0
            start = time.time()
            while elapsed <= reboot_time:
                time.sleep(ping_time)
                self._log.debug('Trying to connect to [%s] target...', self.ip)
                if os.system(ping_cmd) == 0:
                    break
                elapsed = time.time() - start
            if elapsed > reboot_time:
                if self.mac:
                    self._log.warning('target [%s] not responding to PINGs, '
                                      'trying to resolve MAC address...',
                                      self.ip)
                    (self.mac, self.ip) = self.resolv_host(self.mac)
                else:
                    self._log.warning('target [%s] not responding to PINGs, '
                                      'trying to continue...',
                                      self.ip)

        # Force re-initialization of all the devlib modules
        force = True

        # Reset the connection to the target
        self._init(force)

        # Initialize FTrace events collection
        self._init_ftrace(force)

        # Initialize energy probe instrument
        self._init_energy(force)

    def install_kernel(self, tc, reboot=False):
        """
        Deploy kernel and DTB via TFTP, optionally rebooting

        :param tc: Dicionary containing optional keys 'kernel' and 'dtb'. Values
                   are paths to the binaries to deploy.
        :type tc: dict

        :param reboot: Reboot thet target after deployment
        :type reboot: bool
        """

        # Default initialize the kernel/dtb settings
        tc.setdefault('kernel', None)
        tc.setdefault('dtb', None)

        if self.kernel == tc['kernel'] and self.dtb == tc['dtb']:
            return

        self._log.info('Install kernel [%s] on target...', tc['kernel'])

        # Install kernel/dtb via FTFP
        if self._feature('no-kernel'):
            self._log.warning('Kernel deploy disabled by conf features')

        elif 'tftp' in self.conf:
            self._log.info('Deploy kernel via TFTP...')

            # Deploy kernel in TFTP folder (mandatory)
            if 'kernel' not in tc or not tc['kernel']:
                raise ValueError('Missing "kernel" parameter in conf: %s',
                        'KernelSetup', tc)
            self.tftp_deploy(tc['kernel'])

            # Deploy DTB in TFTP folder (if provided)
            if 'dtb' not in tc or not tc['dtb']:
                self._log.debug('DTB not provided, using existing one')
                self._log.debug('Current conf:\n%s', tc)
                self._log.warning('Using pre-installed DTB')
            else:
                self.tftp_deploy(tc['dtb'])

        else:
            raise ValueError('Kernel installation method not supported')

        # Keep track of last installed kernel
        self.kernel = tc['kernel']
        if 'dtb' in tc:
            self.dtb = tc['dtb']

        if not reboot:
            return

        # Reboot target
        self._log.info('Rebooting taget...')
        self.reboot()


    def tftp_deploy(self, src):
        """
        .. TODO
        """

        tftp = self.conf['tftp']

        dst = tftp['folder']
        if 'kernel' in src:
            dst = os.path.join(dst, tftp['kernel'])
        elif 'dtb' in src:
            dst = os.path.join(dst, tftp['dtb'])
        else:
            dst = os.path.join(dst, os.path.basename(src))

        cmd = 'cp {} {} && sync'.format(src, dst)
        self._log.info('Deploy %s into %s', src, dst)
        result = os.system(cmd)
        if result != 0:
            self._log.error('Failed to deploy image: %s', src)
            raise ValueError('copy error')

    def _feature(self, feature):
        return feature in self.conf['__features__']

IFCFG_BCAST_RE = re.compile(
    r'Bcast:(.*) '
)

# vim :set tabstop=4 shiftwidth=4 expandtab
