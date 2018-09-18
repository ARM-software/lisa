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
import os
import contextlib
import logging
from pathlib import Path

import devlib
from devlib.utils.misc import which
from devlib import Platform, TargetError
from trappy.stats.Topology import Topology

from lisa.wlgen.rta import RTA
from lisa.energy import EnergyMeter
from lisa.energy_model import EnergyModel
from lisa.conf import JsonConf, BASEPATH
from lisa.utilities import Loggable
from lisa.platforms.juno_r0_energy import juno_r0_energy
from lisa.platforms.hikey_energy import hikey_energy
from lisa.platforms.pixel_energy import pixel_energy

USERNAME_DEFAULT = 'root'
PASSWORD_DEFAULT = ''
FTRACE_EVENTS_DEFAULT = ['sched:*']
FTRACE_BUFSIZE_DEFAULT = 10240
OUT_PREFIX = 'results'
LATEST_LINK = 'results_latest'

if BASEPATH:
    platforms_path = os.path.join(BASEPATH, 'lisa', 'platforms')

class ArtifactPath(Path):
    """Path to a folder that can be used to store artifacts of a function.
    This must be a clean folder, already created on disk.
    """
    pass

class TargetConfig(JsonConf):
    pass

class TestEnv(Loggable):
    """
    Represents the environment configuring LISA, the target, and the test setup

    :param target_conf: Configuration defining the target to use. It may be:

            - A dict defining the values directly
            - A path to a JSON file containing the configuration
            - ``None``, in which case:
        - $LISA_TARGET_CONF environment variable is read to locate a
                  config file.
                - If the variable is not set, $LISA_HOME/target.config is used.

        You need to provide the information needed to connect to the
        target. For SSH targets that means "host", "username" and
        either "password" or "keyfile". All other fields are optional if
      the relevant features aren't needed.

    :target_conf parameters:
        :platform: Type of target, can be either of:
          - "linux" (ssh connection)
          - "android" (adb connection)
          - "host" for localhost
        :host: Target IP or hostname for SSH access
        :username: For SSH access
        :keyfile: Path to SSH key (alternative to password)
        :password: SSH password (alternative to keyfile)
        :device:  Target Android device ID if using ADB
        :port: Port for Android connection default port is 5555
        :rtapp-calib: Calibration values for RT-App. If unspecified, LISA will
            calibrate RT-App on the target. A message will be logged with
            a value that can be copied here to avoid having to re-run
            calibration on subsequent tests.
        :ftrace: Ftrace configuration - see :meth:`configure_ftrace`.

    :param force_new: Create a new TestEnv object even if there is one available
                      for this session.  By default, TestEnv only creates one
                      object per session, use this to override this behaviour.
    :type force_new: bool

    The role of :class:`TestEnv` is to bundle together Devlib
    features (such as :mod:`libs.devlib.devlib.instrument`,
    :mod:`libs.devlib.devlib.trace`, ...) and provide some helper methods to
    manipulate them.
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
    freeze when using :meth:`freeze_userspace`.
    """

    def __init__(self, target_conf:TargetConfig=None):
        super().__init__()

        # Compute base installation path
        self.logger.info('Using base path: %s', BASEPATH)

        self._pre_target_init(target_conf)
        self._init_target()
        self._post_target_init()

    def _load_em(self, board):
        em_path = os.path.join(
            platforms_path, board.lower() + '.json')
        self.logger.debug('Trying to load default EM from %s', em_path)
        if not os.path.exists(em_path):
            return None
        self.logger.info('Loading default EM:')
        self.logger.info('   %s', em_path)
        board = JsonConf(em_path)
        board.load()
        if 'nrg_model' not in board.json:
            return None
        return board.json['nrg_model']

    def _load_board(self, board):
        board_path = os.path.join(
            platforms_path, board.lower() + '.json')
        self.logger.debug('Trying to load board descriptor from %s', board_path)
        if not os.path.exists(board_path):
            return None
        self.logger.info('Loading board:')
        self.logger.info('   %s', board_path)
        board = JsonConf(board_path)
        board.load()
        if 'board' not in board.json:
            return None
        return board.json['board']

    def _build_topology(self):
        # Initialize target Topology for behavior analysis
        clusters = []

        # Build topology for a big.LITTLE systems
        if self.target.big_core and \
           (self.target.abi == 'arm64' or self.target.abi == 'armeabi'):
            # Populate cluster for a big.LITTLE platform
            if self.target.big_core:
                # Load cluster of LITTLE cores
                clusters.append(
                    [i for i,t in enumerate(self.target.core_names)
                                if t == self.target.little_core])
                # Load cluster of big cores
                clusters.append(
                    [i for i,t in enumerate(self.target.core_names)
                                if t == self.target.big_core])
        # Build topology for an SMP systems
        elif not self.target.big_core or \
             self.target.abi == 'x86_64':
            for core in set(self.target.core_clusters):
                clusters.append(
                    [i for i,v in enumerate(self.target.core_clusters)
                     if v == core])
        self.topology = Topology(clusters=clusters)
        self.logger.info('Topology:')
        self.logger.info('   %s', clusters)

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
            self.logger.warning('Unable to identify cluster frequencies')

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
        self.platform['kernel'] = {
            t: getattr(kver, t, None) for t in [
                'release', 'version',
                'version_number', 'major', 'minor',
                'rc', 'sha1', 'parts'
            ]
        }
        self.platform['abi'] = self.target.abi
        self.platform['os'] = self.target.os

        self.logger.debug('Platform descriptor initialized\n%s', self.platform)
        # self.platform_dump('./')

    def _init_energy(self, force):
        # Initialize energy probe to board default
        self.emeter = EnergyMeter.getInstance(self.target, self.conf, force)

    def _pre_target_init(self, target_conf):
        """
        Initialize everything that does not require a live target
        """
        self.conf = {}
        self.target = None
        self.ftrace = None
        self.workdir = None
        self.__installed_tools = set()
        self.__modules = []
        self.__connection_settings = None
        self._calib = None

        # Keep track of target IP
        self.ip = None

        # Energy meter configuration
        self.emeter = None

        # The platform descriptor to be saved into the results folder
        self.platform = {}

        # Setup target configuration
        if isinstance(target_conf, dict):
            self.logger.info('Loading custom (inline) target configuration')
            self.conf = target_conf
        elif isinstance(target_conf, str):
            self.logger.info('Loading %s target configuration', target_conf)
            self.conf = self.load_target_config(target_conf)
        else:
            target_conf = os.environ.get('LISA_TARGET_CONF', '')
            self.logger.info('Loading [%s] target configuration',
                    target_conf or 'default')
            self.conf = self.load_target_config(target_conf)

        self.logger.debug('Target configuration %s', self.conf)

        # Setup target working directory
        if 'workdir' in self.conf:
            self.workdir = self.conf['workdir']

        # Initialize binary tools to deploy
        self.__tools = list(set(self.conf.get('tools', [])))

        # Initialize ftrace events
        # test configuration override target one
        ftrace_conf = self.conf.get('ftrace', {})

        # Merge the events from target config and test config
        ftrace_conf['events'] = sorted(set(ftrace_conf.get('events', [])))
        self.conf['ftrace'] = ftrace_conf

        # Initialize features
        if '__features__' not in self.conf:
            self.conf['__features__'] = []

    def _init_target(self):
        """
        Initialize the Target
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

        # Configure the host IP
        if 'host' in self.conf:
            self.ip = self.conf['host']
            self.__connection_settings['host'] = self.ip

        try:
            platform_type = self.conf['platform']
        except KeyError:
            raise ValueError('Config error: missing [platform] parameter')

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
        self.logger.info('Devlib modules to load: %s', self.__modules)

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
            self.logger.info('Connecting Android target [%s]', device)
        else:
            self.logger.info('Connecting %s target:', platform_type)
            for key in self.__connection_settings:
                self.logger.info('%10s : %s', key,
                               self.__connection_settings[key])

        self.logger.info('Connection settings:')
        self.logger.info('   %s', self.__connection_settings)

        if platform_type.lower() == 'linux':
            self.logger.debug('Setup LINUX target...')
            if "host" not in self.__connection_settings:
                raise ValueError('Missing "host" param in Linux target conf')

            self.target = devlib.LinuxTarget(
                    platform = platform,
                    connection_settings = self.__connection_settings,
                    working_directory = self.workdir,
                    load_default_modules = False,
                    modules = self.__modules)

        elif platform_type.lower() == 'android':
            self.logger.debug('Setup ANDROID target...')

            self.target = devlib.AndroidTarget(
                    platform = platform,
                    connection_settings = self.__connection_settings,
                    working_directory = self.workdir,
                    load_default_modules = False,
                    modules = self.__modules)

        elif platform_type.lower() == 'host':
            self.logger.debug('Setup HOST target...')

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

        self.logger.debug('Checking target connection...')
        self.logger.debug('Target info:')
        self.logger.debug('      ABI: %s', self.target.abi)
        self.logger.debug('     CPUs: %s', self.target.cpuinfo)
        self.logger.debug(' Clusters: %s', self.target.core_clusters)

        self.logger.info('Initializing target workdir:')
        self.logger.info('   %s', self.target.working_directory)

        self.target.setup()
        self.install_tools(self.__tools)

        # Verify that all the required modules have been initialized
        for module in self.__modules:
            self.logger.debug('Check for module [%s]...', module)
            if not hasattr(self.target, module):
                self.logger.warning('Unable to initialize [%s] module', module)
                self.logger.error('Fix your target kernel configuration or '
                                'disable module from configuration')
                raise RuntimeError('Failed to initialized [{}] module, '
                        'update your kernel or test configurations'.format(module))

        if not self.nrg_model:
            try:
                self.logger.info('Attempting to read energy model from target')
                self.nrg_model = EnergyModel.from_target(self.target)
            except (TargetError, RuntimeError, ValueError) as err:
                self.logger.error("Couldn't read target energy model: %s", err)

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
            host_output_dir = str(self.get_res_dir('gem5')),
            core_names = board['cores'] if board else None,
            core_clusters = self._get_clusters(board['cores']) if board else None,
            big_core = board.get('big_core', None) if board else None,
        )

        return platform

    def _post_target_init(self):
        """
        Initialize everything that requires a live target
        """
        self._build_topology()

        # Initialize the platform descriptor
        self._init_platform()

        # Initialize energy probe instrument
        self._init_energy(True)

    def load_target_config(self, filepath=None):
        """
        Load the target configuration from the specified file.

        :param filepath: Path of the target configuration file. Relative to the
                         root folder of the test suite.
        :type filepath: str

        """

        # "" and None are replaced by the default 'target.config' value
        filepath = filepath or 'target.config'

        # Loading default target configuration
        conf_file = os.path.join(BASEPATH, filepath)

        self.logger.info('Loading target configuration [%s]...', conf_file)
        conf = JsonConf(conf_file)
        conf.load()
        return conf.json

    def get_res_dir(self, name=None, append_time=True, symlink=True):
        """
        Returns a directory managed by LISA to store results

        :param name: Name of the results directory
        :type name: str

        :param append_time: If True and :attr:`name` is not None, the
          current datetime will be appended to :attr:`name`
        :type append_time: bool

        :param symlink: Create a symlink named ``results_latest`` to the newly
          create results directory
        :type symlink: bool
        """

        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not name:
            name = time_str
        elif name and append_time:
            name = "{}-{}".format(name, time_str)

        res_dir = Path(BASEPATH, OUT_PREFIX, name)

        if not res_dir.exists():
            res_dir.mkdir()

        if symlink:
            res_lnk = Path(BASEPATH, LATEST_LINK)
            with contextlib.suppress(FileNotFoundError):
                res_lnk.unlink()
            res_lnk.symlink_to(res_dir)

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
            binary = '{}/tools/scripts/{}'.format(BASEPATH, tool)
            if not os.path.isfile(binary):
                binary = '{}/tools/{}/{}'\
                         .format(BASEPATH, self.target.abi, tool)
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
            return sorted(set(value) | set(target_conf.get(index, default)))

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
            self.logger.info('Enabled tracepoints:')
            for event in events:
                self.logger.info('   %s', event)
        if functions:
            self.logger.info('Kernel functions profiled:')
            for function in functions:
                self.logger.info('   %s', function)

    def platform_dump(self, dest_dir, dest_file='platform.json'):
        plt_file = os.path.join(dest_dir, dest_file)
        self.logger.debug('Dump platform descriptor in [%s]', plt_file)
        with open(plt_file, 'w') as ofile:
            json.dump(self.platform, ofile, sort_keys=True, indent=4)
        return (self.platform, plt_file)

    def get_rtapp_calibration(self, force=False, policy=lambda x: min(x.values())):
        """
        Get rt-app calibration. Run calibration on target if necessary.

        :param force: Always run calibration on target, even if we have already
          stored some calibration result
        :type force: bool

        :param policy: Policy to select calibration value. Defaults to the
          smallest calibration value, which means calibration will be based on
          the biggest CPU in the system.
        :type policy: callable

        :returns: int
        """
        if not force and self._calib:
            return policy(self._calib)

        if not force and 'rtapp-calib' in self.conf:
            self.logger.info('Using configuration provided RTApp calibration')
            self._calib = {
                    int(key): int(value)
                    for key, value in list(self.conf['rtapp-calib'].items())
                }
        else:
            self.logger.info('Calibrating RTApp...')
            self._calib = RTA.get_cpu_calibrations(self)

        self.logger.info('Using RT-App calibration values:')
        self.logger.info('   %s',
                       "{" + ", ".join('"%r": %r' % (key, self._calib[key])
                                       for key in sorted(self._calib)) + "}")
        return policy(self._calib)

    @contextlib.contextmanager
    def freeze_userspace(self):
        """
        Context manager that lets you freeze the userspace
        """
        if 'cgroups' not in self.target.modules:
            raise RuntimeError(
                'Failed to freeze userspace. Ensure "cgroups" module is listed '
                'among modules in target/test configuration')

        controllers = [s.name for s in self.target.cgroups.list_subsystems()]
        if 'freezer' not in controllers:
            self.logger.warning('No freezer cgroup controller on target. '
                              'Not freezing userspace')
            yield
            return

        exclude = self.critical_tasks[self.target.os]
        self.logger.info('Freezing all tasks except: %s', ','.join(exclude))
        self.target.cgroups.freeze(exclude)

        try:
            yield

        finally:
            self.logger.info('Un-freezing userspace tasks')
            self.target.cgroups.freeze(thaw=True)

    @contextlib.contextmanager
    def record_ftrace(self, output_file=None):
        """
        Context manager that lets you record an Ftrace trace

        :param output_file: Filepath for the trace to be created
        :type output_file: str
        """
        if not output_file:
            output_file = self.get_res_dir().joinpath('trace.dat')

        self.ftrace.start()

        yield

        self.ftrace.stop()
        self.ftrace.get_trace(str(output_file))

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
