# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
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
import os.path
import contextlib
import logging
from pathlib import Path
import shlex
from collections.abc import Mapping
import copy

import devlib
from devlib.utils.misc import which
from devlib import Platform
from devlib.platform.gem5 import Gem5SimulationPlatform

from lisa.wlgen.rta import RTA
from lisa.energy_meter import EnergyMeter
from lisa.conf import BASEPATH
from lisa.utils import Loggable, MultiSrcConf, HideExekallID, resolve_dotted_name, get_all_subclasses, import_all_submodules, TypedList

from lisa.platform import PlatformInfo

USERNAME_DEFAULT = 'root'
ADB_PORT_DEFAULT = 5555
SSH_PORT_DEFAULT = 22
FTRACE_EVENTS_DEFAULT = ['sched:*']
FTRACE_BUFSIZE_DEFAULT = 10240
OUT_PREFIX = 'results'
LATEST_LINK = 'results_latest'
DEFAULT_DEVLIB_MODULES = ['sched', 'cpufreq', 'cpuidle']

class ArtifactPath(str):
    """Path to a folder that can be used to store artifacts of a function.
    This must be a clean folder, already created on disk.
    """
    pass

class TargetConf(MultiSrcConf, HideExekallID):
    YAML_MAP_TOP_LEVEL_KEY = 'target-conf'

    STRUCTURE = {
        'kind': str,
        'host': str,
        'board': str,
        'username': str,
        'password': str,
        'port': int,
        'keyfile': str,
        'workdir': str,
        'tools': TypedList[str],
        'ftrace': {
            'events': TypedList[str],
            'functions': TypedList[str],
            'buffsize': int,
        },
        'devlib': {
            'platform': {
                'class': str,
                'args': Mapping,
            },
            'excluded-modules': TypedList[str],
        }
    }

    DEFAULT_CONF = {
        'devlib': {
            'platform': {
                'class': 'devlib.platform.Platform'
            }
        }
    }

    def __init__(self, conf, src='user'):
        super().__init__(conf=conf, src=src)
        # Give some preset in the the lowest prio source
        self.add_src('default', self.DEFAULT_CONF, fallback=True)

    # We do not allow overriding source for this kind of configuration to keep
    # the YAML interface simple and dict-like
    @classmethod
    def from_map(cls, mapping):
        return cls(mapping)

    def to_map(self):
        return dict(self._get_chainmap())

class TestEnv(Loggable):
    """
    Represents the environment configuring LISA, the target, and the test setup

    :param target_conf: Configuration defining the target to use.
    :type target_conf: TargetConf

    You need to provide the information needed to connect to the
    target. For SSH targets that means "host", "username" and
    either "password" or "keyfile". All other fields are optional if
    the relevant features aren't needed.

    The role of :class:`TestEnv` is to wrap :class:`devlib.target.Target` to
    provide additional features on top of it.
    """

    CRITICAL_TASKS = {
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

    def __init__(self, target_conf:TargetConf, plat_info:PlatformInfo=None, res_dir:ArtifactPath=None):
        super().__init__()
        logger = self.get_logger()

        # That res_dir is for the exclusive use of TestEnv itself, it must not
        # be used by users of TestEnv
        self._res_dir = res_dir

        self.target_conf = target_conf
        logger.debug('Target configuration %s', self.target_conf)

        if plat_info is None:
            plat_info = PlatformInfo()
        else:
            # Make a copy of the PlatformInfo so we don't modify the original
            # one we were passed when adding the target source to it
            plat_info = copy.copy(plat_info)
        self.plat_info = plat_info
        logger.info('Pre-configured platform information:\n%s', self.plat_info)

        # Compute base installation path
        logger.info('Using base path: %s', BASEPATH)

        self.ftrace = None
        self._installed_tools = set()
        self.target = self._init_target(self.target_conf, self._res_dir)

        # Initialize binary tools to deploy
        tools = self.target_conf.get('tools', [])
        if tools:
            logger.info('Tools to install: %s', tools)
            self.install_tools(target, tools)

        board_name = target_conf.get('board', None)
        self.tags = [board_name] if board_name else []

        # Autodetect information from the target, after the TestEnv is
        # initialized. Expensive computations are deferred so they will only be
        # computed when actually needed.
        self.plat_info.add_target_src(self, fallback=True)

        # Update the PlatformInfo with keys derived from the energy model
        with contextlib.suppress(KeyError):
            self.plat_info.add_from_nrg_model_src()

        logger.info('Effective platform information:\n%s', self.plat_info)

    @classmethod
    def from_default_conf(cls):
        path = os.getenv('LISA_CONF')
        return cls.from_one_conf(path)

    @classmethod
    def from_one_conf(cls, path):
        target_conf = TargetConf.from_yaml_map(path)
        plat_info = PlatformInfo.from_yaml_map(path)
        return cls(target_conf=target_conf, plat_info=plat_info)

    def _init_target(self, target_conf, res_dir):
        """
        Initialize the Target
        """
        logger = self.get_logger()
        target_kind = target_conf['kind']
        target_workdir = target_conf.get('workdir', None)
        conn_settings = {}

        # If the target is Android, we need just (eventually) the device
        if target_kind == 'android':
            logger.debug('Setting up Android target...')
            devlib_target_cls = devlib.AndroidTarget

            # Workaround for ARM-software/devlib#225
            target_workdir = target_workdir or '/data/local/tmp/devlib-target'

            if 'device' in target_conf:
                device = target_conf['device']
            elif 'host' in target_conf:
                host = target_conf['host']
                port = target_conf.get('port', ADB_PORT_DEFAULT)
                device = '{}:{}'.format(host, port)
            else:
                device = 'DEFAULT'

            conn_settings['device'] = device

        elif target_kind == 'linux':
            logger.debug('Setting up Linux target...')
            devlib_target_cls = devlib.LinuxTarget

            conn_settings['username'] = target_conf.get('username', USERNAME_DEFAULT)
            conn_settings['port'] = target_conf.get('port', SSH_PORT_DEFAULT)
            conn_settings['host'] = target_conf['host']

            # Configure password or SSH keyfile
            if 'keyfile' in target_conf:
                conn_settings['keyfile'] = target_conf['keyfile']
            else:
                conn_settings['password'] = target_conf.get('password', None)
        elif target_kind == 'host':
            logger.debug('Setting up localhost Linux target...')
            devlib_target_cls = devlib.LocalLinuxTarget
            conn_settings['unrooted'] = True
        else:
            raise ValueError('Unsupported platform type {}'.format(target_kind))

        board_name = target_conf.get('board', '')
        logger.info('%s %s target connection settings:', target_kind, board_name)
        for key, val in conn_settings.items():
            logger.info('%10s : %s', key, val)

        ########################################################################
        # Devlib Platform configuration
        ########################################################################

        devlib_platform_cls_name = target_conf['devlib']['platform']['class']
        devlib_platform_kwargs = target_conf['devlib']['platform'].get('args', {})

        devlib_platform_cls = resolve_dotted_name(devlib_platform_cls_name)

        # Hack for Gem5 devlib Platform, that requires a "host_output_dir"
        # argument computed at runtime.
        # Note: lisa.env.Gem5SimulationPlatformWrapper should be used instead
        # of the original one to benefit from mapping configuration
        if issubclass(devlib_platform_cls, Gem5SimulationPlatform):
            devlib_platform_kwargs.setdefault('host_output_dir', res_dir)

        # Actually build the devlib Platform object
        devlib_platform = devlib_platform_cls(**devlib_platform_kwargs)

        ########################################################################
        # Devlib modules configuration
        ########################################################################

        # Make sure all submodules of devlib.module are imported so the classes
        # are all created
        import_all_submodules(devlib.module)
        # Get all devlib Module subclasses that exist
        devlib_module_set = {
            cls.name
            for cls in get_all_subclasses(devlib.module.Module)
            if (
                getattr(cls, 'name', None)
                # early modules try to connect to UART and do very
                # platform-specific things we are not interested in
                and getattr(cls, 'stage') != 'early'
            )
        }

        # The platform can define a list of modules to exclude, in case a given
        # module really have troubles on a given platform.
        excluded_devlib_modules = set(target_conf['devlib'].get('excluded-modules', []))
        devlib_module_set.difference_update(excluded_devlib_modules)

        devlib_module_list = sorted(devlib_module_set)
        logger.info('Devlib modules to load: %s', devlib_module_list)

        ########################################################################
        # Create devlib Target object
        ########################################################################

        target = devlib_target_cls(
            platform = devlib_platform,
            modules = devlib_module_list,
            load_default_modules = False,
            connection_settings = conn_settings,
            working_directory = target_workdir,
        )

        logger.debug('Checking target connection...')
        logger.debug('Target info:')
        logger.debug('      ABI: %s', target.abi)
        logger.debug('     CPUs: %s', target.cpuinfo)
        logger.debug(' clusters: %s', target.core_clusters)
        logger.debug('  workdir: %s', target.working_directory)

        target.setup()

        # Verify that all the required modules have been initialized
        for module in devlib_module_list:
            if not hasattr(target, module):
                logger.warning('Failed to initialized "{}" devlib Module'.format(module))
        return target

    def get_res_dir(self, name=None, append_time=True, symlink=True):
        """
        Returns a directory managed by LISA to store results.

        Usage of that function is reserved to interactive use or simple scripts.
        Tests should not rely on that as the created folder will not be tracked
        by any external entity, which means the results will be lost in some
        automated environment.

        :param name: Name of the results directory
        :type name: str

        :param append_time: If True and :attr:`name` is not None, the
          current datetime will be appended to :attr:`name`
        :type append_time: bool

        :param symlink: Create a symlink named ``results_latest`` to the newly
          create results directory
        :type symlink: bool
        """

        time_str = datetime.now().strftime('%Y%m%d_%H%M%S.%f')
        if not name:
            name = time_str
        elif append_time:
            name = "{}-{}".format(name, time_str)

        res_dir = os.path.join(BASEPATH, OUT_PREFIX, name)

        try:
            os.mkdir(res_dir)
        except FileExistsError:
            pass

        if symlink:
            res_lnk = Path(BASEPATH, LATEST_LINK)
            with contextlib.suppress(FileNotFoundError):
                res_lnk.unlink()

            # There may be a race condition with another tool trying to create
            # the link
            with contextlib.suppress(FileExistsError):
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
        tools.difference_update(self._installed_tools)

        tools_to_install = []
        for tool in tools:
            binary = '{}/tools/scripts/{}'.format(BASEPATH, tool)
            if not os.path.isfile(binary):
                binary = '{}/tools/{}/{}'\
                         .format(BASEPATH, self.target.abi, tool)
            tools_to_install.append(binary)

        for tool_to_install in tools_to_install:
            self.target.install(tool_to_install)

        self._installed_tools.update(tools)

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
        logger = self.get_logger()

        # Merge with setup from target config
        target_conf = self.target_conf.get('ftrace', {})

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
        if 'trace-cmd' not in self._installed_tools:
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
            logger.info('Enabled tracepoints:')
            for event in events:
                logger.info('   %s', event)
        if functions:
            logger.info('Kernel functions profiled:')
            for function in functions:
                logger.info('   %s', function)

    @contextlib.contextmanager
    def freeze_userspace(self):
        """
        Context manager that lets you freeze the userspace
        """
        logger = self.get_logger()
        if 'cgroups' not in self.target.modules:
            raise RuntimeError(
                'Failed to freeze userspace. Ensure "cgroups" module is listed '
                'among modules in target/test configuration')

        controllers = [s.name for s in self.target.cgroups.list_subsystems()]
        if 'freezer' not in controllers:
            logger.warning('No freezer cgroup controller on target. '
                              'Not freezing userspace')
            yield
            return

        exclude = self.CRITICAL_TASKS[self.target.os]
        logger.info('Freezing all tasks except: %s', ','.join(exclude))
        self.target.cgroups.freeze(exclude)

        try:
            yield

        finally:
            logger.info('Un-freezing userspace tasks')
            self.target.cgroups.freeze(thaw=True)

    @contextlib.contextmanager
    def record_ftrace(self, output_file):
        """
        Context manager that lets you record an Ftrace trace

        :param output_file: Filepath for the trace to be created
        :type output_file: str
        """
        self.ftrace.start()
        yield
        self.ftrace.stop()
        self.ftrace.get_trace(output_file)

    @contextlib.contextmanager
    def disable_idle_states(self):
        """
        Context manager that lets you disable all idle states
        """
        # This assumes that freq domains are tied to "idle domains"
        # We'll have to change this if this assumption no longer holds true
        for domain in self.target.cpufreq.iter_domains():
            self.target.cpuidle.disable_all(domain[0])

        try:
            yield

        finally:
            for domain in self.target.cpufreq.iter_domains():
                self.target.cpuidle.enable_all(domain[0])

class Gem5SimulationPlatformWrapper(Gem5SimulationPlatform):
    def __init__(self, system, simulator, **kwargs):
            simulator_args = copy.copy(simulator.get('args', []))
            system_platform = system['platform']

            # Get gem5 binary arguments
            simulator_args.append('--listener-mode=on')

            simulator_args.append(system_platform['description'])
            simulator_args.extend(system_platform.get('args', []))

            simulator_args += ['--kernel {}'.format(system['kernel']),
                     '--dtb {}'.format(system['dtb']),
                     '--disk-image {}'.format(system['disk'])]

            # Quote/escape arguments and build the command line
            gem5_args = ' '.join(shlex.quote(a) for a in simulator_args)

            diod_path = which('diod')
            if diod_path is None:
                raise RuntimeError('Failed to find "diod" on your host machine, check your installation or your PATH variable')

            # Setup virtio
            # Brackets are there to let the output dir be created automatically
            virtio_args = '--which-diod={} --workload-automation-vio={{}}'.format(diod_path)

            super().__init__(
                gem5_args=gem5_args,
                gem5_bin=simulator['bin'],
                virtio_args=virtio_args,
                **kwargs
            )
# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
