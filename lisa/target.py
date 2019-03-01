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
import shlex
from collections.abc import Mapping
import copy
import sys
import argparse
import textwrap
import functools
import inspect
import abc

import devlib
from devlib.utils.misc import which
from devlib import Platform
from devlib.platform.gem5 import Gem5SimulationPlatform

from lisa.wlgen.rta import RTA
from lisa.utils import Loggable, HideExekallID, resolve_dotted_name, get_all_subclasses, import_all_submodules, LISA_HOME, RESULT_DIR, LATEST_LINK, setup_logging, ArtifactPath
from lisa.conf import SimpleMultiSrcConf, KeyDesc, LevelKeyDesc, TopLevelKeyDesc, StrList, Configurable

from lisa.platforms.platinfo import PlatformInfo

class PasswordKeyDesc(KeyDesc):
    def pretty_format(self, v):
        return '<password>'

class TargetConf(SimpleMultiSrcConf, HideExekallID):
    """
    Target connection settings.

    Only keys defined below are allowed, with the given meaning and type:

    {generated_help}

    An instance can be created by calling :class:`~TargetConf` with a
    dictionary. The top-level `target-conf` key is not needed here:

    .. code-block:: python

        TargetConf({{
            'name': 'myboard',
            'host': 192.0.2.1,
            'kind': 'linux',
            'username': 'foo',
            'password': 'bar',
        }})

    Or alternatively, from a YAML configuration file:

    Content of target_conf.yml:

    .. literalinclude:: ../target_conf.yml
        :language: YAML

    ::

        TargetConf.from_yaml_map('target_conf.yml')


    The following special YAML tags can be used in the configuration file:

    .. code-block:: YAML

        target-conf:
            # "!env:<type> ENV_VAR_NAME" can be used to reference an
            # environment variable.
            name: !env:str BOARD_NAME
            port: !env:int PORT

    .. note:: That structure in a YAML file is allowed and will work:

        * file foo.yml::

            target-conf:
                name: myboard

        * file bar.yml::

            target-conf:
                !include foo.yml

        This will result in that structure which would normally be invalid, but
        is handled as a special case::

            target-conf:
                target-conf:
                    name: myboard
    """

    STRUCTURE = TopLevelKeyDesc('target-conf', 'target connection settings', (
        KeyDesc('name', 'Board name, free-form value only used to embelish logs', [str]),
        KeyDesc('kind', 'Target kind. Can be "linux" (ssh) or "android" (adb)', [str]),

        KeyDesc('host', 'Hostname or IP address of the host', [str, None]),
        KeyDesc('username', 'SSH username', [str, None]),
        PasswordKeyDesc('password', 'SSH password', [str, None]),
        KeyDesc('port', 'SSH or ADB server port', [int, None]),
        KeyDesc('device', 'ADB device. Takes precedence over "host"', [str, None]),
        KeyDesc('keyfile', 'SSH private key file', [str, None]),
        KeyDesc('workdir', 'Remote target workdir', [str]),
        KeyDesc('tools', 'List of tools to install on the target', [StrList]),
        LevelKeyDesc('devlib', 'devlib configuration', (
            # Using textual name of the Platform allows this YAML configuration
            # to not use any python-specific YAML tags, so TargetConf files can
            # be parsed and produced by any other third-party code
            LevelKeyDesc('platform', 'devlib.platform.Platform subclass specification', (
                KeyDesc('class', 'Name of the class to use', [str]),
                KeyDesc('args', 'Keyword arguments to build the Platform object', [Mapping]),
            )),
            KeyDesc('excluded-modules', 'List of devlib modules to *not* load', [StrList]),
        ))
    ))

    DEFAULT_SRC = {
        'devlib': {
            'platform': {
                'class': 'devlib.platform.Platform'
            }
        }
    }

    def __init__(self, conf=None, src='user'):
        super().__init__(conf=conf, src=src)

class Target(Loggable, HideExekallID, Configurable):
    """
    Wrap :class:`devlib.target.Target` to provide additional features on top of
    it.

    {configurable_params}

    :param devlib_platform: Instance of :class:`devlib.platform.Platform` to
        use to build the :class:`devlib.target.Target`
    :type devlib_platform: devlib.platform.Platform

    You need to provide the information needed to connect to the
    target. For SSH targets that means "host", "username" and
    either "password" or "keyfile". All other fields are optional if
    the relevant features aren't needed.

    .. note:: The wrapping of :class:`devlib.target.Target` is done using
        composition, as opposed to inheritance. This allows swapping the exact
        class used under the hood, and avoids messing up with ``devlib``
        internal members.
    """

    ADB_PORT_DEFAULT = 5555
    SSH_PORT_DEFAULT = 22

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

    CONF_CLASS = TargetConf
    INIT_KWARGS_KEY_MAP = {
        'devlib_excluded_modules': ['devlib', 'excluded-modules']
    }

    def __init__(self, kind, name='<noname>', tools=[], res_dir=None,
        plat_info=None, workdir=None, device=None, host=None, port=None,
        username='root', password=None, keyfile=None, devlib_platform=None,
        devlib_excluded_modules=[]
    ):

        super().__init__()
        logger = self.get_logger()

        self.name = name

        res_dir = res_dir if res_dir else self._get_res_dir(
            root=os.path.join(LISA_HOME, RESULT_DIR),
            relative='',
            name='{}-{}'.format(self.__class__.__qualname__, self.name),
            append_time=True,
            symlink=True
        )

        self._res_dir = res_dir
        os.makedirs(self._res_dir, exist_ok=True)
        if os.listdir(self._res_dir):
            raise ValueError('res_dir must be empty: {}'.format(self._res_dir))

        if plat_info is None:
            plat_info = PlatformInfo()
        else:
            # Make a copy of the PlatformInfo so we don't modify the original
            # one we were passed when adding the target source to it
            plat_info = copy.copy(plat_info)
            logger.info('User-defined platform information:\n%s', plat_info)

        self.plat_info = plat_info

        # Take the board name from the target configuration so it becomes
        # available for later inspection. That board name is mostly free form
        # and no specific value should be expected for a given kind of board
        # (i.e. a Juno board might be named "foo-bar-juno-on-my-desk")
        if name:
            self.plat_info.add_src('target-conf', dict(name=name))


        self._installed_tools = set()
        self.target = self._init_target(
                kind=kind,
                name=name,
                workdir=workdir,
                device=device,
                host=host,
                port=port,
                username=username,
                password=password,
                keyfile=keyfile,
                devlib_platform=devlib_platform,
                devlib_excluded_modules=devlib_excluded_modules,
            )

        # Initialize binary tools to deploy
        if tools:
            logger.info('Tools to install: %s', tools)
            self.install_tools(target, tools)

        # Autodetect information from the target, after the Target is
        # initialized. Expensive computations are deferred so they will only be
        # computed when actually needed.

        rta_calib_res_dir = os.path.join(self._res_dir, 'rta_calib')
        os.makedirs(rta_calib_res_dir)
        self.plat_info.add_target_src(self, rta_calib_res_dir, fallback=True)

        logger.info('Effective platform information:\n%s', self.plat_info)

    def __getattr__(self, attr):
        """
        Forward all non-overriden attributes/method accesses to the underlying
        :class:`devlib.target.Target`.

        .. note:: That will not forward special methods like __str__, since the
            interpreter bypasses __getattr__ when looking them up.
        """
        return getattr(self.target, attr)

    @classmethod
    def from_conf(cls, conf:TargetConf, res_dir:ArtifactPath=None, plat_info:PlatformInfo=None) -> 'Target':
        cls.get_logger().info('Target configuration:\n{}'.format(conf))
        kwargs = cls.conf_to_init_kwargs(conf)
        kwargs['res_dir'] = res_dir
        kwargs['plat_info'] = plat_info

        # Create a devlib Platform instance out of the configuration file
        devlib_platform_conf = conf['devlib']['platform']

        devlib_platform_cls = resolve_dotted_name(devlib_platform_conf['class'])
        devlib_platform_kwargs = copy.copy(devlib_platform_conf.get('args', {}))

        # Hack for Gem5 devlib Platform, that requires a "host_output_dir"
        # argument computed at runtime.
        # Note: lisa.target.Gem5SimulationPlatformWrapper should be used instead
        # of the original one to benefit from mapping configuration
        if issubclass(devlib_platform_cls, Gem5SimulationPlatform):
            devlib_platform_kwargs.setdefault('host_output_dir', res_dir)

        # Actually build the devlib Platform object
        devlib_platform = devlib_platform_cls(**devlib_platform_kwargs)
        kwargs['devlib_platform'] = devlib_platform

        return cls(**kwargs)

    @classmethod
    def from_default_conf(cls):
        """
        Create a :class:`Target` from the YAML configuration file pointed by
        ``LISA_CONF`` environment variable.
        """
        path = os.environ['LISA_CONF']
        return cls.from_one_conf(path)

    @classmethod
    def from_one_conf(cls, path):
        """
        Create a :class:`Target` from a single YAML configuration file.

        This file will be used to provide a :class:`TargetConf` and
        :class:`lisa.platforms.platinfo.PlatformInfo` instances.
        """
        conf = TargetConf.from_yaml_map(path)
        try:
            plat_info = PlatformInfo.from_yaml_map(path)
        except Exception as e:
            cls.get_logger().warn('No platform information could be found: {}'.format(e))
            plat_info = None
        return cls.from_conf(conf=conf, plat_info=plat_info)

    @classmethod
    def from_cli(cls, argv=None) -> 'Target':
        """
        Create a Target from command line arguments.

        :param argv: The list of arguments. ``sys.argv[1:]`` will be used if
          this is ``None``.
        :type argv: list(str)

        Trying to use this in a script that expects extra arguments is bound
        to be confusing (help message woes, argument clashes...), so for now
        this should only be used in scripts that only expect Target args.
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(
                """
                Connect to a target using the provided configuration in order
                to run a test.

                EXAMPLES

                --target-conf can point to a YAML target configuration file
                with all the necessary connection information:
                $ {script} --target-conf my_target.yml

                Alternatively, --kind must be set along the relevant credentials:
                $ {script} --kind linux --host 192.0.2.1 --username root --password root

                In both cases, --platform-info can point to a PlatformInfo YAML
                file.

                """.format(
                    script=os.path.basename(sys.argv[0])
                )))


        kind_group = parser.add_mutually_exclusive_group(required=True)
        kind_group.add_argument("--kind", "-k",
            choices=["android", "linux", "host"],
            help="The kind of target to connect to.")

        kind_group.add_argument("--target-conf", "-t",
                            help="Path to a TargetConf yaml file. Superseeds other target connection related options.")

        device_group = parser.add_mutually_exclusive_group()
        device_group.add_argument("--device", "-d",
                            help="The ADB ID of the target. Superseeds --host. Only applies to Android kind.")
        device_group.add_argument("--host", "-n",
                            help="The hostname/IP of the target.")

        parser.add_argument("--username", "-u",
                            help="Login username. Only applies to Linux kind.")
        parser.add_argument("--password", "-p",
                            help="Login password. Only applies to Linux kind.")

        parser.add_argument("--platform-info", "-pi",
                            help="Path to a PlatformInfo yaml file.")

        parser.add_argument("--log-level",
                            choices=('warning', 'info', 'debug'),
                            help="Verbosity level of the logs.")

        parser.add_argument("--res-dir", "-o",
                            help="Result directory of the created Target. If no directory is specified, a default location under $LISA_HOME will be used.")

        # Options that are not a key in TargetConf must be listed here
        not_target_conf_opt = (
            'platform_info', 'log_level', 'res_dir', 'target_conf',
        )

        args = parser.parse_args(argv)
        if args.log_level:
            setup_logging(level=args.log_level.upper())

        if args.kind == 'android':
            if not (args.host or args.device):
                parser.error('--host or --device must be specified')
        if args.kind == 'linux':
            for required in ['host', 'username', 'password']:
                if getattr(args, required) is None:
                    parser.error('--{} must be specified'.format(required))

        platform_info = PlatformInfo.from_yaml_map(args.platform_info) if args.platform_info else None
        if args.target_conf:
            target_conf = TargetConf.from_yaml_map(args.target_conf)
        else:
            target_conf = TargetConf(
                {k : v for k, v in vars(args).items()
                 if v is not None and k not in not_target_conf_opt})

        return cls.from_conf(conf=target_conf, plat_info=platform_info, res_dir=args.res_dir)

    def _init_target(self, kind, name, workdir, device, host,
            port, username, password, keyfile,
            devlib_platform, devlib_excluded_modules):
        """
        Initialize the Target
        """
        logger = self.get_logger()
        conn_settings = {}

        # If the target is Android, we need just (eventually) the device
        if kind == 'android':
            logger.debug('Setting up Android target...')
            devlib_target_cls = devlib.AndroidTarget

            # Workaround for ARM-software/devlib#225
            workdir = workdir or '/data/local/tmp/devlib-target'

            if device:
                pass
            elif host:
                port = port or self.ADB_PORT_DEFAULT
                device = '{}:{}'.format(host, port)
            else:
                device = 'DEFAULT'

            conn_settings['device'] = device

        elif kind == 'linux':
            logger.debug('Setting up Linux target...')
            devlib_target_cls = devlib.LinuxTarget

            conn_settings['username'] = username
            conn_settings['port'] = port or self.SSH_PORT_DEFAULT
            conn_settings['host'] = host


            # Configure password or SSH keyfile
            if keyfile:
                conn_settings['keyfile'] = keyfile
            else:
                conn_settings['password'] = password
        elif kind == 'host':
            logger.debug('Setting up localhost Linux target...')
            devlib_target_cls = devlib.LocalLinuxTarget
            conn_settings['unrooted'] = True
        else:
            raise ValueError('Unsupported platform type {}'.format(kind))

        logger.info('%s %s target connection settings:', kind, name)
        for key, val in conn_settings.items():
            logger.info('%10s : %s', key, val)

        ########################################################################
        # Devlib Platform configuration
        ########################################################################

        if not devlib_platform:
            devlib_platform = devlib.platform.Platform()

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
        devlib_module_set.difference_update(devlib_excluded_modules)

        devlib_module_list = sorted(devlib_module_set)
        logger.info('Devlib modules to load: %s', ', '.join(devlib_module_list))

        ########################################################################
        # Create devlib Target object
        ########################################################################

        target = devlib_target_cls(
            platform = devlib_platform,
            modules = devlib_module_list,
            load_default_modules = False,
            connection_settings = conn_settings,
            working_directory = workdir,
        )

        logger.debug('Checking target connection...')
        logger.debug('Target info:')
        logger.debug('      ABI: %s', target.abi)
        logger.debug('     CPUs: %s', target.cpuinfo)
        logger.debug(' clusters: %s', target.core_clusters)
        logger.debug('  workdir: %s', target.working_directory)

        target.setup()

        if isinstance(target, devlib.AndroidTarget):
            target.adb_root(force=True)

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

        :param append_time: If True, the current datetime will be appended to
          the given ``name``. If ``name`` is None, the directory name will be
          the current datetime.
        :type append_time: bool

        :param symlink: Create a symlink named ``results_latest`` to the newly
          created results directory
        :type symlink: bool
        """

        if isinstance(self._res_dir, ArtifactPath):
            root = self._res_dir.root
            relative = self._res_dir.relative
        else:
            root = self._res_dir
            relative = ''

        return self._get_res_dir(
            root=root,
            relative=relative,
            name=name,
            append_time=append_time,
            symlink=symlink,
        )

    def _get_res_dir(self, root, relative, name, append_time, symlink):
        logger = self.get_logger()
        while True:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S.%f')
            if not name:
                name = time_str
            elif append_time:
                name = "{}-{}".format(name, time_str)

            # If we were given an ArtifactPath with an existing root, we
            # preserve that root so it can be relocated as the caller wants it
            res_dir = ArtifactPath(root, os.path.join(relative,name))

            # Compute base installation path
            logger.info('Creating result directory: %s', str(res_dir))

            # It will fail if the folder already exists. In that case,
            # append_time should be used to ensure we get a unique name.
            try:
                os.makedirs(res_dir)
                break
            except FileExistsError:
                # If the time is used in the name, there is some hope that the
                # next time it will succeed
                if append_time:
                    logger.info('Directory already exists, retrying ...')
                    continue
                else:
                    raise

        if symlink:
            res_lnk = os.path.join(LISA_HOME, LATEST_LINK)
            with contextlib.suppress(FileNotFoundError):
                os.remove(res_lnk)

            # There may be a race condition with another tool trying to create
            # the link
            with contextlib.suppress(FileExistsError):
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
        tools.difference_update(self._installed_tools)

        tools_to_install = set()
        for tool in tools:
            binary = '{}/tools/scripts/{}'.format(LISA_HOME, tool)
            if not os.path.isfile(binary):
                binary = '{}/tools/{}/{}'\
                         .format(LISA_HOME, self.target.abi, tool)
            tools_to_install.add(binary)

        # TODO: compute the checksum of the tool + install location and keep
        # that in _installed_tools, so we are sure to be correct
        for tool in tools_to_install - self._installed_tools:
            self.target.install(tool)
            self._installed_tools.add(tool)


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
