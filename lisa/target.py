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
"""
Target manipulation helpers.

A target is a device such as an android phone that can be manipulated
programmatically using :class:`lisa.target.Target`.
"""

from datetime import datetime
import os
import os.path
from pathlib import Path
import contextlib
import shlex
from collections.abc import Mapping
import copy
import sys
import argparse
import textwrap
import functools
import inspect
import pickle
import tempfile
import hashlib
import shutil
from types import ModuleType, FunctionType
from operator import itemgetter
import warnings
import typing
import uuid

import devlib
from devlib.exception import TargetStableError
from devlib.utils.misc import which, to_identifier
from devlib.platform.gem5 import Gem5SimulationPlatform

from lisa.utils import Loggable, HideExekallID, resolve_dotted_name, get_subclasses, import_all_submodules, LISA_HOME, RESULT_DIR, LATEST_LINK, setup_logging, ArtifactPath, nullcontext, ExekallTaggable, memoized, destroyablecontextmanager, ContextManagerExit, update_params_from, delegate_getattr, DelegateToAttr, DirCache
from lisa._assets import ASSETS_PATH
from lisa.conf import SimpleMultiSrcConf, KeyDesc, LevelKeyDesc, TopLevelKeyDesc, Configurable, DelegatedLevelKeyDesc, ConfigKeyError
from lisa._kmod import _KernelBuildEnv, DynamicKmod, _KernelBuildEnvConf

from lisa.platforms.platinfo import PlatformInfo


try:
    import devlib.utils.asyn
except ImportError:
    _DEVLIB_HAS_ASYNC = False
else:
    _DEVLIB_HAS_ASYNC = True


class PasswordKeyDesc(KeyDesc):
    """
    :class:`~lisa.conf.KeyDesc` used to describe the target password.
    """

    def pretty_format(self, v):
        """
        Hide the password with a generic placeholder.
        """
        return '<password>'


# Make sure all submodules of devlib.module are imported so the classes
# are all created before we list them
import_all_submodules(devlib.module)
_DEVLIB_AVAILABLE_MODULE_CLASSES = {
    to_identifier(cls.name): cls
    for cls in get_subclasses(devlib.module.Module)
    if (
        getattr(cls, 'name', None)
        # early modules try to connect to UART and do very
        # platform-specific things we are not interested in
        and getattr(cls, 'stage') != 'early'
    )
}


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

    .. literalinclude:: ../../../../target_conf.yml
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

    .. note:: Only load trusted YAML files as it can lead to abritrary code
        execution.

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
        KeyDesc('kind', 'Target kind. Can be "linux" (ssh) or "android" (adb)', [typing.Literal['linux', 'android', 'host']]),

        KeyDesc('host', 'Hostname or IP address of the SSH or ADB server', [str, None]),
        KeyDesc('username', 'SSH username. On ADB connections, "root" username will root adb upon target connection', [str, None]),
        PasswordKeyDesc('password', 'SSH password', [str, None]),
        KeyDesc('port', 'SSH or ADB server port', [int, None]),
        KeyDesc('device', 'ADB device', [str, None]),
        KeyDesc('keyfile', 'SSH private key file', [str, None]),
        KeyDesc('strict-host-check', 'Equivalent to StrictHostKeyChecking option of OpenSSH', [bool, None]),
        KeyDesc('workdir', 'Remote target workdir', [str]),
        KeyDesc('tools', 'List of tools to install on the target', [typing.Sequence[str]]),
        KeyDesc('lazy-platinfo', 'Lazily autodect the platform information to speed up the connection', [bool]),
        LevelKeyDesc('kernel', 'kernel information', (
            KeyDesc('src', 'Path to kernel source tree matching the kernel running on the target used to build modules', [str, None]),
            DelegatedLevelKeyDesc(
                'modules', 'Kernel module build environment',
                _KernelBuildEnvConf,
            ),
        )),
        LevelKeyDesc('wait-boot', 'Wait for the target to finish booting', (
            KeyDesc('enable', 'Enable the boot check', [bool]),
            KeyDesc('timeout', 'Timeout of the boot check', [int]),
        )),
        LevelKeyDesc('devlib', 'devlib configuration', (
            # Using textual name of the Platform allows this YAML configuration
            # to not use any python-specific YAML tags, so TargetConf files can
            # be parsed and produced by any other third-party code
            LevelKeyDesc('platform', 'devlib.platform.Platform subclass specification', (
                KeyDesc('class', 'Name of the class to use', [str]),
                KeyDesc('args', 'Keyword arguments to build the Platform object', [Mapping]),
            )),
            KeyDesc('excluded-modules', 'List of devlib modules to *not* load', [typing.Sequence[str]]),
            KeyDesc('file-xfer', 'File transfer method. Can be "sftp" (default) or "scp". (Only valid for linux targets)', [typing.Sequence[str]]),
            KeyDesc('max-async', 'Maximum number of asynchronous commands in flight at any time', [int, None]),

        ))
    ))

    DEFAULT_SRC = {
        'devlib': {
            'platform': {
                'class': 'devlib.platform.Platform'
            }
        }
    }


class Target(
    DelegateToAttr('target', [devlib.Target]),
    Loggable,
    HideExekallID,
    ExekallTaggable,
    Configurable,
):
    """
    Wrap :class:`devlib.target.Target` to provide additional features on top of
    it.

    {configurable_params}

    :param devlib_platform: Instance of :class:`devlib.platform.Platform` to
        use to build the :class:`devlib.target.Target`
    :type devlib_platform: devlib.platform.Platform

    :param plat_info: Platform information attached to this target, for the
        benefits of user code.
    :type plat_info: lisa.platforms.platinfo.PlatformInfo

    You need to provide the information needed to connect to the
    target. For SSH targets that means "host", "username" and
    either "password" or "keyfile". All other fields are optional if
    the relevant features aren't needed.

    .. note:: The wrapping of :class:`devlib.target.Target` is done using
        composition, as opposed to inheritance. This allows swapping the exact
        class used under the hood, and avoids messing up with ``devlib``
        internal members.
    """

    ADB_PORT_DEFAULT = 5037
    SSH_PORT_DEFAULT = 22

    CRITICAL_TASKS = {
        'linux': [
            'init',
            # We want to freeze everything except PID 1, we don't want to let
            # sysmted-journald or systemd-timesyncd running.
            'systemd[^-]',
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
            # AOSP 'mount' command needs to communicate with 'emulated'
            # threads, the threads are spawn by the process 'rs.media.module',
            # so need to avoid freezing it to avoid stuck with execute 'mount'
            # command.
            'rs.media.module',
        ]
    }
    """
    Dictionary mapping OS name to list of task names that we can't afford to
    freeze when using :meth:`freeze_userspace`.
    """

    CONF_CLASS = TargetConf
    INIT_KWARGS_KEY_MAP = {
        'devlib_excluded_modules': ['devlib', 'excluded-modules'],
        'devlib_file_xfer': ['devlib', 'file-xfer'],
        'devlib_max_async': ['devlib', 'max-async'],
        'wait_boot': ['wait-boot', 'enable'],
        'wait_boot_timeout': ['wait-boot', 'timeout'],

        'kernel_src': ['kernel', 'src'],
        'kmod_build_env': ['kernel', 'modules'],
    }

    def __init__(self, kind, name='<noname>', tools=[], res_dir=None,
        plat_info=None, lazy_platinfo=True, workdir=None, device=None, host=None, port=None,
        username=None, password=None, keyfile=None, strict_host_check=None,
        devlib_platform=None, devlib_excluded_modules=[], devlib_file_xfer=None,
        wait_boot=True, wait_boot_timeout=10, kernel_src=None, kmod_build_env=None,
        kmod_make_vars=None, kmod_overlay_backend=None, devlib_max_async=None,
    ):
        # Determine file transfer method. Currently avaliable options
        # are 'sftp' and 'scp', defaults to sftp.
        if devlib_file_xfer and devlib_file_xfer not in ('scp', 'sftp'):
            raise ValueError(f'Invalid file transfer method: {devlib_file_xfer}')
        use_scp = devlib_file_xfer == 'scp'

        target = self._init_target(
            kind=kind,
            name=name,
            workdir=workdir,
            device=device,
            host=host,
            port=port,
            username=username,
            password=password,
            keyfile=keyfile,
            strict_host_check=strict_host_check,
            use_scp=use_scp,
            devlib_platform=devlib_platform,
            wait_boot=wait_boot,
            wait_boot_timeout=wait_boot_timeout,
            max_async=devlib_max_async,
        )
        self._init_post_devlib(
            name=name, res_dir=res_dir,
            target=target, tools=tools, plat_info=plat_info,
            lazy_platinfo=lazy_platinfo,
            devlib_excluded_modules=devlib_excluded_modules,
            kernel_src=kernel_src, kmod_build_env=kmod_build_env,
            kmod_make_vars=kmod_make_vars,
            kmod_overlay_backend=kmod_overlay_backend,
        )

    @classmethod
    def _from_devlib_target(cls, target, **kwargs):
        self = cls.__new__(cls)
        self._init_post_devlib(target=target, **kwargs)
        return self

    @update_params_from(__init__)
    def _init_post_devlib(self, *, name, res_dir, target,
        tools, plat_info, lazy_platinfo, devlib_excluded_modules, kernel_src,
        kmod_build_env, kmod_make_vars, kmod_overlay_backend,
    ):
        self._uuid = uuid.uuid4().hex
        self._user_res_dir = res_dir

        # Set it temporarily to avoid breaking __getattr__
        self._devlib_loadable_modules = set()

        # pylint: disable=dangerous-default-value
        super().__init__()
        logger = self.logger
        self.name = name

        self._installed_tools = set()
        self.target = target

        devlib_excluded_modules = set(devlib_excluded_modules)
        # Sorry, can't let you do that. Messing with cgroups in a systemd
        # system is pretty bad idea.
        if self._uses_systemd:
            logger.warning('Will not load cgroups devlib module: target is using systemd, which already uses cgroups')
            devlib_excluded_modules.add('cgroups')

        self._devlib_loadable_modules = set(_DEVLIB_AVAILABLE_MODULE_CLASSES.keys()) - devlib_excluded_modules

        # Initialize binary tools to deploy
        if tools:
            logger.info(f'Tools to install: {tools}')
            self.install_tools(tools)

        # Autodetect information from the target, after the Target is
        # initialized. Expensive computations are deferred so they will only be
        # computed when actually needed.
        self._init_plat_info(plat_info, name, deferred=lazy_platinfo, fallback=True)

        logger.info(f'Effective platform information:\n{self.plat_info}')
        self._kmod_build_env = None

        def _make_kernel_build_env_spec(*args, **kwargs):
            kmod_build_env, *_ = _KernelBuildEnv._resolve_conf(*args, **kwargs)
            deprecated = {
                'overlay-backend': kmod_overlay_backend,
                'make-variables': kmod_make_vars,
            }

            cls_name = self.__class__.__qualname__
            if any(v is not None for v in deprecated.values()):
                warnings.warn(f'{cls_name} kmod_overlay_backend and kmod_make_vars parameters are deprecated, please pass the information inside build_env instead using keys: {", ".join(deprecated.keys())}', DeprecationWarning)

            kmod_build_env.add_src(
                src=f'deprecated-{cls_name}-params',
                conf=deprecated,
                filter_none=True
            )

            return dict(
                tree_path=kernel_src,
                build_conf=kmod_build_env,
            )

        self._kmod_build_env_spec = _make_kernel_build_env_spec(
            conf=kmod_build_env,
            abi=self.plat_info['abi'],
            target=self,
        )

    def _init_plat_info(self, plat_info=None, name=None, **kwargs):

        if plat_info is None:
            plat_info = PlatformInfo()
        else:
            # Make a copy of the PlatformInfo so we don't modify the original
            # one we were passed when adding the target source to it
            plat_info = copy.copy(plat_info)
            self.logger.info(f'User-defined platform information:\n{plat_info}')

        # Take the board name from the target configuration so it becomes
        # available for later inspection. That board name is mostly free form
        # and no specific value should be expected for a given kind of board
        # (i.e. a Juno board might be named "foo-bar-juno-on-my-desk")
        if name:
            plat_info.add_src('target-conf', dict(name=name))

        plat_info.add_target_src(self, **kwargs)
        self.plat_info = plat_info

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in {
                # this _KernelBuildEnv contains a reference to ourselves, and will
                # fail to unpickle because of the circular dependency and the
                # fact that it will call its constructor again, trying to make
                # use of the Target before it is ready.
                '_kmod_build_env',
            }
        }

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._init_plat_info(deferred=True)
        self._kmod_build_env = None

    @property
    def _res_dir(self):
        res_dir = self._user_res_dir if self._user_res_dir else self._get_res_dir(
            root=os.path.join(LISA_HOME, RESULT_DIR),
            relative='',
            name=f'{self.__class__.__qualname__}-{self.name}',
            append_time=True,
            symlink=True
        )

        os.makedirs(res_dir, exist_ok=True)
        if os.listdir(res_dir):
            raise ValueError(f'res_dir must be empty: {self._res_dir}')

        return res_dir

    def get_kmod(self, mod_cls=DynamicKmod, **kwargs):
        """
        Build a :class:`lisa._kmod.DynamicKmod` instance of the given
        subclass.

        :param mod_cls: Subclass of :class:`lisa._kmod.DynamicKmod` used to
            build the module.
        :type mod_cls: type

        :Variable keyword arguments: Forwarded to the ``from_target`` class
            method of ``mod_cls``.
        """
        try:
            build_env = kwargs['kernel_build_env']
        except KeyError:
            memoize_build_env = True
            if self._kmod_build_env:
                build_env = self._kmod_build_env
            else:
                build_env = _KernelBuildEnv.from_target(
                    target=self,
                    **self._kmod_build_env_spec
                )
        else:
            memoize_build_env = False

        kwargs['kernel_build_env'] = build_env

        mod = mod_cls.from_target(
            target=self,
            **kwargs,
        )
        if memoize_build_env:
            # Memoize the KernelTree, as it is a reusable object. Memoizing
            # allows remembering its checksum across calls, which will allow
            # hitting the .ko cache without having to setup a kernel tree.
            self._kmod_build_env = mod.kernel_build_env
        return mod


    def cached_pull(self, src, dst, key=None, **kwargs):
        """
        Same as ``lisa.target.Target.pull`` but will cache the file in the
        ``target.res_dir`` folder, based on the source path.

        :param key: Cache key to check against when deciding to pull the file :
            again. If ``None``, the file will be pulled once in the lifetime of the
            :class:`Target` instance it is called on.
        :type key: object or None

        :Variable keyword arguments: Forwarded to ``Target.pull``.
        """

        def populate(key, path):
            src, kwargs, meta = key
            self.pull(src, path / 'data', **kwargs)

        dir_cache = DirCache(
            category='target_pull',
            populate=populate,
            fmt_version='1',
        )

        src = Path(src)
        dst = Path(dst)

        key = self._uuid if key is None else key
        key = (src, kwargs, str(key))
        cached_path = dir_cache.get_entry(key) / 'data'

        if cached_path.is_dir():
            # We would need to mirror devlib.target.Target.pull() behavior
            # (when the destination exists as a folder, as a file, or does not
            # exist yet) for folders, which is more complex and we don't really
            # need it for now.
            raise NotImplementedError('Folders are not supported in cached_pull() for now')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_dir():
                dst = dst / src.name

            shutil.copy2(cached_path, dst)

    @property
    @memoized
    def _uses_systemd(self):
        try:
            # Check if systemd is being used, according to:
            # https://www.freedesktop.org/software/systemd/man/sd_booted.html
            self.execute('test -d /run/systemd/system/', check_exit_code=True)
        except TargetStableError:
            return False
        else:
            return True

    def is_module_available(self, module):
        """
        Check if the given devlib module is available.

        :returns: ``True`` if module is available, ``False`` otherwise.

        :param module: Devlib module to check.
        :type module: str

        .. note:: This will attempt to load the module if it's not loaded
            already, and bail out if it fails to load.
        """
        if module not in _DEVLIB_AVAILABLE_MODULE_CLASSES.keys():
            raise ValueError(f'"{module}" is not a devlib module')

        try:
            getattr(self, module)
        except Exception: # pylint: disable=broad-except
            return False
        else:
            return True

    def __getattr__(self, attr):
        """
        Forward all non-overriden attributes/method accesses to the underlying
        :class:`devlib.target.Target`.

        .. note:: That will not forward special methods like __str__, since the
            interpreter bypasses __getattr__ when looking them up.

        .. note:: Devlib modules are loaded on demand when accessed.
        """
        if attr in ('_devlib_loadable_modules', 'target'):
            raise AttributeError(attr)

        # If it was not in the loadable list, it
        # has been excluded explicitly
        if attr in (set(_DEVLIB_AVAILABLE_MODULE_CLASSES.keys()) - self._devlib_loadable_modules):
            # pylint: disable=raise-missing-from
            raise AttributeError(f'Devlib target module {attr} was explicitly excluded, not loading it')

        get = super().__getattr__

        try:
            return get(attr)
        except AttributeError:
            # Load the module on demand
            if attr in self._devlib_loadable_modules:
                self.logger.info(f'Loading target devlib module {attr}')
                self.target.install_module(attr)
                return get(attr)
            # Something else that does not exist ...
            else:
                raise

    @classmethod
    def __instance_dir__(cls):
        return {
            **super().__instance_dir__(),
            **_DEVLIB_AVAILABLE_MODULE_CLASSES
        }

    def __dir__(self):
        """
        List our attributes plus the ones from the underlying target, and the
        devlib modules that could be loaded on-demand.
        """
        attrs = set(super().__dir__()) | set(self._devlib_loadable_modules)
        return sorted(attrs)

    @classmethod
    def from_conf(cls, conf: TargetConf, res_dir: ArtifactPath = None, plat_info: PlatformInfo = None, **kwargs) -> 'Target':
        cls.get_logger().info(f'Target configuration:\n{conf}')
        _kwargs = cls.conf_to_init_kwargs(conf)
        _kwargs['res_dir'] = res_dir
        _kwargs['plat_info'] = plat_info

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
        _kwargs['devlib_platform'] = devlib_platform

        _kwargs.update(kwargs)

        cls.check_init_param(**_kwargs)
        return cls(**_kwargs)

    @classmethod
    def from_default_conf(cls):
        """
        Create a :class:`Target` from the YAML configuration file pointed by
        ``LISA_CONF`` environment variable.

        .. note:: Only load trusted YAML files as it can lead to abritrary code
            execution.
        """
        path = os.environ['LISA_CONF']
        return cls.from_one_conf(path)

    @classmethod
    def from_one_conf(cls, path):
        """
        Create a :class:`Target` from a single YAML configuration file.

        This file will be used to provide a :class:`TargetConf` and
        :class:`lisa.platforms.platinfo.PlatformInfo` instances.

        .. note:: Only load trusted YAML files as it can lead to abritrary code
            execution.
        """
        conf = TargetConf.from_yaml_map(path)
        try:
            plat_info = PlatformInfo.from_yaml_map(path)
        except Exception as e: # pylint: disable=broad-except
            cls.get_logger().warning(f'No platform information could be found: {e}')
            plat_info = None
        return cls.from_conf(conf=conf, plat_info=plat_info)

    @classmethod
    # Keep the signature without *args and **kwargs so that it's usable by exekall
    def from_cli(cls, argv=None, params=None) -> 'Target':
        """
        Same as :meth:`from_custom_cli` without the custom parameters
        capabilities.

        :return: A connected :class:`Target`
        """
        _, target = cls.from_custom_cli(argv=argv, params=params)
        return target

    @classmethod
    def from_custom_cli(cls, argv=None, params=None, description=None):
        """
        Create a Target from command line arguments.

        :param argv: The list of arguments. ``sys.argv[1:]`` will be used if
          this is ``None``.
        :type argv: list(str)

        :param params: Dictionary of custom parameters to add to the parser. It
            is in the form of
            ``{param_name: {dict of ArgumentParser.add_argument() options}}``.
        :type params: dict(str, dict)

        :param description: Description passed to the argument parser. If
            ``None``, a default one is provided.
        :type description: str or None

        :return: A tuple ``(args, target)``

        .. note:: This method should not be relied upon to implement long-term
            scripts, it's more designed for quick scripting.
        """

        if description is None:
            description = textwrap.dedent(
                """
                Connect to a target using the provided configuration in order
                to run a test.

                EXAMPLES

                --conf can point to a YAML target configuration file
                with all the necessary connection information:
                $ {script} --conf my_target.yml

                Alternatively, --kind must be set along the relevant credentials:
                $ {script} --kind linux --host 192.0.2.1 --username root --password root

                In both cases, --conf can also contain a PlatformInfo YAML description.

                Note: only load trusted YAML files as it can lead to abritrary
                code execution.
                """.format(
                    script=os.path.basename(sys.argv[0])
                )
            )

        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=description,
        )

        parser.add_argument("--conf", '-c',
            action='append',
            default=[],
            help="Path to a TargetConf and PlatformInfo yaml file. Other options will override what is specified in the file."
        )

        parser.add_argument("--kind", "-k",
            choices=["android", "linux", "host"],
            help="The kind of target to connect to.")

        device_group = parser.add_mutually_exclusive_group()
        device_group.add_argument("--device", "-d",
                            help="The ADB ID of the target. Superseeds --host. Only applies to Android kind.")
        device_group.add_argument("--host", "-n",
                            help="The hostname/IP of the target.")

        parser.add_argument("--username", "-u",
                            help="Login username. Only applies to Linux kind.")
        parser.add_argument("--password", "-p",
                            help="Login password. Only applies to Linux kind.")

        parser.add_argument("--log-level",
                            default='info',
                            choices=('warning', 'info', 'debug'),
                            help="Verbosity level of the logs.")

        parser.add_argument("--res-dir", "-o",
                            help="Result directory of the created Target. If no directory is specified, a default location under $LISA_HOME will be used.")

        params = params or {}
        for param, settings in params.items():
            if settings.get('nargs') == argparse.REMAINDER:
                _param = param
            else:
                _param = f'--{param}'
            parser.add_argument(_param, **settings)

        custom_params = {k.replace('-', '_') for k in params.keys()}

        # Options that are not a key in TargetConf must be listed here
        not_target_conf_opt = {
            'platform_info', 'log_level', 'res_dir', 'conf',
        }
        not_target_conf_opt.update(custom_params)

        args = parser.parse_args(argv)
        setup_logging(level=args.log_level.upper())

        target_conf = TargetConf()
        platform_info = None

        if args.conf:
            try:
                conf_map = SimpleMultiSrcConf.from_yaml_map_list(args.conf)
            except ConfigKeyError as e:
                parser.error(f'Could not load {" or ".join(args.conf)}: {e}')

            platform_info = conf_map.get(PlatformInfo)
            target_conf = conf_map.get(TargetConf, target_conf)

        target_conf.add_src('command-line', {
            k: v for k, v in vars(args).items()
            if v is not None and k not in not_target_conf_opt
        })

        # Some sanity check to get better error messages
        if 'kind' not in target_conf:
            parser.error('--conf with target configuration or any of the connection options is required')

        if args.kind == 'android':
            if ('host' not in target_conf) and ('device' not in target_conf):
                parser.error('--host or --device must be specified')

        if args.kind == 'linux':
            for required in ['host', 'username', 'password']:
                if required not in target_conf:
                    parser.error(f'--{required} must be specified')

        custom_args = {
            param: value
            for param, value in vars(args).items()
            if param in custom_params
        }
        custom_args = argparse.Namespace(**custom_args)

        return custom_args, cls.from_conf(conf=target_conf, plat_info=platform_info, res_dir=args.res_dir)

    @classmethod
    def _init_target(cls, kind, name, workdir, device, host,
            port, username, password, keyfile, strict_host_check, use_scp,
            devlib_platform, wait_boot, wait_boot_timeout, max_async,
    ):
        """
        Initialize the Target
        """
        logger = cls.get_logger()
        conn_settings = {}
        resolved_username = username or 'root'

        logger.debug(f'Setting up {kind} target...')

        # If the target is Android, we need just (eventually) the device
        if kind == 'android':
            devlib_target_cls = devlib.AndroidTarget

            # Workaround for ARM-software/devlib#225
            workdir = workdir or '/data/local/tmp/devlib-target'
            device = device or 'DEFAULT'
            conn_settings.update(
                device=device,
                adb_server=host,
                adb_port=port,
                # If the username was explicitly set to "root", root the target
                # as early as possible
                adb_as_root=(username == 'root'),
            )
        elif kind == 'linux':
            def resolve_strict_host_check(strict_host_check):
                strict_host_check = True if strict_host_check is None else strict_host_check

                # Expand the path to known_hosts so that the devlib objects can
                # be reused unchanged in another user namespace where the home
                # directory would not expand to the expected folder.
                if strict_host_check and isinstance(strict_host_check, bool):
                    strict_host_check = str(Path('~/.ssh/known_hosts').expanduser().resolve())

                return strict_host_check

            devlib_target_cls = devlib.LinuxTarget
            conn_settings.update(
                username=resolved_username,
                port=port or cls.SSH_PORT_DEFAULT,
                host=host,
                strict_host_check=resolve_strict_host_check(strict_host_check),
                use_scp=False if use_scp is None else use_scp,
            )

            # Configure password or SSH keyfile
            if keyfile:
                conn_settings['keyfile'] = keyfile
            else:
                conn_settings['password'] = password
        elif kind == 'host':
            devlib_target_cls = devlib.LocalLinuxTarget
            # If we are given a password, assume we can use it as a sudo
            # password.
            conn_settings.update(
                unrooted=password is None,
                password=password,
            )
        else:
            raise ValueError(f'Unsupported platform type {kind}')

        conn_settings = {
            k: v
            for k, v in conn_settings.items()
            if v is not None
        }

        settings = '\n    '.join(
            f'    {key}: {val}'
            for key, val in conn_settings.items()
            if key != 'password'
        )
        logger.debug(f'{kind} {name} target connection settings:\n    {settings}')

        ########################################################################
        # Devlib Platform configuration
        ########################################################################

        if not devlib_platform:
            devlib_platform = devlib.platform.Platform()


        ########################################################################
        # Create devlib Target object
        ########################################################################

        max_async = 50 if max_async is None else max_async
        devlib_kwargs = dict(
            platform=devlib_platform,
            load_default_modules=False,
            connection_settings=conn_settings,
            working_directory=workdir,
            connect=False,
        )

        connect_kwargs = dict(
            check_boot_completed=wait_boot,
            timeout=wait_boot_timeout,
        )

        if _DEVLIB_HAS_ASYNC:
            devlib_kwargs['max_async'] = max_async
            connect_kwargs['max_async'] = max_async

        devlib_kwargs = {
            k: v
            for k, v in devlib_kwargs.items()
            if v is not None
        }

        target = devlib_target_cls(**devlib_kwargs)

        target.connect(**connect_kwargs)
        logger.debug(f'Target info: {dict(abi=target.abi, cpuinfo=target.cpuinfo, workdir=target.working_directory)}')
        target.setup()
        logger.info(f"Connected to target {(name or '')}")
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
        logger = self.logger
        while True:
            time_str = datetime.now().strftime('%Y%m%d_%H%M%S.%f')
            if not name:
                name = time_str
            elif append_time:
                name = f"{name}-{time_str}"

            # If we were given an ArtifactPath with an existing root, we
            # preserve that root so it can be relocated as the caller wants it
            res_dir = ArtifactPath(root, os.path.join(relative, name))

            # Compute base installation path
            logger.info(f'Creating result directory: {res_dir}')

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

        def bin_path(tool):
            binary = os.path.join(ASSETS_PATH, 'binaries', self.abi, tool)
            if not os.path.isfile(binary):
                binary = os.path.join(ASSETS_PATH, 'binaries', 'scripts', tool)
            return binary

        tools = sorted(set(tools) - self._installed_tools)

        # TODO: compute the checksum of the tool + install location and keep
        # that in _installed_tools, so we are sure to be correct
        for tool in tools:
            self.target.install(bin_path(tool))
            self._installed_tools.add(tool)

    def freeze_userspace(self):
        """
        Context manager that lets you freeze the userspace.

        .. note:: A number of situations prevent from freezing anything. When
            that happens, a warning is logged but no exception is raised, so
            it's a best-effort approach.
        """
        logger = self.logger
        if not self.is_rooted:
            logger.warning('Could not freeze userspace: target is not rooted')
            cm = nullcontext
        elif not self.is_module_available('cgroups'):
            logger.warning('Could not freeze userspace: "cgroups" devlib module is necessary')
            cm = nullcontext
        else:
            controllers = [s.name for s in self.cgroups.list_subsystems()]
            if 'freezer' not in controllers:
                logger.warning('Could not freeze userspace: freezer cgroup controller not available on the target')
                cm = nullcontext

            else:
                exclude = copy.copy(self.CRITICAL_TASKS[self.target.os])

                # Do not freeze the process in charge of de-freezing, otherwise we
                # will freeze to death and a machine hard reboot will be required
                if isinstance(self.target, devlib.LocalLinuxTarget):
                    exclude.append(str(os.getpid()))

                @destroyablecontextmanager
                def cm():
                    logger.info(f"Freezing all tasks except: {','.join(exclude)}")
                    try:
                        yield self.cgroups.freeze(exclude)
                    except ContextManagerExit:
                        logger.info('Un-freezing userspace tasks')
                        self.cgroups.freeze(thaw=True)

        return cm()

    def disable_idle_states(self):
        """
        Context manager that lets you disable all idle states
        """
        logger = self.logger
        logger.info('Disabling idle states for all domains')

        try:
            cpuidle = self.cpuidle
        except AttributeError:
            logger.warning('Could not disable idle states, cpuidle devlib module is not loaded')
            cm = nullcontext
        else:
            @destroyablecontextmanager
            def cm():
                try:
                    for cpu in range(self.plat_info['cpus-count']):
                        cpuidle.disable_all(cpu)
                    yield
                except ContextManagerExit:
                    logger.info('Re-enabling idle states for all domains')
                    for cpu in range(self.plat_info['cpus-count']):
                        cpuidle.enable_all(cpu)

        return cm()

    def get_tags(self):
        return {'board': self.name}

    @classmethod
    def _make_remote_snippet(cls, name, code_str, module, kwargs, global_vars, out_tempfiles):
        # Inject the parameters inside the wrapper's globals so that it can
        # access them. It's harmless as they would shadow any global name
        # anyway, and it's restricted to the wrapper using eval()
        global_vars = {
            **global_vars,
            **kwargs,
        }

        # Treat the modules separately as they cannot be pickled
        modules = {
            name: mod
            for name, mod in global_vars.items()
            if isinstance(mod, ModuleType)
        }

        def can_include(f):
            return (
                isinstance(f, FunctionType) and
                # Only allow inlining of functions defined in the same module so that:
                # 1. there is no name clash risk
                # 2. we don't inline the whole world, which could lead to a
                #    number of problems that could appear after another module
                #    is updated or so. We only inline local things that are in
                #    direct control
                f.__module__ == module
            )

        def add_func(f, name):
            # Disallow decorated functions since their definition depends on
            # external callable we cannot control
            if hasattr(f, '__wrapped__'):
                raise TypeError('Decorated functions cannot be called from remote functions')

            closure_vars = {
                name: val
                for var_dct in inspect.getclosurevars(f)
                if isinstance(var_dct, Mapping)
                for name, val in var_dct.items()
            }

            funcs[name] = (f, cls._get_code(f)[1])

            for _name, _f in closure_vars.items():
                if _f is not f and can_include(_f):
                    add_func(_f, _name)

            modules.update(
                (name, mod)
                for name, mod in closure_vars.items()
                if isinstance(mod, ModuleType)
            )

        funcs = {}
        for f_name, f in global_vars.items():
            if can_include(f):
                add_func(f, f_name)

        code_str += '\n' + '\n'.join(map(itemgetter(1), funcs.values()))

        non_pickled = set(modules.keys()) | set(funcs.keys())
        global_vars = {
            name: val
            for name, val in global_vars.items()
            if name not in non_pickled
        }

        if modules:
            modules = f"import {', '.join(sorted(modules))}"
        else:
            modules = ''

        script = textwrap.dedent('''
            import pickle
            import sys

            def wrapper():
                {modules}

                {code}
                return {f}({kwargs})

            try:
                out = eval(wrapper.__code__, pickle.loads({globals}))
            except BaseException as e:
                out = e
                out_is_excep = True
            else:
                out_is_excep = False

            out = pickle.dumps(out)
            out_tempfile = {out_tempfiles}[1] if out_is_excep else {out_tempfiles}[0]

            with open(out_tempfile, 'wb') as f:
                f.write(out)
        ''').format(
            f=name,
            code=textwrap.dedent(code_str).replace('\n', '\n' + ' ' * 4),
            modules=modules,
            out_tempfiles=repr(out_tempfiles),
            globals=repr(pickle.dumps(global_vars)),
            kwargs=', '.join(
                f'{name}={name}'
                for name in kwargs.keys()
            )
        )
        return script

    @staticmethod
    def _get_code(f):
        lines, _ = inspect.getsourcelines(f)
        # Remove decorators, as they are either undefined or just were used to
        # feed the function to us
        lines = [
            line
            for line in lines
            if not line.strip().startswith('@')
        ]
        code_str = textwrap.dedent(''.join(lines))
        name = f.__name__
        return (name, code_str)

    def execute_python(self, f, args, kwargs, **execute_kwargs):
        """
        Executes the given Python function ``f`` with the provided positional
        and keyword arguments.

        The return value or any exception is pickled back and is
        returned/raised in the host caller.

        :Variable keyword arguments: Forwarded to :meth:`execute` that
            will spawn the Python interpreter on the target

        .. note:: Closure variables are supported, but mutating them will not
            be reflected in the caller's context. Also, functions that are
            referred to will be:

                * bundled in the script if it is defined in the same module
                * referred to by name, assuming it comes from a module that is
                  installed on the target and that this module is in scope. If
                  that is not the case, a :exc:`NameError` will be raised.

        .. attention:: Decorators are ignored and not applied.
        """
        sig = inspect.signature(f)
        kwargs = sig.bind(*args, **kwargs).arguments
        closure_vars = inspect.getclosurevars(f)

        name, code_str = self._get_code(f)

        def mktemp():
            return self.execute(
                f'mktemp -p {shlex.quote(self.working_directory)}'
            ).strip()

        def read_output(path):
            with tempfile.TemporaryDirectory() as d:
                name = os.path.join(d, 'out')
                self.pull(path, name)
                with open(name, 'rb') as f:
                    return pickle.loads(f.read())

        def parse_output(paths, err):
            val, excep = paths
            try:
                return read_output(val)
            # If the file is empty, we probably got an exception
            except EOFError:
                # pylint: disable=raise-missing-from
                try:
                    excep = read_output(excep)
                # If we can't even read the exception, raise the initial one
                # from devlib
                except EOFError:
                    raise err if err is not None else ValueError('No exception was raised or value returned by the function')
                else:
                    raise excep


        out_tempfiles = tuple()
        try:
            out_tempfiles = (mktemp(), mktemp())
            snippet = self._make_remote_snippet(
                name=name,
                code_str=code_str,
                module=f.__module__,
                kwargs=kwargs,
                global_vars={
                    **closure_vars.globals,
                    **closure_vars.nonlocals,
                },
                out_tempfiles=out_tempfiles
            )
            cmd = ['python3', '-c', snippet]
            cmd = ' '.join(map(shlex.quote, cmd))
            try:
                self.execute(cmd, **execute_kwargs)
            except Exception as e: # pylint: disable=broad-except
                err = e
            else:
                err = None
            return parse_output(out_tempfiles, err)
        finally:
            for path in out_tempfiles:
                self.remove(path)


    def remote_func(self, **kwargs):
        """
        Decorates a given function to execute remotely using
        :meth:`execute_python`.

        ::

            target = Target(...)

            @target.remote_func(timeout=42)
            def foo(x, y):
                return x + y

            # Execute the function on the target transparently
            val = foo(1, y=2)

        :Variable keyword arguments: Forwarded to :meth:`execute` that
            will spawn the Python interpreter on the target
        """
        def wrapper_param(f):
            @functools.wraps(f)
            def wrapper(*f_args, **f_kwargs):
                return self.execute_python(f, f_args, f_kwargs, **kwargs)
            return wrapper
        return wrapper_param


    def closing(self):
        """
        Returns a context manager that will disconnect the target automatically.
        """
        # Do not use contextlib.contextmanager() as it would force the __exit__
        # path to run, since destroying a suspended generator raises
        # GeneratorExit at the yield point.
        class _ClosingCM:
            def __enter__(_self):
                return self

            def __exit__(_self, *args, **kwargs):
                self.disconnect()

        return _ClosingCM()


class Gem5SimulationPlatformWrapper(Gem5SimulationPlatform):
    def __init__(self, system, simulator, **kwargs):
        simulator_args = copy.copy(simulator.get('args', []))
        system_platform = system['platform']

        # Get gem5 binary arguments
        simulator_args.append('--listener-mode=on')

        simulator_args.append(system_platform['description'])
        simulator_args.extend(system_platform.get('args', []))

        simulator_args.extend((
            f"--kernel {system['kernel']}",
            f"--dtb {system['dtb']}",
            f"--disk-image {system['disk']}"
        ))

        diod_path = which('diod')
        if diod_path is None:
            raise RuntimeError('Failed to find "diod" on your host machine, check your installation or your PATH variable')

        # Setup virtio
        # Brackets are there to let the output dir be created automatically
        virtio_args = [
            f'--which-diod={diod_path}',
            '--workload-automation-vio={}',
        ]
        simulator_args.extend(virtio_args)

        # Quote/escape arguments and build the command line
        gem5_args = ' '.join(shlex.quote(a) for a in simulator_args)

        super().__init__(
            gem5_args=gem5_args,
            gem5_bin=simulator['bin'],
            **kwargs
        )

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
