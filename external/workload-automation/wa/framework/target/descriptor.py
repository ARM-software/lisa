#    Copyright 2018 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect

from devlib import (LinuxTarget, AndroidTarget, LocalLinuxTarget,
                    ChromeOsTarget, Platform, Juno, TC2, Gem5SimulationPlatform,
                    AdbConnection, SshConnection, LocalConnection,
                    TelnetConnection, Gem5Connection)
from devlib.target import DEFAULT_SHELL_PROMPT
from devlib.utils.ssh import DEFAULT_SSH_SUDO_COMMAND

from wa.framework import pluginloader
from wa.framework.configuration.core import get_config_point_map
from wa.framework.exception import PluginLoaderError
from wa.framework.plugin import Plugin, Parameter
from wa.framework.target.assistant import LinuxAssistant, AndroidAssistant, ChromeOsAssistant
from wa.utils.types import list_of_strings, list_of_ints, regex, identifier, caseless_string
from wa.utils.misc import isiterable


def list_target_descriptions(loader=pluginloader):
    targets = {}
    for cls in loader.list_target_descriptors():
        descriptor = cls()
        for desc in descriptor.get_descriptions():
            if desc.name in targets:
                msg = 'Duplicate target "{}" returned by {} and {}'
                prev_dtor = targets[desc.name].source
                raise PluginLoaderError(msg.format(desc.name, prev_dtor.name,
                                                   descriptor.name))
            targets[desc.name] = desc
    return list(targets.values())


def get_target_description(name, loader=pluginloader):
    for tdesc in list_target_descriptions(loader):
        if tdesc.name == name:
            return tdesc
    raise ValueError('Could not find target descriptor "{}"'.format(name))


def instantiate_target(tdesc, params, connect=None, extra_platform_params=None):
    # pylint: disable=too-many-locals,too-many-branches
    target_params = get_config_point_map(tdesc.target_params)
    platform_params = get_config_point_map(tdesc.platform_params)
    conn_params = get_config_point_map(tdesc.conn_params)
    assistant_params = get_config_point_map(tdesc.assistant_params)

    tp, pp, cp = {}, {}, {}

    for supported_params, new_params in (target_params, tp), (platform_params, pp), (conn_params, cp):
        for name, value in supported_params.items():
            if value.default and name == value.name:
                new_params[name] = value.default

    for name, value in params.items():
        if name in target_params:
            if not target_params[name].deprecated:
                tp[name] = value
        elif name in platform_params:
            if not platform_params[name].deprecated:
                pp[name] = value
        elif name in conn_params:
            if not conn_params[name].deprecated:
                cp[name] = value
        elif name in assistant_params:
            pass
        else:
            msg = 'Unexpected parameter for {}: {}'
            raise ValueError(msg.format(tdesc.name, name))

    for pname, pval in (extra_platform_params or {}).items():
        if pname in pp:
            raise RuntimeError('Platform parameter clash: {}'.format(pname))
        pp[pname] = pval

    tp['platform'] = (tdesc.platform or Platform)(**pp)
    if cp:
        tp['connection_settings'] = cp
    if tdesc.connection:
        tp['conn_cls'] = tdesc.connection
    if connect is not None:
        tp['connect'] = connect

    return tdesc.target(**tp)


def instantiate_assistant(tdesc, params, target):
    assistant_params = {}
    for param in tdesc.assistant_params:
        if param.name in params:
            assistant_params[param.name] = params[param.name]
        elif param.default:
            assistant_params[param.name] = param.default
    return tdesc.assistant(target, **assistant_params)


class TargetDescription(object):

    def __init__(self, name, source, description=None, target=None, platform=None,
                 conn=None, assistant=None, target_params=None, platform_params=None,
                 conn_params=None, assistant_params=None):
        self.name = name
        self.source = source
        self.description = description
        self.target = target
        self.platform = platform
        self.connection = conn
        self.assistant = assistant
        self._set('target_params', target_params)
        self._set('platform_params', platform_params)
        self._set('conn_params', conn_params)
        self._set('assistant_params', assistant_params)

    def get_default_config(self):
        param_attrs = ['target_params', 'platform_params',
                       'conn_params', 'assistant_params']
        config = {}
        for pattr in param_attrs:
            for p in getattr(self, pattr):
                if not p.deprecated:
                    config[p.name] = p.default
        return config

    def _set(self, attr, vals):
        if vals is None:
            vals = []
        elif isiterable(vals):
            if hasattr(vals, 'values'):
                vals = list(vals.values())
        else:
            msg = '{} must be iterable; got "{}"'
            raise ValueError(msg.format(attr, vals))
        setattr(self, attr, vals)


class TargetDescriptor(Plugin):

    kind = 'target_descriptor'

    def get_descriptions(self):  # pylint: disable=no-self-use
        return []


COMMON_TARGET_PARAMS = [
    Parameter('working_directory', kind=str,
              description='''
              On-target working directory that will be used by WA. This
              directory must be writable by the user WA logs in as without
              the need for privilege elevation.
              '''),
    Parameter('executables_directory', kind=str,
              description='''
              On-target directory where WA will install its executable
              binaries.  This location must allow execution. This location does
              *not* need to be writable by unprivileged users or rooted devices
              (WA will install with elevated privileges as necessary).
              '''),
    Parameter('modules', kind=list,
              description='''
              A list of additional modules to be installed for the target.

              ``devlib`` implements functionality for particular subsystems as
              modules.  A number of "default" modules (e.g. for cpufreq
              subsystem) are loaded automatically, unless explicitly disabled.
              If additional modules need to be loaded, they may be specified
              using this parameter.

              Please see ``devlib`` documentation for information on the available
              modules.
              '''),
    Parameter('load_default_modules', kind=bool, default=True,
              description='''
              A number of modules (e.g. for working with the cpufreq subsystem) are
              loaded by default when a Target is instantiated. Setting this to
              ``True`` would suppress that, ensuring that only the base Target
              interface is initialized.

              You may want to set this to ``False`` if there is a problem with one
              or more default modules on your platform (e.g. your device is
              unrooted and cpufreq is not accessible to unprivileged users), or
              if ``Target`` initialization is taking too long for your platform.
              '''),
    Parameter('shell_prompt', kind=regex, default=DEFAULT_SHELL_PROMPT,
              description='''
              A regex that matches the shell prompt on the target.
              '''),

    Parameter('max_async', kind=int, default=50,
        description='''
            The maximum number of concurent asynchronous connections to the
            target maintained at any time.
            '''),
]

COMMON_PLATFORM_PARAMS = [
    Parameter('core_names', kind=list_of_strings,
              description='''
              List of names of CPU cores in the order that they appear to the
              kernel. If not specified, it will be inferred from the platform.
              '''),
    Parameter('core_clusters', kind=list_of_ints,
              description='''
              Cluster mapping corresponding to the cores in ``core_names``.
              Cluster indexing starts at ``0``.  If not specified, this will be
              inferred from ``core_names`` -- consecutive cores with the same
              name will be assumed to share a cluster.
              '''),
    Parameter('big_core', kind=str,
              description='''
              The name of the big cores in a big.LITTLE system. If not
              specified, this will be inferred, either from the name (if one of
              the names in ``core_names`` matches known big cores), or by
              assuming that the last cluster is big.
              '''),
    Parameter('model', kind=str,
              description='''
              Hardware model of the platform. If not specified, an attempt will
              be made to read it from target.
              '''),
    Parameter('modules', kind=list,
              description='''
              An additional list of modules to be loaded into the target.
              '''),
]

VEXPRESS_PLATFORM_PARAMS = [
    Parameter('serial_port', kind=str,
              description='''
              The serial device/port on the host for the initial connection to
              the target (used for early boot, flashing, etc).
              '''),
    Parameter('baudrate', kind=int,
              description='''
              Baud rate for the serial connection.
              '''),
    Parameter('vemsd_mount', kind=str,
              description='''
              VExpress MicroSD card mount location. This is a MicroSD card in
              the VExpress device that is mounted on the host via USB. The card
              contains configuration files for the platform and firmware and
              kernel images to be flashed.
              '''),
    Parameter('bootloader', kind=str,
              allowed_values=['uefi', 'uefi-shell', 'u-boot', 'bootmon'],
              description='''
              Selects the bootloader mechanism used by the board. Depending on
              firmware version, a number of possible boot mechanisms may be use.

              Please see ``devlib`` documentation for descriptions.
              '''),
    Parameter('hard_reset_method', kind=str,
              allowed_values=['dtr', 'reboottxt'],
              description='''
              There are a couple of ways to reset VersatileExpress board if the
              software running on the board becomes unresponsive. Both require
              configuration to be enabled (please see ``devlib`` documentation).

              ``dtr``: toggle the DTR line on the serial connection
              ``reboottxt``: create ``reboot.txt`` in the root of the VEMSD mount.
              '''),
]

GEM5_PLATFORM_PARAMS = [
    Parameter('gem5_bin', kind=str, mandatory=True,
              description='''
              Path to the gem5 binary
              '''),
    Parameter('gem5_args', kind=str, mandatory=True,
              description='''
              Arguments to be passed to the gem5 binary
              '''),
    Parameter('gem5_virtio', kind=str, mandatory=True,
              description='''
              VirtIO device setup arguments to be passed to gem5. VirtIO is used
              to transfer files between the simulation and the host.
              '''),
    Parameter('name', kind=str, default='gem5',
              description='''
              The name for the gem5 "device".
              '''),
]


CONNECTION_PARAMS = {
    AdbConnection: [
        Parameter(
            'device', kind=str,
            aliases=['adb_name'],
            description="""
            ADB device name
            """),
        Parameter(
            'adb_server', kind=str,
            description="""
            ADB server to connect to.
            """),
        Parameter(
            'adb_port', kind=int,
            description="""
            ADB port to connect to.
            """),
        Parameter(
            'poll_transfers', kind=bool,
            default=True,
            description="""
            File transfers will be polled for activity. Inactive
            file transfers are cancelled.
            """),
        Parameter(
            'start_transfer_poll_delay', kind=int,
            default=30,
            description="""
            How long to wait (s) for a transfer to complete
            before polling transfer activity. Requires ``poll_transfers``
            to be set.
            """),
        Parameter(
            'total_transfer_timeout', kind=int,
            default=3600,
            description="""
            The total time to elapse before a transfer is cancelled, regardless
            of its activity. Requires ``poll_transfers`` to be set.
            """),
        Parameter(
            'transfer_poll_period', kind=int,
            default=30,
            description="""
            The period at which transfer activity is sampled. Requires
            ``poll_transfers`` to be set. Too small values may cause
            the destination size to appear the same over one or more sample
            periods, causing improper transfer cancellation.
            """),
        Parameter(
            'adb_as_root', kind=bool,
            default=False,
            description="""
            Specify whether the adb server should be started in root mode.
            """)
    ],
    SshConnection: [
        Parameter(
            'host', kind=str, mandatory=True,
            description="""
            Host name or IP address of the target.
            """),
        Parameter(
            'username', kind=str, mandatory=True,
            description="""
            User name to connect with
            """),
        Parameter(
            'password', kind=str,
            description="""
            Password to use.
            (When connecting to a passwordless machine set to an
            empty string to prevent attempting ssh key authentication.)
            """),
        Parameter(
            'keyfile', kind=str,
            description="""
            Key file to use
            """),
        Parameter(
            'port', kind=int,
            default=22,
            description="""
            The port SSH server is listening on on the target.
            """),
        Parameter(
            'strict_host_check', kind=bool, default=False,
            description="""
            Specify whether devices should be connected to if
            their host key does not match the systems known host keys. """),
        Parameter(
            'sudo_cmd', kind=str,
            default=DEFAULT_SSH_SUDO_COMMAND,
            description="""
            Sudo command to use. Must have ``{}`` specified
            somewhere in the string it indicate where the command
            to be run via sudo is to go.
            """),
        Parameter(
            'use_scp', kind=bool,
            default=False,
            description="""
            Allow using SCP as method of file transfer instead
            of the default SFTP.
            """),
        Parameter(
            'poll_transfers', kind=bool,
            default=True,
            description="""
            File transfers will be polled for activity. Inactive
            file transfers are cancelled.
            """),
        Parameter(
            'start_transfer_poll_delay', kind=int,
            default=30,
            description="""
            How long to wait (s) for a transfer to complete
            before polling transfer activity. Requires ``poll_transfers``
            to be set.
            """),
        Parameter(
            'total_transfer_timeout', kind=int,
            default=3600,
            description="""
            The total time to elapse before a transfer is cancelled, regardless
            of its activity. Requires ``poll_transfers`` to be set.
            """),
        Parameter(
            'transfer_poll_period', kind=int,
            default=30,
            description="""
            The period at which transfer activity is sampled. Requires
            ``poll_transfers`` to be set. Too small values may cause
            the destination size to appear the same over one or more sample
            periods, causing improper transfer cancellation.
            """),
        # Deprecated Parameters
        Parameter(
            'telnet', kind=str,
            description="""
            Original shell prompt to expect.
            """,
            deprecated=True),
        Parameter(
            'password_prompt', kind=str,
            description="""
            Password prompt to expect
            """,
            deprecated=True),
        Parameter(
            'original_prompt', kind=str,
            description="""
            Original shell prompt to expect.
            """,
            deprecated=True),
    ],
    TelnetConnection: [
        Parameter(
            'host', kind=str, mandatory=True,
            description="""
            Host name or IP address of the target.
            """),
        Parameter(
            'username', kind=str, mandatory=True,
            description="""
            User name to connect with
            """),
        Parameter(
            'password', kind=str,
            description="""
            Password to use.
            """),
        Parameter(
            'port', kind=int,
            description="""
            The port SSH server is listening on on the target.
            """),
        Parameter(
            'password_prompt', kind=str,
            description="""
            Password prompt to expect
            """),
        Parameter(
            'original_prompt', kind=str,
            description="""
            Original shell prompt to expect.
            """),
        Parameter(
            'sudo_cmd', kind=str,
            default="sudo -- sh -c {}",
            description="""
            Sudo command to use. Must have ``{}`` specified
            somewhere in the string it indicate where the command
            to be run via sudo is to go.
            """),
    ],
    Gem5Connection: [
        Parameter(
            'host', kind=str, mandatory=False,
            description="""
            Host name or IP address of the target.
            """),
        Parameter(
            'username', kind=str, default='root',
            description="""
            User name to connect to gem5 simulation.
            """),
        Parameter(
            'password', kind=str,
            description="""
            Password to use.
            """),
        Parameter(
            'port', kind=int,
            description="""
            The port SSH server is listening on on the target.
            """),
        Parameter(
            'password_prompt', kind=str,
            description="""
            Password prompt to expect
            """),
        Parameter(
            'original_prompt', kind=str,
            description="""
            Original shell prompt to expect.
            """),
    ],
    LocalConnection: [
        Parameter(
            'password', kind=str,
            description="""
            Password to use for sudo. if not specified, the user will
            be prompted during intialization.
            """),
        Parameter(
            'keep_password', kind=bool, default=True,
            description="""
            If ``True`` (the default), the password will be cached in
            memory after it is first obtained from the user, so that the
            user would not be prompted for it again.
            """),
        Parameter(
            'unrooted', kind=bool, default=False,
            description="""
            Indicate that the target should be considered unrooted; do not
            attempt sudo or ask the user for their password.
            """),
    ],
}

CONNECTION_PARAMS['ChromeOsConnection'] = \
    CONNECTION_PARAMS[AdbConnection] + CONNECTION_PARAMS[SshConnection]


# name --> ((target_class, conn_class, unsupported_platforms), params_list, defaults)
TARGETS = {
    'linux': ((LinuxTarget, SshConnection, []), COMMON_TARGET_PARAMS, None),
    'android': ((AndroidTarget, AdbConnection, []), COMMON_TARGET_PARAMS +
               [Parameter('package_data_directory', kind=str, default='/data/data',
                          description='''
                          Directory containing Android data
                          '''),
               ], None),
    'chromeos': ((ChromeOsTarget, 'ChromeOsConnection', []), COMMON_TARGET_PARAMS +
                [Parameter('package_data_directory', kind=str, default='/data/data',
                           description='''
                           Directory containing Android data
                           '''),
                Parameter('android_working_directory', kind=str,
                          description='''
                          On-target working directory that will be used by WA for the
                          android container. This directory must be writable by the user
                          WA logs in as without the need for privilege elevation.
                          '''),
                Parameter('android_executables_directory', kind=str,
                          description='''
                          On-target directory where WA will install its executable
                          binaries for the android container. This location must allow execution.
                          This location does *not* need to be writable by unprivileged users or
                          rooted devices (WA will install with elevated privileges as necessary).
                          directory must be writable by the user WA logs in as without
                          the need for privilege elevation.
                          '''),
                ], None),
    'local': ((LocalLinuxTarget, LocalConnection, [Juno, Gem5SimulationPlatform, TC2]),
              COMMON_TARGET_PARAMS, None),
}

# name --> assistant
ASSISTANTS = {
    'linux': LinuxAssistant,
    'android': AndroidAssistant,
    'local': LinuxAssistant,
    'chromeos': ChromeOsAssistant
}

# Platform specific parameter overrides.
JUNO_PLATFORM_OVERRIDES = [
        Parameter('baudrate', kind=int, default=115200,
                description='''
                Baud rate for the serial connection.
                '''),
        Parameter('vemsd_mount', kind=str, default='/media/JUNO',
                description='''
                VExpress MicroSD card mount location. This is a MicroSD card in
                the VExpress device that is mounted on the host via USB. The card
                contains configuration files for the platform and firmware and
                kernel images to be flashed.
                '''),
        Parameter('bootloader', kind=str, default='u-boot',
                allowed_values=['uefi', 'uefi-shell', 'u-boot', 'bootmon'],
                description='''
                Selects the bootloader mechanism used by the board. Depending on
                firmware version, a number of possible boot mechanisms may be use.

                Please see ``devlib`` documentation for descriptions.
                '''),
        Parameter('hard_reset_method', kind=str, default='dtr',
                allowed_values=['dtr', 'reboottxt'],
                description='''
                There are a couple of ways to reset VersatileExpress board if the
                software running on the board becomes unresponsive. Both require
                configuration to be enabled (please see ``devlib`` documentation).

                ``dtr``: toggle the DTR line on the serial connection
                ``reboottxt``: create ``reboot.txt`` in the root of the VEMSD mount.
                '''),
]
TC2_PLATFORM_OVERRIDES = [
        Parameter('baudrate', kind=int, default=38400,
                description='''
                Baud rate for the serial connection.
                '''),
        Parameter('vemsd_mount', kind=str, default='/media/VEMSD',
                description='''
                VExpress MicroSD card mount location. This is a MicroSD card in
                the VExpress device that is mounted on the host via USB. The card
                contains configuration files for the platform and firmware and
                kernel images to be flashed.
                '''),
        Parameter('bootloader', kind=str, default='bootmon',
                allowed_values=['uefi', 'uefi-shell', 'u-boot', 'bootmon'],
                description='''
                Selects the bootloader mechanism used by the board. Depending on
                firmware version, a number of possible boot mechanisms may be use.

                Please see ``devlib`` documentation for descriptions.
                '''),
        Parameter('hard_reset_method', kind=str, default='reboottxt',
                allowed_values=['dtr', 'reboottxt'],
                description='''
                There are a couple of ways to reset VersatileExpress board if the
                software running on the board becomes unresponsive. Both require
                configuration to be enabled (please see ``devlib`` documentation).

                ``dtr``: toggle the DTR line on the serial connection
                ``reboottxt``: create ``reboot.txt`` in the root of the VEMSD mount.
                '''),
]

# name --> ((platform_class, conn_class, conn_overrides), params_list, defaults, target_overrides)
# Note: normally, connection is defined by the Target name, but
#       platforms may choose to override it
# Note: the target_overrides allows you to override common target_params for a
# particular platform. Parameters you can override are in COMMON_TARGET_PARAMS
# Example of overriding one of the target parameters: Replace last `None` with
# a list of `Parameter` objects to be used instead.
PLATFORMS = {
    'generic': ((Platform, None, None), COMMON_PLATFORM_PARAMS, None, None),
    'juno': ((Juno, None, [
                            Parameter('host', kind=str, mandatory=False,
                            description="Host name or IP address of the target."),
                          ]
            ), COMMON_PLATFORM_PARAMS + VEXPRESS_PLATFORM_PARAMS, JUNO_PLATFORM_OVERRIDES, None),
    'tc2': ((TC2, None, None), COMMON_PLATFORM_PARAMS + VEXPRESS_PLATFORM_PARAMS,
            TC2_PLATFORM_OVERRIDES, None),
    'gem5': ((Gem5SimulationPlatform, Gem5Connection, None), GEM5_PLATFORM_PARAMS, None, None),
}


class DefaultTargetDescriptor(TargetDescriptor):

    name = 'devlib_targets'

    description = """
    The default target descriptor that provides descriptions in the form
    <platform>_<target>.

    These map directly onto ``Target``\ s and ``Platform``\ s supplied by ``devlib``.

    """

    def get_descriptions(self):
        # pylint: disable=attribute-defined-outside-init,too-many-locals
        result = []
        for target_name, target_tuple in TARGETS.items():
            (target, conn, unsupported_platforms), target_params = self._get_item(target_tuple)
            assistant = ASSISTANTS[target_name]
            conn_params = CONNECTION_PARAMS[conn]
            for platform_name, platform_tuple in PLATFORMS.items():
                platform_target_defaults = platform_tuple[-1]
                platform_tuple = platform_tuple[0:-1]
                (platform, plat_conn, conn_defaults), platform_params = self._get_item(platform_tuple)
                if platform in unsupported_platforms:
                    continue
                # Add target defaults specified in the Platform tuple
                target_params = self._override_params(target_params, platform_target_defaults)
                name = '{}_{}'.format(platform_name, target_name)
                td = TargetDescription(name, self)
                td.target = target
                td.platform = platform
                td.assistant = assistant
                td.target_params = target_params
                td.platform_params = platform_params
                td.assistant_params = assistant.parameters

                if plat_conn:
                    td.conn = plat_conn
                    td.conn_params = self._override_params(CONNECTION_PARAMS[plat_conn],
                                                           conn_defaults)
                else:
                    td.conn = conn
                    td.conn_params = self._override_params(conn_params, conn_defaults)

                result.append(td)
        return result

    def _override_params(self, params, overrides): # pylint: disable=no-self-use
        ''' Returns a new list of parameters replacing any parameter with the
        corresponding parameter in overrides'''
        if not overrides:
            return params
        param_map = {p.name: p for p in params}
        for override in overrides:
            if override.name in param_map:
                param_map[override.name] = override
        # Return the list of overriden parameters
        return list(param_map.values())

    def _get_item(self, item_tuple):
        cls_tuple, params, defaults = item_tuple
        updated_params = self._override_params(params, defaults)
        return cls_tuple, updated_params


_adhoc_target_descriptions = []


def create_target_description(name, *args, **kwargs):
    name = identifier(name)
    for td in _adhoc_target_descriptions:
        if caseless_string(name) == td.name:
            msg = 'Target with name "{}" already exists (from source: {})'
            raise ValueError(msg.format(name, td.source))

    stack = inspect.stack()
    # inspect.stack() returns a list of call frame records for the current thread
    # in reverse call order. So the first entry is for the current frame and next one
    # for the immediate caller. Each entry is a tuple in the format
    #  (frame_object, module_path, line_no, function_name, source_lines, source_lines_index)
    #
    # Here we assign the path of the calling module as the "source" for this description.
    # because this might be invoked via the add_scription_for_target wrapper, we need to
    # check for that, and make sure that we get the info for *its* caller in that case.
    if stack[1][3] == 'add_description_for_target':
        source = stack[2][1]
    else:
        source = stack[1][1]

    _adhoc_target_descriptions.append(TargetDescription(name, source, *args, **kwargs))


def _get_target_defaults(target):
    specificity = 0
    res = ('linux', TARGETS['linux'])  # fallback to a generic linux target
    for name, ttup in TARGETS.items():
        if issubclass(target, ttup[0][0]):
            new_spec = len(inspect.getmro(ttup[0][0]))
            if new_spec > specificity:
                res = (name, ttup)
                specificity = new_spec
    return res


def add_description_for_target(target, description=None, **kwargs):
    (base_name, ((_, base_conn, _), base_params, _)) = _get_target_defaults(target)

    if 'target_params' not in kwargs:
        kwargs['target_params'] = base_params

    if 'platform' not in kwargs:
        kwargs['platform'] = Platform
    if 'platform_params' not in kwargs:
        for (plat, conn, _), params, _, _ in PLATFORMS.values():
            if plat == kwargs['platform']:
                kwargs['platform_params'] = params
                if conn is not None and kwargs['conn'] is None:
                    kwargs['conn'] = conn
                break

    if 'conn' not in kwargs:
        kwargs['conn'] = base_conn
    if 'conn_params' not in kwargs:
        kwargs['conn_params'] = CONNECTION_PARAMS.get(kwargs['conn'])

    if 'assistant' not in kwargs:
        kwargs['assistant'] = ASSISTANTS.get(base_name)

    create_target_description(target.name, target=target, description=description, **kwargs)


class SimpleTargetDescriptor(TargetDescriptor):

    name = 'adhoc_targets'

    description = """
    Returns target descriptions added with ``create_target_description``.

    """

    def get_descriptions(self):
        return _adhoc_target_descriptions
