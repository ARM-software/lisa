from collections import OrderedDict
from copy import copy

from devlib import (LinuxTarget, AndroidTarget, LocalLinuxTarget,
                    Platform, Juno, TC2, Gem5SimulationPlatform,
                    AdbConnection, SshConnection, LocalConnection,
                    Gem5Connection)

from wa.framework import pluginloader
from wa.framework.configuration.core import get_config_point_map
from wa.framework.exception import PluginLoaderError
from wa.framework.plugin import Plugin, Parameter
from wa.framework.target.assistant import LinuxAssistant, AndroidAssistant
from wa.utils.types import list_of_strings, list_of_ints
from wa.utils.misc import isiterable


def get_target_descriptions(loader=pluginloader):
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
    return targets.values()


def instantiate_target(tdesc, params, connect=None, extra_platform_params=None):
    target_params = get_config_point_map(tdesc.target_params)
    platform_params = get_config_point_map(tdesc.platform_params)
    conn_params = get_config_point_map(tdesc.conn_params)
    assistant_params = get_config_point_map(tdesc.assistant_params)

    tp, pp, cp = {}, {}, {}

    for supported_params, new_params in (target_params, tp), (platform_params, pp), (conn_params, cp):
        for name, value in supported_params.iteritems():
            if value.default and name == value.name:
                new_params[name] = value.default

    for name, value in params.iteritems():
        if name in target_params:
            tp[name] = value
        elif name in platform_params:
            pp[name] = value
        elif name in conn_params:
            cp[name] = value
        elif name in assistant_params:
            pass
        else:
            msg = 'Unexpected parameter for {}: {}'
            raise ValueError(msg.format(tdesc.name, name))

    for pname, pval in (extra_platform_params or {}).iteritems():
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
            for n, p in getattr(self, pattr).itervalues():
                config[n] = p.default
        return config

    def _set(self, attr, vals):
        if vals is None:
            vals = {}
        elif isiterable(vals):
            if not hasattr(vals, 'iteritems'):
                vals = {v.name: v for v in vals}
        else:
            msg = '{} must be iterable; got "{}"'
            raise ValueError(msg.format(attr, vals))
        setattr(self, attr, vals)


class TargetDescriptor(Plugin):

    kind = 'target_descriptor'

    def get_descriptions(self):
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
    Parameter('modules', kind=list_of_strings,
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
    Parameter('modules', kind=list_of_strings,
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
        Parameter('device', kind=str,
                aliases=['adb_name'],
                description="""
                ADB device name
                """),
        Parameter('adb_server', kind=str,
                description="""
                ADB server to connect to.
                """),
    ],
    SshConnection: [
        Parameter('host', kind=str, mandatory=True,
                description="""
                Host name or IP address of the target.
                """),
        Parameter('username', kind=str, mandatory=True,
                description="""
                User name to connect with
                """),
        Parameter('password', kind=str,
                description="""
                Password to use.
                """),
        Parameter('keyfile', kind=str,
                description="""
                Key file to use
                """),
        Parameter('port', kind=int,
                description="""
                The port SSH server is listening on on the target.
                """),
        Parameter('telnet', kind=bool, default=False,
                description="""
                If set to ``True``, a Telnet connection, rather than
                SSH will be used.
                """),
        Parameter('password_prompt', kind=str,
                description="""
                Password prompt to expect
                """),
        Parameter('original_prompt', kind=str,
                description="""
                Original shell prompt to expect.
                """),
        Parameter('sudo_cmd', kind=str,
                default="sudo -- sh -c '{}'",
                description="""
                Sudo command to use. Must have ``"{}"``` specified
                somewher in the string it indicate where the command
                to be run via sudo is to go.
                """),
    ],
    Gem5Connection: [
        Parameter('host', kind=str, mandatory=False,
                description="""
                Host name or IP address of the target.
                """),
        Parameter('username', kind=str, default='root',
                description="""
                User name to connect to gem5 simulation.
                """),
        Parameter('password', kind=str,
                description="""
                Password to use.
                """),
        Parameter('port', kind=int,
                description="""
                The port SSH server is listening on on the target.
                """),
        Parameter('password_prompt', kind=str,
                description="""
                Password prompt to expect
                """),
        Parameter('original_prompt', kind=str,
                description="""
                Original shell prompt to expect.
                """),
    ],
    LocalConnection: [
        Parameter('password', kind=str,
                description="""
                Password to use for sudo. if not specified, the user will
                be prompted during intialization.
                """),
        Parameter('keep_password', kind=bool, default=True,
                description="""
                If ``True`` (the default), the password will be cached in
                memory after it is first obtained from the user, so that the
                user would not be prompted for it again.
                """),
        Parameter('unrooted', kind=bool, default=False,
                description="""
                Indicate that the target should be considered unrooted; do not
                attempt sudo or ask the user for their password.
                """),
    ],
}

# name --> ((target_class, conn_class), params_list, defaults, assistant_class)
TARGETS = {
    'linux': ((LinuxTarget, SshConnection), COMMON_TARGET_PARAMS, None),
    'android': ((AndroidTarget, AdbConnection), COMMON_TARGET_PARAMS +
                [Parameter('package_data_directory', kind=str, default='/data/data',
                           description='''
                           Directory containing Android data
                           '''),
                ], None),
    'local': ((LocalLinuxTarget, LocalConnection), COMMON_TARGET_PARAMS, None),
}

# name --> assistant
ASSISTANTS = {
    'linux': LinuxAssistant,
    'android': AndroidAssistant,
    'local': LinuxAssistant,
}

# name --> ((platform_class, conn_class), params_list, defaults)
# Note: normally, connection is defined by the Target name, but
#       platforms may choose to override it
PLATFORMS = {
    'generic': ((Platform, None), COMMON_PLATFORM_PARAMS, None),
    'juno': ((Juno, None), COMMON_PLATFORM_PARAMS + VEXPRESS_PLATFORM_PARAMS,
            {
                 'vemsd_mount': '/media/JUNO',
                 'baudrate': 115200,
                 'bootloader': 'u-boot',
                 'hard_reset_method': 'dtr',
            }),
    'tc2': ((TC2, None), COMMON_PLATFORM_PARAMS + VEXPRESS_PLATFORM_PARAMS,
            {
                 'vemsd_mount': '/media/VEMSD',
                 'baudrate': 38400,
                 'bootloader': 'bootmon',
                 'hard_reset_method': 'reboottxt',
            }),
    'gem5': ((Gem5SimulationPlatform, Gem5Connection), GEM5_PLATFORM_PARAMS, None),
}


class DefaultTargetDescriptor(TargetDescriptor):

    name = 'devlib_targets'

    description = """
    The default target descriptor that provides descriptions in the form
    <platform>_<target>.

    These map directly onto ``Target``\ s and ``Platform``\ s supplied by ``devlib``.

    """

    def get_descriptions(self):
        result = []
        for target_name, target_tuple in TARGETS.iteritems():
            (target, conn), target_params = self._get_item(target_tuple)
            assistant = ASSISTANTS[target_name]
            conn_params =  CONNECTION_PARAMS[conn]
            for platform_name, platform_tuple in PLATFORMS.iteritems():
                (platform, plat_conn), platform_params = self._get_item(platform_tuple)
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
                    td.conn_params =  CONNECTION_PARAMS[plat_conn]
                else:
                    td.conn = conn
                    td.conn_params = conn_params

                result.append(td)
        return result

    def _get_item(self, item_tuple):
        cls, params, defaults = item_tuple
        if not defaults:
            return cls, params

        param_map = OrderedDict((p.name, copy(p)) for p in params)
        for name, value in defaults.iteritems():
            if name not in param_map:
                raise ValueError('Unexpected default "{}"'.format(name))
            param_map[name].default = value
        return cls, param_map.values()
