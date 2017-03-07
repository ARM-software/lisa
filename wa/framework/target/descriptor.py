from collections import OrderedDict
from copy import copy

from devlib import (LinuxTarget, AndroidTarget, LocalLinuxTarget,
                    Platform, Juno, TC2, Gem5SimulationPlatform)

from wa.framework import pluginloader
from wa.framework.exception import PluginLoaderError
from wa.framework.plugin import Plugin, Parameter
from wa.utils.types import list_of_strings, list_of_ints


def get_target_descriptions(loader=pluginloader):
    targets = {}
    for cls in loader.list_target_descriptors():
        descriptor = cls()
        for desc in descriptor.get_descriptions():
            if desc.name in targets:
                msg = 'Duplicate target "{}" returned by {} and {}'
                prev_dtor = targets[desc.name].source
                raise PluginLoaderError(msg.format(dsc.name, prev_dtor.name,
                                                   descriptor.name))
            targets[desc.name] = desc
    return targets.values()


def instantiate_target(tdesc, params):
    target_params = {p.name: p for p in tdesc.target_params}
    platform_params = {p.name: p for p in tdesc.platform_params}
    conn_params = {p.name: p for p in tdesc.conn_params}

    tp, pp, cp = {}, {}, {}

    for name, value in params.iteritems():
        if name in target_params:
            tp[name] = value
        elif name in platform_params:
            pp[name] = value
        elif name in conn_params:
            cp[name] = value
        else:
            msg = 'Unexpected parameter for {}: {}'
            raise ValueError(msg.format(tdesc.name, name))

    tp['platform'] = (tdesc.platform or Platform)(**pp)
    if cp:
        tp['connection_settings'] = cp
    if tdesc.connection:
        tp['conn_cls'] = tdesc.connection

    return tdesc.target(**tp)


class TargetDescription(object):

    def __init__(self, name, source, description=None, target=None, platform=None, 
                 conn=None, target_params=None, platform_params=None,
                 conn_params=None):
        self.name = name
        self.source = source
        self.description = description
        self.target = target
        self.platform = platform
        self.connection = conn
        self._set('target_params', target_params)
        self._set('platform_params', platform_params)
        self._set('conn_params', conn_params)

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

              Please see ``devlab`` documentation for information on the available
              modules.
              '''),
    Parameter('load_default_modules', kind=bool, default=True,
              description='''
              A number of modules (e.g. for working with the cpufreq subsystem) are
              loaded by default when a Target is instantiated. Setting this to
              ``True`` would suppress that, ensuring that only the base Target
              interface is initialized.

              You may want to set this if there is a problem with one or more default
              modules on your platform (e.g. your device is unrooted and cpufreq is
              not accessible to unprivileged users), or if Target initialization is
              taking too long for your platform.
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
    Parameter('host_output_dir', kind=str, mandatory=True,
              description='''
              Path on the host where gem5 output (e.g. stats file) will be placed.
              '''),
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
]

# name --> (target_class, params_list, defaults)
TARGETS = {
    'linux': (LinuxTarget, COMMON_TARGET_PARAMS, None),
    'android': (AndroidTarget, COMMON_TARGET_PARAMS +
                [Parameter('package_data_directory', kind=str, default='/data/data',
                           description='''
                           Directory containing Android data
                           '''),
                ], None),
    'local': (LocalLinuxTarget, COMMON_TARGET_PARAMS, None),
}

# name --> (platform_class, params_list, defaults)
PLATFORMS = {
    'generic': (Platform, COMMON_PLATFORM_PARAMS, None),
    'juno': (Juno, COMMON_PLATFORM_PARAMS + VEXPRESS_PLATFORM_PARAMS,
            {
                 'vemsd_mount': '/media/JUNO',
                 'baudrate': 115200,
                 'bootloader': 'u-boot',
                 'hard_reset_method': 'dtr',
            }),
    'tc2': (TC2, COMMON_PLATFORM_PARAMS + VEXPRESS_PLATFORM_PARAMS,
            {
                 'vemsd_mount': '/media/VEMSD',
                 'baudrate': 38400,
                 'bootloader': 'bootmon',
                 'hard_reset_method': 'reboottxt',
            }),
    'gem5': (Gem5SimulationPlatform, GEM5_PLATFORM_PARAMS, None),
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
            target, target_params = self._get_item(target_tuple)
            for platform_name, platform_tuple in PLATFORMS.iteritems():
                platform, platform_params = self._get_item(platform_tuple)

                name = '{}_{}'.format(platform_name, target_name)
                td = TargetDescription(name, self)
                td.target = target
                td.platform = platform
                td.target_params = target_params
                td.platform_params = platform_params
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

