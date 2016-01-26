#    Copyright 2013-2015 ARM Limited
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

"""
Base classes for device interfaces.

    :Device: The base class for all devices. This defines the interface that must be
             implemented by all devices and therefore any workload and instrumentation
             can always rely on.
    :AndroidDevice: Implements most of the :class:`Device` interface, and extends it
                    with a number of Android-specific methods.
    :BigLittleDevice: Subclasses :class:`AndroidDevice` to implement big.LITTLE-specific
                      runtime parameters.
    :SimpleMulticoreDevice: Subclasses :class:`AndroidDevice` to implement homogeneous cores
                          device runtime parameters.

"""

import os
import imp
import string
from collections import OrderedDict
from contextlib import contextmanager

from wlauto.core.extension import Extension, ExtensionMeta, AttributeCollection, Parameter
from wlauto.core.extension_loader import ExtensionLoader
from wlauto.exceptions import DeviceError, ConfigError
from wlauto.utils.types import list_of_integers, list_of, caseless_string


__all__ = ['RuntimeParameter', 'CoreParameter', 'Device', 'DeviceMeta']


class RuntimeParameter(object):
    """
    A runtime parameter which has its getter and setter methods associated it
    with it.

    """

    def __init__(self, name, getter, setter,
                 getter_args=None, setter_args=None,
                 value_name='value', override=False):
        """
        :param name: the name of the parameter.
        :param getter: the getter method which returns the value of this parameter.
        :param setter: the setter method which sets the value of this parameter. The setter
                       always expects to be passed one argument when it is called.
        :param getter_args: keyword arguments to be used when invoking the getter.
        :param setter_args: keyword arguments to be used when invoking the setter.
        :param override: A ``bool`` that specifies whether a parameter of the same name further up the
                            hierarchy should be overridden. If this is ``False`` (the default), an exception
                            will be raised by the ``AttributeCollection`` instead.

        """
        self.name = name
        self.getter = getter
        self.setter = setter
        self.getter_args = getter_args or {}
        self.setter_args = setter_args or {}
        self.value_name = value_name
        self.override = override

    def __str__(self):
        return self.name

    __repr__ = __str__


class CoreParameter(RuntimeParameter):
    """A runtime parameter that will get expanded into a RuntimeParameter for each core type."""

    def get_runtime_parameters(self, core_names):
        params = []
        for core in set(core_names):
            name = string.Template(self.name).substitute(core=core)
            getter = string.Template(self.getter).substitute(core=core)
            setter = string.Template(self.setter).substitute(core=core)
            getargs = dict(self.getter_args.items() + [('core', core)])
            setargs = dict(self.setter_args.items() + [('core', core)])
            params.append(RuntimeParameter(name, getter, setter, getargs, setargs, self.value_name, self.override))
        return params


class DynamicModuleSpec(dict):

    @property
    def name(self):
        return self.keys()[0]

    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        if args:
            if len(args) > 1:
                raise ValueError(args)
            value = args[0]
        else:
            value = kwargs
        if isinstance(value, basestring):
            self[value] = {}
        elif isinstance(value, dict) and len(value) == 1:
            for k, v in value.iteritems():
                self[k] = v
        else:
            raise ValueError(value)


class DeviceMeta(ExtensionMeta):

    to_propagate = ExtensionMeta.to_propagate + [
        ('runtime_parameters', RuntimeParameter, AttributeCollection),
        ('dynamic_modules', DynamicModuleSpec, AttributeCollection),
    ]


class Device(Extension):
    """
    Base class for all devices supported by Workload Automation. Defines
    the interface the rest of WA uses to interact with devices.

        :name: Unique name used to identify the device.
        :platform: The name of the device's platform (e.g. ``Android``) this may
                   be used by workloads and instrumentation to assess whether they
                   can run on the device.
        :working_directory: a string of the directory which is
                            going to be used by the workloads on the device.
        :binaries_directory: a string of the binary directory for
                             the device.
        :has_gpu:     Should be ``True`` if the device as a separate GPU, and
                    ``False`` if graphics processing is done on a CPU.

                    .. note:: Pretty much all devices currently on the market
                                have GPUs, however this may not be the case for some
                                development boards.

        :path_module: The name of one of the modules implementing the os.path
                      interface, e.g. ``posixpath`` or ``ntpath``. You can provide
                      your own implementation rather than relying on one of the
                      standard library modules, in which case you need to specify
                      the *full* path to you module. e.g. '/home/joebloggs/mypathimp.py'
        :parameters: A list of RuntimeParameter objects. The order of the objects
                     is very important as the setters and getters will be called
                     in the order the RuntimeParameter objects inserted.
        :active_cores: This should be a list of all the currently active cpus in
                      the device in ``'/sys/devices/system/cpu/online'``. The
                      returned list should be read from the device at the time
                      of read request.

    """
    __metaclass__ = DeviceMeta

    parameters = [
        Parameter('core_names', kind=list_of(caseless_string), mandatory=True, default=None,
                  description="""
                  This is a list of all cpu cores on the device with each
                  element being the core type, e.g. ``['a7', 'a7', 'a15']``. The
                  order of the cores must match the order they are listed in
                  ``'/sys/devices/system/cpu'``. So in this case, ``'cpu0'`` must
                  be an A7 core, and ``'cpu2'`` an A15.'
                  """),
        Parameter('core_clusters', kind=list_of_integers, mandatory=True, default=None,
                  description="""
                  This is a list indicating the cluster affinity of the CPU cores,
                  each element correponding to the cluster ID of the core coresponding
                  to its index. E.g. ``[0, 0, 1]`` indicates that cpu0 and cpu1 are on
                  cluster 0, while cpu2 is on cluster 1. If this is not specified, this
                  will be inferred from ``core_names`` if possible (assuming all cores with
                  the same name are on the same cluster).
                  """),
    ]

    runtime_parameters = []
    # dynamic modules are loaded or not based on whether the device supports
    # them (established at runtime by module probling the device).
    dynamic_modules = []

    # These must be overwritten by subclasses.
    name = None
    platform = None
    default_working_directory = None
    has_gpu = None
    path_module = None
    active_cores = None

    def __init__(self, **kwargs):  # pylint: disable=W0613
        super(Device, self).__init__(**kwargs)
        if not self.path_module:
            raise NotImplementedError('path_module must be specified by the deriving classes.')
        libpath = os.path.dirname(os.__file__)
        modpath = os.path.join(libpath, self.path_module)
        if not modpath.lower().endswith('.py'):
            modpath += '.py'
        try:
            self.path = imp.load_source('device_path', modpath)
        except IOError:
            raise DeviceError('Unsupported path module: {}'.format(self.path_module))

    def validate(self):
        # pylint: disable=access-member-before-definition,attribute-defined-outside-init
        if self.core_names and not self.core_clusters:
            self.core_clusters = []
            clusters = []
            for cn in self.core_names:
                if cn not in clusters:
                    clusters.append(cn)
                self.core_clusters.append(clusters.index(cn))
        if len(self.core_names) != len(self.core_clusters):
            raise ConfigError('core_names and core_clusters are of different lengths.')

    def initialize(self, context):
        """
        Initialization that is performed at the begining of the run (after the device has
        been connecte).

        """
        loader = ExtensionLoader()
        for module_spec in self.dynamic_modules:
            module = self._load_module(loader, module_spec)
            if not hasattr(module, 'probe'):
                message = 'Module {} does not have "probe" attribute; cannot be loaded dynamically'
                raise ValueError(message.format(module.name))
            if module.probe(self):
                self.logger.debug('Installing module "{}"'.format(module.name))
                self._install_module(module)
            else:
                self.logger.debug('Module "{}" is not supported by the device'.format(module.name))

    def reset(self):
        """
        Initiate rebooting of the device.

        Added in version 2.1.3.

        """
        raise NotImplementedError()

    def boot(self, *args, **kwargs):
        """
        Perform the seteps necessary to boot the device to the point where it is ready
        to accept other commands.

        Changed in version 2.1.3: no longer expected to wait until boot completes.

        """
        raise NotImplementedError()

    def connect(self, *args, **kwargs):
        """
        Establish a connection to the device that will be used for subsequent commands.

        Added in version 2.1.3.
        """
        raise NotImplementedError()

    def disconnect(self):
        """ Close the established connection to the device. """
        raise NotImplementedError()

    def ping(self):
        """
        This must return successfully if the device is able to receive commands, or must
        raise :class:`wlauto.exceptions.DeviceUnresponsiveError` if the device cannot respond.

        """
        raise NotImplementedError()

    def get_runtime_parameter_names(self):
        return [p.name for p in self._expand_runtime_parameters()]

    def get_runtime_parameters(self):
        """ returns the runtime parameters that have been set. """
        # pylint: disable=cell-var-from-loop
        runtime_parameters = OrderedDict()
        for rtp in self._expand_runtime_parameters():
            if not rtp.getter:
                continue
            getter = getattr(self, rtp.getter)
            rtp_value = getter(**rtp.getter_args)
            runtime_parameters[rtp.name] = rtp_value
        return runtime_parameters

    def set_runtime_parameters(self, params):
        """
        The parameters are taken from the keyword arguments and are specific to
        a particular device. See the device documentation.

        """
        runtime_parameters = self._expand_runtime_parameters()
        rtp_map = {rtp.name.lower(): rtp for rtp in runtime_parameters}

        params = OrderedDict((k.lower(), v) for k, v in params.iteritems() if v is not None)

        expected_keys = rtp_map.keys()
        if not set(params.keys()).issubset(set(expected_keys)):
            unknown_params = list(set(params.keys()).difference(set(expected_keys)))
            raise ConfigError('Unknown runtime parameter(s): {}'.format(unknown_params))

        for param in params:
            self.logger.debug('Setting runtime parameter "{}"'.format(param))
            rtp = rtp_map[param]
            setter = getattr(self, rtp.setter)
            args = dict(rtp.setter_args.items() + [(rtp.value_name, params[rtp.name.lower()])])
            setter(**args)

    def capture_screen(self, filepath):
        """Captures the current device screen into the specified file in a PNG format."""
        raise NotImplementedError()

    def get_properties(self, output_path):
        """Captures and saves the device configuration properties version and
         any other relevant information. Return them in a dict"""
        raise NotImplementedError()

    def listdir(self, path, **kwargs):
        """ List the contents of the specified directory. """
        raise NotImplementedError()

    def push_file(self, source, dest):
        """ Push a file from the host file system onto the device. """
        raise NotImplementedError()

    def pull_file(self, source, dest):
        """ Pull a file from device system onto the host file system. """
        raise NotImplementedError()

    def delete_file(self, filepath):
        """ Delete the specified file on the device. """
        raise NotImplementedError()

    def file_exists(self, filepath):
        """ Check if the specified file or directory exist on the device. """
        raise NotImplementedError()

    def get_pids_of(self, process_name):
        """ Returns a list of PIDs of the specified process name. """
        raise NotImplementedError()

    def kill(self, pid, as_root=False):
        """ Kill the  process with the specified PID. """
        raise NotImplementedError()

    def killall(self, process_name, as_root=False):
        """ Kill all running processes with the specified name. """
        raise NotImplementedError()

    def install(self, filepath, **kwargs):
        """ Install the specified file on the device. What "install" means is device-specific
        and may possibly also depend on the type of file."""
        raise NotImplementedError()

    def uninstall(self, filepath):
        """ Uninstall the specified file on the device. What "uninstall" means is device-specific
        and may possibly also depend on the type of file."""
        raise NotImplementedError()

    def execute(self, command, timeout=None, **kwargs):
        """
        Execute the specified command command on the device and return the output.

        :param command: Command to be executed on the device.
        :param timeout: If the command does not return after the specified time,
                        execute() will abort with an error. If there is no timeout for
                        the command, this should be set to 0 or None.

        Other device-specific keyword arguments may also be specified.

        :returns: The stdout output from the command.

        """
        raise NotImplementedError()

    def set_sysfile_value(self, filepath, value, verify=True):
        """
        Write the specified value to the specified file on the device
        and verify that the value has actually been written.

        :param file: The file to be modified.
        :param value: The value to be written to the file. Must be
                      an int or a string convertable to an int.
        :param verify: Specifies whether the value should be verified, once written.

        Should raise DeviceError if could write value.

        """
        raise NotImplementedError()

    def get_sysfile_value(self, sysfile, kind=None):
        """
        Get the contents of the specified sysfile.

        :param sysfile: The file who's contents will be returned.

        :param kind: The type of value to be expected in the sysfile. This can
                     be any Python callable that takes a single str argument.
                     If not specified or is None, the contents will be returned
                     as a string.

        """
        raise NotImplementedError()

    def start(self):
        """
        This gets invoked before an iteration is started and is endented to help the
        device manange any internal supporting functions.

        """
        pass

    def stop(self):
        """
        This gets invoked after iteration execution has completed and is endented to help the
        device manange any internal supporting functions.

        """
        pass

    def __str__(self):
        return 'Device<{}>'.format(self.name)

    __repr__ = __str__

    def _expand_runtime_parameters(self):
        expanded_params = []
        for param in self.runtime_parameters:
            if isinstance(param, CoreParameter):
                expanded_params.extend(param.get_runtime_parameters(self.core_names))  # pylint: disable=no-member
            else:
                expanded_params.append(param)
        return expanded_params

    @contextmanager
    def _check_alive(self):
        try:
            yield
        except Exception as e:
            self.ping()
            raise e
