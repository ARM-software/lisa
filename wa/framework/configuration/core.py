#    Copyright 2014-2016 ARM Limited
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

import os
import re
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict

from wa.framework.exception import ConfigError, NotFoundError
from wa.framework.configuration.tree import SectionNode
from wa.utils.misc import (get_article, merge_config_values)
from wa.utils.types import (identifier, integer, boolean, list_of_strings,
                            list_of, toggle_set, obj_dict, enum)
from wa.utils.serializer import is_pod


# Mapping for kind conversion; see docs for convert_types below
KIND_MAP = {
    int: integer,
    bool: boolean,
    dict: OrderedDict,
}

Status = enum(['UNKNOWN', 'NEW', 'PENDING',
               'STARTED', 'CONNECTED', 'INITIALIZED', 'RUNNING',
               'OK', 'PARTIAL', 'FAILED', 'ABORTED', 'SKIPPED'])



##########################
### CONFIG POINT TYPES ###
##########################


class RebootPolicy(object):
    """
    Represents the reboot policy for the execution -- at what points the device
    should be rebooted. This, in turn, is controlled by the policy value that is
    passed in on construction and would typically be read from the user's settings.
    Valid policy values are:

    :never: The device will never be rebooted.
    :as_needed: Only reboot the device if it becomes unresponsive, or needs to be flashed, etc.
    :initial: The device will be rebooted when the execution first starts, just before
              executing the first workload spec.
    :each_spec: The device will be rebooted before running a new workload spec.
    :each_iteration: The device will be rebooted before each new iteration.

    """

    valid_policies = ['never', 'as_needed', 'initial', 'each_spec', 'each_iteration']

    def __init__(self, policy):
        policy = policy.strip().lower().replace(' ', '_')
        if policy not in self.valid_policies:
            message = 'Invalid reboot policy {}; must be one of {}'.format(policy, ', '.join(self.valid_policies))
            raise ConfigError(message)
        self.policy = policy

    @property
    def can_reboot(self):
        return self.policy != 'never'

    @property
    def perform_initial_boot(self):
        return self.policy not in ['never', 'as_needed']

    @property
    def reboot_on_each_spec(self):
        return self.policy in ['each_spec', 'each_iteration']

    @property
    def reboot_on_each_iteration(self):
        return self.policy == 'each_iteration'

    def __str__(self):
        return self.policy

    __repr__ = __str__

    def __cmp__(self, other):
        if isinstance(other, RebootPolicy):
            return cmp(self.policy, other.policy)
        else:
            return cmp(self.policy, other)

    def to_pod(self):
        return self.policy

    @staticmethod
    def from_pod(pod):
        return RebootPolicy(pod)


class status_list(list):

    def append(self, item):
        list.append(self, str(item).upper())


class LoggingConfig(dict):

    defaults = {
        'file_format': '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        'verbose_format': '%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        'regular_format': '%(levelname)-8s %(message)s',
        'color': True,
    }

    @staticmethod
    def from_pod(pod):
        return LoggingConfig(pod)

    def __init__(self, config=None):
        dict.__init__(self)
        if isinstance(config, dict):
            config = {identifier(k.lower()): v for k, v in config.iteritems()}
            self['regular_format'] = config.pop('regular_format', self.defaults['regular_format'])
            self['verbose_format'] = config.pop('verbose_format', self.defaults['verbose_format'])
            self['file_format'] = config.pop('file_format', self.defaults['file_format'])
            self['color'] = config.pop('colour_enabled', self.defaults['color'])  # legacy
            self['color'] = config.pop('color', self.defaults['color'])
            if config:
                message = 'Unexpected logging configuration parameters: {}'
                raise ValueError(message.format(bad_vals=', '.join(config.keys())))
        elif config is None:
            for k, v in self.defaults.iteritems():
                self[k] = v
        else:
            raise ValueError(config)

    def to_pod(self):
        return self


def get_type_name(kind):
    typename = str(kind)
    if '\'' in typename:
        typename = typename.split('\'')[1]
    elif typename.startswith('<function'):
        typename = typename.split()[1]
    return typename


class ConfigurationPoint(object):
    """
    This defines a generic configuration point for workload automation. This is
    used to handle global settings, plugin parameters, etc.

    """

    def __init__(self, name,
                 kind=None,
                 mandatory=None,
                 default=None,
                 override=False,
                 allowed_values=None,
                 description=None,
                 constraint=None,
                 merge=False,
                 aliases=None,
                 global_alias=None):
        """
        Create a new Parameter object.

        :param name: The name of the parameter. This will become an instance
                     member of the plugin object to which the parameter is
                     applied, so it must be a valid python  identifier. This
                     is the only mandatory parameter.
        :param kind: The type of parameter this is. This must be a callable
                     that takes an arbitrary object and converts it to the
                     expected type, or raised ``ValueError`` if such conversion
                     is not possible. Most Python standard types -- ``str``,
                     ``int``, ``bool``, etc. -- can be used here. This
                     defaults to ``str`` if not specified.
        :param mandatory: If set to ``True``, then a non-``None`` value for
                          this parameter *must* be provided on plugin
                          object construction, otherwise ``ConfigError``
                          will be raised.
        :param default: The default value for this parameter. If no value
                        is specified on plugin construction, this value
                        will be used instead. (Note: if this is specified
                        and is not ``None``, then ``mandatory`` parameter
                        will be ignored).
        :param override: A ``bool`` that specifies whether a parameter of
                         the same name further up the hierarchy should
                         be overridden. If this is ``False`` (the
                         default), an exception will be raised by the
                         ``AttributeCollection`` instead.
        :param allowed_values: This should be the complete list of allowed
                               values for this parameter.  Note: ``None``
                               value will always be allowed, even if it is
                               not in this list.  If you want to disallow
                               ``None``, set ``mandatory`` to ``True``.
        :param constraint: If specified, this must be a callable that takes
                           the parameter value as an argument and return a
                           boolean indicating whether the constraint has been
                           satisfied. Alternatively, can be a two-tuple with
                           said callable as the first element and a string
                           describing the constraint as the second.
        :param merge: The default behaviour when setting a value on an object
                      that already has that attribute is to overrided with
                      the new value. If this is set to ``True`` then the two
                      values will be merged instead. The rules by which the
                      values are merged will be determined by the types of
                      the existing and new values -- see
                      ``merge_config_values`` documentation for details.
        :param aliases: Alternative names for the same configuration point.
                        These are largely for backwards compatibility.
        :param global_alias: An alias for this parameter that can be specified at
                            the global level. A global_alias can map onto many
                            ConfigurationPoints.
        """
        self.name = identifier(name)
        if kind in KIND_MAP:
            kind = KIND_MAP[kind]
        if kind is not None and not callable(kind):
            raise ValueError('Kind must be callable.')
        self.kind = kind
        self.mandatory = mandatory
        if not is_pod(default):
            msg = "The default for '{}' must be a Plain Old Data type, but it is of type '{}' instead."
            raise TypeError(msg.format(self.name, type(default)))
        self.default = default
        self.override = override
        self.allowed_values = allowed_values
        self.description = description
        if self.kind is None and not self.override:
            self.kind = str
        if constraint is not None and not callable(constraint) and not isinstance(constraint, tuple):
            raise ValueError('Constraint must be callable or a (callable, str) tuple.')
        self.constraint = constraint
        self.merge = merge
        self.aliases = aliases or []
        self.global_alias = global_alias

        if self.default is not None:
            try:
                self.validate_value("init", self.default)
            except ConfigError:
                raise ValueError('Default value "{}" is not valid'.format(self.default))

    def match(self, name):
        if name == self.name or name in self.aliases:
            return True
        elif name == self.global_alias:
            return True
        return False

    def set_value(self, obj, value=None, check_mandatory=True):
        if value is None:
            if self.default is not None:
                value = self.kind(self.default)
            elif check_mandatory and self.mandatory:
                msg = 'No values specified for mandatory parameter "{}" in {}'
                raise ConfigError(msg.format(self.name, obj.name))
        else:
            try:
                value = self.kind(value)
            except (ValueError, TypeError):
                typename = get_type_name(self.kind)
                msg = 'Bad value "{}" for {}; must be {} {}'
                article = get_article(typename)
                raise ConfigError(msg.format(value, self.name, article, typename))
        if value is not None:
            self.validate_value(self.name, value)
        if self.merge and hasattr(obj, self.name):
            value = merge_config_values(getattr(obj, self.name), value)
        setattr(obj, self.name, value)

    def validate(self, obj, check_mandatory=True):
        value = getattr(obj, self.name, None)
        if value is not None:
            self.validate_value(obj.name, value)
        else:
            if check_mandatory and self.mandatory:
                msg = 'No value specified for mandatory parameter "{}" in {}.'
                raise ConfigError(msg.format(self.name, obj.name))

    def validate_value(self, name, value):
        if self.allowed_values:
            self.validate_allowed_values(name, value)
        if self.constraint:
            self.validate_constraint(name, value)

    def validate_allowed_values(self, name, value):
        if 'list' in str(self.kind):
            for v in value:
                if v not in self.allowed_values:
                    msg = 'Invalid value {} for {} in {}; must be in {}'
                    raise ConfigError(msg.format(v, self.name, name, self.allowed_values))
        else:
            if value not in self.allowed_values:
                msg = 'Invalid value {} for {} in {}; must be in {}'
                raise ConfigError(msg.format(value, self.name, name, self.allowed_values))

    def validate_constraint(self, name, value):
        msg_vals = {'value': value, 'param': self.name, 'plugin': name}
        if isinstance(self.constraint, tuple) and len(self.constraint) == 2:
            constraint, msg = self.constraint  # pylint: disable=unpacking-non-sequence
        elif callable(self.constraint):
            constraint = self.constraint
            msg = '"{value}" failed constraint validation for "{param}" in "{plugin}".'
        else:
            raise ValueError('Invalid constraint for "{}": must be callable or a 2-tuple'.format(self.name))
        if not constraint(value):
            raise ConfigError(value, msg.format(**msg_vals))

    def __repr__(self):
        d = copy(self.__dict__)
        del d['description']
        return 'ConfigurationPoint({})'.format(d)

    __str__ = __repr__


class RuntimeParameter(object):

    def __init__(self, name,
                 kind=None,
                 description=None,
                 merge=False):

        self.name = re.compile(name)
        if kind is not None:
            if kind in KIND_MAP:
                kind = KIND_MAP[kind]
            if not callable(kind):
                raise ValueError('Kind must be callable.')
        else:
            kind = str
        self.kind = kind
        self.description = description
        self.merge = merge

    def validate_kind(self, value, name):
        try:
            value = self.kind(value)
        except (ValueError, TypeError):
            typename = get_type_name(self.kind)
            msg = 'Bad value "{}" for {}; must be {} {}'
            article = get_article(typename)
            raise ConfigError(msg.format(value, name, article, typename))

    def match(self, name):
        if self.name.match(name):
            return True
        return False

    def update_value(self, name, new_value, source, dest):
        self.validate_kind(new_value, name)

        if name in dest:
            old_value, sources = dest[name]
        else:
            old_value = None
            sources = {}
        sources[source] = new_value

        if self.merge:
            new_value = merge_config_values(old_value, new_value)

        dest[name] = (new_value, sources)


class RuntimeParameterManager(object):

    runtime_parameters = []

    def __init__(self, target_manager):
        self.state = {}
        self.target_manager = target_manager

    def get_initial_state(self):
        """
        Should be used to load the starting state from the device. This state
        should be updated if any changes are made to the device, and they are successful.
        """
        pass

    def match(self, name):
        for rtp in self.runtime_parameters:
            if rtp.match(name):
                return True
        return False

    def update_value(self, name, value, source, dest):
        for rtp in self.runtime_parameters:
            if rtp.match(name):
                rtp.update_value(name, value, source, dest)
                break
        else:
            msg = 'Unknown runtime parameter "{}"'
            raise ConfigError(msg.format(name))

    def static_validation(self, params):
        """
        Validate values that do not require a active device connection.
        This method should also pop all runtime parameters meant for this manager
        from params, even if they are not being statically validated.
        """
        pass

    def dynamic_validation(self, params):
        """
        Validate values that require an active device connection
        """
        pass

    def commit(self):
        """
        All values have been validated, this will now actually set values
        """
        pass

################################
### RuntimeParameterManagers ###
################################


class CpuFreqParameters(object):

    runtime_parameters = {
        "cores": RuntimeParameter("(.+)_cores"),
        "min_frequency": RuntimeParameter("(.+)_min_frequency", kind=int),
        "max_frequency": RuntimeParameter("(.+)_max_frequency", kind=int),
        "frequency": RuntimeParameter("(.+)_frequency", kind=int),
        "governor": RuntimeParameter("(.+)_governor"),
        "governor_tunables": RuntimeParameter("(.+)_governor_tunables"),
    }

    def __init__(self, target):
        super(CpuFreqParameters, self).__init__(target)
        self.core_names = set(target.core_names)

    def match(self, name):
        for param in self.runtime_parameters.itervalues():
            if param.match(name):
                return True
        return False

    def update_value(self, name, value, source):
        for param in self.runtime_parameters.iteritems():
            core_name_match = param.name.match(name)
            if not core_name_match:
                continue

            core_name = core_name_match.groups()[0]
            if core_name not in self.core_names:
                msg = '"{}" in {} is not a valid core name, must be in: {}'
                raise ConfigError(msg.format(core_name, name, ", ".join(self.core_names)))

            param.update_value(name, value, source)
            break
        else:
            RuntimeError('"{}" does not belong to CpuFreqParameters'.format(name))

    def _get_merged_value(self, core, param_name):
        return self.runtime_parameters[param_name].merged_values["{}_{}".format(core, param_name)]

    def _cross_validate(self, core):
        min_freq = self._get_merged_value(core, "min_frequency")
        max_frequency = self._get_merged_value(core, "max_frequency")
        if max_frequency < min_freq:
            msg = "{core}_max_frequency must be larger than {core}_min_frequency"
            raise ConfigError(msg.format(core=core))
        frequency = self._get_merged_value(core, "frequency")
        if not min_freq < frequency < max_frequency:
            msg = "{core}_frequency must be between {core}_min_frequency and {core}_max_frequency"
            raise ConfigError(msg.format(core=core))
        #TODO: more checks

    def commit_to_device(self, target):
        pass
        # TODO: Write values to device is correct order ect

#####################
### Configuration ###
#####################


def _to_pod(cfg_point, value):
    if is_pod(value):
        return value
    if hasattr(cfg_point.kind, 'to_pod'):
        return value.to_pod()
    msg = '{} value "{}" is not serializable'
    raise ValueError(msg.format(cfg_point.name, value))


class Configuration(object):

    config_points = []
    name = ''

    # The below line must be added to all subclasses
    configuration = {cp.name: cp for cp in config_points}

    @classmethod
    def from_pod(cls, pod):
        instance = cls()
        for cfg_point in cls.config_points:
            if cfg_point.name in pod:
                value = pod.pop(cfg_point.name)
                if hasattr(cfg_point.kind, 'from_pod'):
                    value = cfg_point.kind.from_pod(value)
                cfg_point.set_value(instance, value)
        if pod:
            msg = 'Invalid entry(ies) for "{}": "{}"'
            raise ValueError(msg.format(cls.name, '", "'.join(pod.keys())))
        return instance

    def __init__(self):
        for confpoint in self.config_points:
            confpoint.set_value(self, check_mandatory=False)

    def set(self, name, value, check_mandatory=True):
        if name not in self.configuration:
            raise ConfigError('Unknown {} configuration "{}"'.format(self.name,
                                                                     name))
        self.configuration[name].set_value(self, value,
                                           check_mandatory=check_mandatory)

    def update_config(self, values, check_mandatory=True):
        for k, v in values.iteritems():
            self.set(k, v, check_mandatory=check_mandatory)

    def validate(self):
        for cfg_point in self.config_points:
            cfg_point.validate(self)

    def to_pod(self):
        pod = {}
        for cfg_point in self.config_points:
            value = getattr(self, cfg_point.name, None)
            pod[cfg_point.name] = _to_pod(cfg_point, value)
        return pod


# This configuration for the core WA framework
class MetaConfiguration(Configuration):

    name = "Meta Configuration"

    plugin_packages = [
        'wa.commands',
        'wa.framework.getters',
        'wa.framework.target.descriptor',
        'wa.instrumentation',
        'wa.processors',
        'wa.workloads',
    ]

    config_points = [
        ConfigurationPoint(
            'user_directory',
            description="""
            Path to the user directory. This is the location WA will look for
            user configuration, additional plugins and plugin dependencies.
            """,
            kind=str,
            default=os.path.join(os.path.expanduser('~'), '.workload_automation'),
        ),
        ConfigurationPoint(
            'assets_repository',
            description="""
            The local mount point for the filer hosting WA assets.
            """,
        ),
        ConfigurationPoint(
            'logging',
            kind=LoggingConfig,
            default=LoggingConfig.defaults,
            description="""
            WA logging configuration. This should be a dict with a subset
            of the following keys::

            :normal_format: Logging format used for console output
            :verbose_format: Logging format used for verbose console output
            :file_format: Logging format used for run.log
            :color: If ``True`` (the default), console logging output will
                    contain bash color escape codes. Set this to ``False`` if
                    console output will be piped somewhere that does not know
                    how to handle those.
            """,
        ),
        ConfigurationPoint(
            'verbosity',
            kind=int,
            default=0,
            description="""
            Verbosity of console output.
            """,
        ),
        ConfigurationPoint(  # TODO: Needs some format for dates etc/ comes from cfg
            'default_output_directory',
            default="wa_output",
            description="""
            The default output directory that will be created if not
            specified when invoking a run.
            """,
        ),
    ]
    configuration = {cp.name: cp for cp in config_points}

    @property
    def dependencies_directory(self):
        return os.path.join(self.user_directory, 'dependencies')

    @property
    def plugins_directory(self):
        return os.path.join(self.user_directory, 'plugins')

    @property
    def user_config_file(self):
        return os.path.join(self.user_directory, 'config.yaml')

    def __init__(self, environ=os.environ):
        super(MetaConfiguration, self).__init__()
        user_directory = environ.pop('WA_USER_DIRECTORY', '')
        if user_directory:
            self.set('user_directory', user_directory)


# This is generic top-level configuration for WA runs.
class RunConfiguration(Configuration):

    name = "Run Configuration"

    # Metadata is separated out because it is not loaded into the auto
    # generated config file
    meta_data = [
        ConfigurationPoint(
            'run_name',
            kind=str,
            description='''
            A string that labels the WA run that is being performed. This would
            typically be set in the ``config`` section of an agenda (see
            :ref:`configuration in an agenda <configuration_in_agenda>`) rather
            than in the config file.
            ''',
        ),
        ConfigurationPoint(
            'project',
            kind=str,
            description='''
            A string naming the project for which data is being collected. This
            may be useful, e.g. when uploading data to a shared database that
            is populated from multiple projects.
            ''',
        ),
        ConfigurationPoint(
            'project_stage',
            kind=dict,
            description='''
            A dict or a string that allows adding additional identifier. This
            is may be useful for long-running projects.
            ''',
        ),
    ]
    config_points = [
        ConfigurationPoint(
            'execution_order',
            kind=str,
            default='by_iteration',
            allowed_values=['by_iteration', 'by_spec', 'by_section', 'random'],
            description='''
            Defines the order in which the agenda spec will be executed. At the
            moment, the following execution orders are supported:

            ``"by_iteration"``
                The first iteration of each workload spec is executed one after
                the other, so all workloads are executed before proceeding on
                to the second iteration.  E.g. A1 B1 C1 A2 C2 A3. This is the
                default if no order is explicitly specified.

                In case of multiple sections, this will spread them out, such
                that specs from the same section are further part. E.g. given
                sections X and Y, global specs A and B, and two iterations,
                this will run ::

                        X.A1, Y.A1, X.B1, Y.B1, X.A2, Y.A2, X.B2, Y.B2

            ``"by_section"``
                Same  as ``"by_iteration"``, however this will group specs from
                the same section together, so given sections X and Y, global
                specs A and B, and two iterations, this will run ::

                        X.A1, X.B1, Y.A1, Y.B1, X.A2, X.B2, Y.A2, Y.B2

            ``"by_spec"``
                All iterations of the first spec are executed before moving on
                to the next spec. E.g. A1 A2 A3 B1 C1 C2 This may also be
                specified as ``"classic"``, as this was the way workloads were
                executed in earlier versions of WA.

            ``"random"``
                Execution order is entirely random.
            ''',
        ),
        ConfigurationPoint(
            'reboot_policy',
            kind=RebootPolicy,
            default='as_needed',
            allowed_values=RebootPolicy.valid_policies,
            description='''
            This defines when during execution of a run the Device will be
            rebooted. The possible values are:

            ``"never"``
                The device will never be rebooted.

            ``"initial"``
                The device will be rebooted when the execution first starts,
                just before executing the first workload spec.

            ``"each_spec"``
                The device will be rebooted before running a new workload spec.

                .. note:: this acts the same as each_iteration when execution order
                          is set to by_iteration

            ``"each_iteration"``
                The device will be rebooted before each new iteration.
            '''),
        ConfigurationPoint(
            'device',
            kind=str,
            mandatory=True,
            description='''
            This setting defines what specific Device subclass will be used to
            interact the connected device. Obviously, this must match your
            setup.
            ''',
        ),
        ConfigurationPoint(
            'retry_on_status',
            kind=list_of(Status),
            default=['FAILED', 'PARTIAL'],
            allowed_values=Status.levels[Status.RUNNING.value:],
            description='''
            This is list of statuses on which a job will be considered to have
            failed and will be automatically retried up to ``max_retries``
            times. This defaults to ``["FAILED", "PARTIAL"]`` if not set.
            Possible values are::

            ``"OK"``
            This iteration has completed and no errors have been detected

            ``"PARTIAL"``
            One or more instruments have failed (the iteration may still be running).

            ``"FAILED"``
            The workload itself has failed.

            ``"ABORTED"``
            The user interrupted the workload
            ''',
        ),
        ConfigurationPoint(
            'max_retries',
            kind=int,
            default=2,
            description='''
            The maximum number of times failed jobs will be retried before
            giving up. If not set.

            .. note:: this number does not include the original attempt
            ''',
        ),
        ConfigurationPoint(
            'bail_on_init_failure',
            kind=bool,
            default=True,
            description='''
            When jobs fail during their main setup and run phases, WA will
            continue attempting to run the remaining jobs. However, by default,
            if they fail during their early initialization phase, the entire run
            will end without continuing to run jobs. Setting this to ``False``
            means that WA will instead skip all the jobs from the job spec that
            failed, but continue attempting to run others.
            '''
        ),
        ConfigurationPoint(
            'result_processors',
            kind=toggle_set,
            default=['csv', 'status'],
            description='''
            The list of output processors to be used for this run. Output processors
            post-process data generated by workloads and instruments, e.g. to
            generate additional reports, format the output in a certain way, or
            export the output to an exeternal location.
            ''',
        ),
        ConfigurationPoint(
            'allow_phone_home',
            kind=bool, default=True,
            description='''
            Setting this to ``False`` prevents running any workloads that are marked
            with 'phones_home', meaning they are at risk of exposing information
            about the device to the outside world. For example, some benchmark
            applications upload device data to a database owned by the
            maintainers.

            This can be used to minimise the risk of accidentally running such
            workloads when testing confidential devices.
            '''),
    ]
    configuration = {cp.name: cp for cp in config_points + meta_data}

    def __init__(self):
        super(RunConfiguration, self).__init__()
        for confpoint in self.meta_data:
            confpoint.set_value(self, check_mandatory=False)
        self.device_config = None

    def merge_device_config(self, plugin_cache):
        """
        Merges global device config and validates that it is correct for the
        selected device.
        """
        # pylint: disable=no-member
        if self.device is None:
            msg = 'Attempting to merge device config with unspecified device'
            raise RuntimeError(msg)
        self.device_config = plugin_cache.get_plugin_config(self.device,
                                                            generic_name="device_config")

    def to_pod(self):
        pod = super(RunConfiguration, self).to_pod()
        pod['device_config'] = dict(self.device_config or {})
        return pod

    @classmethod
    def from_pod(cls, pod):
        meta_pod = {}
        for cfg_point in cls.meta_data:
            meta_pod[cfg_point.name] = pod.pop(cfg_point.name, None)

        instance = super(RunConfiguration, cls).from_pod(pod)
        for cfg_point in cls.meta_data:
            cfg_point.set_value(instance, meta_pod[cfg_point.name])

        return instance


class JobSpec(Configuration):

    name = "Job Spec"

    config_points = [
        ConfigurationPoint('iterations', kind=int, default=1,
                           description='''
                           How many times to repeat this workload spec
                           '''),
        ConfigurationPoint('workload_name', kind=str, mandatory=True,
                           aliases=["name"],
                           description='''
                           The name of the workload to run.
                           '''),
        ConfigurationPoint('workload_parameters', kind=obj_dict,
                           aliases=["params", "workload_params"],
                           description='''
                           Parameter to be passed to the workload
                           '''),
        ConfigurationPoint('runtime_parameters', kind=obj_dict,
                           aliases=["runtime_params"],
                           description='''
                           Runtime parameters to be set prior to running
                           the workload.
                           '''),
        ConfigurationPoint('boot_parameters', kind=obj_dict,
                           aliases=["boot_params"],
                           description='''
                           Parameters to be used when rebooting the target
                           prior to running the workload.
                           '''),
        ConfigurationPoint('label', kind=str,
                           description='''
                           Similar to IDs but do not have the uniqueness restriction.
                           If specified, labels will be used by some result
                           processes instead of (or in addition to) the workload
                           name. For example, the csv result processor will put
                           the label in the "workload" column of the CSV file.
                           '''),
        ConfigurationPoint('augmentations', kind=toggle_set, merge=True,
                           aliases=["instruments", "processors", "instrumentation",
                                    "result_processors", "augment"],
                           description='''
                           The instruments and result processors to enable (or
                           disabled using a ~) during this workload spec. This combines the
                           "instrumentation" and "result_processors" from
                           previous versions of WA (the old entries are now
                           aliases for this).
                           '''),
        ConfigurationPoint('flash', kind=dict, merge=True,
                           description='''

                           '''),
        ConfigurationPoint('classifiers', kind=dict, merge=True,
                           description='''
                           Classifiers allow you to tag metrics from this workload
                           spec to help in post processing them. Theses are often
                           used to help identify what runtime_parameters were used
                           for results when post processing.
                           '''),
    ]
    configuration = {cp.name: cp for cp in config_points}

    @classmethod
    def from_pod(cls, pod):
        job_id = pod.pop('id')
        instance = super(JobSpec, cls).from_pod(pod)
        instance.id = job_id
        return instance

    @property
    def section_id(self):
        if self.id is not None:
            self.id.rsplit('-', 1)[0]

    @property
    def workload_id(self):
        if self.id is not None:
            self.id.rsplit('-', 1)[-1]

    def __init__(self):
        super(JobSpec, self).__init__()
        self.to_merge = defaultdict(OrderedDict)
        self._sources = []
        self.id = None

    def to_pod(self):
        pod = super(JobSpec, self).to_pod()
        pod['id'] = self.id
        return pod

    def update_config(self, source, check_mandatory=True):
        self._sources.append(source)
        values = source.config
        for k, v in values.iteritems():
            if k == "id":
                continue
            elif k.endswith('_parameters'):
                if v:
                    self.to_merge[k][source] = copy(v)
            else:
                try:
                    self.set(k, v, check_mandatory=check_mandatory)
                except ConfigError as e:
                    msg = 'Error in {}:\n\t{}'
                    raise ConfigError(msg.format(source.name, e.message))

    def merge_workload_parameters(self, plugin_cache):
        # merge global generic and specific config
        workload_params = plugin_cache.get_plugin_config(self.workload_name,
                                                         generic_name="workload_parameters",
                                                         is_final=False)

        cfg_points = plugin_cache.get_plugin_parameters(self.workload_name)
        for source in self._sources:
            config = dict(self.to_merge["workload_parameters"].get(source, {}))
            if not config:
                continue

            for name, cfg_point in cfg_points.iteritems():
                if name in config:
                    value = config.pop(name)
                    cfg_point.set_value(workload_params, value,
                                        check_mandatory=False)
            if config:
                msg = 'Unexpected config "{}" for "{}"'
                raise ConfigError(msg.format(config, self.workload_name))

        self.workload_parameters = workload_params

    def merge_runtime_parameters(self, plugin_cache, target_manager):

        # Order global runtime parameters
        runtime_parameters = OrderedDict()
        try:
            global_runtime_params = plugin_cache.get_plugin_config("runtime_parameters")
        except NotFoundError:
            global_runtime_params = {}
        for source in plugin_cache.sources:
            if source in global_runtime_params:
                runtime_parameters[source] = global_runtime_params[source]

        # Add runtime parameters from JobSpec
        for source, values in self.to_merge['runtime_parameters'].iteritems():
            runtime_parameters[source] = values

        # Merge
        self.runtime_parameters = target_manager.merge_runtime_parameters(runtime_parameters)

    def finalize(self):
        self.id = "-".join([source.config['id']
                            for source in self._sources[1:]])  # ignore first id, "global"
        if self.label is None:
            self.label = self.workload_name



# This is used to construct the list of Jobs WA will run
class JobGenerator(object):

    name = "Jobs Configuration"

    @property
    def enabled_instruments(self):
        self._read_enabled_instruments = True
        return self._enabled_instruments

    @property
    def enabled_processors(self):
        self._read_enabled_processors = True
        return self._enabled_processors

    def __init__(self, plugin_cache):
        self.plugin_cache = plugin_cache
        self.ids_to_run = []
        self.sections = []
        self.workloads = []
        self._enabled_instruments = set()
        self._enabled_processors = set()
        self._read_enabled_instruments = False
        self._read_enabled_processors = False
        self.disabled_augmentations = []

        self.job_spec_template = obj_dict(not_in_dict=['name'])
        self.job_spec_template.name = "globally specified job spec configuration"
        self.job_spec_template.id = "global"
        # Load defaults
        for cfg_point in JobSpec.configuration.itervalues():
            cfg_point.set_value(self.job_spec_template, check_mandatory=False)

        self.root_node = SectionNode(self.job_spec_template)

    def set_global_value(self, name, value):
        JobSpec.configuration[name].set_value(self.job_spec_template, value,
                                              check_mandatory=False)
        if name == "augmentations":
            self.update_augmentations(value)

    def add_section(self, section, workloads):
        new_node = self.root_node.add_section(section)
        for workload in workloads:
            new_node.add_workload(workload)

    def add_workload(self, workload):
        self.root_node.add_workload(workload)

    def disable_augmentations(self, augmentations):
        #TODO: Validate
        self.disabled_augmentations = ["~{}".format(i) for i in augmentations]

    def update_augmentations(self, value):
        for entry in value:
            entry_cls = self.plugin_cache.get_plugin_class(entry)
            if entry_cls.kind == 'instrument':
                if self._read_enabled_instruments:
                    msg = "'enabled_instruments' cannot be updated after it has been accessed"
                    raise RuntimeError(msg)
                self._enabled_instruments.add(entry)
            elif entry_cls.kind == 'result_processor':
                if self._read_enabled_processors:
                    msg = "'enabled_processors' cannot be updated after it has been accessed"
                    raise RuntimeError(msg)
                self._enabled_processors.add(entry)
            else:
                msg = 'Unknown augmentation type: {}'
                raise ConfigError(msg.format(entry_cls.kind))

    def only_run_ids(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        self.ids_to_run = ids

    def generate_job_specs(self, target_manager):
        specs = []
        for leaf in self.root_node.leaves():
            workload_entries = leaf.workload_entries
            sections = [leaf]
            for ancestor in leaf.ancestors():
                workload_entries = ancestor.workload_entries + workload_entries
                sections.insert(0, ancestor)

            for workload_entry in workload_entries:
                job_spec = create_job_spec(deepcopy(workload_entry), sections,
                                           target_manager, self.plugin_cache,
                                           self.disabled_augmentations)
                if self.ids_to_run:
                    for job_id in self.ids_to_run:
                        if job_id in job_spec.id:
                            break
                    else:
                        continue
                self.update_augmentations(job_spec.augmentations.values())
                specs.append(job_spec)
        return specs


def create_job_spec(workload_entry, sections, target_manager, plugin_cache,
                    disabled_augmentations):
    job_spec = JobSpec()

    # PHASE 2.1: Merge general job spec configuration
    for section in sections:
        job_spec.update_config(section, check_mandatory=False)
    job_spec.update_config(workload_entry, check_mandatory=False)

    # PHASE 2.2: Merge global, section and workload entry "workload_parameters"
    job_spec.merge_workload_parameters(plugin_cache)

    # TODO: PHASE 2.3: Validate device runtime/boot parameters
    job_spec.merge_runtime_parameters(plugin_cache, target_manager)
    target_manager.validate_runtime_parameters(job_spec.runtime_parameters)

    # PHASE 2.4: Disable globally disabled augmentations
    job_spec.set("augmentations", disabled_augmentations)
    job_spec.finalize()

    return job_spec


def get_config_point_map(params):
    pmap = {}
    for p in params:
        pmap[p.name] = p
        for alias in p.aliases:
            pmap[alias] = p
    return pmap


settings = MetaConfiguration(os.environ)
