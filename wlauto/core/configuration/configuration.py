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
from copy import copy
from collections import OrderedDict, defaultdict
from devlib import TargetError

from wlauto.exceptions import ConfigError
from wlauto.utils.misc import (get_article, merge_config_values)
from wlauto.utils.types import (identifier, integer, boolean,
                                list_of_strings, toggle_set,
                                obj_dict)
from wlauto.core.configuration.tree import SectionNode

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
                message = 'Unexpected logging configuation parameters: {}'
                raise ValueError(message.format(bad_vals=', '.join(config.keys())))
        elif config is None:
            for k, v in self.defaults.iteritems():
                self[k] = v
        else:
            raise ValueError(config)


# Mapping for kind conversion; see docs for convert_types below
KIND_MAP = {
    int: integer,
    bool: boolean,
    dict: OrderedDict,
}


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
                 aliases=None):
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
        """
        self.name = identifier(name)
        if kind is not None and not callable(kind):
            raise ValueError('Kind must be callable.')
        if kind in KIND_MAP:
            kind = KIND_MAP[kind]
        self.kind = kind
        self.mandatory = mandatory
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

        if self.default is not None:
            try:
                self.validate_value("init", self.default)
            except ConfigError:
                raise ValueError('Default value "{}" is not valid'.format(self.default))

    def match(self, name):
        if name == self.name:
            return True
        elif name in self.aliases:
            return True
        return False

    def set_value(self, obj, value=None, check_mandatory=True):
        if value is None:
            if self.default is not None:
                value = self.default
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
            self.validate_value(obj.name, value)
        if self.merge and hasattr(obj, self.name):
            value = merge_config_values(getattr(obj, self.name), value)
        setattr(obj, self.name, value)

    def validate(self, obj):
        value = getattr(obj, self.name, None)
        if value is not None:
            self.validate_value(obj.name, value)
        else:
            if self.mandatory:
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
        return 'ConfPoint({})'.format(d)

    __str__ = __repr__


class RuntimeParameter(object):

    def __init__(self, name,
                 kind=None,
                 description=None,
                 merge=False):

        self.name = re.compile(name)
        if kind is not None:
            if not callable(kind):
                raise ValueError('Kind must be callable.')
            if kind in KIND_MAP:
                kind = KIND_MAP[kind]
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
        from params, even if they are not beign statically validated.
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


# pylint: disable=too-many-nested-blocks, too-many-branches
def merge_using_priority_specificity(generic_name, specific_name, plugin_cache):
    """
    WA configuration can come from various sources of increasing priority, as well
    as being specified in a generic and specific manner (e.g. ``device_config``
    and ``nexus10`` respectivly). WA has two rules for the priority of configuration:

        - Configuration from higher priority sources overrides configuration from
          lower priority sources.
        - More specific configuration overrides less specific configuration.

    There is a situation where these two rules come into conflict. When a generic
    configuration is given in config source of high priority and a specific
    configuration is given in a config source of lower priority. In this situation
    it is not possible to know the end users intention and WA will error.

    :param generic_name: The name of the generic configuration e.g ``device_config``
    :param specific_name: The name of the specific configuration used, e.g ``nexus10``
    :param cfg_point: A dict of ``ConfigurationPoint``s to be used when merging configuration.
                      keys=config point name, values=config point

    :rtype: A fully merged and validated configuration in the form of a obj_dict.
    """
    generic_config = plugin_cache.get_plugin_config(generic_name)
    specific_config = plugin_cache.get_plugin_config(specific_name)
    cfg_points = plugin_cache.get_plugin_parameters(specific_name)
    sources = plugin_cache.sources
    final_config = obj_dict(not_in_dict=['name'])
    seen_specific_config = defaultdict(list)

    # set_value uses the 'name' attribute of the passed object in it error
    # messages, to ensure these messages make sense the name will have to be
    # changed several times during this function.
    final_config.name = specific_name

    # Load default config
    for cfg_point in cfg_points.itervalues():
        cfg_point.set_value(final_config, check_mandatory=False)

    # pylint: disable=too-many-nested-blocks
    for source in sources:
        try:
            if source in generic_config:
                for name, cfg_point in cfg_points.iteritems():
                    final_config.name = generic_name
                    if name in generic_config[source]:
                        if name in seen_specific_config:
                            msg = ('"{generic_name}" configuration "{config_name}" has already been '
                                   'specified more specifically for {specific_name} in:\n\t\t{sources}')
                            msg = msg.format(generic_name=generic_name,
                                             config_name=name,
                                             specific_name=specific_name,
                                             sources="\n\t\t".join(seen_specific_config[name]))
                            raise ConfigError(msg)
                        value = generic_config[source].pop(name)
                        cfg_point.set_value(final_config, value, check_mandatory=False)
                if generic_config[source]:
                    msg = 'Invalid entry(ies) for "{}" in "{}": "{}"'
                    msg = msg.format(specific_name, generic_name, '", "'.join(generic_config[source]))
                    raise ConfigError(msg)

            if source in specific_config:
                final_config.name = specific_name
                for name, cfg_point in cfg_points.iteritems():
                    if name in specific_config[source]:
                        seen_specific_config[name].append(source)
                        value = specific_config[source].pop(name)
                        cfg_point.set_value(final_config, value, check_mandatory=False)
                if specific_config[source]:
                    msg = 'Invalid entry(ies) for "{}": "{}"'
                    raise ConfigError(msg.format(specific_name, '", "'.join(specific_config[source])))

        except ConfigError as e:
            raise ConfigError('Error in "{}":\n\t{}'.format(source, str(e)))

    # Validate final configuration
    final_config.name = specific_name
    for cfg_point in cfg_points.itervalues():
        cfg_point.validate(final_config)

    return final_config


class Configuration(object):

    __configuration = []
    name = ""
    # The below line must be added to all subclasses
    configuration = {cp.name: cp for cp in __configuration}

    def __init__(self):
        # Load default values for configuration points
        for confpoint in self.configuration.itervalues():
            confpoint.set_value(self, check_mandatory=False)

    def set(self, name, value, check_mandatory=True):
        if name not in self.configuration:
            raise ConfigError('Unknown {} configuration "{}"'.format(self.name, name))
        self.configuration[name].set_value(self, value, check_mandatory=check_mandatory)

    def update_config(self, values, check_mandatory=True):
        for k, v in values.iteritems():
            self.set(k, v, check_mandatory=check_mandatory)

    def validate(self):
        for cfg_point in self.configuration.itervalues():
            cfg_point.validate(self)

    def to_pod(self):
        pod = {}
        for cfg_point_name in self.configuration.iterkeys():
            value = getattr(self, cfg_point_name, None)
            if value is not None:
                pod[cfg_point_name] = value
        return pod

    @classmethod
    # pylint: disable=unused-argument
    def from_pod(cls, pod, plugin_cache):
        instance = cls()
        for name, cfg_point in cls.configuration.iteritems():
            if name in pod:
                cfg_point.set_value(instance, pod.pop(name))
        if pod:
            msg = 'Invalid entry(ies) for "{}": "{}"'
            raise ConfigError(msg.format(cls.name, '", "'.join(pod.keys())))
        instance.validate()
        return instance


# This configuration for the core WA framework
class WAConfiguration(Configuration):

    name = "WA Configuration"
    __configuration = [
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
            'plugin_packages',
            kind=list_of_strings,
            default=[
                'wlauto.commands',
                'wlauto.workloads',
                'wlauto.instrumentation',
                'wlauto.result_processors',
                'wlauto.managers',
                'wlauto.resource_getters',
            ],
            description="""
            List of packages that will be scanned for WA plugins.
            """,
        ),
        ConfigurationPoint(
            'plugin_paths',
            kind=list_of_strings,
            default=[
                'workloads',
                'instruments',
                'targets',
                'processors',

                # Legacy
                'managers',
                'result_processors',
            ],
            description="""
            List of paths that will be scanned for WA plugins.
            """,
            merge=True
        ),
        ConfigurationPoint(
            'plugin_ignore_paths',
            kind=list_of_strings,
            default=[],
            description="""
            List of (sub)paths that will be ignored when scanning
            ``plugin_paths`` for WA plugins.
            """,
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
        ConfigurationPoint(  # TODO: Needs some format for dates ect/ comes from cfg
            'default_output_directory',
            default="wa_output",
            description="""
            The default output directory that will be created if not
            specified when invoking a run.
            """,
        ),
    ]
    configuration = {cp.name: cp for cp in __configuration}

    dependencies_directory = None  # TODO: What was this for?


# This is generic top-level configuration for WA runs.
class RunConfiguration(Configuration):

    name = "Run Configuration"
    __configuration = [
        ConfigurationPoint('run_name', kind=str,  # TODO: Can only come from an agenda
                           description='''
                           A descriptive name for this WA run.
                           '''),
        ConfigurationPoint('project', kind=str,
                           description='''
                           The project this WA run belongs too.
                           '''),
        ConfigurationPoint('project_stage', kind=dict,
                           description='''
                           The stage of the project this WA run is from.
                           '''),
        ConfigurationPoint('execution_order', kind=str, default='by_iteration',
                           allowed_values=None,  # TODO:
                           description='''
                           The order that workload specs will be executed
                           '''),
        ConfigurationPoint('reboot_policy', kind=str, default='as_needed',
                           allowed_values=RebootPolicy.valid_policies,
                           description='''
                           How the device will be rebooted during the run.
                           '''),
        ConfigurationPoint('device', kind=str, mandatory=True,
                           description='''
                           The type of device this WA run will be executed on.
                           '''),
        ConfigurationPoint('retry_on_status', kind=status_list,
                           default=status_list(['FAILED', 'PARTIAL']),
                           allowed_values=None,  # TODO: - can it even be done?
                           description='''
                           Which iteration results will lead to WA retrying.
                           '''),
        ConfigurationPoint('max_retries', kind=int, default=3,
                           description='''
                           The number of times WA will attempt to retry a failed
                           iteration.
                           '''),
    ]
    configuration = {cp.name: cp for cp in __configuration}

    def __init__(self):
        super(RunConfiguration, self).__init__()
        self.device_config = None

    def merge_device_config(self, plugin_cache):
        """
        Merges global device config and validates that it is correct for the
        selected device.
        """
        # pylint: disable=no-member
        self.device_config = merge_using_priority_specificity("device_config",
                                                              self.device,
                                                              plugin_cache)

    def to_pod(self):
        pod = super(RunConfiguration, self).to_pod()
        pod['device_config'] = self.device_config
        return pod

    # pylint: disable=no-member
    @classmethod
    def from_pod(cls, pod, plugin_cache):
        try:
            device_config = obj_dict(values=pod.pop("device_config"), not_in_dict=['name'])
        except KeyError as e:
            msg = 'No value specified for mandatory parameter "{}".'
            raise ConfigError(msg.format(e.args[0]))

        instance = super(RunConfiguration, cls).from_pod(pod, plugin_cache)

        device_config.name = "device_config"
        cfg_points = plugin_cache.get_plugin_parameters(instance.device)
        for entry_name in device_config.iterkeys():
            if entry_name not in cfg_points.iterkeys():
                msg = 'Invalid entry "{}" for device "{}".'
                raise ConfigError(msg.format(entry_name, instance.device, cls.name))
            else:
                cfg_points[entry_name].validate(device_config)

        instance.device_config = device_config
        return instance


# This is the configuration for WA jobs
class JobSpec(Configuration):

    name = "Job Spec"

    __configuration = [
        ConfigurationPoint('iterations', kind=int, default=1,
                           description='''
                           How many times to repeat this workload spec
                           '''),
        ConfigurationPoint('workload_name', kind=str, mandatory=True,
                           aliases=["name"],
                           description='''
                           The name of the workload to run.
                           '''),
        ConfigurationPoint('label', kind=str,
                           description='''
                           Similar to IDs but do not have the uniqueness restriction.
                           If specified, labels will be used by some result
                           processes instead of (or in addition to) the workload
                           name. For example, the csv result processor will put
                           the label in the "workload" column of the CSV file.
                           '''),
        ConfigurationPoint('instrumentation', kind=toggle_set, merge=True,
                           aliases=["instruments"],
                           description='''
                           The instruments to enable (or disabled using a ~)
                           during this workload spec.
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
    configuration = {cp.name: cp for cp in __configuration}

    def __init__(self):
        super(JobSpec, self).__init__()
        self._to_merge = defaultdict(dict)
        self._sources = []
        self.id = None
        self.workload_parameters = None
        self.runtime_parameters = None
        self.boot_parameters = None

    def update_config(self, source, check_mandatory=True):
        self._sources.append(source)
        values = source.config
        for k, v in values.iteritems():
            if k == "id":
                continue
            elif k in ["workload_parameters", "runtime_parameters", "boot_parameters"]:
                if v:
                    self._to_merge[k][source] = copy(v)
            else:
                try:
                    self.set(k, v, check_mandatory=check_mandatory)
                except ConfigError as e:
                    msg = 'Error in {}:\n\t{}'
                    raise ConfigError(msg.format(source.name, e.message))

    # pylint: disable=no-member
    # Only call after the rest of the JobSpec is merged
    def merge_workload_parameters(self, plugin_cache):
        # merge global generic and specific config
        workload_params = merge_using_priority_specificity("workload_parameters",
                                                           self.workload_name,
                                                           plugin_cache)

        # Merge entry "workload_parameters"
        # TODO: Wrap in - "error in [agenda path]"
        cfg_points = plugin_cache.get_plugin_parameters(self.workload_name)
        for source in self._sources:
            if source in self._to_merge["workload_params"]:
                config = self._to_merge["workload_params"][source]
                for name, cfg_point in cfg_points.iteritems():
                    if name in config:
                        value = config.pop(name)
                        cfg_point.set_value(workload_params, value, check_mandatory=False)
                if config:
                    msg = 'conflicting entry(ies) for "{}" in {}: "{}"'
                    msg = msg.format(self.workload_name, source.name,
                                     '", "'.join(workload_params[source]))

        self.workload_parameters = workload_params

    def merge_runtime_parameters(self, plugin_cache, target_manager):

        # Order global runtime parameters
        runtime_parameters = OrderedDict()
        global_runtime_params = plugin_cache.get_plugin_config("runtime_parameters")
        for source in plugin_cache.sources:
            runtime_parameters[source] = global_runtime_params[source]

        # Add runtime parameters from JobSpec
        for source, values in self.to_merge['runtime_parameters'].iteritems():
            runtime_parameters[source] = values

        # Merge
        self.runtime_parameters = target_manager.merge_runtime_parameters(runtime_parameters)


    def finalize(self):
        self.id = "-".join([source.config['id'] for source in self._sources[1:]])  # ignore first id, "global"

    def to_pod(self):
        pod = super(JobSpec, self).to_pod()
        pod['workload_parameters'] = self.workload_parameters
        pod['runtime_parameters'] = self.runtime_parameters
        pod['boot_parameters'] = self.boot_parameters
        return pod

    @classmethod
    def from_pod(cls, pod, plugin_cache):
        try:
            workload_parameters = pod['workload_parameters']
            runtime_parameters = pod['runtime_parameters']
            boot_parameters = pod['boot_parameters']
        except KeyError as e:
            msg = 'No value specified for mandatory parameter "{}}".'
            raise ConfigError(msg.format(e.args[0]))

        instance = super(JobSpec, cls).from_pod(pod, plugin_loader)

        # TODO: validate parameters and construct the rest of the instance


# This is used to construct the list of Jobs WA will run
class JobGenerator(object):

    name = "Jobs Configuration"

    @property
    def enabled_instruments(self):
        self._read_enabled_instruments = True
        return self._enabled_instruments

    def update_enabled_instruments(self, value):
        if self._read_enabled_instruments:
            msg = "'enabled_instruments' cannot be updated after it has been accessed"
            raise RuntimeError(msg)
        self._enabled_instruments.update(value)

    def __init__(self, plugin_cache):
        self.plugin_cache = plugin_cache
        self.ids_to_run = []
        self.sections = []
        self.workloads = []
        self._enabled_instruments = set()
        self._read_enabled_instruments = False
        self.disabled_instruments = []

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

    def add_section(self, section, workloads):
        new_node = self.root_node.add_section(section)
        for workload in workloads:
            new_node.add_workload(workload)

    def add_workload(self, workload):
        self.root_node.add_workload(workload)

    def disable_instruments(self, instruments):
        #TODO: Validate
        self.disabled_instruments = ["~{}".format(i) for i in instruments]

    def only_run_ids(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        self.ids_to_run = ids

    def generate_job_specs(self, target_manager):

        for leaf in self.root_node.leaves():
            # PHASE 1: Gather workload and section entries for this leaf
            workload_entries = leaf.workload_entries
            sections = [leaf]
            for ancestor in leaf.ancestors():
                workload_entries = ancestor.workload_entries + workload_entries
                sections.insert(0, ancestor)

            # PHASE 2: Create job specs for this leaf
            for workload_entry in workload_entries:
                job_spec = JobSpec()  # Loads defaults

                # PHASE 2.1: Merge general job spec configuration
                for section in sections:
                    job_spec.update_config(section, check_mandatory=False)
                job_spec.update_config(workload_entry, check_mandatory=False)

                # PHASE 2.2: Merge global, section and workload entry "workload_parameters"
                job_spec.merge_workload_parameters(self.plugin_cache)
                target_manager.static_runtime_parameter_validation(job_spec.runtime_parameters)

                # TODO: PHASE 2.3: Validate device runtime/boot paramerers
                job_spec.merge_runtime_parameters(self.plugin_cache, target_manager)
                target_manager.validate_runtime_parameters(job_spec.runtime_parameters)

                # PHASE 2.4: Disable globally disabled instrumentation
                job_spec.set("instrumentation", self.disabled_instruments)
                job_spec.finalize()

                # PHASE 2.5: Skip job_spec if part of it's ID is not in self.ids_to_run
                if self.ids_to_run:
                    for job_id in self.ids_to_run:
                        if job_id in job_spec.id:
                            #TODO: logging
                            break
                    else:
                        continue

                # PHASE 2.6: Update list of instruments that need to be setup
                # pylint: disable=no-member
                self.update_enabled_instruments(job_spec.instrumentation.values())

                yield job_spec

settings = WAConfiguration()
