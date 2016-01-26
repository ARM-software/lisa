#    Copyright 2014-2015 ARM Limited
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


import os
import json
from copy import copy
from collections import OrderedDict

from wlauto.exceptions import ConfigError
from wlauto.utils.misc import merge_dicts, merge_lists, load_struct_from_file
from wlauto.utils.types import regex_type, identifier


class SharedConfiguration(object):

    def __init__(self):
        self.number_of_iterations = None
        self.workload_name = None
        self.label = None
        self.boot_parameters = OrderedDict()
        self.runtime_parameters = OrderedDict()
        self.workload_parameters = OrderedDict()
        self.instrumentation = []


class ConfigurationJSONEncoder(json.JSONEncoder):

    def default(self, obj):  # pylint: disable=E0202
        if isinstance(obj, WorkloadRunSpec):
            return obj.to_dict()
        elif isinstance(obj, RunConfiguration):
            return obj.to_dict()
        elif isinstance(obj, RebootPolicy):
            return obj.policy
        elif isinstance(obj, regex_type):
            return obj.pattern
        else:
            return json.JSONEncoder.default(self, obj)


class WorkloadRunSpec(object):
    """
    Specifies execution of a workload, including things like the number of
    iterations, device runtime_parameters configuration, etc.

    """

    # These should be handled by the framework if not explicitly specified
    # so it's a programming error if they're not
    framework_mandatory_parameters = ['id', 'number_of_iterations']

    # These *must* be specified by the user (through one mechanism or another)
    # and it is a configuration error if they're not.
    mandatory_parameters = ['workload_name']

    def __init__(self,
                 id=None,  # pylint: disable=W0622
                 number_of_iterations=None,
                 workload_name=None,
                 boot_parameters=None,
                 label=None,
                 section_id=None,
                 workload_parameters=None,
                 runtime_parameters=None,
                 instrumentation=None,
                 flash=None,
                 classifiers=None,
                 ):  # pylint: disable=W0622
        self.id = id
        self.number_of_iterations = number_of_iterations
        self.workload_name = workload_name
        self.label = label or self.workload_name
        self.section_id = section_id
        self.boot_parameters = boot_parameters or OrderedDict()
        self.runtime_parameters = runtime_parameters or OrderedDict()
        self.workload_parameters = workload_parameters or OrderedDict()
        self.instrumentation = instrumentation or []
        self.flash = flash or OrderedDict()
        self.classifiers = classifiers or OrderedDict()
        self._workload = None
        self._section = None
        self.enabled = True

    def set(self, param, value):
        if param in ['id', 'section_id', 'number_of_iterations', 'workload_name', 'label']:
            if value is not None:
                setattr(self, param, value)
        elif param in ['boot_parameters', 'runtime_parameters', 'workload_parameters', 'flash', 'classifiers']:
            setattr(self, param, merge_dicts(getattr(self, param), value, list_duplicates='last',
                                             dict_type=OrderedDict, should_normalize=False))
        elif param in ['instrumentation']:
            setattr(self, param, merge_lists(getattr(self, param), value, duplicates='last'))
        else:
            raise ValueError('Unexpected workload spec parameter: {}'.format(param))

    def validate(self):
        for param_name in self.framework_mandatory_parameters:
            param = getattr(self, param_name)
            if param is None:
                msg = '{} not set for workload spec.'
                raise RuntimeError(msg.format(param_name))
        for param_name in self.mandatory_parameters:
            param = getattr(self, param_name)
            if param is None:
                msg = '{} not set for workload spec for workload {}'
                raise ConfigError(msg.format(param_name, self.id))

    def match_selectors(self, selectors):
        """
        Returns ``True`` if this spec matches the specified selectors, and
        ``False`` otherwise. ``selectors`` must be a dict-like object with
        attribute names mapping onto selector values. At the moment, only equality
        selection is supported; i.e. the value of the attribute of the spec must
        match exactly the corresponding value specified in the ``selectors`` dict.

        """
        if not selectors:
            return True
        for k, v in selectors.iteritems():
            if getattr(self, k, None) != v:
                return False
        return True

    @property
    def workload(self):
        if self._workload is None:
            raise RuntimeError("Workload for {} has not been loaded".format(self))
        return self._workload

    @property
    def secition(self):
        if self.section_id and self._section is None:
            raise RuntimeError("Section for {} has not been loaded".format(self))
        return self._section

    def load(self, device, ext_loader):
        """Loads the workload for the specified device using the specified loader.
        This must be done before attempting to execute the spec."""
        self._workload = ext_loader.get_workload(self.workload_name, device, **self.workload_parameters)

    def to_dict(self):
        d = copy(self.__dict__)
        del d['_workload']
        del d['_section']
        return d

    def copy(self):
        other = WorkloadRunSpec()
        other.id = self.id
        other.number_of_iterations = self.number_of_iterations
        other.workload_name = self.workload_name
        other.label = self.label
        other.section_id = self.section_id
        other.boot_parameters = copy(self.boot_parameters)
        other.runtime_parameters = copy(self.runtime_parameters)
        other.workload_parameters = copy(self.workload_parameters)
        other.instrumentation = copy(self.instrumentation)
        other.flash = copy(self.flash)
        other.classifiers = copy(self.classifiers)
        other._section = self._section  # pylint: disable=protected-access
        other.enabled = self.enabled
        return other

    def __str__(self):
        return '{} {}'.format(self.id, self.label)

    def __cmp__(self, other):
        if not isinstance(other, WorkloadRunSpec):
            return cmp('WorkloadRunSpec', other.__class__.__name__)
        return cmp(self.id, other.id)


class _SpecConfig(object):
    # TODO: This is a bit of HACK for alias resolution. This formats Alias
    #       params as if they came from config.

    def __init__(self, name, params=None):
        setattr(self, name, params or {})


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


class RunConfigurationItem(object):
    """
    This represents a predetermined "configuration point" (an individual setting)
    and describes how it must be handled when encountered.

    """

    # Also defines the NULL value for each category
    valid_categories = {
        'scalar': None,
        'list': [],
        'dict': {},
    }

    # A callable that takes an arbitrary number of positional arguments
    # is also valid.
    valid_methods = ['keep', 'replace', 'merge']

    def __init__(self, name, category, method):
        if category not in self.valid_categories:
            raise ValueError('Invalid category: {}'.format(category))
        if not callable(method) and method not in self.valid_methods:
            raise ValueError('Invalid method: {}'.format(method))
        if category == 'scalar' and method == 'merge':
            raise ValueError('Method cannot be "merge" for a scalar')
        self.name = name
        self.category = category
        self.method = method

    def combine(self, *args):
        """
        Combine the provided values according to the method for this
        configuration item. Order matters -- values are assumed to be
        in the order they were specified by the user. The resulting value
        is also checked to patch the specified type.

        """
        args = [a for a in args if a is not None]
        if not args:
            return self.valid_categories[self.category]
        if self.method == 'keep' or len(args) == 1:
            value = args[0]
        elif self.method == 'replace':
            value = args[-1]
        elif self.method == 'merge':
            if self.category == 'list':
                value = merge_lists(*args, duplicates='last', dict_type=OrderedDict)
            elif self.category == 'dict':
                value = merge_dicts(*args,
                                    should_merge_lists=True,
                                    should_normalize=False,
                                    list_duplicates='last',
                                    dict_type=OrderedDict)
            else:
                raise ValueError('Unexpected category for merge : "{}"'.format(self.category))
        elif callable(self.method):
            value = self.method(*args)
        else:
            raise ValueError('Unexpected method: "{}"'.format(self.method))

        return value


def _combine_ids(*args):
    return '_'.join(args)


class status_list(list):

    def append(self, item):
        list.append(self, str(item).upper())


class RunConfiguration(object):
    """
    Loads and maintains the unified configuration for this run. This includes configuration
    for WA execution as a whole, and parameters for specific specs.

    WA configuration mechanism aims to be flexible and easy to use, while at the same
    time providing storing validation and early failure on error. To meet these requirements,
    the implementation gets rather complicated. This is going to be a quick overview of
    the underlying mechanics.

    .. note:: You don't need to know this to use WA, or to write extensions for it. From
              the point of view of extension writers, configuration from various sources
              "magically" appears as attributes of their classes. This explanation peels
              back the curtain and is intended for those who, for one reason or another,
              need to understand how the magic works.

    **terminology**

    run

        A single execution of a WA agenda.

    run config(uration) (object)

        An instance of this class. There is one per run.

    config(uration) item

        A single configuration entry or "setting", e.g. the device interface to use. These
        can be for the run as a whole, or for a specific extension.

    (workload) spec

        A specification of a single workload execution. This combines workload configuration
        with things like the number of iterations to run, which instruments to enable, etc.
        More concretely, this is an instance of :class:`WorkloadRunSpec`.

    **overview**

    There are three types of WA configuration:

        1. "Meta" configuration that determines how the rest of the configuration is
           processed (e.g. where extensions get loaded from). Since this does not pertain
           to *run* configuration, it will not be covered further.
        2. Global run configuration, e.g. which workloads, result processors and instruments
           will be enabled for a run.
        3. Per-workload specification configuration, that determines how a particular workload
           instance will get executed (e.g. what workload parameters will be used, how many
           iterations.

    **run configuration**

    Run configuration may appear in a config file (usually ``~/.workload_automation/config.py``),
    or in the ``config`` section of an agenda. Configuration is specified as a nested structure
    of dictionaries (associative arrays, or maps) and lists in the syntax following the format
    implied by the file extension (currently, YAML and Python are supported). If the same
    configuration item appears in more than one source, they are merged with conflicting entries
    taking the value from the last source that specified them.

    In addition to a fixed set of global configuration items, configuration for any WA
    Extension (instrument, result processor, etc) may also be specified, namespaced under
    the extension's name (i.e. the extensions name is a key in the global config with value
    being a dict of parameters and their values). Some Extension parameters also specify a
    "global alias" that may appear at the top-level of the config rather than under the
    Extension's name. It is *not* an error to specify configuration for an Extension that has
    not been enabled for a particular run; such configuration will be ignored.


    **per-workload configuration**

    Per-workload configuration can be specified in three places in the agenda: the
    workload entry in the ``workloads`` list, the ``global`` entry (configuration there
    will be applied to every workload entry), and in a section entry in ``sections`` list
    ( configuration in every section will be applied to every workload entry separately,
    creating a "cross-product" of section and workload configurations; additionally,
    sections may specify their own workload lists).

    If they same configuration item appears in more than one of the above places, they will
    be merged in the following order: ``global``, ``section``, ``workload``, with conflicting
    scalar values in the later overriding those from previous locations.


    **Global parameter aliases**

    As mentioned above, an Extension's parameter may define a global alias, which will be
    specified and picked up from the top-level config, rather than config for that specific
    extension. It is an error to specify the value for a parameter both through a global
    alias and through extension config dict in the same configuration file. It is, however,
    possible to use a global alias in one file, and specify extension configuration for the
    same parameter in another file, in which case, the usual merging rules would apply.

    **Loading and validation of configuration**

    Validation of user-specified configuration happens at several stages of run initialisation,
    to ensure that appropriate context for that particular type of validation is available and
    that meaningful errors can be reported, as early as is feasible.

    - Syntactic validation is performed when configuration is first loaded.
      This is done by the loading mechanism (e.g. YAML parser), rather than WA itself. WA
      propagates any errors encountered as ``ConfigError``\ s.
    - Once a config file is loaded into a Python structure, it scanned to
      extract settings. Static configuration is validated and added to the config. Extension
      configuration is collected into a collection of "raw" config, and merged as appropriate, but
      is not processed further at this stage.
    - Once all configuration sources have been processed, the configuration as a whole
      is validated (to make sure there are no missing settings, etc).
    - Extensions are loaded through the run config object, which instantiates
      them with appropriate parameters based on the "raw" config collected earlier. When an
      Extension is instantiated in such a way, its config is "officially" added to run configuration
      tracked by the run config object. Raw config is discarded at the end of the run, so
      that any config that wasn't loaded in this way is not recoded (as it was not actually used).
    - Extension parameters a validated individually (for type, value ranges, etc) as they are
      loaded in the Extension's __init__.
    - An extension's ``validate()`` method is invoked before it is used (exactly when this
      happens depends on the extension's type) to perform any final validation *that does not
      rely on the target being present* (i.e. this would happen before WA connects to the target).
      This can be used perform inter-parameter validation for an extension (e.g. when valid range for
      one parameter depends on another), and more general WA state assumptions (e.g. a result
      processor can check that an instrument it depends on has been installed).
    - Finally, it is the responsibility of individual extensions to validate any assumptions
      they make about the target device (usually as part of their ``setup()``).

    **Handling of Extension aliases.**

    WA extensions can have zero or more aliases (not to be confused with global aliases for extension
    *parameters*). An extension allows associating an alternative name for the extension with a set
    of parameter values. In other words aliases associate common configurations for an extension with
    a name, providing a shorthand for it. For example, "t-rex_offscreen" is an alias for "glbenchmark"
    workload that specifies that "use_case" should be "t-rex" and "variant" should be "offscreen".

    **special loading rules**

    Note that as a consequence of being able to specify configuration for *any* Extension namespaced
    under the Extension's name in the top-level config, two distinct mechanisms exist form configuring
    devices and workloads. This is valid, however due to their nature, they are handled in a special way.
    This may be counter intuitive, so configuration of devices and workloads creating entries for their
    names in the config is discouraged in favour of using the "normal" mechanisms of configuring them
    (``device_config`` for devices and workload specs in the agenda for workloads).

    In both cases (devices and workloads), "normal" config will always override named extension config
    *irrespective of which file it was specified in*. So a ``adb_name`` name specified in ``device_config``
    inside ``~/.workload_automation/config.py`` will override ``adb_name`` specified for ``juno`` in the
    agenda (even when device is set to "juno").

    Again, this ignores normal loading rules, so the use of named extension configuration for devices
    and workloads is discouraged. There maybe some situations where this behaviour is useful however
    (e.g. maintaining configuration for different devices in one config file).

    """

    default_reboot_policy = 'as_needed'
    default_execution_order = 'by_iteration'

    # This is generic top-level configuration.
    general_config = [
        RunConfigurationItem('run_name', 'scalar', 'replace'),
        RunConfigurationItem('project', 'scalar', 'replace'),
        RunConfigurationItem('project_stage', 'dict', 'replace'),
        RunConfigurationItem('execution_order', 'scalar', 'replace'),
        RunConfigurationItem('reboot_policy', 'scalar', 'replace'),
        RunConfigurationItem('device', 'scalar', 'replace'),
        RunConfigurationItem('flashing_config', 'dict', 'replace'),
        RunConfigurationItem('retry_on_status', 'list', 'replace'),
        RunConfigurationItem('max_retries', 'scalar', 'replace'),
    ]

    # Configuration specified for each workload spec. "workload_parameters"
    # aren't listed because they are handled separately.
    workload_config = [
        RunConfigurationItem('id', 'scalar', _combine_ids),
        RunConfigurationItem('number_of_iterations', 'scalar', 'replace'),
        RunConfigurationItem('workload_name', 'scalar', 'replace'),
        RunConfigurationItem('label', 'scalar', 'replace'),
        RunConfigurationItem('section_id', 'scalar', 'replace'),
        RunConfigurationItem('boot_parameters', 'dict', 'merge'),
        RunConfigurationItem('runtime_parameters', 'dict', 'merge'),
        RunConfigurationItem('instrumentation', 'list', 'merge'),
        RunConfigurationItem('flash', 'dict', 'merge'),
        RunConfigurationItem('classifiers', 'dict', 'merge'),
    ]

    # List of names that may be present in configuration (and it is valid for
    # them to be there) but are not handled buy RunConfiguration.
    ignore_names = ['logging', 'remote_assets_mount_point']

    def get_reboot_policy(self):
        if not self._reboot_policy:
            self._reboot_policy = RebootPolicy(self.default_reboot_policy)
        return self._reboot_policy

    def set_reboot_policy(self, value):
        if isinstance(value, RebootPolicy):
            self._reboot_policy = value
        else:
            self._reboot_policy = RebootPolicy(value)

    reboot_policy = property(get_reboot_policy, set_reboot_policy)

    @property
    def all_instrumentation(self):
        result = set()
        for spec in self.workload_specs:
            result = result.union(set(spec.instrumentation))
        return result

    def __init__(self, ext_loader):
        self.ext_loader = ext_loader
        self.device = None
        self.device_config = None
        self.execution_order = None
        self.project = None
        self.project_stage = None
        self.run_name = None
        self.instrumentation = {}
        self.result_processors = {}
        self.workload_specs = []
        self.flashing_config = {}
        self.other_config = {}  # keeps track of used config for extensions other than of the four main kinds.
        self.retry_on_status = status_list(['FAILED', 'PARTIAL'])
        self.max_retries = 3
        self._used_config_items = []
        self._global_instrumentation = []
        self._reboot_policy = None
        self._agenda = None
        self._finalized = False
        self._general_config_map = {i.name: i for i in self.general_config}
        self._workload_config_map = {i.name: i for i in self.workload_config}
        # Config files may contains static configuration for extensions that
        # would not be part of this of this run (e.g. DB connection settings
        # for a result processor that has not been enabled). Such settings
        # should not be part of configuration for this run (as they will not
        # be affecting it), but we still need to keep track it in case a later
        # config (e.g. from the agenda) enables the extension.
        # For this reason, all extension config is first loaded into the
        # following dict and when an extension is identified as need for the
        # run, its config is picked up from this "raw" dict and it becomes part
        # of the run configuration.
        self._raw_config = {'instrumentation': [], 'result_processors': []}

    def get_extension(self, ext_name, *args):
        self._check_finalized()
        self._load_default_config_if_necessary(ext_name)
        ext_config = self._raw_config[ext_name]
        ext_cls = self.ext_loader.get_extension_class(ext_name)
        if ext_cls.kind not in ['workload', 'device', 'instrument', 'result_processor']:
            self.other_config[ext_name] = ext_config
        return self.ext_loader.get_extension(ext_name, *args, **ext_config)

    def to_dict(self):
        d = copy(self.__dict__)
        to_remove = ['ext_loader', 'workload_specs'] + [k for k in d.keys() if k.startswith('_')]
        for attr in to_remove:
            del d[attr]
        d['workload_specs'] = [s.to_dict() for s in self.workload_specs]
        d['reboot_policy'] = self.reboot_policy  # this is a property so not in __dict__
        return d

    def load_config(self, source):
        """Load configuration from the specified source. The source must be
        either a path to a valid config file or a dict-like object. Currently,
        config files can be either python modules (.py extension) or YAML documents
        (.yaml extension)."""
        if self._finalized:
            raise ValueError('Attempting to load a config file after run configuration has been finalized.')
        try:
            config_struct = _load_raw_struct(source)
            self._merge_config(config_struct)
        except ConfigError as e:
            message = 'Error in {}:\n\t{}'
            raise ConfigError(message.format(getattr(source, 'name', None), e.message))

    def set_agenda(self, agenda, selectors=None):
        """Set the agenda for this run; Unlike with config files, there can only be one agenda."""
        if self._agenda:
            # note: this also guards against loading an agenda after finalized() has been called,
            #       as that would have required an agenda to be set.
            message = 'Attempting to set a second agenda {};\n\talready have agenda {} set'
            raise ValueError(message.format(agenda.filepath, self._agenda.filepath))
        try:
            self._merge_config(agenda.config or {})
            self._load_specs_from_agenda(agenda, selectors)
            self._agenda = agenda
        except ConfigError as e:
            message = 'Error in {}:\n\t{}'
            raise ConfigError(message.format(agenda.filepath, e.message))

    def finalize(self):
        """This must be invoked once all configuration sources have been loaded. This will
        do the final processing, setting instrumentation and result processor configuration
        for the run And making sure that all the mandatory config has been specified."""
        if self._finalized:
            return
        if not self._agenda:
            raise ValueError('Attempting to finalize run configuration before an agenda is loaded.')
        self._finalize_config_list('instrumentation')
        self._finalize_config_list('result_processors')
        if not self.device:
            raise ConfigError('Device not specified in the config.')
        self._finalize_device_config()
        if not self.reboot_policy.reboot_on_each_spec:
            for spec in self.workload_specs:
                if spec.boot_parameters:
                    message = 'spec {} specifies boot_parameters; reboot policy must be at least "each_spec"'
                    raise ConfigError(message.format(spec.id))
        for spec in self.workload_specs:
            for globinst in self._global_instrumentation:
                if globinst not in spec.instrumentation:
                    spec.instrumentation.append(globinst)
            spec.validate()
        self._finalized = True

    def serialize(self, wfh):
        json.dump(self, wfh, cls=ConfigurationJSONEncoder, indent=4)

    def _merge_config(self, config):
        """
        Merge the settings specified by the ``config`` dict-like object into current
        configuration.

        """
        if not isinstance(config, dict):
            raise ValueError('config must be a dict; found {}'.format(config.__class__.__name__))

        for k, v in config.iteritems():
            k = identifier(k)
            if k in self.ext_loader.global_param_aliases:
                self._resolve_global_alias(k, v)
            elif k in self._general_config_map:
                self._set_run_config_item(k, v)
            elif self.ext_loader.has_extension(k):
                self._set_extension_config(k, v)
            elif k == 'device_config':
                self._set_raw_dict(k, v)
            elif k in ['instrumentation', 'result_processors']:
                # Instrumentation can be enabled and disabled by individual
                # workloads, so we need to track it in two places: a list of
                # all instruments for the run (as they will all need to be
                # initialized and installed, and a list of only the "global"
                # instruments which can then be merged into instrumentation
                # lists of individual workload specs.
                self._set_raw_list('_global_{}'.format(k), v)
                self._set_raw_list(k, v)
            elif k in self.ignore_names:
                pass
            else:
                raise ConfigError('Unknown configuration option: {}'.format(k))

    def _resolve_global_alias(self, name, value):
        ga = self.ext_loader.global_param_aliases[name]
        for param, ext in ga.iteritems():
            for name in [ext.name] + [a.name for a in ext.aliases]:
                self._load_default_config_if_necessary(name)
                self._raw_config[identifier(name)][param.name] = value

    def _set_run_config_item(self, name, value):
        item = self._general_config_map[name]
        combined_value = item.combine(getattr(self, name, None), value)
        setattr(self, name, combined_value)

    def _set_extension_config(self, name, value):
        default_config = self.ext_loader.get_default_config(name)
        self._set_raw_dict(name, value, default_config)

    def _set_raw_dict(self, name, value, default_config=None):
        existing_config = self._raw_config.get(name, default_config or {})
        new_config = _merge_config_dicts(existing_config, value)
        self._raw_config[identifier(name)] = new_config

    def _set_raw_list(self, name, value):
        old_value = self._raw_config.get(name, [])
        new_value = merge_lists(old_value, value, duplicates='last')
        self._raw_config[identifier(name)] = new_value

    def _finalize_config_list(self, attr_name):
        """Note: the name is somewhat misleading. This finalizes a list
        form the specified configuration (e.g. "instrumentation"); internal
        representation is actually a dict, not a list..."""
        ext_config = {}
        raw_list = self._raw_config.get(attr_name, [])
        for extname in raw_list:
            default_config = self.ext_loader.get_default_config(extname)
            ext_config[extname] = self._raw_config.get(identifier(extname), default_config)
        list_name = '_global_{}'.format(attr_name)
        global_list = self._raw_config.get(list_name, [])
        setattr(self, list_name, global_list)
        setattr(self, attr_name, ext_config)

    def _finalize_device_config(self):
        self._load_default_config_if_necessary(self.device)
        config = _merge_config_dicts(self._raw_config.get(self.device, {}),
                                     self._raw_config.get('device_config', {}),
                                     list_duplicates='all')
        self.device_config = config

    def _load_default_config_if_necessary(self, name):
        name = identifier(name)
        if name not in self._raw_config:
            self._raw_config[name] = self.ext_loader.get_default_config(name)

    def _load_specs_from_agenda(self, agenda, selectors):
        global_dict = agenda.global_.to_dict() if agenda.global_ else {}
        if agenda.sections:
            for section_entry in agenda.sections:
                section_dict = section_entry.to_dict()
                for workload_entry in agenda.workloads + section_entry.workloads:
                    workload_dict = workload_entry.to_dict()
                    self._load_workload_spec(global_dict, section_dict, workload_dict, selectors)
        else:  # no sections were specified
            for workload_entry in agenda.workloads:
                workload_dict = workload_entry.to_dict()
                self._load_workload_spec(global_dict, {}, workload_dict, selectors)

    def _load_workload_spec(self, global_dict, section_dict, workload_dict, selectors):
        spec = WorkloadRunSpec()
        for name, config in self._workload_config_map.iteritems():
            value = config.combine(global_dict.get(name), section_dict.get(name), workload_dict.get(name))
            spec.set(name, value)
        if section_dict:
            spec.set('section_id', section_dict.get('id'))

        realname, alias_config = self.ext_loader.resolve_alias(spec.workload_name)
        if not spec.label:
            spec.label = spec.workload_name
        spec.workload_name = realname
        dicts = [self.ext_loader.get_default_config(realname),
                 alias_config,
                 self._raw_config.get(spec.workload_name),
                 global_dict.get('workload_parameters'),
                 section_dict.get('workload_parameters'),
                 workload_dict.get('workload_parameters')]
        dicts = [d for d in dicts if d is not None]
        value = _merge_config_dicts(*dicts)
        spec.set('workload_parameters', value)

        if not spec.number_of_iterations:
            spec.number_of_iterations = 1

        if spec.match_selectors(selectors):
            instrumentation_config = self._raw_config['instrumentation']
            for instname in spec.instrumentation:
                if instname not in instrumentation_config:
                    instrumentation_config.append(instname)
            self.workload_specs.append(spec)

    def _check_finalized(self):
        if not self._finalized:
            raise ValueError('Attempting to access configuration before it has been finalized.')


def _load_raw_struct(source):
    """Load a raw dict config structure from the specified source."""
    if isinstance(source, basestring):
        if os.path.isfile(source):
            raw = load_struct_from_file(filepath=source)
        else:
            raise ConfigError('File "{}" does not exit'.format(source))
    elif isinstance(source, dict):
        raw = source
    else:
        raise ConfigError('Unknown config source: {}'.format(source))
    return raw


def _merge_config_dicts(*args, **kwargs):
    """Provides a different set of default settings for ```merge_dicts`` """
    return merge_dicts(*args,
                       should_merge_lists=kwargs.get('should_merge_lists', False),
                       should_normalize=kwargs.get('should_normalize', False),
                       list_duplicates=kwargs.get('list_duplicates', 'last'),
                       dict_type=kwargs.get('dict_type', OrderedDict))
