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
from copy import copy
from collections import OrderedDict
import logging
import shutil
from glob import glob
from itertools import chain

from wlauto.exceptions import ConfigError
from wlauto.utils.misc import merge_dicts, merge_lists, load_struct_from_file
from wlauto.utils.types import regex_type, identifier, integer, boolean, list_of_strings
from wlauto.core import pluginloader
from wlauto.utils.serializer import json, WAJSONEncoder
from wlauto.exceptions import ConfigError
from wlauto.utils.misc import isiterable, get_article
from wlauto.utils.serializer import read_pod, yaml


class ConfigurationPoint(object):
    """
    This defines a gneric configuration point for workload automation. This is
    used to handle global settings, plugin parameters, etc.

    """

    # Mapping for kind conversion; see docs for convert_types below
    kind_map = {
        int: integer,
        bool: boolean,
    }

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
                 convert_types=True):
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
        :param convert_types: If ``True`` (the default), will automatically
                              convert ``kind`` values from native Python
                              types to WA equivalents. This allows more
                              ituitive interprestation of parameter values,
                              e.g. the string ``"false"`` being interpreted
                              as ``False`` when specifed as the value for
                              a boolean Parameter.

        """
        self.name = identifier(name)
        if kind is not None and not callable(kind):
            raise ValueError('Kind must be callable.')
        if convert_types and kind in self.kind_map:
            kind = self.kind_map[kind]
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

    def match(self, name):
        if name == self.name:
            return True
        elif name in self.aliases:
            return True
        return False

    def set_value(self, obj, value=None):
        if value is None:
            if self.default is not None:
                value = self.default
            elif self.mandatory:
                msg = 'No values specified for mandatory parameter {} in {}'
                raise ConfigError(msg.format(self.name, obj.name))
        else:
            try:
                value = self.kind(value)
            except (ValueError, TypeError):
                typename = self.get_type_name()
                msg = 'Bad value "{}" for {}; must be {} {}'
                article = get_article(typename)
                raise ConfigError(msg.format(value, self.name, article, typename))
        if self.merge and hasattr(obj, self.name):
            value = merge_config_values(getattr(obj, self.name), value)
        setattr(obj, self.name, value)

    def validate(self, obj):
        value = getattr(obj, self.name, None)

        if value is not None:
            if self.allowed_values:
                self._validate_allowed_values(obj, value)
            if self.constraint:
                self._validate_constraint(obj, value)
        else:
            if self.mandatory:
                msg = 'No value specified for mandatory parameter {} in {}.'
                raise ConfigError(msg.format(self.name, obj.name))

    def get_type_name(self):
        typename = str(self.kind)
        if '\'' in typename:
            typename = typename.split('\'')[1]
        elif typename.startswith('<function'):
            typename = typename.split()[1]
        return typename

    def _validate_allowed_values(self, obj, value):
        if 'list' in str(self.kind):
            for v in value:
                if v not in self.allowed_values:
                    msg = 'Invalid value {} for {} in {}; must be in {}'
                    raise ConfigError(msg.format(v, self.name, obj.name, self.allowed_values))
        else:
            if value not in self.allowed_values:
                msg = 'Invalid value {} for {} in {}; must be in {}'
                raise ConfigError(msg.format(value, self.name, obj.name, self.allowed_values))

    def _validate_constraint(self, obj, value):
        msg_vals = {'value': value, 'param': self.name, 'plugin': obj.name}
        if isinstance(self.constraint, tuple) and len(self.constraint) == 2:
            constraint, msg = self.constraint  # pylint: disable=unpacking-non-sequence
        elif callable(self.constraint):
            constraint = self.constraint
            msg = '"{value}" failed constraint validation for {param} in {plugin}.'
        else:
            raise ValueError('Invalid constraint for {}: must be callable or a 2-tuple'.format(self.name))
        if not constraint(value):
            raise ConfigError(value, msg.format(**msg_vals))

    def __repr__(self):
        d = copy(self.__dict__)
        del d['description']
        return 'ConfPoint({})'.format(d)

    __str__ = __repr__


class ConfigurationPointCollection(object):

    def __init__(self):
        self._configs = []
        self._config_map = {}

    def get(self, name, default=None):
        return self._config_map.get(name, default)

    def add(self, point):
        if not isinstance(point, ConfigurationPoint):
            raise ValueError('Mustbe a ConfigurationPoint, got {}'.format(point.__class__))
        existing = self.get(point.name)
        if existing:
            if point.override:
                new_point = copy(existing)
                for a, v in point.__dict__.iteritems():
                    if v is not None:
                        setattr(new_point, a, v)
                self.remove(existing)
                point = new_point
            else:
                raise ValueError('Duplicate ConfigurationPoint "{}"'.format(point.name))
        self._add(point)

    def remove(self, point):
        self._configs.remove(point)
        del self._config_map[point.name]
        for alias in point.aliases:
            del self._config_map[alias]

    append = add

    def _add(self, point):
        self._configs.append(point)
        self._config_map[point.name] = point
        for alias in point.aliases:
            if alias in self._config_map:
                message = 'Clashing alias "{}" between "{}" and "{}"'
                raise ValueError(message.format(alias, point.name,
                                                self._config_map[alias].name))

    def __str__(self):
        str(self._configs)

    __repr__ = __str__

    def __iadd__(self, other):
        for p in other:
            self.add(p)
        return self

    def __iter__(self):
        return iter(self._configs)

    def __contains__(self, p):
        if isinstance(p, basestring):
            return p in self._config_map
        return p.name in self._config_map

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._configs[i]
        return self._config_map[i]

    def __len__(self):
        return len(self._configs)


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


__WA_CONFIGURATION = [
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
    ConfigurationPoint(
        'default_output_directory',
        default="wa_output",
        description="""
        The default output directory that will be created if not
        specified when invoking a run.
        """,
    ),
]

WA_CONFIGURATION = {cp.name: cp for cp in __WA_CONFIGURATION}

ENVIRONMENT_VARIABLES = {
    'WA_USER_DIRECTORY': WA_CONFIGURATION['user_directory'],
    'WA_PLUGIN_PATHS': WA_CONFIGURATION['plugin_paths'],
    'WA_EXTENSION_PATHS': WA_CONFIGURATION['plugin_paths'],  # plugin_paths (legacy)
}


class WAConfiguration(object):
    """
    This is configuration for Workload Automation framework as a whole. This
    does not track configuration for WA runs. Rather, this tracks "meta"
    configuration, such as various locations WA looks for things, logging
    configuration etc.

    """

    basename = 'config'

    @property
    def dependencies_directory(self):
        return os.path.join(self.user_directory, 'dependencies')

    def __init__(self):
        self.user_directory = ''
        self.plugin_packages = []
        self.plugin_paths = []
        self.plugin_ignore_paths = []
        self.config_paths = []
        self.logging = {}
        self._logger = logging.getLogger('settings')
        for confpoint in WA_CONFIGURATION.itervalues():
            confpoint.set_value(self)

    def load_environment(self):
        for name, confpoint in ENVIRONMENT_VARIABLES.iteritems():
            value = os.getenv(name)
            if value:
                confpoint.set_value(self, value)
        self._expand_paths()

    def load_config_file(self, path):
        self.load(read_pod(path))
        if path not in self.config_paths:
            self.config_paths.append(path)

    def load_user_config(self):
        globpath = os.path.join(self.user_directory, '{}.*'.format(self.basename))
        for path in glob(globpath):
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.pyc', '.pyo', '.py~']:
                continue
            self.load_config_file(path)

    def load(self, config):
        for name, value in config.iteritems():
            if name in WA_CONFIGURATION:
                confpoint = WA_CONFIGURATION[name]
                confpoint.set_value(self, value)
        self._expand_paths()

    def set(self, name, value):
        if name not in WA_CONFIGURATION:
            raise ConfigError('Unknown WA configuration "{}"'.format(name))
        WA_CONFIGURATION[name].set_value(self, value)

    def initialize_user_directory(self, overwrite=False):
        """
        Initialize a fresh user environment creating the workload automation.

        """
        if os.path.exists(self.user_directory):
            if not overwrite:
                raise ConfigError('Environment {} already exists.'.format(self.user_directory))
            shutil.rmtree(self.user_directory)

        self._expand_paths()
        os.makedirs(self.dependencies_directory)
        for path in self.plugin_paths:
            os.makedirs(path)

        with open(os.path.join(self.user_directory, 'config.yaml'), 'w') as _:
            yaml.dump(self.to_pod())

        if os.getenv('USER') == 'root':
            # If running with sudo on POSIX, change the ownership to the real user.
            real_user = os.getenv('SUDO_USER')
            if real_user:
                import pwd  # done here as module won't import on win32
                user_entry = pwd.getpwnam(real_user)
                uid, gid = user_entry.pw_uid, user_entry.pw_gid
                os.chown(self.user_directory, uid, gid)
                # why, oh why isn't there a recusive=True option for os.chown?
                for root, dirs, files in os.walk(self.user_directory):
                    for d in dirs:
                        os.chown(os.path.join(root, d), uid, gid)
                    for f in files:
                        os.chown(os.path.join(root, f), uid, gid)

    @staticmethod
    def from_pod(pod):
        instance = WAConfiguration()
        instance.load(pod)
        return instance

    def to_pod(self):
        return dict(
            user_directory=self.user_directory,
            plugin_packages=self.plugin_packages,
            plugin_paths=self.plugin_paths,
            plugin_ignore_paths=self.plugin_ignore_paths,
            logging=self.logging,
        )

    def _expand_paths(self):
        expanded = []
        for path in self.plugin_paths:
            path = os.path.expanduser(path)
            path = os.path.expandvars(path)
            expanded.append(os.path.join(self.user_directory, path))
        self.plugin_paths = expanded
        expanded = []
        for path in self.plugin_ignore_paths:
            path = os.path.expanduser(path)
            path = os.path.expandvars(path)
            expanded.append(os.path.join(self.user_directory, path))
        self.plugin_ignore_paths = expanded


class PluginConfiguration(object):
    """ Maintains a mapping of plugin_name --> plugin_config. """

    def __init__(self, loader=pluginloader):
        self.loader = loader
        self.config = {}

    def update(self, name, config):
        if not hasattr(config, 'get'):
            raise ValueError('config must be a dict-like object got: {}'.format(config))
        name, alias_config = self.loader.resolve_alias(name)
        existing_config = self.config.get(name)
        if existing_config is None:
            existing_config = alias_config

        new_config = config or {}
        self.config[name] = merge_config_values(existing_config, new_config)


def merge_config_values(base, other):
    """
    This is used to merge two objects, typically when setting the value of a
    ``ConfigurationPoint``. First, both objects are categorized into

        c: A scalar value. Basically, most objects. These values
           are treated as atomic, and not mergeable.
        s: A sequence. Anything iterable that is not a dict or
           a string (strings are considered scalars).
        m: A key-value mapping. ``dict`` and it's derivatives.
        n: ``None``.
        o: A mergeable object; this is an object that implements both
          ``merge_with`` and ``merge_into`` methods.

    The merge rules based on the two categories are then as follows:

        (c1, c2) --> c2
        (s1, s2) --> s1 . s2
        (m1, m2) --> m1 . m2
        (c, s) --> [c] . s
        (s, c) --> s . [c]
        (s, m) --> s . [m]
        (m, s) --> [m] . s
        (m, c) --> ERROR
        (c, m) --> ERROR
        (o, X) --> o.merge_with(X)
        (X, o) --> o.merge_into(X)
        (X, n) --> X
        (n, X) --> X

    where:

        '.'  means concatenation (for maps, contcationation of (k, v) streams
             then converted back into a map). If the types of the two objects
             differ, the type of ``other`` is used for the result.
        'X'  means "any category"
        '[]' used to indicate a literal sequence (not necessarily a ``list``).
             when this is concatenated with an actual sequence, that sequencies
             type is used.

    notes:

        - When a mapping is combined with a sequence, that mapping is
          treated as a scalar value.
        - When combining two mergeable objects, they're combined using
          ``o1.merge_with(o2)`` (_not_ using o2.merge_into(o1)).
        - Combining anything with ``None`` yields that value, irrespective
          of the order. So a ``None`` value is eqivalent to the corresponding
          item being omitted.
        - When both values are scalars, merging is equivalent to overwriting.
        - There is no recursion (e.g. if map values are lists, they will not
          be merged; ``other`` will overwrite ``base`` values). If complicated
          merging semantics (such as recursion) are required, they should be
          implemented within custom mergeable types (i.e. those that implement
          ``merge_with`` and ``merge_into``).

    While this can be used as a generic "combine any two arbitry objects"
    function, the semantics have been selected specifically for merging
    configuration point values.

    """
    cat_base = categorize(base)
    cat_other = categorize(other)

    if cat_base == 'n':
        return other
    elif cat_other == 'n':
        return base

    if cat_base == 'o':
        return base.merge_with(other)
    elif cat_other == 'o':
        return other.merge_into(base)

    if cat_base == 'm':
        if cat_other == 's':
            return merge_sequencies([base], other)
        elif cat_other == 'm':
            return merge_maps(base, other)
        else:
            message = 'merge error ({}, {}): "{}" and "{}"'
            raise ValueError(message.format(cat_base, cat_other, base, other))
    elif cat_base == 's':
        if cat_other == 's':
            return merge_sequencies(base, other)
        else:
            return merge_sequencies(base, [other])
    else:  # cat_base == 'c'
        if cat_other == 's':
            return merge_sequencies([base], other)
        elif cat_other == 'm':
            message = 'merge error ({}, {}): "{}" and "{}"'
            raise ValueError(message.format(cat_base, cat_other, base, other))
        else:
            return other


def merge_sequencies(s1, s2):
    return type(s2)(chain(s1, s2))


def merge_maps(m1, m2):
    return type(m2)(chain(m1.iteritems(), m2.iteritems()))


def categorize(v):
    if hasattr(v, 'merge_with') and hasattr(v, 'merge_into'):
        return 'o'
    elif hasattr(v, 'iteritems'):
        return 'm'
    elif isiterable(v):
        return 's'
    elif v is None:
        return 'n'
    else:
        return 'c'

settings = WAConfiguration()

class SharedConfiguration(object):

    def __init__(self):
        self.number_of_iterations = None
        self.workload_name = None
        self.label = None
        self.boot_parameters = OrderedDict()
        self.runtime_parameters = OrderedDict()
        self.workload_parameters = OrderedDict()
        self.instrumentation = []


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

    def to_pod(self):
        d = copy(self.__dict__)
        del d['_workload']
        del d['_section']
        return d

    @staticmethod
    def from_pod(pod):
        instance = WorkloadRunSpec(id=pod['id'],  # pylint: disable=W0622
                                   number_of_iterations=pod['number_of_iterations'],
                                   workload_name=pod['workload_name'],
                                   boot_parameters=pod['boot_parameters'],
                                   label=pod['label'],
                                   section_id=pod['section_id'],
                                   workload_parameters=pod['workload_parameters'],
                                   runtime_parameters=pod['runtime_parameters'],
                                   instrumentation=pod['instrumentation'],
                                   flash=pod['flash'],
                                   classifiers=pod['classifiers'],
                                   )
        return instance

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

    def to_pod(self):
        return self.policy

    @staticmethod
    def from_pod(pod):
        return RebootPolicy(pod)


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

    def __str__(self):
        return "RCI(name: {}, category: {}, method: {})".format(self.name, self.category, self.method)

    __repr__ = __str__


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

    .. note:: You don't need to know this to use WA, or to write plugins for it. From
              the point of view of plugin writers, configuration from various sources
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
        can be for the run as a whole, or for a specific plugin.

    (workload) spec

        A specification of a single workload execution. This combines workload configuration
        with things like the number of iterations to run, which instruments to enable, etc.
        More concretely, this is an instance of :class:`WorkloadRunSpec`.

    **overview**

    There are three types of WA configuration:

        1. "Meta" configuration that determines how the rest of the configuration is
           processed (e.g. where plugins get loaded from). Since this does not pertain
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
    implied by the file plugin (currently, YAML and Python are supported). If the same
    configuration item appears in more than one source, they are merged with conflicting entries
    taking the value from the last source that specified them.

    In addition to a fixed set of global configuration items, configuration for any WA
    Plugin (instrument, result processor, etc) may also be specified, namespaced under
    the plugin's name (i.e. the plugins name is a key in the global config with value
    being a dict of parameters and their values). Some Plugin parameters also specify a
    "global alias" that may appear at the top-level of the config rather than under the
    Plugin's name. It is *not* an error to specify configuration for an Plugin that has
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

    As mentioned above, an Plugin's parameter may define a global alias, which will be
    specified and picked up from the top-level config, rather than config for that specific
    plugin. It is an error to specify the value for a parameter both through a global
    alias and through plugin config dict in the same configuration file. It is, however,
    possible to use a global alias in one file, and specify plugin configuration for the
    same parameter in another file, in which case, the usual merging rules would apply.

    **Loading and validation of configuration**

    Validation of user-specified configuration happens at several stages of run initialisation,
    to ensure that appropriate context for that particular type of validation is available and
    that meaningful errors can be reported, as early as is feasible.

    - Syntactic validation is performed when configuration is first loaded.
      This is done by the loading mechanism (e.g. YAML parser), rather than WA itself. WA
      propagates any errors encountered as ``ConfigError``\ s.
    - Once a config file is loaded into a Python structure, it scanned to
      extract settings. Static configuration is validated and added to the config. Plugin
      configuration is collected into a collection of "raw" config, and merged as appropriate, but
      is not processed further at this stage.
    - Once all configuration sources have been processed, the configuration as a whole
      is validated (to make sure there are no missing settings, etc).
    - Plugins are loaded through the run config object, which instantiates
      them with appropriate parameters based on the "raw" config collected earlier. When an
      Plugin is instantiated in such a way, its config is "officially" added to run configuration
      tracked by the run config object. Raw config is discarded at the end of the run, so
      that any config that wasn't loaded in this way is not recoded (as it was not actually used).
    - Plugin parameters a validated individually (for type, value ranges, etc) as they are
      loaded in the Plugin's __init__.
    - An plugin's ``validate()`` method is invoked before it is used (exactly when this
      happens depends on the plugin's type) to perform any final validation *that does not
      rely on the target being present* (i.e. this would happen before WA connects to the target).
      This can be used perform inter-parameter validation for an plugin (e.g. when valid range for
      one parameter depends on another), and more general WA state assumptions (e.g. a result
      processor can check that an instrument it depends on has been installed).
    - Finally, it is the responsibility of individual plugins to validate any assumptions
      they make about the target device (usually as part of their ``setup()``).

    **Handling of Plugin aliases.**

    WA plugins can have zero or more aliases (not to be confused with global aliases for plugin
    *parameters*). An plugin allows associating an alternative name for the plugin with a set
    of parameter values. In other words aliases associate common configurations for an plugin with
    a name, providing a shorthand for it. For example, "t-rex_offscreen" is an alias for "glbenchmark"
    workload that specifies that "use_case" should be "t-rex" and "variant" should be "offscreen".

    **special loading rules**

    Note that as a consequence of being able to specify configuration for *any* Plugin namespaced
    under the Plugin's name in the top-level config, two distinct mechanisms exist form configuring
    devices and workloads. This is valid, however due to their nature, they are handled in a special way.
    This may be counter intuitive, so configuration of devices and workloads creating entries for their
    names in the config is discouraged in favour of using the "normal" mechanisms of configuring them
    (``device_config`` for devices and workload specs in the agenda for workloads).

    In both cases (devices and workloads), "normal" config will always override named plugin config
    *irrespective of which file it was specified in*. So a ``adb_name`` name specified in ``device_config``
    inside ``~/.workload_automation/config.py`` will override ``adb_name`` specified for ``juno`` in the
    agenda (even when device is set to "juno").

    Again, this ignores normal loading rules, so the use of named plugin configuration for devices
    and workloads is discouraged. There maybe some situations where this behaviour is useful however
    (e.g. maintaining configuration for different devices in one config file).

    """

    default_reboot_policy = 'as_needed'
    default_execution_order = 'by_iteration'

    # This is generic top-level configuration.
    general_config = [
        RunConfigurationItem('run_name', 'scalar', 'replace'),
        RunConfigurationItem('output_directory', 'scalar', 'replace'),
        RunConfigurationItem('meta_directory', 'scalar', 'replace'),
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
    ignore_names = WA_CONFIGURATION.keys()

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
    def meta_directory(self):
        path = os.path.join(self.output_directory, "__meta")
        if not os.path.exists(path):
            os.makedirs(os.path.abspath(path))
        return path

    @property
    def log_file(self):
        path = os.path.join(self.output_directory, "run.log")
        return os.path.abspath(path)

    @property
    def all_instrumentation(self):
        result = set()
        for spec in self.workload_specs:
            result = result.union(set(spec.instrumentation))
        return result

    def __init__(self, ext_loader=pluginloader):
        self.ext_loader = ext_loader
        self.device = None
        self.device_config = None
        self.execution_order = None
        self.project = None
        self.project_stage = None
        self.run_name = None
        self.output_directory = settings.default_output_directory
        self.instrumentation = {}
        self.result_processors = {}
        self.workload_specs = []
        self.flashing_config = {}
        self.other_config = {}  # keeps track of used config for plugins other than of the four main kinds.
        self.retry_on_status = status_list(['FAILED', 'PARTIAL'])
        self.max_retries = 3
        self._used_config_items = []
        self._global_instrumentation = []
        self._reboot_policy = None
        self.agenda = None
        self._finalized = False
        self._general_config_map = {i.name: i for i in self.general_config}
        self._workload_config_map = {i.name: i for i in self.workload_config}
        # Config files may contains static configuration for plugins that
        # would not be part of this of this run (e.g. DB connection settings
        # for a result processor that has not been enabled). Such settings
        # should not be part of configuration for this run (as they will not
        # be affecting it), but we still need to keep track it in case a later
        # config (e.g. from the agenda) enables the plugin.
        # For this reason, all plugin config is first loaded into the
        # following dict and when an plugin is identified as need for the
        # run, its config is picked up from this "raw" dict and it becomes part
        # of the run configuration.
        self._raw_config = {'instrumentation': [], 'result_processors': []}

    def get_plugin(self, name=None, kind=None, *args, **kwargs):
        self._check_finalized()
        self._load_default_config_if_necessary(name)
        ext_config = self._raw_config[name]
        ext_cls = self.ext_loader.get_plugin_class(name)
        if ext_cls.kind not in ['workload', 'device', 'instrument', 'result_processor']:
            self.other_config[name] = ext_config
        ext_config.update(kwargs)
        return self.ext_loader.get_plugin(name=name, *args, **ext_config)

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
        config files can be either python modules (.py plugin) or YAML documents
        (.yaml plugin)."""
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
        if self.agenda:
            # note: this also guards against loading an agenda after finalized() has been called,
            #       as that would have required an agenda to be set.
            message = 'Attempting to set a second agenda {};\n\talready have agenda {} set'
            raise ValueError(message.format(agenda.filepath, self.agenda.filepath))
        try:
            self._merge_config(agenda.config or {})
            self._load_specs_from_agenda(agenda, selectors)
            self.agenda = agenda
        except ConfigError as e:
            message = 'Error in {}:\n\t{}'
            raise ConfigError(message.format(agenda.filepath, e.message))

    def finalize(self):
        """This must be invoked once all configuration sources have been loaded. This will
        do the final processing, setting instrumentation and result processor configuration
        for the run And making sure that all the mandatory config has been specified."""
        if self._finalized:
            return
        if not self.agenda:
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
            elif self.ext_loader.has_plugin(k):
                self._set_plugin_config(k, v)
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

    def _set_plugin_config(self, name, value):
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

    @staticmethod
    def from_pod(pod, ext_loader=pluginloader):
        instance = RunConfiguration
        self.device = pod['device']
        self.execution_order = pod['execution_order']
        self.project = pod['project']
        self.project_stage = pod['project_stage']
        self.run_name = pod['run_name']
        self.max_retries = pod['max_retries']
        self._reboot_policy.policy = RebootPolicy.from_pod(pod['_reboot_policy'])
        self.output_directory = pod['output_directory']
        self.device_config = pod['device_config']
        self.instrumentation = pod['instrumentation']
        self.result_processors = pod['result_processors']
        self.workload_specs = [WorkloadRunSpec.from_pod(pod) for pod in pod['workload_specs']]
        self.flashing_config = pod['flashing_config']
        self.other_config = pod['other_config']
        self.retry_on_status = pod['retry_on_status']
        self._used_config_items = pod['_used_config_items']
        self._global_instrumentation = pod['_global_instrumentation']

    def to_pod(self):
        if not self._finalized:
            raise Exception("Cannot use `to_pod` until the config is finalis")
        pod = {}
        pod['device'] = self.device
        pod['execution_order'] = self.execution_order
        pod['project'] = self.project
        pod['project_stage'] = self.project_stage
        pod['run_name'] = self.run_name
        pod['max_retries'] = self.max_retries
        pod['_reboot_policy'] = self._reboot_policy.to_pod()
        pod['output_directory'] = os.path.abspath(self.output_directory)
        pod['device_config'] = self.device_config
        pod['instrumentation'] = self.instrumentation
        pod['result_processors'] = self.result_processors
        pod['workload_specs'] = [w.to_pod() for w in self.workload_specs]
        pod['flashing_config'] = self.flashing_config
        pod['other_config'] = self.other_config
        pod['retry_on_status'] = self.retry_on_status
        pod['_used_config_items'] = self._used_config_items
        pod['_global_instrumentation'] = self._global_instrumentation
        return pod

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
