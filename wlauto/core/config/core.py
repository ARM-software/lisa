import os
import logging
import shutil
from glob import glob
from copy import copy
from itertools import chain

from wlauto.core import pluginloader
from wlauto.exceptions import ConfigError
from wlauto.utils.types import integer, boolean, identifier, list_of_strings
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
        'filer_mount_point',
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
            self.config_paths.append(config_paths)

    def load_user_config(self):
        globpath = os.path.join(self.user_directory, '{}.*'.format(self.basename))
        for path in glob(globpath):
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.pyc', '.pyo']:
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
        self.dependencies_directory = os.path.join(self.user_directory,
                                                   self.dependencies_directory)
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
