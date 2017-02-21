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


# pylint: disable=E1101
import os
import sys
import inspect
import imp
import string
import logging
from copy import copy
from itertools import chain
from collections import OrderedDict, defaultdict

from wa.framework import log
from wa.framework.exception import ValidationError, ConfigError, NotFoundError, PluginLoaderError
from wa.framework.configuration.core import ConfigurationPoint, ConfigurationPointCollection
from wa.utils.misc import isiterable, ensure_directory_exists as _d, get_article
from wa.utils.misc import walk_modules, get_article
from wa.utils.types import identifier, integer, boolean, caseless_string


class Parameter(ConfigurationPoint):

    is_runtime = False

    def __init__(self, name, 
                 kind=None, 
                 mandatory=None, 
                 default=None, 
                 override=False,
                 allowed_values=None, 
                 description=None, 
                 constraint=None, 
                 convert_types=True,
                 global_alias=None,
                 reconfigurable=True):
        """
        :param global_alias: This is an alternative alias for this parameter,
                             unlike the name, this alias will not be
                             namespaced under the owning extension's name
                             (hence the global part). This is introduced
                             primarily for backward compatibility -- so that
                             old extension settings names still work. This
                             should not be used for new parameters.

        :param reconfigurable: This indicated whether this parameter may be 
                               reconfigured during the run (e.g. between different
                               iterations). This determines where in run configruation
                               this parameter may appear.

        For other parameters, see docstring for 
        ``wa.framework.configuration.core.ConfigurationPoint``

        """
        super(Parameter, self).__init__(name, kind, mandatory,
                                        default, override, allowed_values,
                                        description, constraint,
                                        convert_types)
        self.global_alias = global_alias
        self.reconfigurable = reconfigurable

    def __repr__(self):
        d = copy(self.__dict__)
        del d['description']
        return 'Param({})'.format(d)


class PluginAliasCollection(object):
    """
    Accumulator for extension attribute objects (such as Parameters). This will
    replace any class member list accumulating such attributes through the magic of
    metaprogramming\ [*]_.

    .. [*] which is totally safe and not going backfire in any way...

    """

    @property
    def values(self):
        return self._attrs.values()

    def __init__(self):
        self._attrs = OrderedDict()

    def add(self, p):
        p = self._to_attrcls(p)
        if p.name in self._attrs:
            if p.override:
                newp = copy(self._attrs[p.name])
                for a, v in p.__dict__.iteritems():
                    if v is not None:
                        setattr(newp, a, v)
                self._attrs[p.name] = newp
            else:
                # Duplicate attribute condition is check elsewhere.
                pass
        else:
            self._attrs[p.name] = p

    append = add

    def __str__(self):
        return 'AC({})'.format(map(str, self._attrs.values()))

    __repr__ = __str__

    def _to_attrcls(self, p):
        if isinstance(p, tuple) or isinstance(p, list):
            # must be in the form (name, {param: value, ...})
            p = Alias(p[1], **p[1])
        elif not isinstance(p, Alias):
            raise ValueError('Invalid parameter value: {}'.format(p))
        if p.name in self._attrs:
            raise ValueError('Attribute {} has already been defined.'.format(p.name))
        return p

    def __iadd__(self, other):
        for p in other:
            self.add(p)
        return self

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, p):
        return p in self._attrs

    def __getitem__(self, i):
        return self._attrs[i]

    def __len__(self):
        return len(self._attrs)


class Alias(object):
    """
    This represents a configuration alias for an extension, mapping an alternative name to
    a set of parameter values, effectively providing an alternative set of default values.

    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.parameters = kwargs
        self.plugin_name = None  # gets set by the MetaClass

    def validate(self, plugin):
        plugin_params = set(p.name for p in plugin.parameters)
        for param in self.parameters:
            if param not in plugin_params:
                # Raising config error because aliases might have come through
                # the config.
                msg = 'Parameter {} (defined in alias {}) is invalid for {}'
                raise ValueError(msg.format(param, self.name, plugin.name))


class PluginMeta(type):
    """
    This basically adds some magic to extensions to make implementing new extensions, such as
    workloads less complicated.

    It ensures that certain class attributes (specified by the ``to_propagate``
    attribute of the metaclass) get propagated down the inheritance hierarchy. The assumption
    is that the values of the attributes specified in the class are iterable; if that is not met,
    Bad Things(tm) will happen.

    This also provides "virtual" method implementations. The ``super``'s version of these
    methods (specified by the ``virtual_methods`` attribute of the metaclass) will be 
    automatically invoked.

    """

    to_propagate = [
        ('parameters', ConfigurationPointCollection),
    ]

    #virtual_methods = ['validate', 'initialize', 'finalize']
    virtual_methods = []

    def __new__(mcs, clsname, bases, attrs):
        mcs._propagate_attributes(bases, attrs)
        cls = type.__new__(mcs, clsname, bases, attrs)
        mcs._setup_aliases(cls)
        mcs._implement_virtual(cls, bases)
        return cls

    @classmethod
    def _propagate_attributes(mcs, bases, attrs):
        """
        For attributes specified by to_propagate, their values will be a union of
        that specified for cls and it's bases (cls values overriding those of bases
        in case of conflicts).

        """
        for prop_attr, attr_collector_cls in mcs.to_propagate:
            should_propagate = False
            propagated = attr_collector_cls()
            for base in bases:
                if hasattr(base, prop_attr):
                    propagated += getattr(base, prop_attr) or []
                    should_propagate = True
            if prop_attr in attrs:
                propagated += attrs[prop_attr] or []
                should_propagate = True
            if should_propagate:
                attrs[prop_attr] = propagated

    @classmethod
    def _setup_aliases(mcs, cls):
        if hasattr(cls, 'aliases'):
            aliases, cls.aliases = cls.aliases, PluginAliasCollection()
            for alias in aliases:
                if isinstance(alias, basestring):
                    alias = Alias(alias)
                alias.validate(cls)
                alias.plugin_name = cls.name
                cls.aliases.add(alias)

    @classmethod
    def _implement_virtual(mcs, cls, bases):
        """
        This implements automatic method propagation to the bases, so
        that you don't have to do something like

            super(cls, self).vmname()

        This also ensures that the methods that have beend identified as
        "globally virtual" are executed exactly once per WA execution, even if
        invoked through instances of different subclasses

        """
        methods = {}
        called_globals = set()
        for vmname in mcs.virtual_methods:
            clsmethod = getattr(cls, vmname, None)
            if clsmethod:
                basemethods = [getattr(b, vmname) for b in bases if hasattr(b, vmname)]
                methods[vmname] = [bm for bm in basemethods if bm != clsmethod]
                methods[vmname].append(clsmethod)

                def generate_method_wrapper(vname):  # pylint: disable=unused-argument
                    # this creates a closure with the method name so that it
                    # does not need to be passed to the wrapper as an argument,
                    # leaving the wrapper to accept exactly the same set of
                    # arguments as the method it is wrapping.
                    name__ = vmname  # pylint: disable=cell-var-from-loop

                    def wrapper(self, *args, **kwargs):
                        for dm in methods[name__]:
                            dm(self, *args, **kwargs)
                    return wrapper

                setattr(cls, vmname, generate_method_wrapper(vmname))


class Plugin(object):
    """
    Base class for all WA plugins.
    A plugin extends the functionality of WA in some way. Plugins are discovered
    and loaded dynamically by the plugin loader upon invocation of WA scripts.
    Adding an extension is a matter of placing a class that implements an appropriate
    interface somewhere it would be discovered by the loader. That "somewhere" is
    typically one of the plugin subdirectories under ``~/.workload_automation/``.

    """
    __metaclass__ = PluginMeta

    name = None
    kind = None
    parameters = []
    aliases = []

    @classmethod
    def get_default_config(cls):
        return {p.name: p.default for p in cls.parameters}

    @classmethod
    def get_parameter(cls, name):
        for param in cls.parameters:
            if param.name == name or name in param.aliases:
                return param

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.name)
        self.capabilities = getattr(self.__class__, 'capabilities', [])
        self.update_config(**kwargs)

    def get_config(self):
        """
        Returns current configuration (i.e. parameter values) of this plugin.

        """
        config = {}
        for param in self.parameters:
            config[param.name] = getattr(self, param.name, None)
        return config

    def update_config(self, **kwargs):
        """
        Updates current configuration (i.e. parameter values) of this plugin.

        """
        for param in self.parameters:
            param.set_value(self, kwargs.get(param.name))
        for key in kwargs:
            if key not in self.parameters:
                message = 'Unexpected parameter "{}" for {}'
                raise ConfigError(message.format(key, self.name))

    def validate(self):
        """
        Perform basic validation to ensure that this extension is capable of running.
        This is intended as an early check to ensure the extension has not been mis-configured,
        rather than a comprehensive check (that may, e.g., require access to the execution
        context).

        This method may also be used to enforce (i.e. set as well as check) inter-parameter
        constraints for the extension (e.g. if valid values for parameter A depend on the value
        of parameter B -- something that is not possible to enforce using ``Parameter``\ 's
        ``constraint`` attribute.

        """
        if self.name is None:
            raise ValidationError('name not set for {}'.format(self.__class__.__name__))
        if self.kind is None:
            raise ValidationError('kind not set for {}'.format(self.name))
        for param in self.parameters:
            param.validate(self)

    def initialize(self, context):
        pass

    def finalize(self, context):
        pass

    def has(self, capability):
        """Check if this extension has the specified capability. The alternative method ``can`` is
        identical to this. Which to use is up to the caller depending on what makes semantic sense
        in the context of the capability, e.g. ``can('hard_reset')`` vs  ``has('active_cooling')``."""
        return capability in self.capabilities

    can = has


class TargetedPluginMeta(PluginMeta):

    to_propagate = PluginMeta.to_propagate + [
        ('supported_targets', list),
        ('supported_platforms', list),
    ]
    virtual_methods = PluginMeta.virtual_methods + [
        'validate_on_target',
    ]


class TargetedPlugin(Plugin):
    """
    A plugin that operates on a target device.  These kinds of plugins are created
    with a ``devlib.Target`` instance and may only support certain kinds of targets.

    """

    __metaclass__ = TargetedPluginMeta

    supported_targets = []
    supported_platforms = []

    def __init__(self, target, **kwargs):
        super(TargetedPlugin, self).__init__(**kwargs)
        if self.supported_targets and target.os not in self.supported_targets:
            raise TargetError('Plugin {} does not support target {}'.format(self.name, target.name))
        if self.supported_platforms and target.platform.name not in self.supported_platforms:
            raise TargetError('Plugin {} does not support platform {}'.format(self.name, target.platform))
        self.target = target

    def validate_on_target(self):
        """
        This will be invoked once at the beginning of a run after a ``Target`` 
        has been connected and initialized. This is intended for validation 
        that cannot be performed offline but does not depend on ephemeral 
        state that is likely to change during the course of a run (validation
        against such states should be done during setup of a particular
        execution.

        """
        pass


class GlobalParameterAlias(object):
    """
    Represents a "global alias" for an plugin parameter. A global alias
    is specified at the top-level of config rather namespaced under an plugin
    name.

    Multiple plugins may have parameters with the same global_alias if they are
    part of the same inheritance hierarchy and one parameter is an override of the
    other. This class keeps track of all such cases in its plugins dict.

    """

    def __init__(self, name):
        self.name = name
        self.plugins = {}

    def iteritems(self):
        for ext in self.plugins.itervalues():
            yield (self.get_param(ext), ext)

    def get_param(self, ext):
        for param in ext.parameters:
            if param.global_alias == self.name:
                return param
        message = 'Plugin {} does not have a parameter with global alias {}'
        raise ValueError(message.format(ext.name, self.name))

    def update(self, other_ext):
        self._validate_ext(other_ext)
        self.plugins[other_ext.name] = other_ext

    def _validate_ext(self, other_ext):
        other_param = self.get_param(other_ext)
        for param, ext in self.iteritems():
            if ((not (issubclass(ext, other_ext) or issubclass(other_ext, ext))) and
                    other_param.kind != param.kind):
                message = 'Duplicate global alias {} declared in {} and {} plugins with different types'
                raise PluginLoaderError(message.format(self.name, ext.name, other_ext.name))
            if not param.name == other_param.name:
                message = 'Two params {} in {} and {} in {} both declare global alias {}'
                raise PluginLoaderError(message.format(param.name, ext.name,
                                                 other_param.name, other_ext.name, self.name))

    def __str__(self):
        text = 'GlobalAlias({} => {})'
        extlist = ', '.join(['{}.{}'.format(e.name, p.name) for p, e in self.iteritems()])
        return text.format(self.name, extlist)


MODNAME_TRANS = string.maketrans(':/\\.', '____')

class PluginLoader(object):
    """
    Discovers, enumerates and loads available devices, configs, etc.
    The loader will attempt to discover things on construction by looking
    in predetermined set of locations defined by default_paths. Optionally,
    additional locations may specified through paths parameter that must
    be a list of additional Python module paths (i.e. dot-delimited).

    """


    def __init__(self, packages=None, paths=None, ignore_paths=None, keep_going=False):
        """
        params::

            :packages: List of packages to load plugins from.
            :paths: List of paths to be searched for Python modules containing
                    WA plugins.
            :ignore_paths: List of paths to ignore when search for WA plugins (these would
                           typically be subdirectories of one or more locations listed in
                           ``paths`` parameter.
            :keep_going: Specifies whether to keep going if an error occurs while loading
                         plugins.
        """
        self.logger = logging.getLogger('pluginloader')
        self.keep_going = keep_going
        self.packages = packages or []
        self.paths = paths or []
        self.ignore_paths = ignore_paths or []
        self.plugins = {}
        self.kind_map = defaultdict(dict)
        self.aliases = {}
        self.global_param_aliases = {}
        self._discover_from_packages(self.packages)
        self._discover_from_paths(self.paths, self.ignore_paths)

    def update(self, packages=None, paths=None, ignore_paths=None):
        """ Load plugins from the specified paths/packages
        without clearing or reloading existing plugin. """
        if packages:
            self.packages.extend(packages)
            self._discover_from_packages(packages)
        if paths:
            self.paths.extend(paths)
            self.ignore_paths.extend(ignore_paths or [])
            self._discover_from_paths(paths, ignore_paths or [])

    def clear(self):
        """ Clear all discovered items. """
        self.plugins = []
        self.kind_map.clear()

    def reload(self):
        """ Clear all discovered items and re-run the discovery. """
        self.clear()
        self._discover_from_packages(self.packages)
        self._discover_from_paths(self.paths, self.ignore_paths)

    def get_plugin_class(self, name, kind=None):
        """
        Return the class for the specified plugin if found or raises ``ValueError``.

        """
        name, _ = self.resolve_alias(name)
        if kind is None:
            try:
                return self.plugins[name]
            except KeyError:
                raise NotFoundError('Plugins {} not found.'.format(name))
        if kind not in self.kind_map:
            raise ValueError('Unknown plugin type: {}'.format(kind))
        store = self.kind_map[kind]
        if name not in store:
            raise NotFoundError('Plugins {} is not {} {}.'.format(name, get_article(kind), kind))
        return store[name]

    def get_plugin(self, name, kind=None, *args, **kwargs):
        """
        Return plugin of the specified kind with the specified name. Any
        additional parameters will be passed to the plugin's __init__.

        """
        name, base_kwargs = self.resolve_alias(name)
        kwargs = OrderedDict(chain(base_kwargs.iteritems(), kwargs.iteritems()))
        cls = self.get_plugin_class(name, kind)
        plugin = cls(*args, **kwargs)
        return plugin

    def get_default_config(self, name):
        """
        Returns the default configuration for the specified plugin name. The
        name may be an alias, in which case, the returned config will be
        augmented with appropriate alias overrides.

        """
        real_name, alias_config = self.resolve_alias(name)
        base_default_config = self.get_plugin_class(real_name).get_default_config()
        return merge_dicts(base_default_config, alias_config, list_duplicates='last', dict_type=OrderedDict)

    def list_plugins(self, kind=None):
        """
        List discovered plugin classes. Optionally, only list plugins of a
        particular type.

        """
        if kind is None:
            return self.plugins.values()
        if kind not in self.kind_map:
            raise ValueError('Unknown plugin type: {}'.format(kind))
        return self.kind_map[kind].values()

    def has_plugin(self, name, kind=None):
        """
        Returns ``True`` if an plugins with the specified ``name`` has been
        discovered by the loader. If ``kind`` was specified, only returns ``True``
        if the plugin has been found, *and* it is of the specified kind.

        """
        try:
            self.get_plugin_class(name, kind)
            return True
        except NotFoundError:
            return False

    def resolve_alias(self, alias_name):
        """
        Try to resolve the specified name as an plugin alias. Returns a
        two-tuple, the first value of which is actual plugin name, and the
        iisecond is a dict of parameter values for this alias. If the name passed
        is already an plugin name, then the result is ``(alias_name, {})``.

        """
        alias_name = identifier(alias_name.lower())
        if alias_name in self.plugins:
            return (alias_name, {})
        if alias_name in self.aliases:
            alias = self.aliases[alias_name]
            return (alias.plugin_name, alias.parameters)
        raise NotFoundError('Could not find plugin or alias "{}"'.format(alias_name))

    # Internal methods.

    def __getattr__(self, name):
        """
        This resolves methods for specific plugins types based on corresponding
        generic plugin methods. So it's possible to say things like ::

            loader.get_device('foo')

        instead of ::

            loader.get_plugin('foo', kind='device')

        """
        if name.startswith('get_'):
            name = name.replace('get_', '', 1)
            if name in self.kind_map:
                def __wrapper(pname, *args, **kwargs):
                    return self.get_plugin(pname, name, *args, **kwargs)
                return __wrapper
        if name.startswith('list_'):
            name = name.replace('list_', '', 1).rstrip('s')
            if name in self.kind_map:
                def __wrapper(*args, **kwargs):
                    return self.list_plugins(name, *args, **kwargs)
                return __wrapper
        if name.startswith('has_'):
            name = name.replace('has_', '', 1)
            if name in self.kind_map:
                def __wrapper(pname, *args, **kwargs):
                    return self.has_plugin(pname, name, *args, **kwargs)
                return __wrapper
        raise AttributeError(name)


    def _discover_from_packages(self, packages):
        self.logger.debug('Discovering plugins in packages')
        try:
            for package in packages:
                for module in walk_modules(package):
                    self._discover_in_module(module)
        except ImportError as e:
            source = getattr(e, 'path', package)
            message = 'Problem loading plugins from {}: {}'
            raise PluginLoaderError(message.format(source, e.message))

    def _discover_from_paths(self, paths, ignore_paths):
        paths = paths or []
        ignore_paths = ignore_paths or []
        self.logger.debug('Discovering plugins in paths')
        for path in paths:
            self.logger.debug('Checking path %s', path)
            if os.path.isfile(path):
                self._discover_from_file(path)
            for root, _, files in os.walk(path, followlinks=True):
                should_skip = False
                for igpath in ignore_paths:
                    if root.startswith(igpath):
                        should_skip = True
                        break
                if should_skip:
                    continue
                for fname in files:
                    if not os.path.splitext(fname)[1].lower() == '.py':
                        continue
                    filepath = os.path.join(root, fname)
                    self._discover_from_file(filepath)

    def _discover_from_file(self, filepath):
        try:
            modname = os.path.splitext(filepath[1:])[0].translate(MODNAME_TRANS)
            module = imp.load_source(modname, filepath)
            self._discover_in_module(module)
        except (SystemExit, ImportError), e:
            if self.keep_going:
                self.logger.warning('Failed to load {}'.format(filepath))
                self.logger.warning('Got: {}'.format(e))
            else:
                raise PluginLoaderError('Failed to load {}'.format(filepath), sys.exc_info())
        except Exception as e:
            message = 'Problem loading plugins from {}: {}'
            raise PluginLoaderError(message.format(filepath, e))

    def _discover_in_module(self, module):  # NOQA pylint: disable=too-many-branches
        self.logger.debug('Checking module %s', module.__name__)
        log.indent()
        try:
            for obj in vars(module).itervalues():
                if inspect.isclass(obj):
                    if not issubclass(obj, Plugin):
                        continue
                    if not obj.kind:
                        message = 'Skipping plugin {} as it does not define a kind'
                        self.logger.debug(message.format(obj.__name__))
                        continue
                    if not obj.name:
                        message = 'Skipping {} {} as it does not define a name'
                        self.logger.debug(message.format(obj.kind, obj.__name__))
                        continue
                    try:
                        self._add_found_plugin(obj)
                    except PluginLoaderError as e:
                        if self.keep_going:
                            self.logger.warning(e)
                        else:
                            raise e
        finally:
            log.dedent()

    def _add_found_plugin(self, obj):
        """
            :obj: Found plugin class
            :ext: matching plugin item.
        """
        self.logger.debug('Adding %s %s', obj.kind, obj.name)
        key = identifier(obj.name.lower())
        if key in self.plugins or key in self.aliases:
            raise PluginLoaderError('{} "{}" already exists.'.format(obj.kind, obj.name))
        # Plugins are tracked both, in a common plugins
        # dict, and in per-plugin kind dict (as retrieving
        # plugins by kind is a common use case.
        self.plugins[key] = obj
        self.kind_map[obj.kind][key] = obj

        for alias in obj.aliases:
            alias_id = identifier(alias.name.lower())
            if alias_id in self.plugins or alias_id in self.aliases:
                raise PluginLoaderError('{} "{}" already exists.'.format(obj.kind, obj.name))
            self.aliases[alias_id] = alias

        # Update global aliases list. If a global alias is already in the list,
        # then make sure this plugin is in the same parent/child hierarchy
        # as the one already found.
        for param in obj.parameters:
            if param.global_alias:
                if param.global_alias not in self.global_param_aliases:
                    ga = GlobalParameterAlias(param.global_alias)
                    ga.update(obj)
                    self.global_param_aliases[ga.name] = ga
                else:  # global alias already exists.
                    self.global_param_aliases[param.global_alias].update(obj)
