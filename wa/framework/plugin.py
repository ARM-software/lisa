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
from collections import OrderedDict, defaultdict
from itertools import chain
from copy import copy

from wa.framework.configuration.core import settings, ConfigurationPoint as Parameter
from wa.framework.exception import (NotFoundError, PluginLoaderError, TargetError,
                                    ValidationError, ConfigError, HostError)
from wa.utils import log
from wa.utils.misc import (ensure_directory_exists as _d, walk_modules, load_class,
                           merge_dicts_simple, get_article)
from wa.utils.types import identifier


MODNAME_TRANS = string.maketrans(':/\\.', '____')


class AttributeCollection(object):
    """
    Accumulator for plugin attribute objects (such as Parameters or Artifacts).

    This will replace any class member list accumulating such attributes
    through the magic of metaprogramming\ [*]_.

    .. [*] which is totally safe and not going backfire in any way...

    """

    @property
    def values(self):
        return self._attrs.values()

    def __init__(self, attrcls):
        self._attrcls = attrcls
        self._attrs = OrderedDict()

    def add(self, p):
        p = self._to_attrcls(p)
        if p.name in self._attrs:
            if p.override:
                newp = copy(self._attrs[p.name])
                for a, v in p.__dict__.iteritems():
                    if v is not None:
                        setattr(newp, a, v)
                if not hasattr(newp, "_overridden"):
                    # pylint: disable=protected-access
                    newp._overridden = p._owner
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
        if not isinstance(p, self._attrcls):
            raise ValueError('Invalid attribute value: {}; must be a {}'.format(p, self._attrcls))
        if p.name in self._attrs and not p.override:
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


class AliasCollection(AttributeCollection):

    def __init__(self):
        super(AliasCollection, self).__init__(Alias)

    def _to_attrcls(self, p):
        if isinstance(p, (list, tuple)):
            # must be in the form (name, {param: value, ...})
            # pylint: disable=protected-access
            p = self._attrcls(p[1], **p[1])
        elif not isinstance(p, self._attrcls):
            raise ValueError('Invalid parameter value: {}'.format(p))
        if p.name in self._attrs:
            raise ValueError('Attribute {} has already been defined.'.format(p.name))
        return p


class ListCollection(list):

    def __init__(self, attrcls):  # pylint: disable=unused-argument
        super(ListCollection, self).__init__()


class Alias(object):
    """
    This represents a configuration alias for an plugin, mapping an alternative
    name to a set of parameter values, effectively providing an alternative set
    of default values.

    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.params = kwargs
        self.plugin_name = None  # gets set by the MetaClass

    def validate(self, ext):
        ext_params = set(p.name for p in ext.parameters)
        for param in self.params:
            if param not in ext_params:
                # Raising config error because aliases might have come through
                # the config.
                msg = 'Parameter {} (defined in alias {}) is invalid for {}'
                raise ConfigError(msg.format(param, self.name, ext.name))


class PluginMeta(type):
    """
    This basically adds some magic to plugins to make implementing new plugins,
    such as workloads less complicated.

    It ensures that certain class attributes (specified by the ``to_propagate``
    attribute of the metaclass) get propagated down the inheritance hierarchy.
    The assumption is that the values of the attributes specified in the class
    are iterable; if that is not met, Bad Things (tm) will happen.

    """

    to_propagate = [
        ('parameters', Parameter, AttributeCollection),
    ]

    def __new__(mcs, clsname, bases, attrs):
        mcs._propagate_attributes(bases, attrs, clsname)
        cls = type.__new__(mcs, clsname, bases, attrs)
        mcs._setup_aliases(cls)
        return cls

    @classmethod
    def _propagate_attributes(mcs, bases, attrs, clsname):
        """
        For attributes specified by to_propagate, their values will be a union of
        that specified for cls and its bases (cls values overriding those of bases
        in case of conflicts).

        """
        for prop_attr, attr_cls, attr_collector_cls in mcs.to_propagate:
            should_propagate = False
            propagated = attr_collector_cls(attr_cls)
            for base in bases:
                if hasattr(base, prop_attr):
                    propagated += getattr(base, prop_attr) or []
                    should_propagate = True
            if prop_attr in attrs:
                pattrs = attrs[prop_attr] or []
                for pa in pattrs:
                    if not isinstance(pa, attr_cls):
                        msg = 'Invalid value "{}" for attribute "{}"; must be a {}'
                        raise ValueError(msg.format(pa, prop_attr, attr_cls))
                    pa._owner = clsname
                propagated += pattrs
                should_propagate = True
            if should_propagate:
                for p in propagated:
                    override = bool(getattr(p, "override", None))
                    overridden = bool(getattr(p, "_overridden", None))
                    if override != overridden:
                        msg = "Overriding non existing parameter '{}' inside '{}'"
                        raise ValueError(msg.format(p.name, p._owner))
                attrs[prop_attr] = propagated

    @classmethod
    def _setup_aliases(mcs, cls):
        if hasattr(cls, 'aliases'):
            aliases, cls.aliases = cls.aliases, AliasCollection()
            for alias in aliases:
                if isinstance(alias, basestring):
                    alias = Alias(alias)
                alias.validate(cls)
                alias.plugin_name = cls.name
                cls.aliases.add(alias)


class Plugin(object):
    """
    Base class for all WA plugins. An plugin is basically a plug-in.  It
    extends the functionality of WA in some way. Plugins are discovered and
    loaded dynamically by the plugin loader upon invocation of WA scripts.
    Adding an plugin is a matter of placing a class that implements an
    appropriate interface somewhere it would be discovered by the loader. That
    "somewhere" is typically one of the plugin subdirectories under
    ``~/.workload_automation/``.

    """
    __metaclass__ = PluginMeta

    kind = None
    name = None
    parameters = []
    artifacts = []
    aliases = []
    core_modules = []

    @classmethod
    def get_default_config(cls):
        return {p.name: p.default for p in cls.parameters}

    @property
    def dependencies_directory(self):
        return _d(os.path.join(settings.dependencies_directory, self.name))

    @property
    def _classname(self):
        return self.__class__.__name__

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.name)
        self._modules = []
        self.capabilities = getattr(self.__class__, 'capabilities', [])
        for param in self.parameters:
            param.set_value(self, kwargs.get(param.name))
        for key in kwargs:
            if key not in self.parameters:
                message = 'Unexpected parameter "{}" for {}'
                raise ConfigError(message.format(key, self.name))

    def get_config(self):
        """
        Returns current configuration (i.e. parameter values) of this plugin.

        """
        config = {}
        for param in self.parameters:
            config[param.name] = getattr(self, param.name, None)
        return config

    def validate(self):
        """
        Perform basic validation to ensure that this plugin is capable of
        running.  This is intended as an early check to ensure the plugin has
        not been mis-configured, rather than a comprehensive check (that may,
        e.g., require access to the execution context).

        This method may also be used to enforce (i.e. set as well as check)
        inter-parameter constraints for the plugin (e.g. if valid values for
        parameter A depend on the value of parameter B -- something that is not
        possible to enfroce using ``Parameter``\ 's ``constraint`` attribute.

        """
        if self.name is None:
            raise ValidationError('Name not set for {}'.format(self._classname))
        for param in self.parameters:
            param.validate(self)

    def __getattr__(self, name):
        if name == '_modules':
            raise ValueError('_modules accessed too early!')
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)

    def load_modules(self, loader):
        """
        Load the modules specified by the "modules" Parameter using the
        provided loader. A loader can be any object that has an atribute called
        "get_module" that implements the following signature::

            get_module(name, owner, **kwargs)

        and returns an instance of :class:`wa.core.plugin.Module`. If the
        module with the specified name is not found, the loader must raise an
        appropriate exception.

        """
        modules = list(reversed(self.core_modules)) +\
                    list(reversed(self.modules or []))
        if not modules:
            return
        for module_spec in modules:
            if not module_spec:
                continue
            module = self._load_module(loader, module_spec)
            self._install_module(module)

    def has(self, capability):
        """
        Check if this plugin has the specified capability. The alternative
        method ``can`` is identical to this. Which to use is up to the caller
        depending on what makes semantic sense in the context of the
        capability, e.g. ``can('hard_reset')`` vs  ``has('active_cooling')``.

        """
        return capability in self.capabilities

    can = has

    def _load_module(self, loader, module_spec):
        if isinstance(module_spec, basestring):
            name = module_spec
            params = {}
        elif isinstance(module_spec, dict):
            if len(module_spec) != 1:
                msg = 'Invalid module spec: {}; dict must have exctly one key -- '\
                      'the module name.'
                raise ValueError(msg.format(module_spec))
            name, params = module_spec.items()[0]
        else:
            message = 'Invalid module spec: {}; must be a string or a one-key dict.'
            raise ValueError(message.format(module_spec))

        if not isinstance(params, dict):
            message = 'Invalid module spec: {}; dict value must also be a dict.'
            raise ValueError(message.format(module_spec))

        module = loader.get_module(name, owner=self, **params)
        module.initialize(None)
        return module

    def _install_module(self, module):
        for capability in module.capabilities:
            if capability not in self.capabilities:
                self.capabilities.append(capability)
        self._modules.append(module)


class TargetedPlugin(Plugin):
    """
    A plugin that interacts with a target device.

    """

    suppoted_targets = []

    @classmethod
    def check_compatible(cls, target):
        if cls.suppoted_targets:
            if target.os not in cls.suppoted_targets:
                msg = 'Incompatible target OS "{}" for {}'
                raise TargetError(msg.format(target.os, cls.name))

    def __init__(self, target, **kwargs):
        super(TargetedPlugin, self).__init__(**kwargs)
        self.check_compatible(target)
        self.target = target


class PluginLoaderItem(object):

    def __init__(self, ext_tuple):
        self.name = ext_tuple.name
        self.default_package = ext_tuple.default_package
        self.default_path = ext_tuple.default_path
        self.cls = load_class(ext_tuple.cls)


class PluginLoader(object):
    """
    Discovers, enumerates and loads available devices, configs, etc.
    The loader will attempt to discover things on construction by looking
    in predetermined set of locations defined by default_paths. Optionally,
    additional locations may specified through paths parameter that must
    be a list of additional Python module paths (i.e. dot-delimited).

    """

    def __init__(self, packages=None, paths=None, ignore_paths=None,
                 keep_going=False):
        """
        params::

            :packages: List of packages to load plugins from.
            :paths: List of paths to be searched for Python modules containing
                    WA plugins.
            :ignore_paths: List of paths to ignore when search for WA plugins
                           (these would typically be subdirectories of one or
                           more locations listed in ``paths`` parameter.
            :keep_going: Specifies whether to keep going if an error occurs while
                         loading plugins.
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
        msg = 'Updating from: packages={} paths={}'
        self.logger.debug(msg.format(packages, paths))
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
        self.logger.debug('Reloading')
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
                raise NotFoundError('plugins {} not found.'.format(name))
        if kind not in self.kind_map:
            raise ValueError('Unknown plugin type: {}'.format(kind))
        store = self.kind_map[kind]
        if name not in store:
            msg = 'plugins {} is not {} {}.'
            raise NotFoundError(msg.format(name, get_article(kind), kind))
        return store[name]

    def get_plugin(self, name=None, kind=None, *args, **kwargs):
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
        return merge_dicts_simple(base_default_config, alias_config)

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
            return (alias.plugin_name, alias.params)
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
        error_msg = 'No plugins of type "{}" discovered'
        if name.startswith('get_'):
            name = name.replace('get_', '', 1)
            if name in self.kind_map:
                def __wrapper(pname, *args, **kwargs):
                    return self.get_plugin(pname, name, *args, **kwargs)
                return __wrapper
            raise NotFoundError(error_msg.format(name))
        if name.startswith('list_'):
            name = name.replace('list_', '', 1).rstrip('s')
            if name in self.kind_map:
                def __wrapper(*args, **kwargs):  # pylint: disable=E0102
                    return self.list_plugins(name, *args, **kwargs)
                return __wrapper
            raise NotFoundError(error_msg.format(name))
        if name.startswith('has_'):
            name = name.replace('has_', '', 1)
            if name in self.kind_map:
                def __wrapper(pname, *args, **kwargs):  # pylint: disable=E0102
                    return self.has_plugin(pname, name, *args, **kwargs)
                return __wrapper
            raise NotFoundError(error_msg.format(name))
        raise AttributeError(name)

    def _discover_from_packages(self, packages):
        self.logger.debug('Discovering plugins in packages')
        try:
            for package in packages:
                for module in walk_modules(package):
                    self._discover_in_module(module)
        except HostError as e:
            message = 'Problem loading plugins from {}: {}'
            raise PluginLoaderError(message.format(e.module, str(e.orig_exc)),
                                    e.exc_info)

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
                    if os.path.splitext(fname)[1].lower() != '.py':
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
                msg = 'Failed to load {}'
                raise PluginLoaderError(msg.format(filepath), sys.exc_info())
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
            pass

    def _add_found_plugin(self, obj):
        """
            :obj: Found plugin class
            :ext: matching plugin item.
        """
        self.logger.debug('Adding %s %s', obj.kind, obj.name)
        key = identifier(obj.name.lower())
        if key in self.plugins or key in self.aliases:
            msg = '{} "{}" already exists.'
            raise PluginLoaderError(msg.format(obj.kind, obj.name))
        # plugins are tracked both, in a common plugins
        # dict, and in per-plugin kind dict (as retrieving
        # plugins by kind is a common use case.
        self.plugins[key] = obj
        self.kind_map[obj.kind][key] = obj

        for alias in obj.aliases:
            alias_id = identifier(alias.name.lower())
            if alias_id in self.plugins or alias_id in self.aliases:
                msg = '{} "{}" already exists.'
                raise PluginLoaderError(msg.format(obj.kind, obj.name))
            self.aliases[alias_id] = alias
