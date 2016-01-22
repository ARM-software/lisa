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


import os
import sys
import inspect
import imp
import string
import logging
from functools import partial
from collections import OrderedDict

from wlauto.core.bootstrap import settings
from wlauto.core.extension import Extension
from wlauto.exceptions import NotFoundError, LoaderError
from wlauto.utils.misc import walk_modules, load_class, merge_lists, merge_dicts, get_article
from wlauto.utils.types import identifier


MODNAME_TRANS = string.maketrans(':/\\.', '____')


class ExtensionLoaderItem(object):

    def __init__(self, ext_tuple):
        self.name = ext_tuple.name
        self.default_package = ext_tuple.default_package
        self.default_path = ext_tuple.default_path
        self.cls = load_class(ext_tuple.cls)


class GlobalParameterAlias(object):
    """
    Represents a "global alias" for an extension parameter. A global alias
    is specified at the top-level of config rather namespaced under an extension
    name.

    Multiple extensions may have parameters with the same global_alias if they are
    part of the same inheritance hierarchy and one parameter is an override of the
    other. This class keeps track of all such cases in its extensions dict.

    """

    def __init__(self, name):
        self.name = name
        self.extensions = {}

    def iteritems(self):
        for ext in self.extensions.itervalues():
            yield (self.get_param(ext), ext)

    def get_param(self, ext):
        for param in ext.parameters:
            if param.global_alias == self.name:
                return param
        message = 'Extension {} does not have a parameter with global alias {}'
        raise ValueError(message.format(ext.name, self.name))

    def update(self, other_ext):
        self._validate_ext(other_ext)
        self.extensions[other_ext.name] = other_ext

    def _validate_ext(self, other_ext):
        other_param = self.get_param(other_ext)
        for param, ext in self.iteritems():
            if ((not (issubclass(ext, other_ext) or issubclass(other_ext, ext))) and
                    other_param.kind != param.kind):
                message = 'Duplicate global alias {} declared in {} and {} extensions with different types'
                raise LoaderError(message.format(self.name, ext.name, other_ext.name))
            if param.kind != other_param.kind:
                message = 'Two params {} in {} and {} in {} both declare global alias {}, and are of different kinds'
                raise LoaderError(message.format(param.name, ext.name,
                                                 other_param.name, other_ext.name, self.name))

    def __str__(self):
        text = 'GlobalAlias({} => {})'
        extlist = ', '.join(['{}.{}'.format(e.name, p.name) for p, e in self.iteritems()])
        return text.format(self.name, extlist)


class ExtensionLoader(object):
    """
    Discovers, enumerates and loads available devices, configs, etc.
    The loader will attempt to discover things on construction by looking
    in predetermined set of locations defined by default_paths. Optionally,
    additional locations may specified through paths parameter that must
    be a list of additional Python module paths (i.e. dot-delimited).

    """

    _instance = None

    # Singleton
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ExtensionLoader, cls).__new__(cls, *args, **kwargs)
        else:
            for k, v in kwargs.iteritems():
                if not hasattr(cls._instance, k):
                    raise ValueError('Invalid parameter for ExtensionLoader: {}'.format(k))
                setattr(cls._instance, k, v)
        return cls._instance

    def set_load_defaults(self, value):
        self._load_defaults = value
        if value:
            self.packages = merge_lists(self.default_packages, self.packages, duplicates='last')

    def get_load_defaults(self):
        return self._load_defaults

    load_defaults = property(get_load_defaults, set_load_defaults)

    def __init__(self, packages=None, paths=None, ignore_paths=None, keep_going=False, load_defaults=True):
        """
        params::

            :packages: List of packages to load extensions from.
            :paths: List of paths to be searched for Python modules containing
                    WA extensions.
            :ignore_paths: List of paths to ignore when search for WA extensions (these would
                           typically be subdirectories of one or more locations listed in
                           ``paths`` parameter.
            :keep_going: Specifies whether to keep going if an error occurs while loading
                         extensions.
            :load_defaults: Specifies whether extension should be loaded from default locations
                            (WA package, and user's WA directory) as well as the packages/paths
                            specified explicitly in ``packages`` and ``paths`` parameters.

        """
        self._load_defaults = None
        self.logger = logging.getLogger('ExtensionLoader')
        self.keep_going = keep_going
        self.extension_kinds = {ext_tuple.name: ExtensionLoaderItem(ext_tuple)
                                for ext_tuple in settings.extensions}
        self.default_packages = [ext.default_package for ext in self.extension_kinds.values()]

        self.packages = packages or []
        self.load_defaults = load_defaults
        self.paths = paths or []
        self.ignore_paths = ignore_paths or []
        self.extensions = {}
        self.aliases = {}
        self.global_param_aliases = {}
        # create an empty dict for each extension type to store discovered
        # extensions.
        for ext in self.extension_kinds.values():
            setattr(self, '_' + ext.name, {})
        self._load_from_packages(self.packages)
        self._load_from_paths(self.paths, self.ignore_paths)

    def update(self, packages=None, paths=None, ignore_paths=None):
        """ Load extensions from the specified paths/packages
        without clearing or reloading existing extension. """
        if packages:
            self.packages.extend(packages)
            self._load_from_packages(packages)
        if paths:
            self.paths.extend(paths)
            self.ignore_paths.extend(ignore_paths or [])
            self._load_from_paths(paths, ignore_paths or [])

    def clear(self):
        """ Clear all discovered items. """
        self.extensions.clear()
        for ext in self.extension_kinds.values():
            self._get_store(ext).clear()

    def reload(self):
        """ Clear all discovered items and re-run the discovery. """
        self.clear()
        self._load_from_packages(self.packages)
        self._load_from_paths(self.paths, self.ignore_paths)

    def get_extension_class(self, name, kind=None):
        """
        Return the class for the specified extension if found or raises ``ValueError``.

        """
        name, _ = self.resolve_alias(name)
        if kind is None:
            return self.extensions[name]
        ext = self.extension_kinds.get(kind)
        if ext is None:
            raise ValueError('Unknown extension type: {}'.format(kind))
        store = self._get_store(ext)
        if name not in store:
            raise NotFoundError('Extensions {} is not {} {}.'.format(name, get_article(kind), kind))
        return store[name]

    def get_extension(self, name, *args, **kwargs):
        """
        Return extension of the specified kind with the specified name. Any additional
        parameters will be passed to the extension's __init__.

        """
        name, base_kwargs = self.resolve_alias(name)
        kind = kwargs.pop('kind', None)
        kwargs = merge_dicts(base_kwargs, kwargs, list_duplicates='last', dict_type=OrderedDict)
        cls = self.get_extension_class(name, kind)
        extension = _instantiate(cls, args, kwargs)
        extension.load_modules(self)
        return extension

    def get_default_config(self, ext_name):
        """
        Returns the default configuration for the specified extension name. The name may be an alias,
        in which case, the returned config will be augmented with appropriate alias overrides.

        """
        real_name, alias_config = self.resolve_alias(ext_name)
        base_default_config = self.get_extension_class(real_name).get_default_config()
        return merge_dicts(base_default_config, alias_config, list_duplicates='last', dict_type=OrderedDict)

    def list_extensions(self, kind=None):
        """
        List discovered extension classes. Optionally, only list extensions of a
        particular type.

        """
        if kind is None:
            return self.extensions.values()
        if kind not in self.extension_kinds:
            raise ValueError('Unknown extension type: {}'.format(kind))
        return self._get_store(self.extension_kinds[kind]).values()

    def has_extension(self, name, kind=None):
        """
        Returns ``True`` if an extensions with the specified ``name`` has been
        discovered by the loader. If ``kind`` was specified, only returns ``True``
        if the extension has been found, *and* it is of the specified kind.

        """
        try:
            self.get_extension_class(name, kind)
            return True
        except NotFoundError:
            return False

    def resolve_alias(self, alias_name):
        """
        Try to resolve the specified name as an extension alias. Returns a
        two-tuple, the first value of which is actual extension name, and the
        second is a dict of parameter values for this alias. If the name passed
        is already an extension name, then the result is ``(alias_name, {})``.

        """
        alias_name = identifier(alias_name.lower())
        if alias_name in self.extensions:
            return (alias_name, {})
        if alias_name in self.aliases:
            alias = self.aliases[alias_name]
            return (alias.extension_name, alias.params)
        raise NotFoundError('Could not find extension or alias "{}"'.format(alias_name))

    # Internal methods.

    def __getattr__(self, name):
        """
        This resolves methods for specific extensions types based on corresponding
        generic extension methods. So it's possible to say things like ::

            loader.get_device('foo')

        instead of ::

            loader.get_extension('foo', kind='device')

        """
        if name.startswith('get_'):
            name = name.replace('get_', '', 1)
            if name in self.extension_kinds:
                return partial(self.get_extension, kind=name)
        if name.startswith('list_'):
            name = name.replace('list_', '', 1).rstrip('s')
            if name in self.extension_kinds:
                return partial(self.list_extensions, kind=name)
        if name.startswith('has_'):
            name = name.replace('has_', '', 1)
            if name in self.extension_kinds:
                return partial(self.has_extension, kind=name)
        raise AttributeError(name)

    def _get_store(self, ext):
        name = getattr(ext, 'name', ext)
        return getattr(self, '_' + name)

    def _load_from_packages(self, packages):
        try:
            for package in packages:
                for module in walk_modules(package):
                    self._load_module(module)
        except ImportError as e:
            message = 'Problem loading extensions from package {}: {}'
            raise LoaderError(message.format(package, e.message))

    def _load_from_paths(self, paths, ignore_paths):
        self.logger.debug('Loading from paths.')
        for path in paths:
            self.logger.debug('Checking path %s', path)
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
                    try:
                        modname = os.path.splitext(filepath[1:])[0].translate(MODNAME_TRANS)
                        module = imp.load_source(modname, filepath)
                        self._load_module(module)
                    except (SystemExit, ImportError), e:
                        if self.keep_going:
                            self.logger.warn('Failed to load {}'.format(filepath))
                            self.logger.warn('Got: {}'.format(e))
                        else:
                            raise LoaderError('Failed to load {}'.format(filepath), sys.exc_info())
                    except Exception as e:
                        message = 'Problem loading extensions from {}: {}'
                        raise LoaderError(message.format(filepath, e))

    def _load_module(self, module):  # NOQA pylint: disable=too-many-branches
        self.logger.debug('Checking module %s', module.__name__)
        for obj in vars(module).itervalues():
            if inspect.isclass(obj):
                if not issubclass(obj, Extension) or not hasattr(obj, 'name') or not obj.name:
                    continue
                try:
                    for ext in self.extension_kinds.values():
                        if issubclass(obj, ext.cls):
                            self._add_found_extension(obj, ext)
                            break
                    else:  # did not find a matching Extension type
                        message = 'Unknown extension type for {} (type: {})'
                        raise LoaderError(message.format(obj.name, obj.__class__.__name__))
                except LoaderError as e:
                    if self.keep_going:
                        self.logger.warning(e)
                    else:
                        raise e

    def _add_found_extension(self, obj, ext):
        """
            :obj: Found extension class
            :ext: matching extension item.
        """
        self.logger.debug('\tAdding %s %s', ext.name, obj.name)
        key = identifier(obj.name.lower())
        obj.kind = ext.name
        if key in self.extensions or key in self.aliases:
            raise LoaderError('{} {} already exists.'.format(ext.name, obj.name))
        # Extensions are tracked both, in a common extensions
        # dict, and in per-extension kind dict (as retrieving
        # extensions by kind is a common use case.
        self.extensions[key] = obj
        store = self._get_store(ext)
        store[key] = obj
        for alias in obj.aliases:
            alias_id = identifier(alias.name)
            if alias_id in self.extensions or alias_id in self.aliases:
                raise LoaderError('{} {} already exists.'.format(ext.name, obj.name))
            self.aliases[alias_id] = alias

        # Update global aliases list. If a global alias is already in the list,
        # then make sure this extension is in the same parent/child hierarchy
        # as the one already found.
        for param in obj.parameters:
            if param.global_alias:
                if param.global_alias not in self.global_param_aliases:
                    ga = GlobalParameterAlias(param.global_alias)
                    ga.update(obj)
                    self.global_param_aliases[ga.name] = ga
                else:  # global alias already exists.
                    self.global_param_aliases[param.global_alias].update(obj)


# Utility functions.

def _instantiate(cls, args=None, kwargs=None):
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    try:
        return cls(*args, **kwargs)
    except Exception:
        raise LoaderError('Could not load {}'.format(cls), sys.exc_info())
