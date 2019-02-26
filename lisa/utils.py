# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import abc
import copy
from collections.abc import Mapping, Sequence
from collections import OrderedDict
import contextlib
import inspect
import logging
import logging.config
import functools
import pickle
import sys
import os
import importlib
import pkgutil
import operator
import numbers
import difflib
import threading
import itertools

import ruamel.yaml
from ruamel.yaml import YAML

# Do not infer the value using __file__, since it will break later on when
# lisa package is installed in the site-package locations using pip, which
# are typically not writable.
LISA_HOME = os.getenv('LISA_HOME')
"""
The detected location of your LISA installation
"""

class Loggable:
    """
    A simple class for uniformly named loggers
    """

    @classmethod
    def get_logger(cls, suffix=None):
        cls_name = cls.__name__
        module = inspect.getmodule(cls)
        if module:
            name = module.__name__ + '.' + cls_name
        else:
            name = cls_name
        if suffix:
            name += '.' + suffix
        return logging.getLogger(name)

    @classmethod
    def log_locals(cls, var_names=None, level='debug'):
        """
        Debugging aid: log the local variables of the calling function

        :param var_names: List of variable names to display, or all of them if
            left to default.
        :type var_names: list(str)

        :param level: log level to use.
        :type level: str
        """
        level = getattr(logging, level.upper())
        call_frame = sys._getframe(1)

        for name, val in call_frame.f_locals.items():
            if var_names and name not in var_names:
                continue
            cls.get_logger().log(level, 'Local variable: {}: {}'.format(name, val))

def get_subclasses(cls, only_leaves=False, cls_set=None):
    """Get all indirect subclasses of the class."""
    if cls_set is None:
        cls_set = set()

    for subcls in cls.__subclasses__():
        if subcls not in cls_set:
            to_be_added = set(get_subclasses(subcls, only_leaves, cls_set))
            to_be_added.add(subcls)
            if only_leaves:
                to_be_added = {
                    cls for cls in to_be_added
                    if not cls.__subclasses__()
                }
            cls_set.update(to_be_added)

    return cls_set

class HideExekallID:
    """Hide the subclasses in the simplified ID format of exekall.

    That is mainly used for uninteresting classes that do not add any useful
    information to the ID. This should not be used on domain-specific classes
    since alternatives may be used by the user while debugging for example.
    Hiding too many classes may lead to ambiguity, which is exactly what the ID
    is fighting against.
    """
    pass

def memoized(callable_):
    """
    Decorator to memoize the result of a callable,
    based on :func:`functools.lru_cache`
    """
    # It is important to have one separate call to lru_cache for every call to
    # memoized, otherwise all uses of the decorator will end up using the same
    # wrapper and all hells will break loose.

    # maxsize should be a power of two for better speed, see:
    # https://docs.python.org/3/library/functools.html#functools.lru_cache
    return functools.lru_cache(maxsize=1024, typed=True)(callable_)

def resolve_dotted_name(name):
    """Only resolve names where __qualname__ == __name__, i.e the callable is a
    module-level name."""
    mod_name, callable_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, callable_name)

def get_all_subclasses(cls, cls_set=None):
    if cls_set is None:
        cls_set = set()
    cls_set.add(cls)
    for subcls in cls.__subclasses__():
        get_all_subclasses(subcls, cls_set)

    return cls_set

def import_all_submodules(pkg):
    """Import all submodules of a given package."""
    return [
        loader.find_module(module_name).load_module(module_name)
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__)
    ]

class UnknownTagPlaceholder:
    def __init__(self, tag, data, location=None):
        self.tag = tag
        self.data = data
        self.location = location

    def __str__(self):
        return '<UnknownTagPlaceholder of {}>'.format(self.tag)

class Serializable(Loggable):
    """
    A helper class for YAML serialization/deserialization

    Not to be used on its own - instead, your class should inherit from this
    class to gain serialization superpowers.
    """
    serialized_whitelist = []
    serialized_blacklist = []
    serialized_placeholders = dict()

    YAML_ENCODING = 'utf-8'
    "Encoding used for YAML files"

    DEFAULT_SERIALIZATION_FMT = 'yaml'
    "Default format used when serializing objects"

    _yaml = YAML(typ='unsafe')

    @classmethod
    def _init_yaml(cls):
        """
        Needs to be called only once when the module is imported. Since that is
        done at module-level, there is no need to do that from user code.
        """
        yaml = cls._yaml
        # If allow_unicode=True, true unicode characters will be written to the
        # file instead of being replaced by escape sequence.
        yaml.allow_unicode = ('utf' in cls.YAML_ENCODING)
        yaml.default_flow_style = False
        yaml.indent = 4
        yaml.constructor.add_constructor('!include', cls._yaml_include_constructor)
        yaml.constructor.add_constructor('!var', cls._yaml_var_constructor)
        yaml.constructor.add_multi_constructor('!env:', cls._yaml_env_var_constructor)
        yaml.constructor.add_multi_constructor('!call:', cls._yaml_call_constructor)

        # Replace unknown tags by a placeholder object containing the data.
        # This happens when the class was not imported at the time the object
        # was deserialized
        yaml.constructor.add_constructor(None, cls._yaml_unknown_tag_constructor)

    @classmethod
    def _yaml_unknown_tag_constructor(cls, loader, node):
        # Get the basic data types that can be expressed using the YAML syntax,
        # without using any tag-specific constructor
        data = None
        for constructor in (
            loader.construct_scalar,
            loader.construct_sequence,
            loader.construct_mapping
        ):
            try:
                data = constructor(node)
            except ruamel.yaml.constructor.ConstructorError:
                continue
            else:
                break

        tag = node.tag
        cls.get_logger().debug('Could not find constructor for YAML tag "{tag}" ({mark}), using a placeholder'.format(
            tag=tag,
            mark=str(node.start_mark).strip()
        ))

        return UnknownTagPlaceholder(tag, data, location=node.start_mark)

    @classmethod
    def _yaml_call_constructor(cls, loader, suffix, node):
        """
        Provide a !call tag in YAML that can be used to call a Python
        callable with a mapping of arguments:
        # There is no space after "call:"
        !call:package.module.Class
            arg1: foo
            arg2: bar
            arg3: 42
        will execute: package.module.Class(arg1='foo', arg2='bar', arg3=42)
        """
        # Restrict to keyword arguments to have improve stability of
        # configuration files.
        kwargs = loader.construct_mapping(node, deep=True)
        return loader.make_python_instance(suffix, node, kwds=kwargs, newobj=False)


    # Allow !include to use relative paths from the current file. Since we
    # introduce a global state, we use thread-local storage.
    _included_path = threading.local()
    _included_path.val = None
    @staticmethod
    @contextlib.contextmanager
    def _set_relative_include_root(path):
        old = Serializable._included_path.val
        Serializable._included_path.val = path
        try:
            yield
        finally:
            Serializable._included_path.val = old

    @classmethod
    def _yaml_include_constructor(cls, loader, node):
        """
        Provide a !include tag in YAML that can be used to include the content of
        another YAML file. Environment variables are expanded in the given path.

        !include /foo/$ENV_VAR/bar.yml
        """
        path = loader.construct_scalar(node)
        assert isinstance(path, str)
        path = os.path.expandvars(path)

        # Paths are relative to the file that is being included
        if not os.path.isabs(path):
            path = os.path.join(Serializable._included_path.val, path)

        with cls._set_relative_include_root(path):
            with open(path, 'r', encoding=cls.YAML_ENCODING) as f:
                return cls._yaml.load(f)

    @classmethod
    def _yaml_env_var_constructor(cls, loader, suffix, node):
        """
        Provide a !include tag in YAML that can be used to include the content
        of an environment variable, and converting it to a Python type:

        !env:int MY_ENV_VAR
        """
        varname = loader.construct_scalar(node)
        assert isinstance(varname, str)

        type_ = loader.find_python_name(suffix, node.start_mark)
        assert callable(type_)
        try:
            value = os.environ[varname]
        except KeyError:
            cls._warn_missing_env(varname)
            return None
        else:
            return type_(value)

    @classmethod
    # memoize to avoid displaying the same message twice
    @memoized
    def _warn_missing_env(cls, varname):
            cls.get_logger().warning('Environment variable "{}" not defined, using None value'.format(varname))

    @classmethod
    def _yaml_var_constructor(cls, loader, node):
        """
        Provide a !var tag in YAML that can be used to reference a module-level
        variable:

        !var package.module.var
        """
        varname = loader.construct_scalar(node)
        assert isinstance(varname, str)
        return loader.find_python_name(varname, node.start_mark)

    def to_path(self, filepath, fmt=None):
        """
        Serialize the object to a file

        :param filepath: The path of the file in which the object will be dumped
        :type filepath: str

        :param fmt: Serialization format.
        :type fmt: str
        """

        data = self
        return self._to_path(data, filepath, fmt)

    @classmethod
    def _to_path(cls, instance, filepath, fmt):
        if fmt is None:
            fmt = cls.DEFAULT_SERIALIZATION_FMT

        if fmt == 'yaml':
            kwargs = dict(mode='w', encoding=cls.YAML_ENCODING)
            dumper = cls._yaml.dump
        elif fmt == 'pickle':
            kwargs = dict(mode='wb')
            dumper = pickle.dump
        else:
            raise ValueError('Unknown format "{}"'.format(fmt))

        with open(str(filepath), **kwargs) as fh:
            dumper(instance, fh)

    @classmethod
    def from_path(cls, filepath, fmt=None):
        """
        Deserialize an object from a file

        :param filepath: The path of file in which the object has been dumped
        :type filepath: str

        :param fmt: Serialization format.
        :type fmt: str

        :raises AssertionError: if the deserialized object is not an instance
                                of the class.
        """
        instance = cls._from_path(filepath, fmt)
        assert isinstance(instance, cls)
        return instance

    @classmethod
    def _from_path(cls, filepath, fmt):
        filepath = str(filepath)
        if fmt is None:
            fmt = cls.DEFAULT_SERIALIZATION_FMT

        if fmt == 'yaml':
            kwargs = dict(mode='r', encoding=cls.YAML_ENCODING)
            loader = cls._yaml.load
        elif fmt == 'pickle':
            kwargs = dict(mode='rb')
            loader = pickle.load
        else:
            raise ValueError('Unknown format "{}"'.format(fmt))

        with cls._set_relative_include_root(os.path.dirname(filepath)):
            with open(filepath, **kwargs) as fh:
                instance = loader(fh)

        return instance

    def __getstate__(self):
       """
       Filter the instance's attributes upon serialization.

       The following class attributes can be used to customize the serialized
       content:
           * :attr:`serialized_whitelist`: list of attribute names to
             serialize. All other attributes will be ignored and will not be
             saved/restored.

           * :attr:`serialized_blacklist`: list of attribute names to not
             serialize.  All other attributes will be saved/restored.

           * serialized_placeholders: Map of attribute names to placeholder
             values. These attributes will not be serialized, and the
             placeholder value will be used upon restoration.

           If both :attr:`serialized_whitelist` and
           :attr:`serialized_blacklist` are specified,
           :attr:`serialized_blacklist` is ignored.
       """

       dct = copy.copy(self.__dict__)
       if self.serialized_whitelist:
           dct = {attr: dct[attr] for attr in self.serialized_whitelist}

       elif self.serialized_blacklist:
           for attr in self.serialized_blacklist:
               dct.pop(attr, None)

       for attr, placeholder in self.serialized_placeholders.items():
           dct.pop(attr, None)

       return dct

    def __setstate__(self, dct):
       if self.serialized_placeholders:
           dct.update(copy.deepcopy(self.serialized_placeholders))
       self.__dict__ = dct

    def __copy__(self):
       """Make sure that copying the class still works as usual, without
       dropping some attributes by defining __copy__
       """
       return super().__copy__()

Serializable._init_yaml()

def setup_logging(filepath='logging.conf', level=logging.INFO):
    """
    Initialize logging used for all the LISA modules.

    :param filepath: the relative or absolute path of the logging
                     configuration to use. Relative path uses
                     :attr:`lisa.utils.LISA_HOME` as base folder.
    :type filepath: str

    :param level: the default log level to enable
    :type level: int
    """

    # Load the specified logfile using an absolute path
    if not os.path.isabs(filepath):
        filepath = os.path.join(LISA_HOME, filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError('Logging configuration file not found in: {}'\
                         .format(filepath))

    # Set the level first, so the config file can override with more details
    logging.getLogger().setLevel(level)
    logging.config.fileConfig(filepath)

    logging.info('Using LISA logging configuration:')
    logging.info('  %s', filepath)

class ArtifactPath(str, Loggable, HideExekallID):
    """Path to a folder that can be used to store artifacts of a function.
    This must be a clean folder, already created on disk.
    """
    def __new__(cls, root, relative, *args, **kwargs):
        root = os.path.realpath(str(root))
        relative = str(relative)
        # we only support paths relative to the root parameter
        assert not os.path.isabs(relative)
        absolute = os.path.join(root, relative)

        # Use a resolved absolute path so it is more convenient for users to
        # manipulate
        path = os.path.realpath(absolute)

        path_str = super().__new__(cls, path, *args, **kwargs)
        # Record the actual root, so we can relocate the path later with an
        # updated root
        path_str.root = root
        path_str.relative = relative
        return path_str

    def __fspath__(self):
        return str(self)

    def __reduce__(self):
        # Serialize the path relatively to the root, so it can be relocated
        # easily
        relative = self.relative_to(self.root)
        return (type(self), (self.root, relative))

    def relative_to(self, path):
        return os.path.relpath(str(self), start=str(path))

    def with_root(self, root):
        # Get the path relative to the old root
        relative = self.relative_to(self.root)

        # Swap-in the new root and return a new instance
        return type(self)(root, relative)

def groupby(iterable, key=None):
    # We need to sort before feeding to groupby, or it will fail to establish
    # the groups as expected.
    iterable = sorted(iterable, key=key)
    return itertools.groupby(iterable, key=key)

def group_by_value(mapping, key_sort=lambda x: x):
    """
    Group a mapping by its values

    :param mapping: Mapping to reverse
    :type mapping: collections.abc.Mapping

    :param key_sort: The ``key`` parameter to a :func:`sorted` call on the
      mapping keys
    :type key_sort: collections.abc.Callable

    :rtype: collections.OrderedDict

    The idea behind this method is to "reverse" a mapping, IOW to create a new
    mapping that has the passed mapping's values as keys. Since different keys
    can point to the same value, the new values will be lists of old keys.

    **Example:**

    >>> group_by_value({0: 42, 1: 43, 2: 42})
    OrderedDict([(42, [0, 2]), (43, [1])])
    """
    if not key_sort:
        # Just conserve the order
        key_sort = lambda x: 0

    return OrderedDict(
        (val, sorted((k for k, v in key_group), key=key_sort))
        for val, key_group in groupby(mapping.items(), key=operator.itemgetter(1))
    )

def deduplicate(seq, keep_last=True, key=lambda x: x):
    """
    Deduplicate items in the given sequence and return a list.
    :param seq: Sequence to deduplicate
    :type Seq: collections.abc.Sequence

    :param key: Key function that will be used to determine duplication.  It
        takes one item at a time, returning a hashable key value
    :type key: collections.abc.Callable

    :param keep_last: If True, will keep the last occurence of each duplicated
        items. Otherwise, keep the first occurence.
    :type keep_last: bool
    """
    reorder = (lambda seq: seq) if keep_last else reversed
    # Use an OrderedDict to keep original ordering of the sequence
    dedup = OrderedDict(
        (key(x), x)
        for x in reorder(seq)
    )
    return list(reorder(dedup.values()))

def get_nested_key(mapping, key_path):
    """
    Get a key in a nested mapping

    :param mapping: The mapping to lookup in
    :type mapping: collections.abc.Mapping

    :param key_path: Path to the key in the mapping, in the form of a list of
        keys.
    :type key_path: list
    """
    if not key_path:
        return mapping
    for key in key_path[:-1]:
        mapping = mapping[key]
    return mapping[key_path[-1]]

def set_nested_key(mapping, key_path, val, level=None):
    """
    Set a key in a nested mapping

    :param mapping: The mapping to update
    :type mapping: collections.abc.MutableMapping

    :param key_path: Path to the key in the mapping, in the form of a list of
        keys.
    :type key_path: list

    :param level: Factory used when creating a level is needed. By default,
        ``type(mapping)`` will be called without any parameter.
    :type level: collections.abc.Callable
    """
    assert key_path

    if level is None:
        # This should work for dict and most basic structures
        level = type(mapping)

    for key in key_path[:-1]:
        try:
            mapping = mapping[key]
        except KeyError:
            new_level = level()
            mapping[key] = new_level
            mapping = new_level

    mapping[key_path[-1]] = val


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
