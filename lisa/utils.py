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

if not LISA_HOME:
    logging.getLogger(__name__).warning('LISA_HOME env var is not set, LISA may misbehave.')

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

def get_subclasses(cls, cls_set=None):
    """Get all indirect subclasses of the class."""
    if cls_set is None:
        cls_set = set()

    for subcls in cls.__subclasses__():
        if subcls not in cls_set:
            cls_set.add(subcls)
            cls_set.update(get_subclasses(subcls, cls_set))

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
        yaml.Constructor.add_constructor('!include', cls._yaml_include_constructor)
        yaml.Constructor.add_constructor('!var', cls._yaml_var_constructor)
        yaml.Constructor.add_multi_constructor('!env:', cls._yaml_env_var_constructor)
        yaml.Constructor.add_multi_constructor('!call:', cls._yaml_call_constructor)

        # Replace unknown tags by a placeholder object containing the data.
        # This happens when the class was not imported at the time the object
        # was deserialized
        yaml.Constructor.add_constructor(None, cls._yaml_unknown_tag_constructor)

        #TODO: remove that once the issue is fixed
        # Workaround for ruamel.yaml bug #244:
        # https://bitbucket.org/ruamel/yaml/issues/244
        yaml.Representer.add_multi_representer(type, yaml.Representer.represent_name)

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

    def __deepcopy__(self):
       return super().__deepcopy__()

Serializable._init_yaml()

class SerializableConfABC(Serializable, abc.ABC):
    _registered_toplevel_keys = {}

    @abc.abstractmethod
    def YAML_MAP_TOP_LEVEL_KEY():
        """Top-level key used when dumping and loading the data to a YAML file.
        This allows using a single file for different purposes.
        """
        pass

    @abc.abstractmethod
    def to_map(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_map(cls, mapping):
        raise NotImplementedError

    @classmethod
    def from_yaml_map(cls, path):
        """
        Allow reloading from a plain mapping, to avoid having to specify a tag
        in the configuration file. The content is hosted under a top-level key
        specified in :attr:`YAML_MAP_TOP_LEVEL_KEY`.
        """

        mapping = cls._from_path(path, fmt='yaml')
        assert isinstance(mapping, Mapping)
        data = mapping[cls.YAML_MAP_TOP_LEVEL_KEY]
        # "unwrap" an extra layer of toplevel key, to play well with !include
        if len(data) == 1 and cls.YAML_MAP_TOP_LEVEL_KEY in data.keys():
            data = data[cls.YAML_MAP_TOP_LEVEL_KEY]
        return cls.from_map(data)

    def to_yaml_map(self, path):
        data = self.to_map()
        mapping = {self.YAML_MAP_TOP_LEVEL_KEY: data}
        return self._to_path(mapping, path, fmt='yaml')

    # Only used with Python >= 3.6, but since that is just a sanity check it
    # should be okay
    @classmethod
    def __init_subclass__(cls, **kwargs):
        # Ensure uniqueness of toplevel key
        toplevel_key = cls.YAML_MAP_TOP_LEVEL_KEY
        if toplevel_key in cls._registered_toplevel_keys:
            raise RuntimeError('Class {name} cannot reuse YAML_MAP_TOP_LEVEL_KEY="{key}" as it is already used by {user}'.format(
                name = cls.__qualname__,
                key = toplevel_key,
                user = cls._registered_toplevel_keys[toplevel_key]
            ))
        else:
            cls._registered_toplevel_keys[toplevel_key] = cls

        super().__init_subclass__(**kwargs)

class DeferredValue:
    """
    Wrapper similar to functools.partial.

    I is a nown class to detect and to derive from to create different
    categories of deferred values.
    """
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.callback(*self.args, **self.kwargs)

    def __str__(self):
        return '<lazy value of {}>'.format(self.callback.__qualname__)

class KeyDescBase(abc.ABC):
    """
    Base class for configuration files key descriptor.

    This allows defining the structure of the configuration file, in order
    to sanitize user input and generate help snippets used in various places.
    """
    INDENTATION = 4 * ' '
    def __init__(self, name, help):
        self.name = name
        self.help = help
        self.parent = None

    @property
    def qualname(self):
        """
        "Qualified" name of the key.

        This is a slash-separated path in the config file from the root to that
        key:
        <parent qualname>/<name>
        """
        if self.parent is None:
            return self.name
        return '/'.join((self.parent.qualname, self.name))

    @staticmethod
    def _get_cls_name(cls, style=None):
        """
        Get a prettily-formated name for the class given as parameter

        :param cls: class to get the name from
        :type cls: type

        :param style: When "rst", a RestructuredText snippet is returned
        :param style: str

        """
        if cls is None:
            return 'None'
        mod_name = inspect.getmodule(cls).__name__
        mod_name = mod_name + '.' if mod_name not in ('builtins', '__main__') else ''
        name = mod_name + cls.__qualname__
        if style == 'rst':
            name = ':class:`~{}`'.format(name)
        return name

    @abc.abstractmethod
    def get_help(self, style=None):
        """
        Get a help message describing the key.

        :param style: When "rst", ResStructuredText formatting may be applied
        :param style: str
        """
        pass

    @abc.abstractmethod
    def validate_val(self, val):
        """
        Validate a value to be used for that key.

        :raises TypeError: When the value has the wrong type
        :raises ValueError: If the value does not comply with some other
            constraints. Note that constraints should ideally be encoded in the
            type itself, to make help message as straightforward as possible.
        """
        pass

class KeyDesc(KeyDescBase):
    """
    Key descriptor describing a leaf key in the configuration.

    :param name: Name of the key

    :param help: Short help message describing the use of that key

    :param classinfo: sequence of allowed types for that key. As a special
        case, `None` is allowed in that sequence of types, even though it is
        not strictly speaking a type.
    :type classinfo: collections.abc.Sequence
    """
    def __init__(self, name, help, classinfo):
        super().__init__(name=name, help=help)
        # isinstance's style classinfo
        self.classinfo = tuple(classinfo)

    def validate_val(self, val):
        """
        Check that the value is an instance of one of the type specified in the
        ``self.classinfo``.

        If the value is not an instance of any of these types, then a
        :exc:`TypeError` is raised corresponding to the first type in the
        tuple, which is assumed to be the main one.
        """
        # Or if that key is supposed to hold a value
        classinfo = self.classinfo
        key = self.qualname
        def get_excep(key, val, classinfo, cls, msg):
            classinfo = ' or '.join(self._get_cls_name(cls) for cls in classinfo)
            msg = ': ' + msg if msg else ''
            return TypeError('Key "{key}" is an instance of {actual_cls}, but should be instance of {classinfo}{msg}. Help: {help}'.format(
                        key=key,
                        actual_cls=self._get_cls_name(type(val)),
                        classinfo=classinfo,
                        msg=msg,
                        help=self.help,
                    ), key)

        def checkinstance(key, val, classinfo):
            excep_list = []
            for cls in classinfo:
                if cls is None:
                    if val is not None:
                        excep_list.append(
                            get_excep(key, val, classinfo, cls, 'Key is not None')
                        )
                # Some classes are able to raise a more detailed
                # exception than just the boolean return value of
                # __instancecheck__
                elif hasattr(cls, 'instancecheck'):
                    try:
                        cls.instancecheck(val)
                    except TypeError as e:
                        excep_list.append(
                            get_excep(key, val, classinfo, cls, str(e))
                        )
                else:
                    if not isinstance(val, cls):
                        excep_list.append(
                            get_excep(key, val, classinfo, cls, None)
                        )

            # If no type was validated, we raise an exception. This will
            # only show the exception for the first class to be tried,
            # which is the primary one.
            if len(excep_list) == len(classinfo):
                raise excep_list[0]

        # DeferredValue will be checked when they are computed
        if not isinstance(val, DeferredValue):
            checkinstance(key, val, classinfo)

    def get_help(self, style=None):
        prefix = '*' if style == 'rst' else '|-'
        return '{prefix} {key} ({classinfo}){help}'.format(
            prefix=prefix,
            key=self.name,
            classinfo=' or '.join(
                self._get_cls_name(key_cls, style='rst')
                for key_cls in self.classinfo
            ),
            help=': ' + self.help if self.help else ''
        )

class LevelKeyDesc(KeyDescBase, Mapping):
    """
    Key descriptor defining a hierarchical level in the configuration.

    :param name: name of the key in the configuration

    :param help: Short help describing the use of the keys inside that level

    :param children: collections.abc.Sequence of :class:`KeyDescBase` defining the allowed keys
        under that level
    :type children: collections.abc.Sequence

    Children keys will get this key assigned as a parent when passed to the
    constructor.

    """
    def __init__(self, name, help, children):
        super().__init__(name=name, help=help)
        self.children = children

        # Fixup parent for easy nested declaration
        for key_desc in self.children:
            key_desc.parent = self

    @property
    def _key_map(self):
        return {
            key_desc.name: key_desc
            for key_desc in self.children
        }
    def __iter__(self):
        return iter(self._key_map)
    def __len__(self):
        return len(self._key_map)
    def __getitem__(self, key):
        self.check_allowed_key(key)
        return self._key_map[key]

    def check_allowed_key(self, key):
        """
        Checks that a given key is allowed under that levels
        """
        try:
            key_desc = self._key_map[key]
        except KeyError:
            try:
                closest_match = difflib.get_close_matches(
                    word=key,
                    possibilities=self._key_map.keys(),
                    n=1,
                )[0]
            except IndexError:
                closest_match = ''
            else:
                closest_match = ', maybe you meant "{}" ?'.format(closest_match)

            parent = self.qualname
            raise KeyError('Key "{key}" is not allowed in {parent}{maybe}'.format(
                key=key,
                parent=parent,
                maybe=closest_match,
            ), parent, key)

    def validate_val(self, conf):
        """Validate a mapping to be used as a configuration source"""
        if not isinstance(conf, Mapping):
            key = self.qualname
            raise TypeError('Configuration of {key} must be a Mapping'.format(
                key=key,
            ), key)
        for key, val in conf.items():
            self[key].validate_val(val)

    def get_help(self, style=None):
        idt = self.INDENTATION
        prefix = '*' if style == 'rst' else '+-'
        # Nasty hack: adding an empty ResStructuredText comment between levels
        # of nested list avoids getting extra blank line between list items.
        # That prevents ResStructuredText from thinking each item must be a
        # paragraph.
        suffix = '\n\n..\n\n' + idt if style == 'rst' else '\n'
        help_ = '{prefix} {key}:{help}{suffix}'.format(
            prefix=prefix,
            suffix=suffix,
            key=self.name,
            help= ' ' + self.help if self.help else '',
            idt=idt,
        )
        nl = '\n' + idt
        help_ += nl.join(
            key_desc.get_help(style=style).replace('\n', nl)
            for key_desc in self.children
        )
        if style == 'rst':
            help_ += '\n\n..\n'

        return help_

class TopLevelKeyDesc(LevelKeyDesc):
    """
    Top-level key descriptor, which defines the top-level key to use in the
    configuration files.

    This top-level key is omitted in all interfaces except for the
    configuration file, since it only reflects the configuration class
    """
    pass

class MultiSrcConfMeta(abc.ABCMeta):
    """
    Metaclass of :class:`MultiSrcConf`.

    It will use the docstring of the class, using it as a ``str.format`` template
    with the ``{generated_help}`` placeholder replaced by a snippet of
    ResStructuredText containing the list of allowed keys.

    .. note:: Since the dosctring is interpreted as a template, "{" and "}"
        characters must be doubled to appear in the final output.
    """
    def __new__(metacls, name, bases, dct, **kwargs):
        new_cls = super().__new__(metacls, name, bases, dct, **kwargs)
        if not inspect.isabstract(new_cls):
            doc = new_cls.__doc__
            if doc:
                # Create a ResStructuredText preformatted block
                generated_help = '\n' + new_cls.get_help(style='rst')
                new_cls.__doc__ = doc.format(generated_help=generated_help)
        return new_cls

class MultiSrcConf(SerializableConfABC, Loggable, Mapping, metaclass=MultiSrcConfMeta):
    """
    Base class providing layered configuration management.

    :param conf: collections.abc.Mapping to initialize the configuration with. This must be
        optional, in which case it is assumed the object will contain a default
        base configuration.
    :type conf: collections.abc.Mapping

    :param src: Name of the source added when passing ``conf``
    :param src: str

    The class inherits from :class:`collections.abc.Mapping`, which means it
    can be used like a readonly dict. Writing to it is handled by a different
    API that allows naming the source of values that are stored.

    Each configuration key can be either a leaf key, that holds a value, or
    a level key that allows to defined nested levels. The allowed keys is set
    by the ``STRUCTURE`` class attribute.

    Each leaf key can hold different values coming from different named
    sources.  By default, the last added source will have the highest priority
    and will be served when looking up that key. A different priority order can
    be defined for a specific key if needed.

    .. seealso:: :class:`KeyDescBase`
    """

    @abc.abstractmethod
    def STRUCTURE():
        """
        Class attribute defining the structure of the configuration file, as a
        instance of :class:`TopLevelKeyDesc`
        """
        pass

    DEFAULT_SRC = {}
    """
    Source added automatically using :meth:`add_src` under the name 'default'
    when instances are built.
    """

    def __init__(self, conf=None, src='conf'):
        self._nested_init(
            parent=None,
            structure=self.STRUCTURE,
            src_prio=[]
        )
        if conf is not None:
            self.add_src(src, conf)

        # Give some preset in the the lowest prio source
        if self.DEFAULT_SRC:
            self.add_src('default', self.DEFAULT_SRC, fallback=True)

    @classmethod
    def get_help(cls, *args, **kwargs):
        return cls.STRUCTURE.get_help(*args, **kwargs)

    def _nested_init(self, parent, structure, src_prio):
        """Called to initialize nested instances of the class for nested
        configuration levels."""
        self._structure = structure
        "Structure of that level of configuration"
        # Make a copy to avoid sharing it with the parent
        self._src_prio = copy.copy(src_prio)
        "List of sources in priority order (1st item is highest prio)"
        self._src_override = {}
        "Map of keys to map of source to values"
        self._key_map = {}
        "Key/value map of leaf values"
        self._sublevel_map = {}
        "Key/sublevel map of nested configuration objects"

        # Build the tree of objects for nested configuration mappings
        for key, key_desc in self._structure.items():
            if isinstance(key_desc, LevelKeyDesc):
                self._sublevel_map[key] = self._nested_new(
                    parent = self,
                    structure = key_desc,
                    src_prio = self._src_prio,
                )

    @classmethod
    def _nested_new(cls, *args, **kwargs):
        new = cls.__new__(cls)
        new._nested_init(*args, **kwargs)
        return new

    def __copy__(self):
        """Shallow copy of the nested configuration tree, without
        duplicating the leaf values."""
        cls = type(self)
        new = cls.__new__(cls)
        new.__dict__ = copy.copy(self.__dict__)

        # make a shallow copy of the attributes
        attr_set = set(self.__dict__.keys())
        # we do not duplicate the structure, since it is a readonly bit of
        # configuration. That would break parent links in it
        attr_set -= {'_structure'}
        for attr in attr_set:
            new.__dict__[attr] = copy.copy(self.__dict__[attr])

        # Do the same with sublevels
        new._sublevel_map = {
            key: sublevel.__copy__()
            for key, sublevel in new._sublevel_map.items()
        }

        return new

    def to_map(self):
        """
        Export the configuration as a mapping.

        The return value should preserve key-specific priority override list,
        which is not done if directly passing that instance to ``dict()``.
        """
        mapping = dict()
        # For each key, get the highest prio value
        mapping['conf'] = self._get_effective_map()
        src_override = self._get_nested_src_override()
        if src_override:
            mapping['source'] = src_override
        return mapping

    @classmethod
    def from_map(cls, mapping):
        """
        Create a new configuration instance, using the output of :meth:`to_map`
        """
        conf = mapping.get('conf', {})
        src_override = mapping.get('source', {})

        plat_conf = cls(conf)
        plat_conf.force_src_nested(src_override)
        return plat_conf

    def add_src(self, src, conf, filter_none=False, fallback=False):
        """
        Add a source of configuration.

        :param src: Name of the soruce to add
        :type src: str

        :param conf: Nested mapping of key/values to overlay
        :type conf: collections.abc.Mapping

        :param filter_none: Ignores the keys that have a ``None`` value. That
            simplifies the creation of the mapping, by having keys always
            present. That should not be used if ``None`` value for a key is
            expected, as opposit to not having that key set at all.
        :type filter_none: bool

        :param fallback: If True, the source will be added as a fallback, which
            means at the end of the priority list. By default, the source will
            have the highest priority and will be used unless a key-specific
            priority override is setup.
        :type fallback: bool

        This method provides a way to update the configuration, by importing a
        mapping as a new source.
        """
        # Filter-out None values, so they won't override actual data from
        # another source
        if filter_none:
            conf = {
                k: v for k, v in conf.items()
                if v is not None
            }

        self._structure.validate_val(conf)

        for key, val in conf.items():
            key_desc = self._structure[key]
            # Dispatch the nested mapping to the right sublevel
            if isinstance(key_desc, LevelKeyDesc):
                # sublevels have already been initialized when the root object
                # was created.
                self._sublevel_map[key].add_src(src, val, filter_none=filter_none, fallback=fallback)
            # Otherwise that is a leaf value that we store at that level
            else:
                self._key_map.setdefault(key, {})[src] = val

        if src not in self._src_prio:
            if fallback:
                self._src_prio.append(src)
            else:
                self._src_prio.insert(0, src)

    def set_default_src(self, src_prio):
        """
        Set the default source priority list.

        :param src_prio: list of source names, first is the highest priority
        :type src_prio: collections.abc.Sequence(str)

        Adding sources using :meth:`add_src` in the right order is preferable,
        but the default priority order can be specified using that method.
        """

        # Make a copy of the list to make sure it is not modified behind our back
        self._src_prio = list(src_prio)
        for sublevel in self._sublevel_map.values():
            sublevel.set_default_src(src_prio)

    def force_src_nested(self, key_src_map):
        """
        Force the source priority list for all the keys defined in the nested
        mapping ``key_src_map``

        :param key_src_map: nested mapping of keys to priority list of sources
        :type key_src_map: collections.abc.Mapping
        """
        for key, src_or_map in key_src_map.items():
            key_desc = self._structure[key]
            if isinstance(key_desc, LevelKeyDesc):
                mapping = src_or_map
                self._sublevel_map[key].force_src_nested(mapping)
            else:
                self.force_src(key, src_or_map)

    def force_src(self, key, src_prio):
        """
        Force the source priority list for a given key

        :param key: name of the key. Only leaf keys are allowed here, since
            level keys have no source on their own.
        :type key: str

        :param src_prio: List of sources in priority order (first is highest
            priority).  Special value ``None`` can be used to remove the
            key-specific priority override, so the default priority list will
            be used instead.
        :type src_prio: collections.abc.Sequence(str) or None
        """

        key_desc = self._structure[key]
        if isinstance(key_desc, LevelKeyDesc):
            key = key_desc.qualname
            raise ValueError('Cannot force source of the sub-level "{key}" in {cls}'.format(
                key=key,
                cls=type(self).__qualname__
            ), key)

        # None means removing the src override for that key
        if src_prio is None:
            self._src_override.pop(key, None)
        else:
            self._src_override[key] = src_prio

    def _get_nested_src_override(self):
        # Make a copy to avoid modifying it
        override = copy.copy(self._src_override)
        for key, sublevel in self._sublevel_map.items():
            sublevel_override = sublevel._get_nested_src_override()
            # Skip sublevel keys if they don't have any override to specify
            if sublevel_override:
                override[key] = sublevel_override
        return override

    def _get_effective_map(self, eval_deferred=True):
        """
        Return the effective mapping by taking values from the highest
        priority source for each key, recursively.
        """
        mapping = {}
        for key in self._key_map.keys():
            try:
                val = self.get_key(key, eval_deferred=eval_deferred)
            # If the source of that key does not exist, we just ignore it
            except KeyError:
                pass
            else:
                mapping[key] = val

        mapping.update(
            (key, sublevel._get_effective_map(eval_deferred=eval_deferred))
            for key, sublevel in self._sublevel_map.items()
        )

        return mapping

    def _resolve_prio(self, key):
        if key not in self._key_map:
            return []

        # Get the priority list from the prio override list, or just the
        # default prio list
        src_list = self._src_override.get(key, self._src_prio)

        # Only include a source if it holds an actual value for that key
        src_list = [
            src for src in src_list
            if src in self._key_map[key]
        ]
        return src_list

    def resolve_src(self, key):
        """
        Get the source name that will be used to serve the value of ``key``.
        """
        key_desc = self._structure[key]

        if isinstance(key_desc, LevelKeyDesc):
            key = key_desc.qualname
            raise ValueError('Key "{key}" is a nested configuration level, it does not have a source on its own.'.format(
                key=key,
            ), key)

        # Get the priority list from the prio override list, or just the
        # default prio list
        src_prio = self._resolve_prio(key)
        if src_prio:
            return src_prio[0]
        else:
            key = key_desc.qualname
            raise KeyError('Could not find any source for key "{key}"'.format(
                key=key,
            ), key)

    def _eval_deferred_val(self, src, key):
        key_desc = self._structure[key]
        val = self._key_map[key][src]
        if isinstance(val, DeferredValue):
            val = val()
            key_desc.validate_val(val)
            self._key_map[key][src] = val
        return val

    def eval_deferred(self, cls=DeferredValue, src=None):
        """
        Evaluate instances of :class:`DeferredValue` that can be used for
        values that are expensive to compute.

        :param cls: Only evaluate values of instances of that class. This can
            be used to have different categories of :class:`DeferredValue` by
            subclassing.
        :type cls: subclass of :class:`DeferredValue`

        :param src: If not ``None``, only evaluate values that were added under
            that source name.
        :type src: str or None
        """
        for key, src_map in self._key_map.items():
            for src_, val in src_map.items():
                if src is not None and src != src_:
                    continue
                if isinstance(val, cls):
                    self._eval_deferred_val(src_, key)

        for sublevel in self._sublevel_map.values():
            sublevel.eval_deferred(cls, src)

    def __getstate__(self):
        """
        Filter instances of :class:`DeferredValue` that are not computed
        already since their runtime parameters will probably not be available
        after deserialization.

        If needed, call :meth:`eval_deferred` before serializing.
        """
        # Filter-out DeferredValue key-value pairs before serialization
        key_map = {
            key: {
                src: v
                for src, v in src_map.items()
                if not isinstance(v, DeferredValue)
            }
            for key, src_map in self._key_map.items()
        }
        # keys without any source are just removed
        key_map = {
            k: src_map for k, src_map in key_map.items()
            if src_map
        }
        state = copy.copy(super().__getstate__())
        state['_key_map'] = key_map

        return state

    def get_key(self, key, src=None, eval_deferred=True):
        """
        Get the value of the given key.

        :param key: name of the key to lookup
        :type key: str

        :param src: If not None, look up the value of the key in that source
        :type src: str or None

        :param eval_deferred: If True, evaluate instances of
            :class:`DeferredValue` if needed
        :type eval_deferred: bool

        .. note:: Using the indexing operator ``self[key]`` is preferable in
            most cases , but this method provides more parameters.
        """
        key_desc = self._structure[key]

        if isinstance(key_desc, LevelKeyDesc):
            return self._sublevel_map[key]

        # Compute the source to use for that key
        if src is None:
            src = self.resolve_src(key)

        try:
            val = self._key_map[key][src]
        except KeyError:
            key = key_desc.qualname
            raise KeyError('Key "{key}" is not available from source "{src}"'.format(
                key=key,
                src=src,
            ), key)

        if eval_deferred:
            val = self._eval_deferred_val(src, key)

        try:
            frame_conf = inspect.stack()[2]
        except Exception:
            caller, filename, lineno = ['<unknown>'] * 3
        else:
            caller = frame_conf.function
            filename = frame_conf.filename
            lineno = frame_conf.lineno

        self.get_logger().debug('{caller} ({filename}:{lineno}) has used key {key} from source "{src}": {val}'.format(
            key=key_desc.qualname,
            src=src,
            val=val,
            caller=caller,
            filename=filename,
            lineno=lineno,
        ))
        return val

    def get_src_map(self, key):
        """
        Get a mapping of all sources for the given ``key``, in priority order
        (first item is the highest priority source).
        """
        key_desc = self._structure[key]
        if isinstance(key_desc, LevelKeyDesc):
            key = key_desc.qualname
            raise ValueError('Key "{key}" is a nested configuration level in {cls}, it does not have a source on its own.'.format(
                key=key,
                cls=type(self).__qualname__,
            ), key)

        return OrderedDict(
            (src, self._eval_deferred_val(src, key))
            for src in self._resolve_prio(key)
        )

    def pretty_format(self, eval_deferred=False):
        """
        Give a pretty string representation of the configuration.

        :param eval_deferred: If True, evaluate all deferred values before
            printing.
        :type eval_deferred: bool
        """
        out = []
        idt_style = ' '
        for k, v in self.items(eval_deferred=eval_deferred):
            v_cls = type(v)
            is_sublevel = k in self._sublevel_map
            if is_sublevel:
                v = v.pretty_format(eval_deferred=eval_deferred)
                # If there is no content, just skip that sublevel entirely
                if not v.strip():
                   continue
            else:
                v = str(v)

            if is_sublevel or '\n' in v:
                v = '\n' + v
            else:
                v = ' ' + v

            if is_sublevel:
                k_str = '+- ' + k
                v_prefix = '    '
            else:
                k_str = '|- ' + k
                v_prefix = '|   '

            v = v.replace('\n', '\n' + v_prefix)

            out.append('{k}{src}{cls}:{v}'.format(
                k=k_str,
                cls='' if is_sublevel else ' ('+v_cls.__qualname__+')',
                src='' if is_sublevel else ' from '+self.resolve_src(k),
                v=v,
            ))
        return '\n'.join(out)

    def __str__(self):
        return self.pretty_format()

    def __getitem__(self, key):
        return self.get_key(key)

    def _get_key_names(self):
        return list(self._key_map.keys()) + list(self._sublevel_map.keys())

    def __iter__(self):
        return iter(self._get_key_names())

    def __len__(self):
        return len(self._get_key_names())

    def items(self, eval_deferred=True):
        """
        Override the default definition of
        ``collections.abc.Mapping.items()`` to allow not evaluating deferred
        values if necessary.
        """

        return (
            (k, self.get_key(k, eval_deferred=eval_deferred))
            for k in self.keys()
        )

    def _ipython_key_completions_(self):
        "Allow Jupyter keys completion in interactive notebooks"
        return self.keys()

class GenericContainerMetaBase(type):
    """
    Base class for the metaclass of generic containers.
    """
    def __instancecheck__(cls, instance):
        try:
            cls.instancecheck(instance)
        except TypeError:
            return False
        else:
            return True

class GenericContainerBase:
    """
    Base class for generic containers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).instancecheck(self)

class GenericMappingMeta(GenericContainerMetaBase, type(Mapping)):
    """
    Metaclass for generic mapping containers.

    It provides an ``__instancecheck__`` implementation that checks the type
    of the keys and values. This make it suitable for input sanitizing based
    on type checking.
    """
    def instancecheck(cls, instance):
        if not isinstance(instance, Mapping):
            raise TypeError('not a Mapping')

        k_type, v_type = cls._type
        for k, v in instance.items():
            if not isinstance(k, k_type):
                raise TypeError('Key "{key}" of type {actual_cls} should be of type {k_type}'.format(
                    key=k,
                    actual_cls=type(k).__qualname__,
                    k_type=k_type.__qualname__,
                ), k)

            if not isinstance(v, v_type):
                raise TypeError('Value of {actual_cls} key "{key}" should be of type {v_type}'.format(
                    key=k,
                    actual_cls=type(v).__qualname__,
                    v_type=v_type.__qualname__,
                ), k)

class TypedDict(GenericContainerBase, dict, metaclass=GenericMappingMeta):
    """
    Subclass of dict providing keys and values type check.
    """
    pass

class GenericSequenceMeta(GenericContainerMetaBase, type(Sequence)):
    """Similar to :class:`GenericMappingMeta` for sequences"""
    def instancecheck(cls, instance):
        if not isinstance(instance, Sequence):
            raise TypeError('not a Sequence')

        type_ = cls._type
        for i, x in enumerate(instance):
            if not isinstance(x, type_):
                raise TypeError('Item #{i} "{val}" of type {actual_cls} should be of type {type_}'.format(
                    i=i,
                    val=x,
                    actual_cls=type(x).__qualname__,
                    type_=type_.__qualname__
                ), i)

class TypedList(GenericContainerBase, list, metaclass=GenericSequenceMeta):
    """
    Subclass of list providing keys and values type check.
    """
    pass

class IntIntDict(TypedDict):
    _type = (int, int)

class IntList(TypedList):
    _type = int

class IntIntListDict(TypedDict):
    _type = (int, IntList)

class IntListList(TypedList):
    _type = IntList

class StrList(TypedList):
    _type = str

class StrIntListDict(TypedDict):
    _type = (str, IntList)

def setup_logging(filepath='logging.conf', level=logging.INFO):
    """
    Initialize logging used for all the LISA modules.

    :param filepath: the relative or absolute path of the logging
                     configuration to use. Relative path uses
                     :attr:`lisa.utils.LISA_HOME` as base folder.
    :type filepath: str

    :param level: the default log level to enable, logging.INFO by default
    :type level: int
    """

    # Load the specified logfile using an absolute path
    if not os.path.isabs(filepath):
        filepath = os.path.join(LISA_HOME, filepath)

    if not os.path.exists(filepath):
        raise ValueError('Logging configuration file not found in: {}'\
                         .format(filepath))
    logging.config.fileConfig(filepath)
    logging.getLogger().setLevel(level)

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

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
