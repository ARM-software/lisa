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

import ruamel.yaml
from ruamel.yaml import YAML

# Do not infer the value using __file__, since it will break later on when
# lisa package is installed in the site-package locations using pip, which
# are typically not writable.
LISA_HOME = os.getenv('LISA_HOME')
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
    return functools.lru_cache(maxsize=None, typed=True)(callable_)

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

class Serializable:
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
        yaml.allow_unicode = (cls.YAML_ENCODING == 'utf-8')
        yaml.default_flow_style = False
        yaml.indent = 4
        yaml.Constructor.add_constructor('!include', cls._yaml_include_constructor)
        yaml.Constructor.add_multi_constructor('!call:', cls._yaml_call_constructor)
        #TODO: remove that once the issue is fixed
        # Workaround for ruamel.yaml bug #244:
        # https://bitbucket.org/ruamel/yaml/issues/244
        yaml.Representer.add_multi_representer(type, yaml.Representer.represent_name)


    @classmethod
    def _yaml_call_constructor(cls, loader, suffix, node):
        """
        Provide a !call tag in YAML that can be used to call a Python
        callable with a mapping of arguments:
        !call:package.module.Class
            arg1: foo
            arg2: bar
            arg3: 42
        is will execute: package.module.Class(arg1='foo', arg2='bar', arg3=42)
        """
        # Restrict to keyword arguments to have improve stability of
        # configuration files.
        kwargs = loader.construct_mapping(node, deep=True)
        return loader.make_python_instance(suffix, node, kwds=kwargs, newobj=False)

    @classmethod
    def _yaml_include_constructor(cls, loader, node):
        """
        Provide a !include tag in YAML that can be used to include the content of
        another YAML file. Environment variables are expanded in the given path.
        """
        path = loader.construct_scalar(node)
        assert isinstance(path, str)
        path = os.path.expandvars(path)
        with open(path, 'r', encoding=cls.YAML_ENCODING) as f:
            return cls._yaml.load(f)

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
        # On Python >= 3.6, __init_subclass__ will take care of that
        if sys.version_info < (3, 6):
            # Better late than never. Doing it here avoids using a metaclass
            # just to register the class. If we don't do that, yaml_tag class
            # attribute will be ignored.
            cls._yaml.register_class(type(instance))

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

        with open(str(filepath), **kwargs) as fh:
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

   # Will only be called with Python >= 3.6
    def __init_subclass__(cls, **kwargs):
        # Register the class to ensure yaml_tag will be used
        cls._yaml.register_class(cls)
        super().__init_subclass__(**kwargs)

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
        specified in :attr:`YAML_MAP_TOP_LEVEL_KEY'.
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

class MultiSrcConf(SerializableConfABC, Loggable, Mapping):
    @abc.abstractmethod
    def STRUCTURE():
        """
        Class attribute defining the structure of the configuration file, as a
        nested dictionary mirroring the allowed keys, with values containing
        the type of the key or a nested dictionary.
        """
        pass

    def __init__(self, conf=None, src='default'):
        self._nested_init(
            parent = None,
            level = None,
            structure = self.STRUCTURE,
            src_prio = []
        )
        if conf is not None:
            self.add_src(src, conf)

    def _nested_init(self, parent, level, structure, src_prio):
        """Called to initialize nested instances of the class for nested
        configuration."""
        self._parent = parent
        "Link to parent object"
        self._level = level
        "Name of the current level in the configuration"
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
        for key, cls_or_map in self._structure.items():
            if self._is_sublevel_key(key):
                self._sublevel_map[key] = self._nested_new(
                    parent = self,
                    level = key,
                    structure = cls_or_map,
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
        # Avoid infinite recursion
        attr_set -= {'_parent'}
        for attr in attr_set:
            new.__dict__[attr] = copy.copy(self.__dict__[attr])

        # Do the same with sublevels
        new._sublevel_map = {
            key: sublevel.__copy__()
            for key, sublevel in new._sublevel_map.items()
        }

        # fixup _parent
        for sublevel in new._sublevel_map.values():
           sublevel._parent = new

        return new

    def to_map(self):
        mapping = dict()
        # For each key, get the highest prio value
        mapping['conf'] = self._get_effective_map()
        src_override = self._get_nested_src_override()
        if src_override:
            mapping['source'] = src_override
        return mapping

    @classmethod
    def from_map(cls, mapping):
        conf = mapping['conf']
        src_override = mapping.get('source', {})

        plat_conf = cls(conf)
        plat_conf.force_src_nested(src_override)
        return plat_conf

    def _get_parent_levels(self):
        if self._parent is not None:
            return self._parent._get_parent_levels() + [self._level]
        else:
            return []

    def _get_key_name(self, key):
        return '/'.join(self._get_parent_levels() + [key])

    def validate_src(self, conf):
        """Validate a mapping to be used as a source"""
        if not isinstance(conf, Mapping):
            raise ValueError('Configuration of {cls} must be a Mapping'.format(
                cls=type(self).__qualname__,
            ))
        for key, val in conf.items():
            self.validate_val(key, val)

    def _get_sublevel(self, key):
        return self._sublevel_map[key]

    def _is_sublevel_key(self, key):
        return isinstance(self._structure[key], Mapping)

    def _check_allowed_key(self, key):
        try:
            cls_or_map = self._structure[key]
        except KeyError:
            raise ValueError('Key "{key}" is not allowed in {cls}'.format(
                key=self._get_key_name(key),
                cls=type(self).__qualname__
            ))

    def validate_val(self, key, val):
        # Check we only have allowed key names
        self._check_allowed_key(key)

        # If that key is supposed to be a level of nested keys
        if self._is_sublevel_key(key):
            return self._get_sublevel(key).validate_src(val)
        # Or if that key is supposed to hold a value
        else:
            cls = self._structure[key]
            def raise_excep(key, val, cls, msg):
                raise ValueError('Key "{key}" is an instance of {actual_cls}, but should be instance of {cls}{msg}'.format(
                            key=self._get_key_name(key),
                            actual_cls=type(val),
                            cls=cls,
                            msg = ': ' + msg if msg else ''
                        ))
            # DeferredValue will be checked when they are computed
            if isinstance(val, DeferredValue):
                pass
            # Some classes are able to raise a more detailed exception than
            # just the boolean return value of __instancecheck__
            elif hasattr(cls, '_instancecheck'):
                try:
                    cls._instancecheck(val)
                except ValueError as e:
                    raise_excep(key, val, cls, str(e))
            else:
                if not isinstance(val, cls):
                    raise_excep(key, val, cls, None)

    def add_src(self, src, conf, filter_none=False, fallback=False):
        # Filter-out None values, so they won't override actual data from
        # another source
        if filter_none:
            conf = {
                k: v for k, v in conf.items()
                if v is not None
            }

        self.validate_src(conf)

        for key, val in conf.items():
            # Dispatch the nested mapping to the right sublevel
            if self._is_sublevel_key(key):
                # sublevels have already been initialized when the root object
                # was created.
                self._get_sublevel(key).add_src(src, val, filter_none=filter_none, fallback=fallback)
            # Otherwise that is a leaf value that we store at that level
            else:
                self._key_map.setdefault(key, {})[src] = val

        if src not in self._src_prio:
            if fallback:
                self._src_prio.append(src)
            else:
                self._src_prio.insert(0, src)

    def set_default_src(self, src_prio):
        # Make a copy of the list to make sure it is not modified behind our back
        self._src_prio = list(src_prio)
        for sublevel in self._sublevel_map.values():
            sublevel.set_default_src(src_prio)

    def force_src_nested(self, key_src_map):
        for key, src_or_map in key_src_map.items():
            if self._is_sublevel_key(key):
                mapping = src_or_map
                self._get_sublevel(key).force_src_nested(mapping)
            else:
                self.force_src(key, src_or_map)

    def force_src(self, key, src_prio):
        self._check_allowed_key(key)
        if self._is_sublevel_key(key):
            raise ValueError('Cannot force source of the sub-level "{level}" in {cls}'.format(
                level=self._get_key_name(key),
                cls=type(self).__qualname__
            ))

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
        self._check_allowed_key(key)

        cls_name = type(self).__qualname__
        key_name = self._get_key_name(key)
        if self._is_sublevel_key(key):
            raise ValueError('Key "{key}" is a nested configuration level in {cls}, it does not have a source on its own.'.format(
                key=key_name,
                cls=cls_name,
            ))

        # Get the priority list from the prio override list, or just the
        # default prio list
        src_prio = self._resolve_prio(key)
        if src_prio:
            return src_prio[0]
        else:
            raise KeyError('Could not find any source for key "{key}" in {cls}'.format(
                key=key_name,
                cls=cls_name
            ))

    def _eval_deferred_val(self, src, key):
        val = self._key_map[key][src]
        if isinstance(val, DeferredValue):
            val = val()
            self.validate_val(key, val)
            self._key_map[key][src] = val
        return val


    def eval_deferred(self, cls=DeferredValue, src=None):
        for key, src_map in self._key_map.items():
            for src_, val in src_map.items():
                if src is not None and src != src_:
                    continue
                if isinstance(val, cls):
                    self._eval_deferred_val(src_, key)

        for sublevel in self._sublevel_map.values():
            sublevel.eval_deferred(cls, src)

    def __getstate__(self):
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
        state = copy.copy(self.__dict__)
        state['_key_map'] = key_map

        return state

    def get_key(self, key, src=None, eval_deferred=True):
        self._check_allowed_key(key)

        with contextlib.suppress(KeyError):
            return self._get_sublevel(key)

        cls_name = type(self).__qualname__
        key_name = self._get_key_name(key)
        # Compute the source to use for that key
        if src is None:
            src = self.resolve_src(key)

        try:
            val = self._key_map[key][src]
        except KeyError:
            raise KeyError('Key "{key}" is not available from source "{src}" in {cls}'.format(
                key=key_name,
                src=src,
                cls=cls_name,
            ))

        if eval_deferred:
            val = self._eval_deferred_val(src, key)

        frame_conf = inspect.stack()[2]
        self.get_logger().debug('{cls} used by {caller} ({filename}:{lineno}) from "{src}" source: {key}={val}'.format(
            key=key_name,
            src=src,
            val=val,
            caller=frame_conf.function,
            filename=frame_conf.filename,
            lineno=frame_conf.lineno,
            cls=cls_name,
        ))
        return val

    def get_src_map(self, key):
        self._check_allowed_key(key)
        if self._is_sublevel_key(key):
            raise ValueError('Key "{key}" is a nested configuration level in {cls}, it does not have a source on its own.'.format(
                key=self._get_key_name(key),
                cls=type(self).__qualname__,
            ))

        return OrderedDict(
            (src, self._eval_deferred_val(src, key))
            for src in self._resolve_prio(key)
        )

    def pretty_format(self, eval_deferred=False):
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
        return (
            (k, self.get_key(k, eval_deferred=eval_deferred))
            for k in self.keys()
        )

class GenericContainerMetaBase(type):
    def __instancecheck__(cls, instance):
        try:
            cls._instancecheck(instance)
        except ValueError:
            return False
        else:
            return True

# That is needed to make ruamel.yaml consider these classes as objects, so it
# uses __reduce_ex__
ruamel.yaml.Representer.add_multi_representer(GenericContainerMetaBase, ruamel.yaml.Representer.represent_object)

class GenericContainerBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self)._instancecheck(self)

    @classmethod
    def __reduce_ex__(cls, version):
        return (cls._build, (cls._type,))

class GenericMappingMeta(GenericContainerMetaBase, type(Mapping)):
    @memoized
    def __getitem__(cls, type_):
        if type_ is None:
            return cls

        class new_cls(cls):
            _type = type_

        suffix = '[{},{}]'.format(*(
            t.__qualname__ for t in type_
        ))
        new_cls.__qualname__ = cls.__qualname__ + suffix
        new_cls.__name__ = cls.__name__ + suffix
        return new_cls

    def _instancecheck(cls, instance):
        if not isinstance(instance, Mapping):
            raise ValueError('not a Mapping')

        k_type, v_type = cls._type
        for k, v in instance.items():
            if not isinstance(k, k_type):
                raise ValueError('Key "{key}" of type {actual_cls} should be of type {k_type}'.format(
                    key=k,
                    actual_cls=type(k).__qualname__,
                    k_type=k_type.__qualname__,
                ))

            if not isinstance(v, v_type):
                raise ValueError('Value of {actual_cls} key "{key}" should be of type {v_type}'.format(
                    key=k,
                    actual_cls=type(v).__qualname__,
                    v_type=v_type.__qualname__,
                ))

class TypedDict(GenericContainerBase, dict, metaclass=GenericMappingMeta):
    # Workaround issues in ruamel.yaml when it comes to complex setups
    @staticmethod
    def _build(types):
        return TypedDict[types]

class GenericSequenceMeta(GenericContainerMetaBase, type(Sequence)):
    @memoized
    def __getitem__(cls, type_):
        if type_ is None:
            return cls

        class new_cls(cls):
            _type = type_

        suffix = '[{}]'.format(type_.__qualname__)
        new_cls.__qualname__ = cls.__qualname__ + suffix
        new_cls.__name__ = cls.__name__ + suffix
        return new_cls

    def _instancecheck(cls, instance):
        if not isinstance(instance, Sequence):
            raise ValueError('not a Sequence')

        type_ = cls._type
        for i, x in enumerate(instance):
            if not isinstance(x, type_):
                raise ValueError('Item #{i} "{val}" of type {actual_cls} should be of type {type_}'.format(
                    i=i,
                    val=x,
                    actual_cls=type(x).__qualname__,
                    type_=type_.__qualname__
                ))

class TypedList(GenericContainerBase, list, metaclass=GenericSequenceMeta):
    # Workaround issues in ruamel.yaml when it comes to complex setups
    @staticmethod
    def _build(types):
        return TypedList[types]

def setup_logging(filepath='logging.conf', level=logging.INFO):
    """
    Initialize logging used for all the LISA modules.

    :param filepath: the relative or absolute path of the logging
                     configuration to use. Relative path uses
                     :attr:`lisa.utils.LISA_HOME` as base folder.
    :type filepath: str

    :param level: the default log level to enable, INFO by default
    :type level: logging.<level> or int in [0..50]
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

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
