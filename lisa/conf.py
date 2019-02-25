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
import difflib
import inspect
import itertools
import textwrap

from lisa.utils import Serializable, Loggable, get_nested_key, set_nested_key

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
                self._get_cls_name(key_cls, style=style)
                for key_cls in self.classinfo
            ),
            help=': ' + self.help if self.help else ''
        )

    def pretty_format(self, v):
        """
        Format the value for pretty printing.

        :param v: Value of the key that is being printed
        :type v: object

        :return: A string
        """
        return str(v)

class MissingBaseKeyError(KeyError):
    """
    Exception raised when a base key needed to compute a derived key is missing.
    """
    pass

class DerivedKeyDesc(KeyDesc):
    """
    Key descriptor describing a key derived from other keys

    Derived keys cannot be added from a source, since they are purely computed
    out of other keys. It is also not possible to change their source
    priorities. To achieve that, set the source priorities on the keys it is
    based on.

    :param base_key_paths: List of paths to the keys this key is derived from.
        The paths in the form of a list of string are relative to the current
        level, and cannot reference any level above the current one.
    :type base_key_paths: list(list(str))

    :param compute: Function used to compute the value of the key. It takes a
        dictionary of base keys specified in ``base_key_paths`` as only
        parameter and is expected to return the key's value.
    :type compute: collections.abc.Callable
    """

    def __init__(self, name, help, classinfo, base_key_paths, compute):
        super().__init__(name=name, help=help, classinfo=classinfo)
        self._base_key_paths = base_key_paths
        self._compute = compute

    @property
    def help(self):
        return '(derived from {}) '.format(
            ', '.join(sorted(
                self._get_base_key_qualname(path)
                for path in self._base_key_paths
            ))
        ) + self._help

    @help.setter
    def help(self, val):
        self._help = val

    @staticmethod
    def _get_base_key_val(conf, path):
        return get_nested_key(conf, path)

    @staticmethod
    def _get_base_key_src(conf, path):
        conf = get_nested_key(conf, path[:-1])
        return conf.resolve_src(path[-1])

    def _get_base_key_qualname(self, key_path):
        return self.parent.qualname + '/' + '/'.join(key_path)

    def _get_base_conf(self, conf):
        try:
            base_conf = {}
            for key_path in self._base_key_paths:
                val = self._get_base_key_val(conf, key_path)
                set_nested_key(base_conf, key_path, val)
            return base_conf
        except KeyError as e:
            raise MissingBaseKeyError('Missing value for base key "{base_key}" in order to compute derived key "{derived_key}": {msg}'.format(
                derived_key=self.qualname,
                base_key=e.args[1],
                msg=e.args[0],
            )) from e

    def compute_val(self, conf):
        base_conf = self._get_base_conf(conf)
        val = self._compute(base_conf)
        self.validate_val(val)
        return val

    def get_src(self, conf):
        return ','.join(
            '{src}({key})'.format(
                src=self._get_base_key_src(conf, path),
                key=self._get_base_key_qualname(path),
            )
            for path in self._base_key_paths
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
        suffix = '\n\n..\n\n' if style == 'rst' else '\n'
        suffix += idt
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

class MultiSrcConfABC(Serializable, abc.ABC, metaclass=MultiSrcConfMeta):
    _registered_toplevel_keys = {}

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
        in the configuration file. The content is hosted under the top-level
        key specified in ``STRUCTURE``.
        """

        toplevel_key = cls.STRUCTURE.name

        mapping = cls._from_path(path, fmt='yaml')
        assert isinstance(mapping, Mapping)
        data = mapping[toplevel_key] or {}
        # "unwrap" an extra layer of toplevel key, to play well with !include
        if len(data) == 1 and toplevel_key in data.keys():
            data = data[toplevel_key]
        return cls.from_map(data)

    def to_yaml_map(self, path):
        data = self.to_map()
        mapping = {self.STRUCTURE.name: data}
        return self._to_path(mapping, path, fmt='yaml')

    # Only used with Python >= 3.6, but since that is just a sanity check it
    # should be okay
    @classmethod
    def __init_subclass__(cls, **kwargs):
        # Ignore abstract classes, since there can be no instance of them
        if not inspect.isabstract(cls):
            # Ensure uniqueness of toplevel key
            toplevel_key = cls.STRUCTURE.name
            if toplevel_key in cls._registered_toplevel_keys:
                raise RuntimeError('Class {name} cannot reuse top level key "{key}" as it is already used by {user}'.format(
                    name = cls.__qualname__,
                    key = toplevel_key,
                    user = cls._registered_toplevel_keys[toplevel_key]
                ))
            else:
                cls._registered_toplevel_keys[toplevel_key] = cls

        super().__init_subclass__(**kwargs)

class MultiSrcConf(MultiSrcConfABC, Loggable, Mapping):
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

    def _nested_init(self, structure, src_prio):
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
            # Derived keys cannot be set, since they are purely derived from
            # other keys
            elif isinstance(key_desc, DerivedKeyDesc):
                raise ValueError('Cannot set a value for a derived key "{key}"'.format(
                    key=key_desc.qualname,
                ), key_desc.qualname)
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
        qual_key = key_desc.qualname
        if isinstance(key_desc, LevelKeyDesc):
            raise ValueError('Cannot force source of the sub-level "{key}"'.format(
                key=qual_key,
            ), qual_key)
        elif isinstance(key_desc, DerivedKeyDesc):
            raise ValueError('Cannot force source of a derived key "{key}"'.format(
                key=qual_key,
            ), qual_key)
        else:
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
        key_desc = self._structure[key]

        if isinstance(key_desc, DerivedKeyDesc):
            return [key_desc.get_src(self)]
        elif key not in self._key_map:
            return []
        else:
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
        elif isinstance(key_desc, DerivedKeyDesc):
            # Specifying a source is an error for a derived key
            if src is not None:
                key = key_desc.qualname
                raise ValueError('Cannot specify the source when getting "{key}" since it is a derived key'.format(
                    key=key,
                    src=src,
                ), key)

            val = key_desc.compute_val(self)
            src = self.resolve_src(key)
        else:
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
            val=key_desc.pretty_format(val),
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
            raise ValueError('Key "{key}" is a nested configuration level, it does not have a source on its own.'.format(
                key=key,
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

        # We add the derived keys when pretty-printing, for the sake of
        # completeness. This will not honor eval_deferred for base keys.
        def derived_items():
            for key in self._get_derived_key_names():
                try:
                    yield key, self[key]
                except MissingBaseKeyError:
                    continue

        for k, v in itertools.chain(
            self.items(eval_deferred=eval_deferred),
            derived_items()
        ):
            v_cls = type(v)

            key_desc = self._structure[k]
            is_sublevel = isinstance(key_desc, LevelKeyDesc)

            if is_sublevel:
                v = v.pretty_format(eval_deferred=eval_deferred)
                # If there is no content, just skip that sublevel entirely
                if not v.strip():
                   continue
            else:
                v = key_desc.pretty_format(v)

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
        return sorted(list(self._key_map.keys()) + list(self._sublevel_map.keys()))

    def _get_derived_key_names(self):
        return sorted(
            key
            for key, key_desc in self._structure.items()
            if isinstance(key_desc, DerivedKeyDesc)
        )

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
        regular_keys = set(self.keys())
        # For autocompletion purposes, we show the derived keys
        derived_keys = set(self._get_derived_key_names())
        return sorted(regular_keys + derived_keys)


class ConfigurableMeta(abc.ABCMeta):
    def __new__(metacls, name, bases, dct, **kwargs):
        new_cls = super().__new__(metacls, name, bases, dct, **kwargs)
        try:
            # inherited CONF_CLASS will not be taken into account if we look at
            # the dictionary directly
            conf_cls = dct['CONF_CLASS']
        except KeyError:
            return new_cls

        # Link the configuration to the signature of __init__
        sig = inspect.signature(new_cls.__init__)
        init_kwargs_key_map = new_cls._get_kwargs_key_map(sig, conf_cls)
        # What was already there has priority over auto-detected bindings
        init_kwargs_key_map.update(dct.get('INIT_KWARGS_KEY_MAP', {}))
        new_cls.INIT_KWARGS_KEY_MAP = init_kwargs_key_map

        # Create an instance with default configuration, to merge it with
        # defaults taken from __init__
        default_conf = conf_cls()
        default_conf.add_src(
            src='__init__-default',
            conf=metacls._get_default_conf(sig, init_kwargs_key_map),
            # Default configuration set in the conf class still has priority
            fallback=True,
            # When an __init__ parameter has a None default value, we don't
            # add any default value. That avoids failing the type check for
            # keys that really need to be of a certain type when specified.
            filter_none=True,
        )
        # Since a MultiSrcConf is a Mapping, it is useable as a source
        conf_cls.DEFAULT_SRC = default_conf

        # Update the docstring by using the configuration help
        docstring = new_cls.__doc__
        if docstring:
            new_cls.__doc__ = textwrap.dedent(docstring).format(
                configurable_params=new_cls._get_rst_param_doc()
            )

        return new_cls

    def _get_kwargs_key_map(cls, sig, conf_cls):
        """
        Map implicitely keys in the conf class that matches param names.
        """
        def iter_param_key(sig):
            return (
                (param, param.replace('_', '-'))
                for param in sig.parameters.keys()
            )

        return {
            param: [key]
            for param, key in iter_param_key(sig)
            if key in conf_cls.STRUCTURE
        }

    @staticmethod
    def _get_default_conf(sig, kwargs_key_map):
        """
        Get a default configuration source based on the the default parameter
        values.
        """
        default_conf = {}
        for param, param_desc in sig.parameters.items():
            try:
                conf_path = kwargs_key_map[param]
            except KeyError:
                continue
            else:
                default = param_desc.default
                if default is not param_desc.empty:
                    set_nested_key(default_conf, conf_path, default)

        return default_conf

    def _get_param_key_desc_map(cls):
        return {
            param: get_nested_key(cls.CONF_CLASS.STRUCTURE, conf_path)
            for param, conf_path in cls.INIT_KWARGS_KEY_MAP.items()
        }

    def _get_rst_param_doc(cls):
        def get_type_name(t):
            if t is None:
                return 'None'
            else:
                return t.__qualname__

        return '\n'.join(
            ':param {param}: {help}\n:type {param}: {type}\n'.format(
                param=param,
                help=key_desc.help,
                type=' or '.join(get_type_name(t) for t in key_desc.classinfo),
            )
            for param, key_desc
            in cls._get_param_key_desc_map().items()
        )
        return out


class Configurable(abc.ABC, metaclass=ConfigurableMeta):
    """
    Pear a regular class with a configuration class.

    The pearing is achieved by inheriting from :class:`Configurable` and
    setting ``CONF_CLASS`` attribute. The benefits are:

        * The docstring of the class is processed as a string template and
          ``{configurable_params}`` is replaced with a Sphinx-compliant list of
          parameters. The help and type of each parameter is extracted from the
          configuration class.
        * The ``DEFAULT_SRC`` attribute of the configuration class is updated
          with non-``None`` default values of the class ``__init__`` parameters.
        * The :meth:`~Configurable.conf_to_init_kwargs` method allows turning a
          configuration object into a dictionary suitable for passing to
          ``__init__`` as ``**kwargs``.
        * The :meth:`~Configurable.check_init_param` method allows checking
          types of ``__init__`` parameters according to what is specified in the
          configuration class.

    Most of the time, the configuration keys and ``__init__`` parameters have
    the same name (modulo underscore/dashes which are handled automatically).
    In that case, the mapping between config keys and ``__init__`` parameters
    is done without user intervention. When that is not the case, the
    ``INIT_KWARGS_KEY_MAP`` class attribute can be used. Its a dictionary with
    keys being ``__init__`` parameter names, and values being path to
    configuration key. That path is a list of strings to take into account
    sublevels like ``['level-key', 'sublevel', 'foo']``.

    .. note:: A given configuration class must be peared to only one class.
        Otherwise, the ``DEFAULT_SRC`` conf class attribute will be updated
        multiple times, leading to unexpected results.

    .. note:: Some services offered by :class:`Configurable` are not extended
        to subclasses of a class using it. For example, it would not make sense
        to update ``DEFAULT_SRC`` using a subclass ``__init__`` parameters.

    """
    INIT_KWARGS_KEY_MAP = {}

    @classmethod
    def conf_to_init_kwargs(cls, conf):
        """
        Turn a configuration object into a dictionary suitable for passing to
        ``__init__`` as ``**kwargs``.
        """
        kwargs = {}
        for param, conf_path in cls.INIT_KWARGS_KEY_MAP.items():
            try:
                val = get_nested_key(conf, conf_path)
            except KeyError:
                continue
            kwargs[param] = val

        return kwargs

    @classmethod
    def check_init_param(cls, **kwargs):
        """
        Take the same parameters as ``__init__``, and check their types
        according to what is specified in the configuration class.

        :raises TypeError: When the wrong type is detected for a parameter.
        """
        for param, key_desc in cls._get_param_key_desc_map().items():
            if param in kwargs:
                key_desc.validate_val(kwargs[param])


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

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
