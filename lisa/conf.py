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
from collections.abc import Mapping
from collections import OrderedDict
import difflib
import inspect
import itertools
import textwrap
import logging
import re
import contextlib
import pprint
import os
import io
import functools
import threading
import weakref

from ruamel.yaml.comments import CommentedMap

import lisa
from lisa.utils import (
    Serializable, Loggable, get_nested_key, set_nested_key, get_call_site,
    is_running_sphinx, get_cls_name, HideExekallID, get_subclasses, groupby,
    import_all_submodules,
)


class DeferredValue:
    """
    Wrapper similar to :func:`functools.partial` allowing to defer computation
    of the value until the key is actually used.

    Once computed, the deferred value is replaced by the value that was
    computed. This is useful for values that are very costly to compute, but
    should be used with care as it means it will usually not be available in
    the offline :class:`lisa.platforms.platinfo.PlatformInfo` instances. This
    means that client code such as submodules of ``lisa.analysis`` will
    typically not have it available (unless :meth:`~MultiSrcConf.eval_deferred`
    was called) although they might need it.
    """

    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self._is_computing = False

    def __call__(self, key_desc=None):
        # Make sure we don't reenter the callback, to avoid infinite loops.
        if self._is_computing:
            key = key_desc.qualname if key_desc else '<unknown>'
            raise KeyComputationRecursionError(f'Recursion error while computing deferred value for key: {key}', key)

        try:
            self._is_computing = True
            return self.callback(*self.args, **self.kwargs)
        finally:
            self._is_computing = False

    def __str__(self):
        return f'<lazy value of {self.callback.__qualname__}>'


class DeferredExcep(DeferredValue):
    """
    Specialization of :class:`DeferredValue` to lazily raise an exception.

    :param excep: Exception to raise when the value is used.
    :type excep: BaseException
    """

    def __init__(self, excep):
        self.excep = excep

        def callback():
            raise self.excep
        super().__init__(callback=callback)

    def __str__(self):
        return f'<lazy {self.excep.__class__.__qualname__} exception>'


class TopLevelKeyError(ValueError):
    """
    Exception raised when no top-level key matches the expected one in the
    given configuration file.
    """
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return f'Key "{self.key}" needs to appear at the top level'


class KeyDescBase(abc.ABC):
    """
    Base class for configuration files key descriptor.

    This allows defining the structure of the configuration file, in order
    to sanitize user input and generate help snippets used in various places.
    """
    INDENTATION = 4 * ' '
    _VALID_NAME_PATTERN = r'^[a-zA-Z0-9-]+$'

    def __init__(self, name, help):
        # pylint: disable=redefined-builtin

        self._check_name(name)
        self.name = name
        self.help = help
        self.parent = None

    @classmethod
    def _check_name(cls, name):
        if not re.match(cls._VALID_NAME_PATTERN, name):
            raise ValueError(f'Invalid key name "{name}". Key names must match: {self._VALID_NAME_PATTERN}')

    @property
    def qualname(self):
        """
        "Qualified" name of the key.

        This is a slash-separated path in the config file from the root to that
        key:
        <parent qualname>/<name>
        """
        return '/'.join(self.path)

    @property
    def path(self):
        """
        Path in the config file from the root to that key.

        .. note:: This includes the top-level key name, which must be removed
            before it's fed to :meth:`MultiSrcConf.get_nested_key`.
        """
        curr = [self.name]
        if self.parent is None:
            return curr
        return self.parent.path + curr

    @abc.abstractmethod
    def get_help(self, style=None):
        """
        Get a help message describing the key.

        :param style: When "rst", ResStructuredText formatting may be applied
        :param style: str
        """

    @abc.abstractmethod
    def validate_val(self, val):
        """
        Validate a value to be used for that key.

        :raises TypeError: When the value has the wrong type
        :raises ValueError: If the value does not comply with some other
            constraints. Note that constraints should ideally be encoded in the
            type itself, to make help message as straightforward as possible.
        """


class KeyDesc(KeyDescBase):
    """
    Key descriptor describing a leaf key in the configuration.

    :param name: Name of the key

    :param help: Short help message describing the use of that key

    :param classinfo: sequence of allowed types for that key. As a special
        case, `None` is allowed in that sequence of types, even though it is
        not strictly speaking a type.
    :type classinfo: collections.abc.Sequence

    :param newtype: If specified, a type with the given name will be created
        for that key with that name. Otherwise, a camel-case name derived from
        the key name will be used: ``toplevel-key/sublevel/mykey`` will give a
        type named `SublevelMykey`. This class will be exposed as an attribute
        of the parent :class:`MultiSrcConf` (which is why the toplevel key is
        omitted from its name). A getter will also be created on the parent
        configuration class, so that the typed key is exposed to ``exekall``.
        If the key is not present in the configuration object, the getter will
        return ``None``.
    :type newtype: str or None

    :param deepcopy_val: If ``True``, the values will be deepcopied upon
        lookup. This prevents accidental modification of mutable types (like
        lists) by the user.
    :type deepcopy_val: bool
    """

    def __init__(self, name, help, classinfo, newtype=None, deepcopy_val=True):
        # pylint: disable=redefined-builtin

        super().__init__(name=name, help=help)
        # isinstance's style classinfo
        self.classinfo = tuple(classinfo)
        self._newtype = newtype
        self.deepcopy_val = deepcopy_val

    @property
    def newtype(self):
        if self._newtype:
            return self._newtype
        else:
            compos = itertools.chain.from_iterable(
                x.split('-')
                for x in self.path[1:]
            )
            return ''.join(x.title() for x in compos)

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
            # pylint: disable=unused-argument
            classinfo = ' or '.join(get_cls_name(cls) for cls in classinfo)
            msg = ': ' + msg if msg else ''
            return TypeError(f'Key "{key}" is an instance of {get_cls_name(type(val))}, but should be instance of {classinfo}{msg}. Help: {self.help}', key)

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
        base_fmt = '{prefix}{key} ({classinfo}){prefixed_help}.'
        if style == 'rst':
            prefix = '* '
            key = self.name
            fmt = base_fmt
        elif style == 'yaml':
            prefix = ''
            key = ''
            fmt = '{key}{help}\ntype: {classinfo}'
        else:
            prefix = '|- '
            key = self.name
            fmt = base_fmt

        if self.help:
            joiner = f"\n{(' ' * len(prefix))}"
            wrapped_lines = textwrap.wrap(self.help, width=60)
            # If more than one line, output a paragraph on its own starting on
            # a new line
            multiline = len(wrapped_lines) > 1
            if multiline:
                wrapped_lines.insert(0, '')
            help_ = joiner.join(wrapped_lines)
            prefixed_help = (':' if multiline else ': ') + help_
        else:
            help_ = ''
            prefixed_help = help_

        return fmt.format(
            prefix=prefix,
            key=key,
            classinfo=' or '.join(
                get_cls_name(
                    key_cls,
                    style=style,
                    fully_qualified=False,
                )
                for key_cls in self.classinfo
            ),
            help=help_,
            prefixed_help=prefixed_help,
        )

    @staticmethod
    def pretty_format(v):
        """
        Format the value for pretty printing.

        :param v: Value of the key that is being printed
        :type v: object

        :return: A string
        """
        return str(v)


class ConfigKeyError(KeyError):
    """
    Exception raised when a key is not found in the config instance.
    """
    def __init__(self, msg, key=None, src=None):
        # pylint: disable=super-init-not-called
        self.msg = msg
        self.key = key
        self.src = src

    def __str__(self):
        return self.msg


class MissingBaseKeyError(ConfigKeyError):
    """
    Exception raised when a base key needed to compute a derived key is missing.
    """


class DeferredValueComputationError(ConfigKeyError):
    """
    Raised when computing the value of :class:`DeferredValue` lead to an
    exception.
    """
    def __init__(self, msg, excep, key=None, src=None):
        self.excep = excep
        super().__init__(msg, key, src)


class KeyComputationRecursionError(ConfigKeyError, RecursionError):
    """
    Raised when :meth:`DerivedKeyDesc.compute_val` is reentered while computing
    a given key on a configuration instance, or when a :class:`DeferredValue`
    callback is reentered.
    """


class DerivedKeyDesc(KeyDesc):
    """
    Key descriptor describing a key derived from other keys

    Derived keys cannot be added from a source, since they are purely computed
    out of other keys. It is also not possible to change their source
    priorities. To achieve that, set the source priorities on the keys it is
    based on.

    :param base_key_paths: List of paths to the keys this key is derived from.
        The paths in the form of a list of string are relative to the current
        level. To reference a level above the current one, use the special key
        ``..``.
    :type base_key_paths: list(list(str))

    :param compute: Function used to compute the value of the key. It takes a
        dictionary of base keys specified in ``base_key_paths`` as only
        parameter and is expected to return the key's value.
    :type compute: collections.abc.Callable
    """

    def __init__(self, name, help, classinfo, base_key_paths, compute, newtype=None):
        # pylint: disable=redefined-builtin
        super().__init__(name=name, help=help, classinfo=classinfo, newtype=newtype)
        self._base_key_paths = base_key_paths
        self._compute = compute
        self._is_computing_in = set()

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
    def make_get_key(conf, **kwargs):
        f = conf.__class__.get_key
        return functools.partial(f, **kwargs)

    @classmethod
    def _get_base_key_val(cls, conf, path, eval_deferred=True):
        get_key = cls.make_get_key(conf, quiet=True, eval_deferred=eval_deferred)
        return get_nested_key(conf, path, getitem=get_key)

    @classmethod
    def _get_base_key_src(cls, conf, path):
        get_key = cls.make_get_key(conf, quiet=True)
        conf = get_nested_key(conf, path[:-1], getitem=get_key)
        return conf.resolve_src(path[-1])

    def _resolve_key_path(self, key_path):
        def should_skip(n, key):
            if key == '..':
                n += 1
                skip = True
            elif n:
                skip = True
                n -= 1
            else:
                skip = False

            return (n, skip)

        # Traverse the path in opposit order so we can know if a component needs to
        # be removed based on previous components
        key_path = self.parent.path + key_path
        key_path = list(reversed(key_path))

        n = 0
        skip_list = []
        for key in key_path:
            n, skip = should_skip(n, key)
            skip_list.append(skip)

        resolved = [
            key for key, skip in zip(key_path, skip_list)
            if not skip
        ]
        return list(reversed(resolved))

    def _get_base_key_qualname(self, key_path):
        path = self._resolve_key_path(key_path)
        return '/'.join(path)

    def _get_base_conf(self, conf, eval_deferred=True):
        try:
            base_conf = {}
            for key_path in self._base_key_paths:
                val = self._get_base_key_val(conf, key_path, eval_deferred=eval_deferred)
                set_nested_key(base_conf, key_path, val)
            return base_conf
        except ConfigKeyError as e:
            key = self.qualname
            raise MissingBaseKeyError(
                f'Missing value for base key "{e.key}" in order to compute derived key "{key}": {e.msg}',
                key=key,
            ) from e

    def get_non_evaluated_base_keys(self, conf):
        """
        Get the :class:`KeyDescBase` of base keys that have a :class:`DeferredValue`
        value.
        """
        def get_key_desc(path):
            path = self._resolve_key_path(path)[1:]
            return get_nested_key(conf.STRUCTURE, path)

        bases = {
            get_key_desc(key_path): self._get_base_key_val(conf, key_path, eval_deferred=False)
            for key_path in self._base_key_paths
        }

        return [
            key_desc
            for key_desc, val in bases.items()
            if isinstance(val, DeferredValue)
        ]

    def can_be_computed(self, conf):
        try:
            self._get_base_conf(conf, eval_deferred=False)
        except MissingBaseKeyError:
            return False
        else:
            return True

    def compute_val(self, conf, eval_deferred=True):
        conf_id = id(conf)
        if conf_id in self._is_computing_in:
            key = self.qualname
            raise KeyComputationRecursionError(f'Recursion error while computing derived key: {key}', key)
        else:
            try:
                self._is_computing_in.add(conf_id)
                # If there is non evaluated base, transitively return a closure rather
                # than computing now.
                if not eval_deferred and self.get_non_evaluated_base_keys(conf):
                    val = DeferredValue(self.compute_val, conf=conf, eval_deferred=True)
                else:
                    base_conf = self._get_base_conf(conf)
                    val = self._compute(base_conf)
                    self.validate_val(val)
            finally:
                self._is_computing_in.remove(conf_id)

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

    :param value_path: Relative path to a sub-key that will receive assignment
        to that level for non-mapping types. This allows turning a leaf key into a
        level while preserving backward compatibility, as long as:

        * The key did not accept mapping values, otherwise it would be
          ambiguous and is therefore rejected.

        * The old leaf key has a matching new leaf key, that is a sub-key
          of the new level key.

        In practice, that allows turning a single knob into a tree of settings.
    :type value_path: list(str) or None

    Children keys will get this key assigned as a parent when passed to the
    constructor.

    """

    def __init__(self, name, help, children, value_path=None):
        # pylint: disable=redefined-builtin
        super().__init__(name=name, help=help)
        self.children = children

        # Fixup parent for easy nested declaration
        for key_desc in self.children:
            key_desc.parent = self

        self.value_path = value_path

    @property
    def key_desc(self):
        path = self.value_path
        if path is None:
            raise AttributeError(f'{self} does not define a value path for direct assignment')
        else:
            return get_nested_key(self, path)

    def __getattr__(self, attr):
        # If the property raised an exception, __getattr__ is tried so we need
        # to fail explicitly in order to avoid infinite recursion
        if attr == 'key_desc':
            raise AttributeError('recursive key_desc lookup')
        else:
            try:
                key_desc = self.key_desc
            except Exception as e:
                raise AttributeError(str(e))
            else:
                return getattr(key_desc, attr)

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
            self._key_map[key]
        except KeyError:
            # pylint: disable=raise-missing-from
            try:
                closest_match = difflib.get_close_matches(
                    word=str(key),
                    possibilities=self._key_map.keys(),
                    n=1,
                )[0]
            except IndexError:
                closest_match = ''
            else:
                closest_match = f', maybe you meant "{closest_match}" ?'

            parent = self.qualname
            raise ConfigKeyError(
                f'Key "{key}" is not allowed in {parent}{closest_match}',
                key=key,
            )

    def validate_val(self, conf):
        """Validate a mapping to be used as a configuration source"""
        if not isinstance(conf, Mapping):
            key = self.qualname
            raise TypeError(f'Configuration of {key} must be a Mapping', key)
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
            help=' ' + self.help if self.help else '',
        )
        nl = '\n' + idt
        help_ += nl.join(
            key_desc.get_help(style=style).replace('\n', nl)
            for key_desc in self.children
        )
        if style == 'rst':
            help_ += '\n\n..\n'

        return help_


class DelegatedLevelKeyDesc(LevelKeyDesc):
    """
    Level key descriptor that imports the keys from another
    :class:`~lisa.conf.MultiSrcConfABC` subclass.

    :param conf: Configuration class to extract keys from.
    :type conf: MultiSrcConfABC

    :Variable keyword arguments: Forwarded to :class:`lisa.conf.LevelKeyDesc`.

    This allows embedding a configuration inside another one, mostly to be able
    to split a configuration class while preserving backward compatibility.

    .. note:: Only the children keys are taken from the passed level, other
        information such as ``value_path`` are ignored and must be set
        explicitly.
    """

    def __init__(self, name, help, conf, **kwargs):
        # Make a deepcopy to ensure we will not change the parent attribute of
        # an existing structure.
        level = copy.deepcopy(conf.STRUCTURE)

        children = level.values()
        super().__init__(
            name=name,
            help=help,
            children=children,
            **kwargs
        )


class TopLevelKeyDescBase(LevelKeyDesc):
    """
    Top-level key descriptor, which defines the top-level key to use in the
    configuration files.

    :param levels: Levels of the top-level key, as a list of strings. Each item
        specifies a level in a mapping, so that multiple classes can share the same
        actual top-level without specific cooperation.
    :type levels: list(str)

    This top-level key is omitted in all interfaces except for the
    configuration file, since it only reflects the configuration class
    """

    def __init__(self, levels, *args, **kwargs):
        levels = tuple(levels)
        for level in levels:
            super()._check_name(level)

        self.levels = levels
        name = '/'.join(levels)
        super().__init__(name, *args, **kwargs)

    @classmethod
    def _check_name(cls, name):
        pass

    def get_help(self, style=None):
        if style == 'yaml':
            return self.help
        else:
            return super().get_help(style=style)


class TopLevelKeyDesc(TopLevelKeyDescBase):
    """
    Regular top-level key descriptor, with only one level.

    :param name: Name of the top-level key, as a string.
    :type name: str
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__([name], *args, **kwargs)


class NestedTopLevelKeyDesc(TopLevelKeyDescBase):
    """
    Top-level key descriptor, with an arbitrary amount of levels.
    """

class MultiSrcConfABC(Serializable, abc.ABC):
    _REGISTERED_TOPLEVEL_KEYS = {}

    @abc.abstractmethod
    def to_map(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_map(cls, mapping, add_default_src=True):
        raise NotImplementedError

    @classmethod
    def from_yaml_map(cls, path, add_default_src=True):
        """
        Allow reloading from a plain mapping, to avoid having to specify a tag
        in the configuration file. The content is hosted under the top-level
        key specified in ``STRUCTURE``.

        :param path: Path to the YAML file
        :type path: str

        :param add_default_src: Add a default source if available for that
            class.
        :type add_default_src: bool

        .. note:: Only load YAML files from trusted source as it can lead to
            arbitrary code execution.
        """

        toplevel_keys = cls.STRUCTURE.levels
        get_content = lambda x: get_nested_key(x, toplevel_keys) or {}
        def has_keys(keys, mapping):
            if keys:
                key, *keys = keys
                try:
                    return has_keys(keys, mapping[key])
                except KeyError:
                    return False
            else:
                return True

        mapping = cls._from_path(path, fmt='yaml')
        if not isinstance(mapping, Mapping):
            raise ValueError(f'Top-level object is expected to be a mapping but got: {mapping.__class__.__qualname__}')

        try:
            data  = get_content(mapping)
        except KeyError:
            # pylint: disable=raise-missing-from
            raise TopLevelKeyError(toplevel_keys)
        # "unwrap" an extra layer of toplevel key, to play well with !include
        if len(data) == 1 and has_keys(toplevel_keys, data):
            data = get_content(data)
        return cls.from_map(data, add_default_src=add_default_src)

    @classmethod
    def from_yaml_map_list(cls, path_list, add_default_src=True):
        """
        Create a mapping of configuration classes to instance, by loading them
        from the list of paths using :meth:`from_yaml_map` and merging them.

        :param path_list: List of paths to YAML configuration files.
        :type path_list: list(str)

        :param add_default_src: See :meth:`from_yaml_map`.

        .. note:: When merging, the configuration coming from the rightmost
            path will win if it defines some keys that were also defined in another
            file. Each file will be mapped to a different sources, named after
            the basename of the file.

        .. note:: Only load YAML files from trusted source as it can lead to
            arbitrary code execution.
        """
        # Make sure that all modules from LISA are loaded, so that
        # get_subclasses will be accurate.
        import_all_submodules(lisa)

        conf_cls_set = set(get_subclasses(cls, only_leaves=True))
        conf_list = []
        for conf_path in path_list:
            # Try to build as many configurations instances from all the files we
            # are given
            for conf_cls in conf_cls_set:
                try:
                    # Do not add the default source, to avoid overriding user
                    # configuration with the default one.
                    conf = conf_cls.from_yaml_map(conf_path, add_default_src=False)
                except TopLevelKeyError:
                    continue
                else:
                    conf_list.append((conf, conf_path))

        def keyfunc(conf_and_path):
            cls = type(conf_and_path[0])
            # Sort according to class qualified name since classes are not
            # comparable directly
            return (cls.__module__ + '.' + cls.__qualname__), cls

        # Then aggregate all the conf from each type, so they just act as
        # alternative sources.
        conf_map = {}
        for (_, conf_cls), conf_and_path_seq in groupby(conf_list, key=keyfunc):
            conf_and_path_list = list(conf_and_path_seq)

            # Get the default configuration, and stack all user-defined keys
            conf = conf_cls(add_default_src=add_default_src)
            for conf_src, conf_path in conf_and_path_list:
                src = os.path.basename(conf_path)
                conf.add_src(src, conf_src)

            conf_map[conf_cls] = conf

        return conf_map

    @property
    def as_yaml_map(self):
        """
        Give a mapping suitable for storing in a YAML configuration file.

        .. seealso:: :meth:`to_yaml_map` and :meth:`from_yaml_map`
        """
        def make_map(keys, data):
            if keys:
                key, *keys = keys
                return {key: make_map(keys, data)}
            else:
                return data

        data = self.to_map()
        mapping = make_map(self.STRUCTURE.levels, data)
        return mapping

    def to_yaml_map(self, path):
        """
        Write a configuration file, with the key descriptions in comments.

        :param path: Path to the file to write to.
        :type path: str
        """
        return self._to_path(self.as_yaml_map, path, fmt='yaml')

    def to_yaml_map_str(self, **kwargs):
        """
        Return the content of the file that would be create by
        :meth:`to_yaml_map` in a string.

        :Variable keyword arguments: Forwarded to :meth:`to_yaml_map`
        """
        content = io.StringIO()
        self.to_yaml_map(content, **kwargs)
        return content.getvalue()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ignore abstract classes, since there can be no instance of them
        if not inspect.isabstract(cls):
            if cls.__doc__:
                doc = inspect.getdoc(cls)
                # Create a ResStructuredText preformatted block when rendering
                # with Sphinx
                style = 'rst' if is_running_sphinx() else None
                generated_help = '\n' + cls.get_help(style=style)
                indent = '\n    '
                try:
                    # Not all classes support these parameters
                    yaml_example = cls().to_yaml_map_str(
                        add_placeholder=True,
                        placeholder='_'
                    )
                except TypeError:
                    yaml_example = cls().to_yaml_map_str()

                if yaml_example:
                    yaml_example = ':Example YAML:\n\n.. code-block:: YAML\n' + indent + yaml_example.replace('\n', indent)
                cls.__doc__ = doc.format(
                    generated_help=generated_help,
                    yaml_example=yaml_example
                )

        # Create the types for the keys that specify it, along with the getters
        # to expose the values to exekall
        if hasattr(cls, 'STRUCTURE') and isinstance(cls.STRUCTURE, TopLevelKeyDescBase):
            # Ensure uniqueness of toplevel key
            toplevel_keys = tuple(cls.STRUCTURE.levels)

            def format_keys(keys):
                return '/'.join(keys)

            def eq_prefix_keys(key1, key2):
                len_ = min(len(key1), len(key2))
                return tuple(key1[:len_]) == tuple(key2[:len_])

            offending = [
                cls_
                for keys, cls_ in cls._REGISTERED_TOPLEVEL_KEYS.items()
                if eq_prefix_keys(toplevel_keys, keys)
            ]

            # If the offending class has the same name and was declared in the
            # same module, we ignore the conflict as this is probably arising
            # from an import error in that module, that lead to the module
            # being re-imported again (by another import statement).
            if offending and not all(
                (
                    cls_.__qualname__ == cls.__qualname__ and
                    cls_.__module__ == cls.__module__
                )
                for cls_ in offending
            ):
                raise RuntimeError(f'Class {cls.__qualname__} cannot reuse top level key "{format_keys(toplevel_keys)}" as it is already used by {", ".join(offending)}')
            else:
                cls._REGISTERED_TOPLEVEL_KEYS[toplevel_keys] = cls

            def flatten(structure):
                for key_desc in structure.values():
                    if isinstance(key_desc, LevelKeyDesc):
                        yield from flatten(key_desc)
                    else:
                        yield key_desc

            for key_desc in flatten(cls.STRUCTURE):
                newtype_name = key_desc.newtype
                if isinstance(key_desc, KeyDesc):

                    # We need a helper to make sure "key_desc" is bound to the
                    # right object, otherwise it will be referred by name only
                    # and will always have the value during the last iteration
                    # of the loop
                    # FIXME: pylint complains about make_metacls being unused,
                    # which is a bug:
                    # https://github.com/PyCQA/pylint/issues/4020
                    def make_metacls(key_desc): # pylint: disable=unused-variable
                        # Implement __instancecheck__ on the metaclass allows
                        # isinstance(x, Newtype) to be true for any instance of any
                        # type given in KeyDesc.__init__(classinfo=...)
                        class NewtypeMeta(type):
                            def __instancecheck__(cls, x):
                                classinfo = tuple(
                                    c if c is not None else type(None)
                                    for c in key_desc.classinfo
                                )
                                return isinstance(x, classinfo)

                        return NewtypeMeta

                    # Inherit from HideExekallID, since we don't want it to be
                    # shown in the exekall IDs.
                    class Newtype(HideExekallID, metaclass=make_metacls(key_desc)):
                        pass

                    Newtype.__name__ = newtype_name
                    Newtype.__qualname__ = f'{cls.__qualname__}.{newtype_name}'
                    Newtype.__module__ = cls.__module__
                    Newtype.__doc__ = key_desc.help
                    setattr(cls, newtype_name, Newtype)

                    def make_getter(cls, type_, key_desc):
                        def getter(self: cls) -> type_:
                            try:
                                return self.get_nested_key(key_desc.path[1:])
                            # We cannot afford to raise here, as the
                            # configuration instance might not hold a value for
                            # that key, but we still need to pass something to
                            # the user function.
                            except KeyError:
                                return None

                        getter_name = f'_get_typed_key_{type_.__name__}'
                        getter.__name__ = getter_name
                        getter.__qualname__ = f'{cls.__qualname__}.{getter_name}'
                        getter.__module__ = cls.__module__
                        return getter

                    newtype_getter = make_getter(cls, Newtype, key_desc)
                    setattr(cls, newtype_getter.__name__, newtype_getter)


class _HashableMultiSrcConf:
    """
    Dummy wrapper to :class:`MultiSrcConf` that is hashable, each wrapper
    instance being equal to other instances wrapping the same configuration
    instance.

    This allows using configuration as keys for instance-oriented usages like
    indexing in dictionaries to hold instance-related information.

    .. warning:: Python does not implement ``__hash__`` for mutable containers
        for good reasons, make sure you understand why before using this class.
    """
    def __init__(self, conf):
        self.conf = conf

    def __hash__(self):
        return id(self.conf)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.conf is other.conf
        else:
            return False


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

    This base class will modify the docstring of subclasses, using it as an
    ``str.format`` template with the following placeholders:

        * ``{generated_help}``: snippet of ResStructuredText containing the
          list of allowed keys.

        * ``{yaml_example}``: example snippet of YAML

    It will also create the types specified using ``newtype`` in the
    :class:`KeyDesc`, along with a getter to expose it to ``exekall``.

    .. note:: Since the dosctring is interpreted as a template, "{" and "}"
        characters must be doubled to appear in the final output.

    .. attention:: The layout of the configuration is typically guaranteed to
        be backward-compatible in terms of accepted shape of input, but layout
        of the configuration might change. This means that the path to a given
        key could change as long as old input is still accepted. Types of
        values can also be widened, so third party code re-using config classes
        from :mod:`lisa` might have to evolve along the changes of
        configuration.
    """

    @abc.abstractmethod
    def STRUCTURE():
        """
        Class attribute defining the structure of the configuration file, as a
        instance of :class:`TopLevelKeyDescBase`
        """

    DEFAULT_SRC = {}
    """
    Source added automatically using :meth:`~lisa.conf.MultiSrcConf.add_src` under the name 'default'
    when instances are built.
    """

    def __init__(self, conf=None, src='user', add_default_src=True):
        self._nested_init(
            key_desc_path=[],
            src_prio=[],
            parent=None,
        )
        self.add_src(src, conf)

        # Give some preset in the the lowest prio source
        if self.DEFAULT_SRC and add_default_src:
            self.add_src('default', self.DEFAULT_SRC, fallback=True)

    @classmethod
    def get_help(cls, *args, **kwargs):
        return cls.STRUCTURE.get_help(*args, **kwargs)

    def _nested_init(self, key_desc_path, src_prio, parent):
        """Called to initialize nested instances of the class for nested
        configuration levels."""
        self._key_desc_path = key_desc_path
        "Path in the structure of that level of configuration"
        # Make a copy to avoid sharing it with the parent
        self._src_prio = copy.copy(src_prio)
        "List of sources in priority order (1st item is highest prio)"
        self._src_override = {}
        "Map of keys to map of source to values"
        self._key_map = {}
        "Key/value map of leaf values"
        self._sublevel_map = {}
        "Key/sublevel map of nested configuration objects"
        self._parent = parent
        "Parent instance of configuration"
        self._as_hashable = _HashableMultiSrcConf(self)
        """
        Hashable proxy, mostly designed to allow instance-oriented lookup in
        mappings. DO NOT USE IT FOR OTHER PURPOSES. You have been warned.
        """

        # Build the tree of objects for nested configuration mappings
        for key, key_desc in self._structure.items():
            if isinstance(key_desc, LevelKeyDesc):
                self._sublevel_map[key] = self._nested_new(
                    key_desc_path=key_desc.path,
                    src_prio=self._src_prio,
                    parent=self,
                )

    @property
    def _structure(self):
        # The first level in the path is the top-level key, which must be
        # skipped
        path = self._key_desc_path[1:]
        return get_nested_key(self.STRUCTURE, path)

    @classmethod
    def _nested_new(cls, *args, **kwargs):
        new = cls.__new__(cls)
        new._nested_init(*args, **kwargs)
        return new

    def __copy__(self):
        """
        Shallow copy of the nested configuration tree, without duplicating the
        leaf values.
         """
        cls = type(self)
        new = cls.__new__(cls)

        # This is eather going to be fixed up by the caller of __copy__ if we
        # are in a recursive copy, or left as it is if that's either the root,
        # or the highest sublevel that was copied (in case someone copies a
        # part of a larger conf).
        new._parent = self._parent

        not_copied = {'_parent', '_sublevel_map', '_as_hashable'}

        # make a shallow copy of the attributes so we don't end up sharing
        # metadata
        new.__dict__.update(
            (key, copy.copy(val))
            for key, val in self.__dict__.items()
            if key not in not_copied
        )

        # Do the same with sublevels recursively, since we consider the nested
        # levels as one "meta object". It's not really a deepcopy either, since
        # we don't copy the values themselves.
        def copy_sublevel(sublevel):
            new_sublevel = copy.copy(sublevel)
            # Fixup the parent
            new_sublevel._parent = new
            return new_sublevel

        new._sublevel_map = {
            key: copy_sublevel(sublevel)
            for key, sublevel in self._sublevel_map.items()
        }

        new._as_hashable = _HashableMultiSrcConf(new)

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
    def from_map(cls, mapping, add_default_src=True):
        """
        Create a new configuration instance, using the output of :meth:`to_map`
        """
        conf_src = mapping.get('conf', {})
        src_override = mapping.get('source', {})

        conf = cls(conf_src, add_default_src=add_default_src)
        conf.force_src_nested(src_override)
        return conf

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
        class PlaceHolder(str):
            def __new__(cls):
                return super().__new__(cls, '...')

            def __repr__(self):
                return str(self)

        class NonEscapedValue(str):
            def __new__(cls, value):
                value = repr(value)

                # Make sure no individual value will print to a string that is too long
                max_len = 50
                if len(value) > max_len:
                    value = value[:max_len] + '...'

                return super().__new__(cls, value)

            def __repr__(self):
                # pylint: disable=invalid-repr-returned
                return self

        def format_conf(conf):
            if isinstance(conf, Mapping):
                # Make sure that mappings won't be too long
                max_mapping_len = 10
                key_val = sorted(conf.items())
                if len(key_val) > max_mapping_len:
                    key_val = key_val[:max_mapping_len]
                    key_val.append((PlaceHolder(), PlaceHolder()))

                def format_val(val):
                    if isinstance(val, Mapping):
                        return format_conf(val)
                    else:
                        return NonEscapedValue(val)

                return {
                    key: format_val(val)
                    for key, val in key_val
                }
            else:
                return conf

        logger = self.logger
        if logger.isEnabledFor(logging.DEBUG):
            caller, filename, lineno = get_call_site(1, exclude_caller_module=True)
            logger.debug('{caller} ({filename}:{lineno}) has set source "{src}":\n{conf}'.format(
                src=src,
                conf=pprint.pformat(
                    format_conf(conf),
                    indent=4,
                    compact=True,
                ),
                caller=caller if caller else '<unknown>',
                filename=filename if filename else '<unknown>',
                lineno=lineno if lineno else '<unknown>',
            ))
        return self._add_src(
            src, conf,
            filter_none=filter_none, fallback=fallback
        )

    def _add_src(self, src, conf, filter_none=False, fallback=False):
        conf = {} if conf is None else conf

        if isinstance(conf, Mapping):
            # Filter-out None values, so they won't override actual data from
            # another source
            if filter_none:
                conf = {
                    k: v for k, v in conf.items()
                    if v is not None
                }

            # only validate at that level, since sublevel will take care of
            # filtering then validating their own level
            validated_conf = {
                k: v for k, v in conf.items()
                if not isinstance(self._structure[k], LevelKeyDesc)
            }
            self._structure.validate_val(validated_conf)

            for key, val in conf.items():
                key_desc = self._structure[key]
                # Dispatch the nested mapping to the right sublevel
                if isinstance(key_desc, LevelKeyDesc):
                    # sublevels have already been initialized when the root object
                    # was created.
                    self._sublevel_map[key]._add_src(src, val, filter_none=filter_none, fallback=fallback)
                # Derived keys cannot be set, since they are purely derived from
                # other keys
                elif isinstance(key_desc, DerivedKeyDesc):
                    raise ValueError(f'Cannot set a value for a derived key "{key_desc.qualname}"', key_desc.qualname)
                # Otherwise that is a leaf value that we store at that level
                else:
                    self._key_map.setdefault(key, {})[src] = val
        else:
            # Non-mapping value are allowed if the level defines a subkey
            # to assign to. We then craft a conf that sets that specific
            # value.
            key_desc = self._structure
            value_path = key_desc.value_path
            if value_path is None:
                raise ValueError(f'Cannot set a value for the key level "{key_desc.qualname}"', key_desc.qualname)
            else:
                conf = set_nested_key({}, list(value_path), conf)
                self._add_src(src, conf, filter_none=filter_none, fallback=fallback)

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
        # pylint: disable=attribute-defined-outside-init

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
            raise ValueError(f'Cannot force source of the sub-level "{qual_key}"', qual_key)
        elif isinstance(key_desc, DerivedKeyDesc):
            raise ValueError(f'Cannot force source of a derived key "{qual_key}"', qual_key)
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
            raise ValueError(f'Key "{key}" is a nested configuration level, it does not have a source on its own.', key)

        # Get the priority list from the prio override list, or just the
        # default prio list
        src_prio = self._resolve_prio(key)
        if src_prio:
            return src_prio[0]
        else:
            key = key_desc.qualname
            raise ConfigKeyError(
                f'Could not find any source for key "{key}"',
                key=key,
            )

    def _eval_deferred_val(self, src, key, error='raise'):
        error = error or 'raise'
        key_desc = self._structure[key]
        val = self._key_map[key][src]
        validate = True
        if isinstance(val, DeferredValue):
            try:
                val = val(key_desc=key_desc)
            # Wrap into a ConfigKeyError so that the user code can easily
            # handle missing keys, and the original exception is still
            # available as excep.__cause__ since it was chained with "from"
            except Exception as e: # pylint: disable=broad-except
                key_qualname = key_desc.qualname
                msg = f'Could not compute "{key_qualname}" from source "{src}": {e}'

                # Propagate ConfigKeyError as-is
                if isinstance(e, ConfigKeyError):
                    excep = e
                else:
                    excep = DeferredValueComputationError(
                        msg,
                        e,
                        key_qualname,
                        src,
                    )
                    # Chain explicitly like "raise X from Y" in case it needs
                    # to be wrapped to be raised later
                    excep.__cause__ = e

                if error == 'raise':
                    raise excep from e
                elif error == 'log':
                    self.logger.error(msg)
                    # Setup a bomb that will explode during a later access
                    val = DeferredExcep(excep)
                    validate = False

            if validate:
                key_desc.validate_val(val)

            self._key_map[key][src] = val
        return val

    def eval_deferred(self, cls=DeferredValue, src=None, resolve_src=True, error='raise'):
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

        :param resolve_src: If ``True``, resolve the source of each key and
            only compute deferred values for this source.
        :type resolve_src: bool

        :param error: If ``'raise'`` or ``None``, exception are raised as
            usual. If ``log``, the exception is logged at error level.
        :type error: str or None
        """
        for key, src_map in self._key_map.items():

            if resolve_src and src is None:
                resolved_src = self.resolve_src(key)
                src_map = {resolved_src: src_map[resolved_src]}

            for src_, val in src_map.items():
                if src is not None and src != src_:
                    continue
                if isinstance(val, cls):
                    self._eval_deferred_val(src_, key, error=error)

        for sublevel in self._sublevel_map.values():
            sublevel.eval_deferred(cls, src, resolve_src, error=error)

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

    def get_key(self, key, src=None, eval_deferred=True, quiet=False):
        """
        Get the value of the given key. It returns a deepcopy of the value.

        The special key ``..`` can be used to refer to the parent in the
        hierarchy.

        :param key: name of the key to lookup
        :type key: str

        :param src: If not None, look up the value of the key in that source
        :type src: str or None

        :param eval_deferred: If True, evaluate instances of
            :class:`DeferredValue` if needed
        :type eval_deferred: bool

        :param quiet: Avoid logging the access
        :type quiet: bool

        .. note:: Using the indexing operator ``self[key]`` is preferable in
            most cases , but this method provides more parameters.
        """
        if key == '..':
            return self._parent

        key_desc = self._structure[key]

        if isinstance(key_desc, LevelKeyDesc):
            return self._sublevel_map[key]
        elif isinstance(key_desc, DerivedKeyDesc):
            # Specifying a source is an error for a derived key
            if src is not None:
                key = key_desc.qualname
                raise ValueError(f'Cannot specify the source when getting "{key}" since it is a derived key', key)

            val = key_desc.compute_val(self, eval_deferred=eval_deferred)
            src = self.resolve_src(key)
        else:
            # Compute the source to use for that key
            if src is None:
                src = self.resolve_src(key)

            try:
                val = self._key_map[key][src]
            except KeyError:
                # pylint: disable=raise-missing-from
                key = key_desc.qualname
                raise ConfigKeyError(
                    f'Key "{key}" is not available from source "{src}"',
                    key=key,
                    src=src,
                )

            if eval_deferred:
                val = self._eval_deferred_val(src, key, error='raise')

        logger = self.logger
        if not quiet and logger.isEnabledFor(logging.DEBUG):
            caller, filename, lineno = get_call_site(2, exclude_caller_module=True)
            logger.debug('{caller} ({filename}:{lineno}) has used key {key} from source "{src}": {val}'.format(
                key=key_desc.qualname,
                src=src,
                val=key_desc.pretty_format(val),
                caller=caller if caller else '<unknown>',
                filename=filename if filename else '<unknown>',
                lineno=lineno if lineno else '<unknown>',
            ))

        if isinstance(val, DeferredValue):
            return val
        else:
            if key_desc.deepcopy_val:
                val = copy.deepcopy(val)
            return val

    def get_nested_key(self, key, *args, **kwargs):
        """
        Same as :meth:`get_key` but works on a list of keys to access nested mappings.

        :param key: List of nested keys.
        :type key: list(str)
        """
        val = self
        for k in key:
            val = val.get_key(k, *args, **kwargs)

        return val

    def get_src_map(self, key):
        """
        Get a mapping of all sources for the given ``key``, in priority order
        (first item is the highest priority source).
        """
        key_desc = self._structure[key]
        if isinstance(key_desc, LevelKeyDesc):
            key = key_desc.qualname
            raise ValueError(f'Key "{key}" is a nested configuration level, it does not have a source on its own.', key)

        return OrderedDict(
            (src, self._eval_deferred_val(src, key, error='raise'))
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

        # We add the derived keys when pretty-printing, for the sake of
        # completeness. This will not honor eval_deferred for base keys.
        def derived_items():
            for key in self._get_derived_key_names():
                non_eval_base_qualnames = self._structure[key].get_non_evaluated_base_keys(self)
                if non_eval_base_qualnames and not eval_deferred:
                    val = '<depends on lazy keys: {}>'.format(
                        ', '.join(
                            base_key.qualname
                            for base_key in non_eval_base_qualnames
                        )
                    )
                else:
                    val = self.get_key(key, quiet=True)

                yield key, val

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
                cls='' if is_sublevel else ' (' + v_cls.__qualname__ + ')',
                src='' if is_sublevel else ' from ' + self.resolve_src(k),
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
            if (
                isinstance(key_desc, DerivedKeyDesc)
                and key_desc.can_be_computed(self)
            )
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
            (k, self.get_key(k, eval_deferred=eval_deferred, quiet=True))
            for k in self.keys()
        )

    def _ipython_key_completions_(self):
        "Allow Jupyter keys completion in interactive notebooks"
        regular_keys = set(self.keys())
        # For autocompletion purposes, we show the derived keys
        derived_keys = set(self._get_derived_key_names())
        return sorted(regular_keys | derived_keys)

    def _repr_pretty_(self, p, cycle):
        "Pretty print instances in Jupyter notebooks"
        # pylint: disable=unused-argument
        p.text(self.pretty_format())

class SimpleMultiSrcConf(MultiSrcConf):
    """
    Like :class:`MultiSrcConf`, with a simpler config file.

    ``conf`` and ``source`` are not available, and the behaviour is as all keys
    were located under a ``conf`` key. We do not allow overriding source for
    this kind of configuration to keep the YAML interface simple and dict-like
    """
    @classmethod
    def from_map(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def to_map(self):
        return dict(self._get_effective_map())

    def to_yaml_map(self, path, add_placeholder=False, placeholder='<no default>'):
        """
        Write a configuration file, with the key descriptions in comments.

        :param path: Path to the file to write to.
        :type path: str

        :param add_placeholder: If ``True``, a placeholder value will be used
            for keys that don't have values. This allows creating template
            configuration files that list all keys.
        :type add_placeholder: bool

        :param placeholder: Placeholder to use for missing values when
            ``add_placeholder`` is used.
        :type placeholder: object
        """

        def format_comment(key_desc):
            comment = key_desc.get_help(style='yaml')

            if comment:
                return (comment[0].upper() + comment[1:]).strip()
            else:
                return comment

        def add_help(key_desc, data, indent=0):
            name = key_desc.name
            if isinstance(key_desc, LevelKeyDesc):
                if isinstance(key_desc, TopLevelKeyDesc):
                    levels = key_desc.levels
                else:
                    levels = [key_desc.name]

                try:
                    level_data = get_nested_key(data, levels)
                except KeyError:
                    level_data = {}

                level_data = CommentedMap(level_data)

                for subkey_desc in key_desc.children:
                    if subkey_desc.name not in level_data:
                        if add_placeholder:
                            if isinstance(subkey_desc, DerivedKeyDesc):
                                continue

                            if not isinstance(subkey_desc, LevelKeyDesc):
                                level_data[subkey_desc.name] = placeholder
                        else:
                            continue

                    idt = indent + len(levels)
                    level_data.yaml_set_comment_before_after_key(
                        subkey_desc.name,
                        indent=idt * 4,
                        before='\n' + format_comment(subkey_desc),
                    )
                    add_help(subkey_desc, level_data, indent=idt)

                if level_data:
                    set_nested_key(data, levels, level_data)
                else:
                    with contextlib.suppress(KeyError):
                        del data[levels[0]]

        data = CommentedMap(self.as_yaml_map)
        data.yaml_set_start_comment(format_comment(self.STRUCTURE))
        add_help(self.STRUCTURE, data)

        if data:
            return self._to_path(data, path, fmt='yaml-roundtrip')
        else:
            return None


class Configurable(abc.ABC):
    """
    Pair a regular class with a configuration class.

    The pairing is achieved by inheriting from :class:`Configurable` and
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

    .. note:: A given configuration class must be paired to only one class.
        Otherwise, the ``DEFAULT_SRC`` conf class attribute will be updated
        multiple times, leading to unexpected results.

    .. note:: Some services offered by :class:`Configurable` are not extended
        to subclasses of a class using it. For example, it would not make sense
        to update ``DEFAULT_SRC`` using a subclass ``__init__`` parameters.

    """
    INIT_KWARGS_KEY_MAP = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dct = cls.__dict__

        try:
            # inherited CONF_CLASS will not be taken into account if we look at
            # the dictionary directly
            conf_cls = dct['CONF_CLASS']
        except KeyError:
            return

        # Link the configuration to the signature of __init__
        sig = inspect.signature(cls.__init__)
        init_kwargs_key_map = cls._get_kwargs_key_map(sig, conf_cls)
        # What was already there has priority over auto-detected bindings
        init_kwargs_key_map.update(dct.get('INIT_KWARGS_KEY_MAP', {}))
        cls.INIT_KWARGS_KEY_MAP = init_kwargs_key_map

        # Create an instance with default configuration, to merge it with
        # defaults taken from __init__
        default_conf = conf_cls()
        default_conf.add_src(
            src='__init__-default',
            conf=cls._get_default_conf(sig, init_kwargs_key_map),
            # Default configuration set in the conf class still has priority
            fallback=True,
            # When an __init__ parameter has a None default value, we don't
            # add any default value. That avoids failing the type check for
            # keys that really need to be of a certain type when specified.
            filter_none=True,
        )
        # Convert to a dict so that the Sphinx documentation is able to show
        # the content of the source
        conf_cls.DEFAULT_SRC = dict(default_conf._get_effective_map())

        # Update the docstring by using the configuration help
        docstring = inspect.getdoc(cls)
        if docstring:
            cls.__doc__ = docstring.format(
                configurable_params=cls._get_rst_param_doc()
            )

    @staticmethod
    def _get_kwargs_key_map(sig, conf_cls):
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

    @classmethod
    def _get_param_key_desc_map(cls):
        return {
            param: get_nested_key(cls.CONF_CLASS.STRUCTURE, conf_path)
            for param, conf_path in cls.INIT_KWARGS_KEY_MAP.items()
        }

    @classmethod
    def _get_rst_param_doc(cls):
        # pylint: disable=no-value-for-parameter
        return '\n'.join(
            ':param {param}: {help}\n:type {param}: {type}\n'.format(
                param=param,
                help=key_desc.help,
                type=(
                    'collections.abc.Mapping'
                    if isinstance(key_desc, LevelKeyDesc) else
                    ' or '.join(get_cls_name(t) for t in key_desc.classinfo)
                ),
            )
            for param, key_desc
            in cls._get_param_key_desc_map().items()
        )

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
        parameters = OrderedDict(inspect.signature(cls.__init__).parameters)
        # pop `self` param
        parameters.popitem(last=False)
        mandatory_args = [
            name
            for name, param in parameters.items()
            if (
                param.default is inspect.Parameter.empty
                and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            )

        ]

        param_key_desc_map = cls._get_param_key_desc_map()

        missing_param = set(mandatory_args) - set(kwargs.keys())
        if missing_param:
            missing_key_paths = sorted(
                key_desc.qualname
                for param, key_desc in param_key_desc_map.items()
                if param in missing_param
            )
            raise ConfigKeyError(f"Missing mandatory keys: {', '.join(missing_key_paths)}")

        for param, key_desc in param_key_desc_map.items():
            if param in kwargs:
                key_desc.validate_val(kwargs[param])


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
