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
from collections.abc import Mapping, MutableMapping, Sequence
from collections import OrderedDict
import contextlib
import inspect
import io
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
import weakref
from weakref import WeakKeyDictionary
import urllib.parse
import warnings
import textwrap
import webbrowser
import mimetypes

import ruamel.yaml
from ruamel.yaml import YAML

# These modules may not be installed as they are only used for notebook usage
try:
    import sphobjinv
    from IPython.display import IFrame
# ModuleNotFoundError does not exist in Python < 3.6
except ImportError:
    pass

import lisa
import lisa.assets
from lisa.version import version_tuple, parse_version, format_version


# Do not infer the value using __file__, since it will break later on when
# lisa package is installed in the site-package locations using pip, which
# are typically not writable.
LISA_HOME = os.getenv('LISA_HOME')
"""
The detected location of your LISA installation
"""

ASSETS_PATH = os.path.dirname(lisa.assets.__file__)
"""
Path in which all assets the ``lisa`` package relies on are located in.
"""

RESULT_DIR = 'results'
LATEST_LINK = 'results_latest'

TASK_COMM_MAX_LEN = 16 - 1
"""
Value of ``TASK_COMM_LEN - 1`` macro in the kernel, to account for ``\0``
terminator.
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


def get_cls_name(cls, style=None):
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

class HideExekallID:
    """Hide the subclasses in the simplified ID format of exekall.

    That is mainly used for uninteresting classes that do not add any useful
    information to the ID. This should not be used on domain-specific classes
    since alternatives may be used by the user while debugging for example.
    Hiding too many classes may lead to ambiguity, which is exactly what the ID
    is fighting against.
    """
    pass

def memoized(f):
    """
    Decorator to memoize the result of a callable, based on
    :func:`functools.lru_cache`

    .. note:: The first parameter of the callable is cached with a weak
        reference. This suits well the method use-case, since we don't want the
        memoization of methods to prevent garbage collection of the instances
        they are bound to.
    """

    def apply_lru(f):
        # maxsize should be a power of two for better speed, see:
        # https://docs.python.org/3/library/functools.html#functools.lru_cache
        return functools.lru_cache(maxsize=1024, typed=True)(f)

    # We need at least one positional parameter for the WeakKeyDictionary
    if inspect.signature(f).parameters:
        cache_map = WeakKeyDictionary()

        @functools.wraps(f)
        def wrapper(first, *args, **kwargs):
            try:
                partial = cache_map[first]
            except KeyError:
                # Only keep a weak reference here for the "partial" closure
                ref = weakref.ref(first)

                # This partial function does not take "first" as parameter, so
                # that the lru_cache will not keep a reference on it
                @apply_lru
                def partial(*args, **kwargs):
                    return f(ref(), *args, **kwargs)

                cache_map[first] = partial

            return partial(*args, **kwargs)

        return wrapper
    else:
        return apply_lru(f)

def resolve_dotted_name(name):
    """Only resolve names where __qualname__ == __name__, i.e the callable is a
    module-level name."""
    mod_name, callable_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, callable_name)

def import_all_submodules(pkg):
    """Import all submodules of a given package."""
    return _import_all_submodules(pkg.__name__, pkg.__path__)

def _import_all_submodules(pkg_name, pkg_path):
    def import_module(module_name):
        # Load module under its right name, so explicit imports of it will hit
        # the sys.module cache instead of importing twice, with two "version"
        # of each classes defined inside.
        full_name = '{}.{}'.format(pkg_name, module_name)
        module = importlib.import_module(full_name)
        return module

    return [
        import_module(module_name)
        for finder, module_name, is_pkg
        in pkgutil.walk_packages(pkg_path)
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

    The following YAML tags are supported on top of what YAML provides out of
    the box:

        * ``!call``: call a Python callable with a mapping of arguments:

            .. code-block:: yaml

                # will execute:
                # package.module.Class(arg1='foo', arg2='bar', arg3=42)
                # NB: there is no space after "call:"
                !call:package.module.Class
                    arg1: foo
                    arg2: bar
                    arg3: 42

        * ``!include``: include the content of another YAML file. Environment
          variables are expanded in the given path:

            .. code-block:: yaml

                !include /foo/$ENV_VAR/bar.yml

        * ``!env``: take the value of an environment variable, and convert
          it to a Python type:

            .. code-block:: yaml

                !env:int MY_ENV_VAR

            If `interpolate` is used as type, the value will be interpolated
            using :func:`os.path.expandvars` and the resulting string
            returned:

                .. code-block:: yaml

                    !env:interpolate /foo/$MY_ENV_VAR/bar

        * ``!var``: reference a module-level variable:

            .. code-block:: yaml

                !var package.module.var

    .. note:: Not to be used on its own - instead, your class should inherit
        from this class to gain serialization superpowers.
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
        string = loader.construct_scalar(node)
        assert isinstance(string, str)

        type_ = suffix
        if type_ == 'interpolate':
            return os.path.expandvars(string)
        else:
            varname = string

            type_ = loader.find_python_name(type_, node.start_mark)
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
    def _to_yaml(cls, data):
        buff = io.StringIO()
        cls._yaml.dump(data, buff)
        return buff.getvalue()

    def to_yaml(self):
        """
        Return a YAML string with the serialized object.
        """
        return self._to_yaml(self)

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
        """
        Make sure that copying the class still works as usual, without
        dropping some attributes by defining __copy__
        """
        try:
            return super().__copy__()
        except AttributeError:
            cls = self.__class__
            new = cls.__new__(cls)
            new.__dict__.update(self.__dict__)
            return new

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

    # Capture the warnings as log entries
    logging.captureWarnings(True)

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

    @classmethod
    def join(cls, path1, path2):
        """
        Join two paths together, similarly to :func:`os.path.join`.

        If ``path1`` is a :class:`ArtifactPath`, the result will also be one,
        and the root of ``path1`` will be used as the root of the new path.
        """
        if isinstance(path1, cls):
            joined = cls(
                root=path1.root,
                relative=os.path.join(path1.relative, str(path2))
            )
        else:
            joined = os.path.join(str(path1), str(path2))

        return joined


def groupby(iterable, key=None):
    # We need to sort before feeding to groupby, or it will fail to establish
    # the groups as expected.
    iterable = sorted(iterable, key=key)
    return itertools.groupby(iterable, key=key)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    # Since the same iterator is used, it will yield a new item every time zip
    # call next() on it
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

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

def get_nested_key(mapping, key_path, getitem=operator.getitem):
    """
    Get a key in a nested mapping

    :param mapping: The mapping to lookup in
    :type mapping: collections.abc.Mapping

    :param key_path: Path to the key in the mapping, in the form of a list of
        keys.
    :type key_path: list

    :param getitem: Function used to get items on the mapping. Defaults to
        :func:`operator.getitem`.
    :type getitem: collections.abc.Callable
    """
    if not key_path:
        return mapping
    for key in key_path[:-1]:
        mapping = getitem(mapping, key)
    return getitem(mapping, key_path[-1])

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

def get_call_site(levels=0, exclude_caller_module=False):
    """
    Get the location of the source that called that function.

    :returns: (caller, filename, lineno) tuples. Any component can be None if
        nothing was found. Caller is a string containing the function name.

    :param levels: How many levels to look at in the stack
    :type levels: int

    :param exclude_caller_module: Return the first function in the stack that
        is not defined in the same module as the direct caller of
        :func:`get_call_site`.

    .. warning:: That function will exclude all source files that are not part
        of the `lisa` package. It will also exclude functions of
        :mod:`lisa.utils` module.
    """

    try:
        # context=0 speeds up a lot the stack retrieval, since it avoids
        # reading the source files
        stack = inspect.stack(context=0)
    # Getting the stack can sometimes fail under IPython for some reason:
    # https://github.com/ipython/ipython/issues/1456/
    except IndexError:
        return (None, None, None)

    # Exclude all functions from lisa.utils
    excluded_files = {
        __file__,
    }
    if exclude_caller_module:
        excluded_files.add(stack[1].filename)

    caller = None
    filename = None
    lineno = None
    for frame in stack[levels + 1:]:
        caller = frame.function
        filename = frame.filename
        lineno = frame.lineno
        # exclude all non-lisa sources
        if not any(
            filename.startswith(path)
            for path in lisa.__path__
        ) or filename in excluded_files:
            continue
        else:
            break

    return (caller, filename, lineno)

def is_running_sphinx():
    """
    Returns True if the module is imported when Sphinx is running, False
    otherwise.
    """
    return 'sphinx' in sys.modules

def non_recursive_property(f):
    """
    Create a property that raises an :exc:`AttributeError` if it is re-entered.

    .. note:: This only guards against single-thread accesses, it is not
        threadsafe.
    """

    # WeakKeyDictionary ensures that instances will not be held alive just for
    # the guards. Since there is one guard_map per property, we only need to
    # index on the instances
    guard_map = WeakKeyDictionary()

    def _get(self):
        return guard_map.get(self, False)

    def _set(self, val):
        guard_map[self] = val

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if _get(self):
            raise AttributeError('Recursive access to property "{}.{}" while computing its value'.format(
                self.__class__.__qualname__, f.__name__,
            ))

        try:
            _set(self, True)
            return f(self, *args, **kwargs)
        finally:
            _set(self, False)

    return property(wrapper)


def get_short_doc(obj):
    """
    Get the short documentation paragraph at the beginning of docstrings.
    """
    docstring = inspect.getdoc(obj)
    if docstring:
        docstring = split_paragraphs(docstring)[0]
        docstring = ' '.join(docstring.splitlines())
        if not docstring.endswith('.'):
            docstring += '.'
    else:
        docstring = ''
    return docstring


def update_wrapper_doc(func, added_by=None, description=None, remove_params=None, include_kwargs=False):
    """
    Equivalent to :func:`functools.wraps` that updates the signature by taking
    into account the wrapper's extra *keyword-only* parameters and the given
    description.

    :param func: callable to decorate
    :type func: collections.abc.Callable

    :param added_by: Add some kind of reference to give a sense of where the
        new behaviour of the wraps function comes from.
    :type added_by: str or None

    :param description: Extra description output in the docstring.
    :type description: str or None

    :param remove_params: Set of parameter names of ``func`` to not include in
        the decorated function signature. This can be used to hide parameters
        that are only used as part of a decorated/decorator protocol, and not
        exposed in the final decorated function.
    :type remove_params: list(str) or None

    :param include_kwargs: If `True`, variable keyword parameter (``**kwargs``)
        of the decorator is kept in the signature. It is usually removed, since
        it's mostly used to transparently forward arguments to the inner
        ``func``, but can also serve other purposes.
    :type include_kwargs: bool

    .. note:: :func:`functools.wraps` is applied by this decorator, which will
        not work if you applied it yourself.
    """

    if description:
        description = '\n{}\n'.format(description)

    remove_params = remove_params if remove_params else set()

    def decorator(f):
        wrapper_sig = inspect.signature(f)
        f = functools.wraps(func)(f)
        f_sig = inspect.signature(f)

        added_params = [
            desc
            for name, desc in wrapper_sig.parameters.items()
            if (
                desc.kind == inspect.Parameter.KEYWORD_ONLY
                or (
                    include_kwargs
                    and desc.kind == inspect.Parameter.VAR_KEYWORD
                )
            )
        ]
        added_names = set(desc.name for desc in added_params)

        if include_kwargs:
            f_var_keyword_params = []
        else:
            f_var_keyword_params = [
                desc
                for name, desc in f_sig.parameters.items()
                if (
                    desc.kind == inspect.Parameter.VAR_KEYWORD
                    and name not in remove_params
                )
            ]

        f_params = [
            desc
            for name, desc in f_sig.parameters.items()
            if (
                desc.name not in added_names
                and desc not in f_var_keyword_params
                and name not in remove_params
            )
        ]

        f.__signature__ = f_sig.replace(
            # added_params are keyword-only, so they need to go before the var
            # keyword param if there is any
            parameters=f_params + added_params + f_var_keyword_params,
        )

        # Replace the one-liner f description
        extra_doc = "\n\n{added_by}{description}".format(
            added_by='**Added by** {}:\n'.format(added_by) if added_by else '',
            description=description if description else '',
        )

        f_doc = inspect.getdoc(f) or ''
        f.__doc__ = f_doc + extra_doc

        return f
    return decorator


DEPRECATED_MAP = {}
"""
Global dictionary of deprecated classes, functions and so on.
"""

def deprecate(msg=None, replaced_by=None, deprecated_in=None, removed_in=None):
    """
    Mark a class, method, function etc as deprecated and update its docstring.

    :param msg: Message to tell more about this deprecation.
    :type msg: str or None

    :param replaced_by: Other object the deprecated object is replaced by.
    :type replaced_by: object

    :param deprecated_in: Version in which the object was flagged as deprecated.
    :type deprecated_in: str

    :param removed_in: Version in which the deprecated object will be removed.
    :type removed_in: str

    .. note:: In order to decorate all the accessors of properties, apply the
        decorator once the property is fully built::

            class C:
                @property
                def foo(self):
                    pass

                @foo.setter
                def foo(self, val):
                    pass

                # Once all getters/setter/deleters are set, apply the decorator
                foo = deprecate()(foo)
    """

    # RestructuredText Sphinx role
    def getrole(obj):
        if isinstance(obj, type):
            return 'class'
        elif callable(obj):
            if '<locals>' in obj.__qualname__:
                return 'code'
            elif '.' in obj.__qualname__:
                return 'meth'
            else:
                return 'func'
        else:
            return 'code'

    def getname(obj, style=None, abbrev=False):
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        elif isinstance(obj, property):
            obj = obj.fget

        try:
            mod = obj.__module__ + '.'
        except AttributeError:
            mod = ''

        try:
            qualname = obj.__qualname__
        except AttributeError:
            qualname = str(obj)

        name = mod + qualname

        if style == 'rst':
            return ':{}:`{}{}{}`'.format(
                getrole(obj),
                '~' if abbrev else '',
                mod, qualname
            )
        else:
            return name

    def get_meth_stacklevel(func_name):
        # Special methods are usually called from another module, so
        # make sure the warning filters set on lisa will pick these up.
        if func_name.startswith('__') and func_name.endswith('__'):
            return 1
        else:
            return 2

    if removed_in:
        removed_in = parse_version(removed_in)
    current_version = lisa.version.version_tuple

    def decorator(obj):
        obj_name = getname(obj)

        if removed_in and current_version >= removed_in:
            raise DeprecationWarning('{name} was marked as being removed in version {removed_in} but is still present in current version {version}'.format(
                name=obj_name,
                removed_in=format_version(removed_in),
                version=format_version(current_version),
            ))

        def make_msg(style=None):
            if replaced_by is not None:
                try:
                    doc_url = ' (see: {})'.format(get_doc_url(replaced_by))
                except Exception:
                    doc_url = ''

                replacement_msg = ', use {} instead{}'.format(
                    getname(replaced_by, style=style), doc_url,
                )
            else:
                replacement_msg = ''

            if removed_in:
                removal_msg = ' and will be removed in version {}'.format(
                    format_version(removed_in)
                )
            else:
                removal_msg = ''

            return '{name} is deprecated{remove}{replace}{msg}'.format(
                name=getname(obj, style=style, abbrev=True),
                replace=replacement_msg,
                remove=removal_msg,
                msg=': ' + msg if msg else '',
            )

        # stacklevel != 1 breaks the filtering for warnings emitted by APIs
        # called from external modules, like __init_subclass__ that is called
        # from other modules like abc.py
        def wrap_func(func, stacklevel=1):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(make_msg(), DeprecationWarning, stacklevel=stacklevel)
                return func(*args, **kwargs)
            return wrapper

        # Make sure we don't accidentally override an existing entry
        assert obj_name not in DEPRECATED_MAP
        DEPRECATED_MAP[obj_name] = {
            'obj': obj,
            'replaced_by': replaced_by,
            'msg': msg,
            'removed_in': removed_in,
            'deprecated_in': deprecated_in,
        }

        # For classes, wrap __new__ and update docstring
        if isinstance(obj, type):
            # Warn on instance creation
            obj.__new__ = wrap_func(obj.__new__)
            # Will show the warning when the class is subclassed
            # in Python >= 3.6
            obj.__init_subclass__ = wrap_func(obj.__init_subclass__)
            return_obj = obj
            update_doc_of = obj

        elif isinstance(obj, property):
            # Since we cannot update the property itself, replace it with a new
            # one that uses a wrapped getter. This should be safe as properties
            # seems to be immutable, so there is no risk of somebody
            # monkey-patching the object and us throwing away the extra
            # attributes.
            # Note that this will only wrap accessors that are visible at the
            # time the decorator is applied.
            obj = property(
                fget=wrap_func(obj.fget, stacklevel=2),
                fset=wrap_func(obj.fset, stacklevel=2),
                fdel=wrap_func(obj.fdel, stacklevel=2),
            )
            return_obj = obj
            update_doc_of = obj

        elif isinstance(obj, (staticmethod, classmethod)):
            func = obj.__func__
            stacklevel = get_meth_stacklevel(func.__name__)
            func = wrap_func(func, stacklevel=stacklevel)
            # Build a new staticmethod/classmethod with the updated function
            return_obj = obj.__class__(func)
            # Updating the __doc__ of the staticmethod/classmethod itself will
            # have no effect, so update the doc of the underlying function
            update_doc_of = func

        # For other callables, emit the warning when called
        else:
            stacklevel = get_meth_stacklevel(obj.__name__)
            return_obj = wrap_func(obj, stacklevel=stacklevel)
            update_doc_of = return_obj

        doc = inspect.getdoc(update_doc_of) or ''
        update_doc_of.__doc__ = doc + '\n\n' + textwrap.dedent(
        """
        .. attention::

            .. deprecated:: {deprecated_in}

            {msg}
        """.format(
            deprecated_in=deprecated_in if deprecated_in else '<unknown>',
            msg=make_msg(style='rst'),
        )).strip()

        return return_obj

    return decorator


def get_doc_url(obj):
    """
    Return an URL to the documentation about the given object.
    """

    # If it does not have a __qualname__, we are probably more interested in
    # its class
    if not hasattr(obj, '__qualname__'):
        obj = obj.__class__

    obj_name = '{}.{}'.format(
        inspect.getmodule(obj).__name__,
        obj.__qualname__
    )

    return _get_doc_url(obj_name)


# Make sure to cache (almost) all the queries with a strong reference over
# `obj_name` values
@functools.lru_cache(maxsize=4096)
def _get_doc_url(obj_name):
    doc_base_url = 'https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/'
    # Use the inventory built by RTD
    inv_url = urllib.parse.urljoin(doc_base_url, 'objects.inv')

    inv = sphobjinv.Inventory(url=inv_url)

    for inv_obj in inv.objects:
        if inv_obj.name == obj_name:
            doc_page = inv_obj.uri.replace('$', inv_obj.name)
            doc_url = urllib.parse.urljoin(doc_base_url, doc_page)
            return doc_url

    raise ValueError('Could not find the doc of: {}'.format(obj_name))


def show_doc(obj, iframe=False):
    """
    Show the online LISA documentation about the given object.

    :param obj: Object to show the doc of. It can be anything, including
        instances.
    :type obj: object

    :param iframe: If ``True``, uses an IFrame, otherwise opens a web browser.
    :type iframe: bool
    """
    doc_url = get_doc_url(obj)

    if iframe:
        print(doc_url)
        return IFrame(src=doc_url, width="100%", height="600em")
    else:
        webbrowser.open(doc_url)


def split_paragraphs(string):
    """
    Split `string` into a list of paragraphs.

    A paragraph is delimited by empty lines, or lines containing only
    whitespace characters.
    """
    para_list = []
    curr_para = []
    for line in string.splitlines(keepends=True):
        if line.strip():
            curr_para.append(line)
        else:
            para_list.append(''.join(curr_para))
            curr_para = []

    if curr_para:
        para_list.append(''.join(curr_para))

    return para_list

mimetypes.add_type('text/rst', '.rst')
def guess_format(path):
    """
    Guess the file format from a `path`, using the mime types database.
    """
    if path is None:
        return None

    mime_type = mimetypes.guess_type(path, strict=False)[0]
    guessed_format = mime_type.split('/')[1].split('.', 1)[-1].split('+')[0]
    return guessed_format

@contextlib.contextmanager
def nullcontext(enter_result=None):
    """
    Backport of Python 3.7 ``contextlib.nullcontext``

    This context manager does nothing, so it can be used as a default
    placeholder for code that needs to select at runtime what context manager
    to use.

    :param enter_result: Object that will be bound to the target of the with
        statement, or `None` if nothing is specified.
    :type enter_result: object
    """
    yield enter_result


class ExekallTaggable:
    """
    Allows tagging the objects produced in exekall expressions ID.

    .. seealso:: :ref:`exekall expression ID<exekall-expression-id>`
    """

    @abc.abstractmethod
    def get_tags(self):
        """
        :return: Dictionary of tags and tag values
        :rtype: dict(str, object)
        """
        return {}

def annotations_from_signature(sig):
    """
    Build a PEP484 ``__annotations__`` dictionary from a :class:`inspect.Signature`.
    """
    annotations = {
        name: param_spec.annotation
        for name, param_spec in sig.parameters.items()
        if param_spec.annotation != inspect.Parameter.empty
    }

    if sig.return_annotation != inspect.Signature.empty:
        annotations['return'] = sig.return_annotation

    return annotations

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
