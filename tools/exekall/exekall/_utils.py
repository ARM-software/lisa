#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
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

import collections
import contextlib
import fnmatch
import functools
import gc
import importlib
import inspect
import io
import itertools
import logging
import pathlib
import pickle
import subprocess
import sys
import tempfile
import traceback
import types
import uuid
import glob
import textwrap
import argparse
import time
import datetime

DB_FILENAME = 'VALUE_DB.pickle.xz'


class NotSerializableError(Exception):
    pass


def get_class_from_name(cls_name, module_map=None):
    """
    Get a class object from its full name (including the module name).
    """
    # Avoid argument default value that would be huge in Sphinx doc
    module_map = module_map if module_map is not None else sys.modules
    possible_mod_set = {
        mod_name
        for mod_name in module_map.keys()
        if cls_name.startswith(mod_name)
    }

    # Longest match in term of number of components
    possible_mod_list = sorted(possible_mod_set, key=lambda name: len(name.split('.')))
    if possible_mod_list:
        mod_name = possible_mod_list[-1]
    else:
        return None

    mod = module_map[mod_name]
    cls_name = cls_name[len(mod_name) + 1:]
    return _get_class_from_name(cls_name, mod)


def _get_class_from_name(cls_name, namespace):
    if isinstance(namespace, collections.abc.Mapping):
        namespace = types.SimpleNamespace(**namespace)

    split = cls_name.split('.', 1)
    try:
        obj = getattr(namespace, split[0])
    except AttributeError as e:
        raise ValueError('Object not found') from e

    if len(split) > 1:
        return _get_class_from_name('.'.join(split[1:]), obj)
    else:
        return obj


def create_uuid():
    """
    Creates a UUID.
    """
    return uuid.uuid4().hex


def get_mro(cls):
    """
    Wrapper on top of :func:`inspect.getmro` that recognizes ``None`` as a
    type (treated like ``type(None)``).
    """
    if cls is type(None) or cls is None:
        return (type(None), object)
    else:
        assert isinstance(cls, type)
        return inspect.getmro(cls)


def get_method_class(function):
    """
    Get the class of a method by analyzing its name.
    """
    cls_name = function.__qualname__.rsplit('.', 1)[0]
    if '<locals>' in cls_name:
        return None
    return eval(cls_name, function.__globals__)


def get_name(obj, full_qual=True, qual=True, pretty=False):
    """
    Get a name for ``obj`` (function or class) that can be used in a generated
    script.

    :param full_qual: Full name of the object, including its module name.
    :type full_qual: bool

    :param qual: Qualified name of the object
    :type qual: bool

    :param pretty: If ``True``, will show a prettier name for some types,
        although it is not guarnateed it will actually be a type name. For example,
        ``type(None)`` will be shown as ``None`` instead of ``NoneType``.
    :type pretty: bool
    """
    # full_qual enabled implies qual enabled
    _qual = qual or full_qual
    # qual disabled implies full_qual disabled
    full_qual = full_qual and qual
    qual = _qual

    if qual:
        def _get_name(x): return x.__qualname__
    else:
        def _get_name(x): return x.__name__

    if obj is None:
        pretty = True
        obj = type(obj)

    if pretty:
        for prettier_obj in {None, NoValue}:
            if obj == type(prettier_obj):
                # For these types, qual=False or qual=True makes no difference
                obj = prettier_obj
                def _get_name(x): return str(x)
                break

    # Add the module's name in front of the name to get a fully
    # qualified name
    if full_qual:
        try:
            module_name = obj.__module__
        except AttributeError:
            module_name = ''
        else:
            module_name = (
                module_name + '.'
                if module_name not in ('__main__', 'builtins', None)
                else ''
            )
    else:
        module_name = ''

    # Classmethods appear as bound method of classes. Since each subclass will
    # get a different bound method object, we want to reflect that in the
    # name we use, instead of always using the same name that the method got
    # when it was defined
    if inspect.ismethod(obj):
        name = _get_name(obj.__self__) + '.' + obj.__name__
    else:
        name = _get_name(obj)

    return module_name + name


def get_toplevel_module(obj):
    """
    Return the outermost module object in which ``obj`` is defined as a tuple
    of source file path and line number.

    This is usually a package.
    """
    module = inspect.getmodule(obj)
    toplevel_module_name = module.__name__.split('.')[0]
    toplevel_module = sys.modules[toplevel_module_name]
    return toplevel_module


def get_src_loc(obj, shorten=True):
    """
    Get the source code location of ``obj``

    :param shorten: Shorten the paths of the source files by only keeping the
        part relative to the top-level package.
    :type shorten: bool
    """
    try:
        src_line = inspect.getsourcelines(obj)[1]
        src_file = inspect.getsourcefile(obj)
        src_file = pathlib.Path(src_file).resolve()
    except (OSError, TypeError):
        src_line, src_file = None, None

    if shorten and src_file:
        mod = get_toplevel_module(obj)
        try:
            paths = mod.__path__
        except AttributeError:
            paths = [mod.__file__]

        paths = [pathlib.Path(p) for p in paths if p]
        paths = [
            p.parent.resolve()
            for p in paths
            if p in src_file.parents
        ]
        if paths:
            src_file = src_file.relative_to(paths[0])

    return (str(src_file), src_line)


class ExceptionPickler(pickle.Pickler):
    """
    Fix pickling of exceptions so they can be reloaded without troubles, even
    when called with keyword arguments:

    https://bugs.python.org/issue40917
    """

    class _DispatchTable:
        """
        For Python < 3.8 (i.e. before Pickler.reducer_override is available)
        """
        def __init__(self, pickler):
            self.pickler = pickler

        def __getitem__(self, cls):
            try:
                return self.pickler._get_reducer(cls)
            except ValueError:
                raise KeyError

    def __init__(self, *args, **kwargs):
        self.dispatch_table = self._DispatchTable(self)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _make_excep(excep_cls, dct):
        new = excep_cls.__new__(excep_cls)

        # "args" is a bit magical: it's not stored in __dict__, and any "arg"
        # key __dict__ will basically be ignored, so it needs to be restored
        # manually.
        dct = copy.copy(dct)
        try:
            args = dct.pop('args')
        except KeyError:
            pass
        else:
            new.args = args

        new.__dict__ = dct
        return new

    @classmethod
    def _reduce_excep(cls, obj):
        # The __dict__ of exceptions does not contain "args", so shoe-horn
        # it in there so it's passed as a regular attribute.
        dct = obj.__dict__.copy()
        try:
            dct['args'] = obj.args
        except AttributeError:
            pass
        return (cls._make_excep, (obj.__class__, dct))

    # Only for Python >= 3.8, otherwise self.dispatch_table is used
    def reducer_override(self, obj):
        try:
            reducer = self._get_reducer(type(obj))
        except ValueError:
            # Fallback to default behaviour
            return NotImplemented
        else:
            return reducer(obj)

    def _get_reducer(self, cls):
        # Workaround this bug:
        # https://bugs.python.org/issue43460
        if issubclass(cls, BaseException):
            return self._reduce_excep
        else:
            raise ValueError('Class not handled')

    @classmethod
    def dump_bytestring(cls, obj, **kwargs):
        f = io.BytesIO()
        cls.dump_file(f, obj, **kwargs)
        return f.getvalue()

    @classmethod
    def dump_file(cls, f, obj, **kwargs):
        pickler = cls(f, **kwargs)
        return pickler.dump(obj)


def is_serializable(obj, raise_excep=False):
    """
    Try to Pickle the object to see if that raises any exception.
    """
    stream = io.StringIO()
    try:
        # This may be slow for big objects but it is the only way to be sure
        # it can actually be serialized
        ExceptionPickler.dump_bytestring(obj)
    except (TypeError, pickle.PickleError, AttributeError) as e:
        debug('Cannot serialize instance of {}: {}'.format(
            type(obj).__qualname__, str(e)
        ))
        if raise_excep:
            raise NotSerializableError(obj) from e
        return False
    else:
        return True


def once(callable_):
    """
    Call the given function at most once per set of parameters
    """
    return functools.lru_cache(maxsize=None, typed=True)(callable_)


def remove_indices(iterable, ignored_indices):
    """
    Filter the given ``iterable`` by removing listed in ``ignored_indices``.
    """
    return [v for i, v in enumerate(iterable) if i not in ignored_indices]


def get_module_basename(path):
    """
    Get the module name of the module defined in source ``path``.
    """
    path = pathlib.Path(path)
    module_name = inspect.getmodulename(str(path))
    # This is either garbage or a package
    if module_name is None:
        module_name = path.name
    return module_name


def iterate_cb(iterator, pre_hook=None, post_hook=None):
    """
    Iterate over ``iterator``, and  call some callbacks.

    :param pre_hook: Callback called right before getting a new value.
    :type pre_hook: collections.abc.Callable

    :param post_hook: Callback called right after getting a new value.
    :type post_hook: collections.abc.Callable
    """
    with contextlib.suppress(StopIteration):
        for i in itertools.count():
            # Do not execute pre_hook on the first iteration
            if pre_hook and i:
                pre_hook()
            val = next(iterator)
            if post_hook:
                post_hook()

            yield val


def format_exception(e):
    """
    Format the traceback of the exception ``e`` in a string.
    """
    elements = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(elements)


# Logging level above CRITICAL that is always displayed and used for output
LOGGING_OUT_LEVEL = 60
"""
Log level used for the ``OUT`` level.

This allows sending all the output through the logging module instead of using
:func:`print`, so it can easily be recorded to a file
"""


class ExekallFormatter(logging.Formatter):
    """
    Custom :class:`logging.Formatter` that takes care of ``OUT`` level.

    This ``OUT`` level allows using :mod:`logging` instead of :func:`print` so
    it can be redirected to a file easily.
    """

    def __init__(self, fmt, *args, **kwargs):
        self.default_fmt = logging.Formatter(fmt, *args, **kwargs)
        self.out_fmt = logging.Formatter('%(message)s', *args, **kwargs)

    def format(self, record):
        # level above CRITICAL, so it is always displayed
        if record.levelno == LOGGING_OUT_LEVEL:
            return self.out_fmt.format(record)
        # regular levels are logged with the regular formatter
        else:
            return self.default_fmt.format(record)


LOGGING_FOMATTER_MAP = {
    'normal': ExekallFormatter('[%(asctime)s][%(name)s] %(levelname)s  %(message)s'),
    'verbose': ExekallFormatter('[%(asctime)s][%(name)s/%(filename)s:%(lineno)s] %(levelname)s  %(message)s'),
}


def setup_logging(log_level, debug_log_file=None, info_log_file=None, verbose=0):
    """
    Setup the :mod:`logging` module.

    :param log_level: Lowest log level name to display.
    :type log_level: str

    :param debug_log_file: Path to a file where logs are collected at the
        ``DEBUG`` level.
    :type debug_log_file: str

    :param info_log_file: Path to a file where logs are collected at the
        ``INFO`` level.
    :type info_log_file: str

    :param verbose: Verbosity level. The format string for log entries will
        contain more information when the level increases.`
    :type verbose: int
    """
    logging.addLevelName(LOGGING_OUT_LEVEL, 'OUT')
    level = getattr(logging, log_level.upper())

    logger = logging.getLogger()
    # We do not filter anything at the logger level, only at the handler level
    logger.setLevel(logging.NOTSET)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = 'verbose' if verbose else 'normal'
    formatter = LOGGING_FOMATTER_MAP[formatter]
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if debug_log_file:
        file_handler = logging.FileHandler(str(debug_log_file), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LOGGING_FOMATTER_MAP['verbose'])
        logger.addHandler(file_handler)

    if info_log_file:
        file_handler = logging.FileHandler(str(info_log_file), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOGGING_FOMATTER_MAP['normal'])
        logger.addHandler(file_handler)

    # Redirect all warnings of the "warnings" module as log entries
    logging.captureWarnings(True)


EXEKALL_LOGGER = logging.getLogger('EXEKALL')


def out(msg):
    """
    To be used as a replacement of :func:`print`.

    This allows easy redirection of the output to a file.
    """
    EXEKALL_LOGGER.log(LOGGING_OUT_LEVEL, msg)


def info(msg):
    """Write a log message at the INFO level."""
    EXEKALL_LOGGER.info(msg)


def debug(msg):
    """Write a log message at the DEBUG level."""
    EXEKALL_LOGGER.debug(msg)


def warn(msg):
    """Write a log message at the WARNING level."""
    EXEKALL_LOGGER.warning(msg)


def error(msg):
    """Write a log message at the ERROR level."""
    EXEKALL_LOGGER.error(msg)


def infer_mod_name(python_src):
    """
    Compute the module name of a Python source file by inferring its top-level
    package
    """
    python_src = pathlib.Path(python_src)
    module_path = None

    # First look for the outermost package we find in the parent directories.
    # If we were supplied a path, it will not try to go past its highest folder.
    for folder in reversed(python_src.parents):
        if pathlib.Path(folder, '__init__.py').exists():
            package_root_parent = folder.parents[0]
            module_path = python_src.relative_to(package_root_parent)
            break
    # If no package was found, we try to find it through sys.path in case it is
    # only using namespace packages
    else:
        for package_root_parent in sys.path:
            try:
                module_path = python_src.relative_to(package_root_parent)
                break
            except ValueError:
                continue

    # If we found the top-level package
    if module_path is not None:
        module_parents = list(module_path.parents)
        module_basename = get_module_basename(module_path)

        # Import all parent package_names before we import the module
        for package_name in reversed(module_parents[:-1]):
            package_name = import_file(
                pathlib.Path(package_root_parent, package_name),
                module_name='.'.join(package_name.parts),
                is_package=True,
            )

        module_dotted_path = list(module_parents[0].parts) + [module_basename]
        module_name = '.'.join(module_dotted_path)

    else:
        module_name = get_module_basename(python_src)

    return module_name


def find_customization_module_set(module_set):
    """
    Find all customization modules, where subclasses of
    :class:`exekall.customization.AdaptorBase` are expected to be found.

    It looks for modules named ``exekall_customize`` present in any enclosing
    package of modules in ``module_set``.
    """
    def build_full_names(l_l):
        """Explode list of lists, and build full package names."""
        for l in l_l:
            for i, _ in enumerate(l):
                i += 1
                yield '.'.join(l[:i])

    # Exception raised changed in 3.7:
    # https://docs.python.org/3/library/importlib.html#importlib.util.find_spec
    if sys.version_info >= (3, 7):
        import_excep = ModuleNotFoundError
    else:
        import_excep = AttributeError

    package_names_list = [
        module.__name__.split('.')
        for module in module_set
    ]
    package_name_set = set(build_full_names(package_names_list))

    customization_module_set = set()

    for name in package_name_set:
        customize_name = name + '.exekall_customize'
        # Only hide ModuleNotFoundError exceptions when looking up that
        # specific module, we don't want to hide issues inside the module
        # itself.
        module_exists = False
        with contextlib.suppress(import_excep, ValueError):
            module_exists = importlib.util.find_spec(customize_name)

        if module_exists:
            # Importing that module is enough to make the adaptor visible
            # to the Adaptor base class
            customize_module = importlib.import_module(customize_name)
            customization_module_set.add(customize_module)

    return customization_module_set


def import_modules(paths_or_names, excep_handler=None):
    """
    Import the modules in the given list of paths.

    If a folder is passed, all Python sources are recursively imported.
    """
    def import_it(path_or_name):
        # Recursively import all modules when passed folders
        if path_or_name.is_dir():
            yield from import_folder(path_or_name, excep_handler=excep_handler)
        # If passed a file, a symlink or something like that
        elif path_or_name.exists():
            try:
                yield import_file(path_or_name)
            except Exception as e:
                if excep_handler:
                    return excep_handler(str(path_or_name), e)
                else:
                    raise
        # Otherwise, assume it is just a module name
        else:
            yield from import_name_recursively(path_or_name, excep_handler=excep_handler)

    return set(itertools.chain.from_iterable(
        import_it(pathlib.Path(path))
        for path in paths_or_names
    ))


def import_name_recursively(name, excep_handler=None):
    """
    Import a module by its name.

    :param name: Full name of the module.
    :type name: str

    If it's a package, import all submodules recursively.
    """

    name_str = str(name)
    try:
        mod = importlib.import_module(name_str)
    except Exception as e:
        if excep_handler:
            return excep_handler(name_str, e)
        else:
            raise
    try:
        paths = mod.__path__
    # This is a plain module
    except AttributeError:
        yield mod
    # This is a package, so we import all the submodules recursively
    else:
        for path in paths:
            yield from import_folder(pathlib.Path(path), excep_handler=excep_handler)


def import_folder(path, excep_handler=None):
    """
    Import all modules contained in the given folder, recurisvely.
    """
    for python_src in glob.iglob(str(path / '**' / '*.py'), recursive=True):
        try:
            yield import_file(python_src)
        except Exception as e:
            if excep_handler:
                excep_handler(python_src, e)
                continue
            else:
                raise


def import_file(python_src, module_name=None, is_package=False):
    """
    Import a module.

    :param python_src: Path to a Python source file.
    :type python_src: str or pathlib.Path

    :param module_name: Name under which to import the module. If ``None``, the
        name is inferred using :func:`infer_mod_name`
    :type module_name: str

    :param is_package: ``True`` if the module is a package. If a folder or
        ``__init__.py`` is passed, this is forcefully set to ``True``.
    :type is_package: bool

    """
    python_src = pathlib.Path(python_src).resolve()

    # Directly importing __init__.py does not really make much sense and may
    # even break, so just import its package instead.
    if python_src.name == '__init__.py':
        return import_file(
            python_src=python_src.parent,
            module_name=module_name,
            is_package=True
        )

    if python_src.is_dir():
        is_package = True

    if module_name is None:
        module_name = infer_mod_name(python_src)

    # Check if the module has already been imported
    if module_name in sys.modules:
        return sys.modules[module_name]

    is_namespace_package = False
    if is_package:
        # Signify that it is a package to
        # importlib.util.spec_from_file_location
        submodule_search_locations = [str(python_src)]
        init_py = pathlib.Path(python_src, '__init__.py')
        # __init__.py does not exists for namespace packages
        if init_py.exists():
            python_src = init_py
        else:
            is_namespace_package = True
    else:
        submodule_search_locations = None

    # Python >= 3.5 style
    if hasattr(importlib.util, 'module_from_spec'):
        # We manually build a ModuleSpec for namespace packages, since
        # spec_from_file_location apparently does not handle them
        if is_namespace_package:
            spec = importlib.machinery.ModuleSpec(
                name=module_name,
                # loader is None for namespace packages
                loader=None,
                is_package=True,
            )
            # Set __path__ for namespace packages
            spec.submodule_search_locations = submodule_search_locations
        else:
            spec = importlib.util.spec_from_file_location(
                module_name,
                str(python_src),
                submodule_search_locations=submodule_search_locations,
            )
            if spec is None:
                raise ModuleNotFoundError(
                    'Could not find module "{module}" at {path}'.format(
                        module=module_name,
                        path=python_src
                    ),
                    name=module_name,
                    path=python_src,
                )

        module = importlib.util.module_from_spec(spec)
        if not is_namespace_package:
            try:
                # Register module before executing it so relative imports will
                # work
                sys.modules[module_name] = module
                # Nothing to execute in a namespace package
                spec.loader.exec_module(module)
            # If the module cannot be imported cleanly regardless of the reason,
            # make sure we remove it from sys.modules since it's broken. Future
            # attempt to import it should raise again, rather than returning the
            # broken module
            except BaseException:
                with contextlib.suppress(KeyError):
                    del sys.modules[module_name]
                raise
            else:

                # Set the attribute on the parent package, so that this works:
                #
                #    import foo.bar
                #    print(foo.bar)
                try:
                    parent_name, last = module_name.rsplit('.', 1)
                except ValueError:
                    pass
                else:
                    parent = sys.modules[parent_name]
                    setattr(parent, last, module)



    #  Python <= v3.4 style
    else:
        module = importlib.machinery.SourceFileLoader(
            module_name, str(python_src)).load_module()

    sys.modules[module_name] = module
    importlib.invalidate_caches()
    return module


def flatten_seq(seq, levels=1):
    """
    Flatten a nested sequence, up to ``levels`` levels.
    """
    if levels == 0:
        return seq
    else:
        seq = list(itertools.chain.from_iterable(seq))
        return flatten_seq(seq, levels=levels - 1)


def take_first(iterable):
    """
    Pick the first item of ``iterable``.
    """
    for i in iterable:
        return i
    return NoValue


class _NoValueType:
    """
    Type of the :attr:`NoValue` singleton.

    This is mostly used like ``None``, in places where ``None`` may be an
    acceptable value.
    """
    # Use a singleton pattern to make sure that even deserialized instances
    # will be the same object
    def __new__(cls):
        try:
            return cls._instance
        except AttributeError:
            obj = super().__new__(cls)
            cls._instance = obj
            return obj

    def __eq__(self, other):
        return isinstance(other, _NoValueType)

    def __hash__(self):
        return 0

    def __bool__(self):
        return False

    def __repr__(self):
        return 'NoValue'

    def __eq__(self, other):
        return type(self) is type(other)


NoValue = _NoValueType()
"""
Singleton with similar purposes as ``None``.
"""


class RestartableIter:
    """
    Wrap an iterator to give a new iterator that is restartable.
    """

    def __init__(self, it):
        self.values = []

        # Wrap the iterator to update the memoized values
        def wrapped(it):
            for x in it:
                self.values.append(x)
                yield x

        self.it = wrapped(it)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            # Use the stored values the next time we try to get an
            # itertor again
            self.it = iter(self.values)
            raise


def get_froz_val_set_set(db, uuid_seq=None, type_pattern_seq=None):
    """
    Get a set of sets of :class:`exekall.engine.FrozenExprVal`.

    :param db: :class:`exekall.engine.ValueDB` to look into
    :type db: exekall.engine.ValueDB

    :param uuid_seq: Sequence of UUIDs to select.
    :type uuid_seq: list(str)

    :param type_pattern_seq: Sequence of :func:`fnmatch.fnmatch` patterns
        matching type names (including module name).
    :type type_pattern_seq: list(str)
    """

    def uuid_predicate(froz_val):
        return froz_val.uuid in uuid_seq

    def type_pattern_predicate(froz_val):
        return match_base_cls(froz_val.type_, type_pattern_seq)

    if type_pattern_seq and not uuid_seq:
        predicate = type_pattern_predicate

    elif uuid_seq and not type_pattern_seq:
        predicate = uuid_predicate

    elif not uuid_seq and not type_pattern_seq:
        def predicate(froz_val): return True

    else:
        def predicate(froz_val):
            return uuid_predicate(froz_val) and type_pattern_predicate(froz_val)

    return db.get_by_predicate(predicate, flatten=False, deduplicate=True)


def match_base_cls(cls, pattern_list):
    """
    Match the name of the class of the object and all its base classes.
    """
    for base_cls in get_mro(cls):
        base_cls_name = get_name(base_cls, full_qual=True)
        if not base_cls_name:
            continue
        if match_name(base_cls_name, pattern_list):
            return True

    return False


def match_name(name, pattern_list):
    """
    Return ``True`` if ``name`` is matched by any pattern in ``pattern_list``.

    If a pattern starts with ``!``, it is taken as a negative pattern.
    """
    if name is None:
        return False

    if not pattern_list:
        return False

    neg_patterns = {
        pattern[1:]
        for pattern in pattern_list
        if pattern.startswith('!')
    }

    pos_patterns = {
        pattern
        for pattern in pattern_list
        if not pattern.startswith('!')
    }

    def invert(x): return not x
    def identity(x): return x

    def check(pattern_set, f):
        if pattern_set:
            ok = any(
                fnmatch.fnmatch(name, pattern)
                for pattern in pattern_set
            )
            return f(ok)
        else:
            return True

    return (check(pos_patterns, identity) and check(neg_patterns, invert))


def get_common_base(cls_list):
    """
    Get the most derived common base class of classes in ``cls_list``.
    """
    # MRO in which "object" will appear first
    def rev_mro(cls):
        return reversed(inspect.getmro(cls))

    def common(cls1, cls2):
        # Get the most derived class that is in common in the MRO of cls1 and
        # cls2
        for b1, b2 in itertools.takewhile(
            lambda b1_b2: b1_b2[0] is b1_b2[1],
            zip(rev_mro(cls1), rev_mro(cls2))
        ):
            pass
        return b1

    return functools.reduce(common, cls_list)


def get_subclasses(cls):
    """
    Get all the (direct and indirect) subclasses of ``cls``.
    """
    subcls_set = {cls}
    for subcls in cls.__subclasses__():
        subcls_set.update(get_subclasses(subcls))
    return subcls_set


def get_package(module):
    """
    Find the package name of a module. If the module has no package, its own
    name is taken.
    """
    # We keep the search local to the packages these modules are defined in, to
    # avoid getting a huge set of uninteresting callables.
    package = module.__package__
    if package:
        name = package.split('.', 1)[0]
    else:
        name = None
    # Standalone modules that are not inside a package will get "" as
    # package name
    return name or module.__name__


def get_recursive_module_set(module_set, package_set, visited_module_set=None):
    """
    Retrieve the set of all modules recursively imported from the modules in
    ``module_set`` if they are (indirectly) part of one of the packages named
    in ``package_set``.
    """
    visited_modules = set(visited_module_set) if visited_module_set else set()

    def select_module(module):
        # We only recurse into modules that are part of the given set
        # of packages
        return any(
            # Either a submodule of one of the packages or one of the
            # packages themselves
            get_package(module) == package
            for package in package_set
        )

    def _get_recursive_module_set(modules, module_set, package_set):
        for module in modules:
            if not isinstance(module, types.ModuleType):
                continue

            if module in visited_modules:
                continue
            else:
                visited_modules.add(module)

            if select_module(module):
                module_set.add(module)
                _get_recursive_module_set(vars(module).values(), module_set, package_set)

    recursive_module_set = set()
    _get_recursive_module_set(module_set, recursive_module_set, package_set)

    return recursive_module_set


@contextlib.contextmanager
def disable_gc():
    """
    Context manager to disable garbage collection.

    This can result in significant speed-up in code creating a lot of objects,
    like :func:`pickle.load`.
    """
    if not gc.isenabled():
        yield
        return

    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def render_graphviz(expr):
    """
    Render the structure of an expression as a graphviz description or SVG.

    :returns: A tuple(bool, content) where the boolean is ``True`` if SVG could
        be rendered or ``False`` if it still a graphviz description.

    :param expr: Expression to render
    :type expr: exekall.engine.ExpressionBase
    """
    graphviz = expr.format_structure(graphviz=True)
    with tempfile.NamedTemporaryFile('wt') as f:
        f.write(graphviz)
        f.flush()
        try:
            svg = subprocess.check_output(
                ['dot', f.name, '-Tsvg'],
                stderr=subprocess.DEVNULL,
            ).decode('utf-8')
        # If "dot" is not installed
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as e:
            debug(f'dot failed to execute: {e}')
        else:
            return (True, svg)

        return (False, graphviz)


def add_argument(parser, *args, help, **kwargs):
    """
    Equivalent to :meth:`argparse.ArgumentParser.add_argument`, with ``help``
    formatting.

    This allows using parsers setup using raw formatters.
    """
    if help is not argparse.SUPPRESS:
        help = textwrap.dedent(help)
        # Preserve all new lines where there are, and only wrap the other lines.
        help = '\n'.join(textwrap.fill(line) for line in help.splitlines())
    return parser.add_argument(*args, **kwargs, help=help)


def create_adaptor_parser_group(parser, adaptor_cls):
    description = f'{adaptor_cls.name} custom options.\nCan only be specified *after* positional parameters.'
    return parser.add_argument_group(adaptor_cls.name, description)


def powerset(iterable):
    """
    Powerset of the given iterable ::
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def measure_time(iterator):
    """
    Measure how long it took to yield each value of the given iterator.
    """
    while True:
        begin = time.monotonic()
        try:
            val = next(iterator)
        except StopIteration:
            return
        else:
            end = time.monotonic()
            yield (end - begin, val)


def capture_log(iterator):
    logger = logging.getLogger()

    def make_handler(level):
        formatter = LOGGING_FOMATTER_MAP['normal']
        string = io.StringIO()
        handler = logging.StreamHandler(string)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return (string, handler)

    def setup():
        handler_map = {
            logging.getLevelName(level): make_handler(level)
            for level in range(logging.NOTSET, logging.CRITICAL, 10)
        }
        for string, handler in handler_map.values():
            logger.addHandler(handler)
        return handler_map

    def teardown(handler_map):
        def extract(string, handler):
            logger.removeHandler(handler)
            return string.getvalue().rstrip()

        return {
            name: extract(string, handler)
            for name, (string, handler) in handler_map.items()
        }

    while True:
        handler_map = setup()
        utc = utc_datetime()
        try:
            val = next(iterator)
        except StopIteration:
            return
        else:
            log_map = teardown(handler_map)
            yield (utc, log_map, val)


class OrderedSetBase:
    """
    Base class for custom ordered sets.
    """
    def __init__(self, items=[]):
        # Make sure to iterate over items only once, in case it's a generator
        self._list = list(items)
        self._set = set(self._list)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._set == other._set
        elif isinstance(other, collections.abc.Sequence):
            self._set == set(other)
        else:
            return False

    def __contains__(self, item):
        return item in self._set

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)


class FrozenOrderedSet(OrderedSetBase, collections.abc.Set):
    """
    Like a regular ``frozenset``, but iterating over it will yield items in insertion
    order.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hash = functools.reduce(
            lambda hash_, item: hash_ ^ hash(item),
            self._list,
            0
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if hash(self) != hash(other):
                return False

        return super().__eq__(other)


class OrderedSet(OrderedSetBase, collections.abc.MutableSet):
    """
    Like a regular ``set``, but iterating over it will yield items in insertion
    order.
    """
    def add(self, item):
        if item in self._set:
            return
        else:
            self._set.add(item)
            self._list.append(item)

    def update(self, *sets):
        for s in sets:
            for x in s:
                self.add(x)

    def discard(self, item):
        self._set.discard(item)
        with contextlib.suppress(ValueError):
            self._list.remove(item)


def utc_datetime():
    """
    Return a UTC :class:`datetime.datetime`.
    """
    return datetime.datetime.now(datetime.timezone.utc)
