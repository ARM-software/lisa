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

import inspect
import collections
import numbers
import importlib
from importlib import util
import itertools
import sys
import pathlib
import logging
import functools
import fnmatch
import contextlib
import types
import traceback
import logging

import exekall.engine as engine

def flatten_nested_seq(seq):
    return list(itertools.chain.from_iterable(seq))

def load_serial_from_db(db, uuid_seq=None, type_pattern_seq=None):

    def uuid_predicate(serial):
        return (
            serial.value_uuid in uuid_seq
            or serial.excep_uuid in uuid_seq
        )

    def type_pattern_predicate(serial):
        return any(
            match_base_cls(type(serial.value), pattern)
            for pattern in type_pattern_seq
        )

    if type_pattern_seq and not uuid_seq:
        predicate = type_pattern_predicate

    elif uuid_seq and not type_pattern_seq:
        predicate = uuid_predicate

    elif not uuid_seq and not type_pattern_seq:
        predicate = lambda serial: True

    else:
        def predicate(serial):
            return uuid_predicate(serial) and type_pattern_predicate(serial)

    return db.obj_store.get_by_predicate(predicate)

def match_base_cls(cls, pattern):
    # Match on the name of the class of the object and all its base classes
    for base_cls in inspect.getmro(cls):
        base_cls_name = engine.get_name(base_cls, full_qual=True)
        if fnmatch.fnmatch(base_cls_name, pattern):
            return True

    return False

def get_recursive_module_set(module_set, package_set):
    """Retrieve the set of all modules recurisvely imported from the modules in
    `module_set`, if they are (indirectly) part of one of the packages named in
    `package_set`.
    """

    recursive_module_set = set()
    for module in module_set:
        _get_recursive_module_set(module, recursive_module_set, package_set)

    return recursive_module_set

def _get_recursive_module_set(module, module_set, package_set):
    if module in module_set:
        return
    module_set.add(module)
    for imported_module in vars(module).values():
        if (
            isinstance(imported_module, types.ModuleType)
            # We only recurse into modules that are part of the given set
            # of packages
            and any(
                # Either a submodule of one of the packages or one of the
                # packages themselves
                imported_module.__name__.split('.', 1)[0] == package
                for package in package_set
            )
        ):
            _get_recursive_module_set(imported_module, module_set, package_set)

def get_callable_set(module_set):
    # We keep the search local to the packages these modules are defined in, to
    # avoid getting a huge set of uninteresting callables.
    package_set = {
        module.__package__.split('.', 1)[0] for module in module_set
    }
    callable_set = set()
    for module in get_recursive_module_set(module_set, package_set):
        callable_set.update(_get_callable_set(module))

    return callable_set

def _get_callable_set(module):
    callable_pool = set()
    for name, obj in vars(module).items():
        # skip internal classes that may end up being exposed as a global
        if inspect.getmodule(obj) is engine:
            continue

        # If it is a class, get the list of methods
        if isinstance(obj, type):
            callable_list = list(dict(inspect.getmembers(obj, predicate=callable)).values())
            callable_list.append(obj)
        else:
            callable_list = [obj]

        callable_list = [c for c in callable_list if callable(c)]

        for callable_ in callable_list:
            try:
                param_list, return_type = engine.Operator(callable_).get_prototype()
            # If something goes wrong, that means it is not properly annotated
            # so we just ignore it
            except (AttributeError, ValueError, KeyError, engine.AnnotationError):
                continue

            # Also make sure we don't accidentally get callables that will
            # return a abstract base class instance, since that would not work
            # anyway.
            if not inspect.isabstract(return_type):
                callable_pool.add(callable_)

    return callable_pool

def import_file(python_src, module_name=None, is_package=False):
    python_src = pathlib.Path(python_src)
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
                is_package=True
            )
        else:
            spec = importlib.util.spec_from_file_location(module_name, python_src,
                submodule_search_locations=submodule_search_locations)
            if spec is None:
                raise ValueError('Could not find module "{module}" at {path}'.format(
                    module=module_name,
                    path=python_src
                ))

        module = importlib.util.module_from_spec(spec)
        # Register module before executing it so relative imports will work
        sys.modules[module_name] = module
        # Nothing to execute in a namespace package
        if not is_namespace_package:
            spec.loader.exec_module(module)
    #  Python <= v3.4 style
    else:
        module = importlib.machinery.SourceFileLoader(
                module_name, str(python_src)).load_module()

    sys.modules[module_name] = module
    importlib.invalidate_caches()
    return module

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
                module_name = '.'.join(package_name.parts),
                is_package = True,
            )

        module_name = '.'.join((
            ('.'.join(module_parents[0].parts)),
            module_basename
        ))
    else:
        module_name = get_module_basename(python_src)

    return module_name

def get_module_basename(path):
    path = pathlib.Path(path)
    module_name = inspect.getmodulename(str(path))
    # This is either garbage or a package
    if module_name is None:
        module_name = path.name
    return module_name

class TaggedNum:
    def __init__(self, *args, **kwargs):
        self.tags = [str(a) for a in args]

class Int(int, TaggedNum):
    pass

class Float(float, TaggedNum):
    pass

def sweep_number(
    callable_, param,
    start, stop=None, step=1):

    annot = engine.Operator(callable_).get_prototype()[0]
    try:
        type_ = annot[param]
    except KeyError:
        type_ = type(start)

    if stop is None:
        stop = start
        start = 0

    # Swap-in the tagged type if possible
    if issubclass(type_, numbers.Integral):
        type_ = Int
    # Must come in 2nd place, since int is a subclass of numbers.Real
    elif issubclass(type_, numbers.Real):
        type_ = Float

    i = type_(start)
    step = type_(step)
    while i <= stop:
        yield type_(i)
        i += step

def _make_tagged_type(name, qualname, mod_name, bases):
    class new_type(*bases):
        def __init__(self, *args, **kwargs):
            self.tags = (
                [str(a) for a in args] +
                [
                    k+'='+str(a)
                    for k, a in kwargs.items()
                ]
            )
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                pass

    new_type.__name__ = name
    new_type.__qualname__ = qualname
    new_type.__module__ = mod_name
    return new_type

def unique_type(*param_list):
    def decorator(f):
        annot = engine.get_type_hints(f)
        for param in param_list:
            type_ = annot[param]
            f_name = engine.get_name(f, full_qual=False)

            new_type_name = '{f}_{name}'.format(
                f = f_name.replace('.', '_'),
                type_name = type_.__name__,
                name = param
            )
            new_type = _make_tagged_type(
                new_type_name, new_type_name, f.__module__,
                (type_,)
            )
            f.__globals__[new_type_name] = new_type
            f.__annotations__[param] = new_type
        return f

    return decorator

# Call the given function at most once per set of parameters
once = functools.lru_cache()

def iterate_cb(iterator, pre_hook=None, post_hook=None):
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
    elements = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(elements)

# Logging level above CRITICAL that is always displayed and used for output
LOGGING_OUT_LEVEL = 60

class ExekallFormatter(logging.Formatter):
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

def setup_logging(log_level, verbose=False):
    logging.addLevelName(LOGGING_OUT_LEVEL, 'OUT')
    level=getattr(logging, log_level.upper())
    if verbose:
        fmt = '[%(name)s/%(filename)s:%(lineno)s][%(asctime)s] %(levelname)s  %(message)s'
    else:
        fmt = '[%(name)s][%(asctime)s] %(levelname)s  %(message)s'
    formatter = ExekallFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

EXEKALL_LOGGER  = logging.getLogger('EXEKALL')

def out(msg):
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


