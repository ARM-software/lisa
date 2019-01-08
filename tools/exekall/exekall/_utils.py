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

import types
import uuid
import inspect
import functools
import fnmatch
import gc
import collections
import contextlib
import importlib
import io
import itertools
import logging
import pathlib
import pickle
import sys
import traceback

class NotSerializableError(Exception):
    pass

def get_class_from_name(cls_name, module_map):
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
    cls_name = cls_name[len(mod_name)+1:]
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
    return uuid.uuid4().hex

def get_mro(cls):
    if cls is type(None) or cls is None:
        return (type(None), object)
    else:
        assert isinstance(cls, type)
        return inspect.getmro(cls)

def get_name(obj, full_qual=True, qual=True):
    # full_qual enabled implies qual enabled
    _qual = qual or full_qual
    # qual disabled implies full_qual disabled
    full_qual = full_qual and qual
    qual = _qual

    # Add the module's name in front of the name to get a fully
    # qualified name
    if full_qual:
        module_name = obj.__module__
        module_name = (
            module_name + '.'
            if module_name != '__main__' and module_name != 'builtins'
            else ''
        )
    else:
        module_name = ''

    if qual:
        _get_name = lambda x: x.__qualname__
    else:
        _get_name = lambda x: x.__name__

    # Classmethods appear as bound method of classes. Since each subclass will
    # get a different bound method object, we want to reflect that in the
    # name we use, instead of always using the same name that the method got
    # when it was defined
    if inspect.ismethod(obj):
        name = _get_name(obj.__self__) + '.' + obj.__name__
    else:
        name = _get_name(obj)

    return module_name + name

def get_src_loc(obj):
    try:
        src_line = inspect.getsourcelines(obj)[1]
        src_file = inspect.getsourcefile(obj)
        src_file = str(pathlib.Path(src_file).resolve())
    except (OSError, TypeError):
        src_line, src_file = None, None

    return (src_file, src_line)

def is_serializable(obj, raise_excep=False):
    """
    Try to Pickle the object to see if that raises any exception.
    """
    stream = io.StringIO()
    try:
        # This may be slow for big objects but it is the only way to be sure
        # it can actually be serialized
        pickle.dumps(obj)
    except (TypeError, pickle.PickleError) as e:
        debug('Cannot serialize instance of {}: {}'.format(
            type(obj).__qualname__, str(e)
        ))
        if raise_excep:
            raise NotSerializableError(obj) from e
        return False
    else:
        return True

# Call the given function at most once per set of parameters
def once(callable_):
    return functools.lru_cache(maxsize=None, typed=True)(callable_)

def remove_indices(iterable, ignored_indices):
    return [v for i, v in enumerate(iterable) if i not in ignored_indices]

# Basic reimplementation of typing.get_type_hints for Python versions that
# do not have a typing module available, and also avoids creating Optional[]
# when the parameter has a None default value.
def resolve_annotations(annotations, module_vars):
    return {
        # If we get a string, evaluate it in the global namespace of the
        # module in which the callable was defined
        param: cls if not isinstance(cls, str) else eval(cls, module_vars)
        for param, cls in annotations.items()
    }

def get_module_basename(path):
    path = pathlib.Path(path)
    module_name = inspect.getmodulename(str(path))
    # This is either garbage or a package
    if module_name is None:
        module_name = path.name
    return module_name

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

def setup_logging(log_level, debug_log_file=None, info_log_file=None, verbose=0):
    logging.addLevelName(LOGGING_OUT_LEVEL, 'OUT')
    level=getattr(logging, log_level.upper())

    verbose_formatter = ExekallFormatter('[%(name)s/%(filename)s:%(lineno)s][%(asctime)s] %(levelname)s  %(message)s')
    normal_formatter = ExekallFormatter('[%(name)s][%(asctime)s] %(levelname)s  %(message)s')

    logger = logging.getLogger()
    # We do not filter anything at the logger level, only at the handler level
    logger.setLevel(logging.NOTSET)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = verbose_formatter if verbose else normal_formatter
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if debug_log_file:
        file_handler = logging.FileHandler(str(debug_log_file), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(verbose_formatter)
        logger.addHandler(file_handler)

    if info_log_file:
        file_handler = logging.FileHandler(str(info_log_file), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(normal_formatter)
        logger.addHandler(file_handler)

    # Redirect all warnings of the "warnings" module as log entries
    logging.captureWarnings(True)

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

        module_dotted_path = list(module_parents[0].parts) + [module_basename]
        module_name = '.'.join(module_dotted_path)

    else:
        module_name = get_module_basename(python_src)

    return module_name

def find_customization_module_set(module_set):
    def build_full_names(l_l):
        """Explode list of lists, and build full package names."""
        for l in l_l:
            for i, _ in enumerate(l):
                i += 1
                yield '.'.join(l[:i])

    try:
        import_excep = ModuleNotFoundError
    # Python < 3.6
    except NameError:
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
        with contextlib.suppress(import_excep):
            module_exists = importlib.util.find_spec(customize_name)

        if module_exists:
            # Importing that module is enough to make the adaptor visible
            # to the Adaptor base class
            customize_module = importlib.import_module(customize_name)
            customization_module_set.add(customize_module)

    return customization_module_set

def import_file(python_src, module_name=None, is_package=False):
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
                is_package=True
            )
        else:
            spec = importlib.util.spec_from_file_location(module_name, str(python_src),
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

def flatten_seq(seq, levels=1):
    if levels == 0:
        return seq
    else:
        seq = list(itertools.chain.from_iterable(seq))
        return flatten_seq(seq, levels=levels - 1)

def get_froz_val_seq_set(db, uuid_seq=None, type_pattern_seq=None):

    def uuid_predicate(froz_val):
        return (
            froz_val.value_uuid in uuid_seq
            or froz_val.excep_uuid in uuid_seq
        )

    def type_pattern_predicate(froz_val):
        return match_base_cls(type(froz_val.value), type_pattern_seq)

    if type_pattern_seq and not uuid_seq:
        predicate = type_pattern_predicate

    elif uuid_seq and not type_pattern_seq:
        predicate = uuid_predicate

    elif not uuid_seq and not type_pattern_seq:
        predicate = lambda froz_val: True

    else:
        def predicate(froz_val):
            return uuid_predicate(froz_val) and type_pattern_predicate(froz_val)

    return db.get_by_predicate(predicate, flatten=False, deduplicate=True)

def match_base_cls(cls, pattern_list):
    # Match on the name of the class of the object and all its base classes
    for base_cls in get_mro(cls):
        base_cls_name = get_name(base_cls, full_qual=True)
        if not base_cls_name:
            continue
        if any(
                fnmatch.fnmatch(base_cls_name, pattern)
                for pattern in pattern_list
            ):
            return True

    return False

def match_name(name, pattern_list):
    if name is None:
        return False
    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in pattern_list
    )

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


@contextlib.contextmanager
def disable_gc():
    """
    Context manager to disable garbage collection.

    This can result in significant speed-up in code creating a lot of objects,
    like ``pickle.load()``.
    """
    if not gc.isenabled():
        yield
        return

    gc.disable()
    try:
        yield
    finally:
        gc.enable()

