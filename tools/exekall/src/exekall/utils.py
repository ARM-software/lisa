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

# This module is allowed to import engine, as engine does not import it (so
# there is no circular dependency). Most utils should go in _utils unless they
# really need to depend on the engine, without being part of it.

import inspect
import typing
import warnings

import exekall.engine as engine

# Re-export all _utils here
from exekall._utils import *


def get_callable_set(module_set, verbose=False):
    """
    Get the set of callables defined in all modules of ``module_set``.

    We ignore any callable that is defined outside of the modules' package.

    :param module_set: Set of modules to scan.
    :type module_set: set(types.ModuleType)
    """

    package_set = set(map(get_package, module_set))
    callable_set = set()
    visited_obj_set = set()
    visited_module_set = set()

    def get(module_set):
        new_module_set = set()
        callable_set = set()
        for module in get_recursive_module_set(module_set, package_set, visited_module_set):
            visited_module_set.add(module)

            callable_set_ = _get_callable_set(
                namespace=module,
                module=module,
                visited_obj_set=visited_obj_set,
                verbose=verbose,
                package_set=package_set,
            )

            def is_class_attr(obj):
                try:
                    name = obj.__qualname__
                except AttributeError:
                    return False
                else:
                    return '.' in name and '<locals>' not in name


            class_attr = {
                callable_
                for callable_ in callable_set_
                if is_class_attr(callable_)
            }

            # We might have included callables that are defined in another
            # module via inherited methods, so recurse in these modules too
            new_module_set.update(
                map(inspect.getmodule, class_attr)
            )
            callable_set.update(callable_set_)

        new_module_set.discard(None)
        new_module_set -= visited_module_set
        return (callable_set, new_module_set)

    while module_set:
        callable_set_, module_set = get(module_set)
        callable_set.update(callable_set_)

    return callable_set

def _get_members(*args, **kwargs):
    """
    Same as :func:`inspect.getmembers` except that it will silence warnings,
    which avoids triggering deprecation warnings just because we are scanning a
    class
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        return inspect.getmembers(*args, **kwargs)

def _get_callable_set(namespace, module, package_set, visited_obj_set, verbose):
    """
    :param namespace: Module or class
    :param module: Module the namespace was defined in, or ``None`` to be ignored.
    """
    log_f = info if verbose else debug
    callable_pool = set()

    if id(namespace) in visited_obj_set:
        return callable_pool
    else:
        visited_obj_set.add(id(namespace))

    attributes = [
        callable_
        for name, callable_ in _get_members(
            namespace,
            predicate=callable
        )
    ]

    if isinstance(namespace, type):
        attributes.append(namespace)

    def select(attr):
        _module = inspect.getmodule(attr)
        return (
            # Module of builtins is None
            _module is None or
            # skip internal classes that may end up being exposed as a global
            _module is not engine and
            (True if module is None else _module is module) and
            get_package(_module) in package_set
        )

    visited_obj_set.update(attributes)
    attributes = [
        attr
        for attr in attributes
        if id(attr) not in visited_obj_set and select(attr)
    ]

    for callable_ in attributes:
        # Explore the class attributes as well for nested types
        if (
            isinstance(callable_, type) and
            # Do not recurse into the "_type" class attribute of namedtuple, as
            # it has broken globals Python 3.6. The crazy checks are a
            # workaround the fact that there is no direct way to detect if a
            # class is a namedtuple.
            not (
                isinstance(namespace, type) and
                callable_.__name__ == '_type' and
                issubclass(callable_, tuple) and
                hasattr(callable_, '_make') and
                hasattr(callable_, '_asdict') and
                hasattr(callable_, '_replace')
            )
        ):
            callable_pool.update(
                _get_callable_set(
                    namespace=callable_,
                    visited_obj_set=visited_obj_set,
                    verbose=verbose,
                    # We want to get all the attributes in classes, regardless
                    # on what module owns them. For example, we want to select
                    # a method inherited from a base class even if that base
                    # class and the method definition lives somewhere else.
                    module=None,
                    package_set=package_set,
                )
            )

        # Functions defined in a class are methods, and have to be wrapped so
        # engine.Operator() can correctly resolve annotations using the class
        # attributes as a context in which the function was defined.
        #
        # Note that we also wrap things like classmethod, as they have the same
        # needs
        if (
            isinstance(namespace, type) and
            isinstance(
                callable_,
                (
                    # Instance and static methods
                    types.FunctionType,
                    # Class methods
                    types.MethodType,
                )
            )
        ):
            callable_ = engine.UnboundMethod(callable_, namespace)

        try:
            op = engine.Operator(callable_)
            # Trigger exceptions if they have to be raised
            op.prototype
        # If the callable is partially annotated, warn about it since it is
        # likely to be a mistake.
        except engine.PartialAnnotationError as e:
            log_f('Partially-annotated callable "{callable}" will not be used: {e}'.format(
                callable=get_name(callable_),
                e=e,
            ))
            continue
        # If some annotations fail to resolve
        except NameError as e:
            log_f('callable "{callable}" with unresolvable annotations will not be used: {e}'.format(
                callable=get_name(callable_),
                e=e,
            ))
            continue
        # If something goes wrong, that means it is not properly annotated
        # so we just ignore it
        except (SyntaxError, AttributeError, ValueError, KeyError, engine.AnnotationError):
            continue

        def has_typevar(op):
            return any(
                isinstance(x, typing.TypeVar)
                for x in {op.value_type, *op.prototype[0].values()}
            )

        def check_typevar_name(cls, name, var):
            if name != var.__name__:
                log_f('__name__ of {cls}.{var.__name__} typing.TypeVar differs from the name it is bound to "{name}", which will prevent using it for polymorphic parameters or return annotation'.format(
                    cls=get_name(cls, full_qual=True),
                    var=var,
                    name=name,
                ))
            return name

        type_vars = sorted(
            check_typevar_name(op.value_type, name, attr)
            for name, attr in _get_members(
                op.value_type,
                lambda x: isinstance(x, typing.TypeVar)
            )
        )

        # Also make sure we don't accidentally get callables that will
        # return a abstract base class instance, since that would not work
        # anyway.
        if inspect.isabstract(op.value_type):
            log_f('Instances of {} will not be created since it has non-implemented abstract methods'.format(
                get_name(op.value_type, full_qual=True)
            ))
        elif type_vars:
            log_f('Instances of {} will not be created since it has non-overridden TypeVar class attributes: {}'.format(
                get_name(op.value_type, full_qual=True),
                ', '.join(type_vars)
            ))
        elif has_typevar(op):
            log_f('callable "{callable}" with non-resolved associated TypeVar annotations will not be used'.format(
                callable=get_name(callable_),
            ))
        else:
            callable_pool.add(callable_)

    return callable_pool


def sweep_param(callable_, param, start, stop, step=1):
    """
    Used to generate a stream of numbers or strings to feed to a callable.

    :param callable_: Callable the numbers will be used by.
    :type callable_: collections.abc.Callable

    :param param: Name of the parameter of the callable the numbers will be
        providing values for.
    :type param: str

    :param start: Starting value.
    :type start: str

    :param stop: End value (inclusive)
    :type stop: str

    :param step: Increment step.
    :type step: str

    If ``start == stop``, only that value will be yielded, and it can be of any
    type.

    The type used will either be one that is annotated on the callable, or the
    one from the default value if no annotation is available, or float if no
    default value is found. The value will then be built by passing the string
    to the type as only parameter.
    """

    op = engine.Operator(callable_)
    annot = op.prototype[0]
    try:
        type_ = annot[param]
    except KeyError:
        sig = op.signature
        default = sig.parameters[param].default
        if default is not inspect.Parameter.default and default is not None:
            type_ = type(default)
        else:
            type_ = str

    if start == stop:
        yield type_(start)
    else:
        i = type_(start)
        step = type_(step)
        stop = type_(stop)
        while i <= stop:
            yield type_(i)
            i += step


def get_method_class(function):
    """
    Get the class in which a ``function`` is defined.
    """
    # Unbound instance methods
    if isinstance(function, engine.UnboundMethod):
        return function.cls

    try:
        obj = function.__self__
    except AttributeError:
        cls_name = function.__qualname__.rsplit('.', 1)[0]
        if '<locals>' in cls_name:
            return None
        else:
            return eval(cls_name, function.__globals__)
    else:
        # bound class methods
        if isinstance(obj, type):
            return obj
        # bound instance method
        else:
            return type(obj)
