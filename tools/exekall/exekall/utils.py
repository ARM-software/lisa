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
import exekall.engine as engine

# Re-export all _utils here
from exekall._utils import *

def get_callable_set(module_set, verbose=False):
    """
    Get the set of callables defined in all modules of ``module_set``.

    :param module_set: Set of modules to scan.
    :type module_set: set(types.ModuleType)
    """
    # We keep the search local to the packages these modules are defined in, to
    # avoid getting a huge set of uninteresting callables.
    package_set = {
        module.__package__.split('.', 1)[0] for module in module_set
    }
    callable_set = set()
    visited_obj_set = set()
    for module in get_recursive_module_set(module_set, package_set):
        callable_set.update(_get_callable_set(
            module,
            visited_obj_set,
            verbose=verbose,
        ))

    return callable_set

def _get_callable_set(module, visited_obj_set, verbose):
    log_f = info if verbose else debug
    callable_pool = set()

    for name, obj in vars(module).items():
        # skip internal classes that may end up being exposed as a global
        if inspect.getmodule(obj) is engine:
            continue

        if id(obj) in visited_obj_set:
            continue
        else:
            visited_obj_set.add(id(obj))

        # If it is a class, get the list of methods
        if isinstance(obj, type):
            callable_list = [
                callable_
                for name, callable_
                in inspect.getmembers(obj, predicate=callable)
            ]
            callable_list.append(obj)
        else:
            callable_list = [obj]

        callable_list = [c for c in callable_list if callable(c)]

        for callable_ in callable_list:
            try:
                op = engine.Operator(callable_)
                param_list, return_type = op.get_prototype()
            # If the callable is partially annotated, warn about it since it is
            # likely to be a mistake.
            except engine.PartialAnnotationError as e:
                log_f('Partially-annotated callable will not be used: {e}'.format(
                    callable=get_name(callable_),
                    e=e,
                ))
                continue
            # If some annotations fail to resolve
            except NameError as e:
                log_f('callable with unresolvable annotations will not be used: {e}'.format(
                    callable=get_name(callable_),
                    e=e,
                ))
                continue
            # If something goes wrong, that means it is not properly annotated
            # so we just ignore it
            except (AttributeError, ValueError, KeyError, engine.AnnotationError):
                continue

            # Swap-in a wrapper object, so we keep track on the class on which
            # the function was looked up
            if op.is_method:
                callable_ = engine.UnboundMethod(callable_, obj)

            # Also make sure we don't accidentally get callables that will
            # return a abstract base class instance, since that would not work
            # anyway.
            if inspect.isabstract(return_type):
                log_f('Instances of {} will not be created since it has non-implemented abstract methods'.format(
                    get_name(return_type, full_qual=True)
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
    annot = op.get_prototype()[0]
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
