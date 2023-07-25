# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020, Arm Limited and contributors.
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

"""
Generic types inspired by the :mod:`typing` module.
"""

import functools
from collections.abc import Mapping, Sequence
from operator import attrgetter

from lisa.utils import sphinx_register_nitpick_ignore

def _isinstance(x, type_):
    if isinstance(type_, tuple):
        return any(map(lambda type_: _isinstance(x, type_), type_))
    elif isinstance(type_, type):
        return isinstance(x, type_)
    # Typing hint
    else:
        try:
            from typing import get_origin, get_args, Union
        except ImportError:
            # We cannot process the typing hint in that version of Python, so
            # we assume the input is correctly typed. It's not ideal but we
            # cannot do much more than that.
            return True
        else:
            combinator = get_origin(type_)
            args = get_args(type_)
            if combinator == Union:
                return any(map(lambda type_: _isinstance(x, type_), args))
            else:
                raise TypeError(f'Cannot handle type hint: {type_}')

class MetaBase(type):
    def __instancecheck__(cls, instance):
        try:
            cls.instancecheck(instance)
        except TypeError:
            return False
        else:
            return True

    # Fully memoize the function so that this always holds:
    # assert Container[Foo] is Container[Foo]
    @functools.lru_cache(maxsize=None, typed=True)
    def __getitem__(cls, type_):
        NewClass = cls.getitem(type_)

        NewClass.__module__ = cls.__module__

        # Since this type name is not resolvable, avoid cross reference
        # warnings from Sphinx
        sphinx_register_nitpick_ignore(NewClass)
        return NewClass


class GenericContainerMetaBase(MetaBase):
    """
    Base class for the metaclass of generic containers.

    They are parameterized with the ``type_`` class attribute, and classes can
    also be created by indexing on classes with :class:`GenericBase`
    metaclass. The ``type_`` class attribute will be set with what is passed as
    the key.
    """

    # Fully memoize the function so that this always holds:
    # assert Container[Foo] is Container[Foo]
    def getitem(cls, type_):
        class NewClass(cls):
            _type = type_

        types = type_ if isinstance(type_, Sequence) else [type_]

        def make_name(self_getter, sub_getter):
            def _sub_getter(type_):
                try:
                    return sub_getter(type_)
                # type hints like typing.Union don't have a name we can introspect,
                # but it can be pretty-printed
                except AttributeError:
                    return str(type_)
            return '{}[{}]'.format(
                self_getter(cls),
                ','.join(_sub_getter(type_) for type_ in types)
            )

        NewClass.__name__ = make_name(
            attrgetter('__name__'),
            attrgetter('__name__')
        )

        def type_param_name(t):
            if t.__module__ == 'builtins':
                return t.__qualname__
            else:
                # Add the module name so that Sphinx can establish cross
                # references
                return f'{t.__module__}.{t.__qualname__}'

        NewClass.__qualname__ = make_name(
            attrgetter('__qualname__'),
            type_param_name,
        )
        return NewClass


class GenericBase:
    """
    Base class for generic containers.
    """
    def __new__(cls, obj):
        cls.instancecheck(obj)
        return obj


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
            if not _isinstance(k, k_type):
                raise TypeError(f'Key "{k}" of type {type(k).__qualname__} should be of type {k_type.__qualname__}', k)

            if not _isinstance(v, v_type):
                raise TypeError(f'Value of {type(v).__qualname__} key "{k}" should be of type {v_type.__qualname__}', k)


class TypedDict(GenericBase, dict, metaclass=GenericMappingMeta):
    """
    Subclass of dict providing keys and values type check.
    """


class GenericSequenceMeta(GenericContainerMetaBase, type(Sequence)):
    """Similar to :class:`GenericMappingMeta` for sequences"""
    def instancecheck(cls, instance):
        if not isinstance(instance, Sequence):
            raise TypeError('not a Sequence')

        type_ = cls._type
        for i, x in enumerate(instance):
            if not _isinstance(x, type_):
                raise TypeError(f'Item #{i} "{x}" of type {type(x).__qualname__} should be of type {type_.__qualname__}', i)

class GenericSortedSequenceMeta(GenericSequenceMeta):
    def instancecheck(cls, instance):
        super().instancecheck(instance)
        for i, (x, y) in enumerate(zip(instance, instance[1:])):
            if x > y:
                raise TypeError(f'Item #{i} "{x}" is higher than the next item "{y}", but the list must be sorted')


class TypedList(GenericBase, list, metaclass=GenericSequenceMeta):
    """
    Subclass of list providing keys and values type check.
    """


class SortedTypedList(GenericBase, list, metaclass=GenericSortedSequenceMeta):
    """
    Subclass of list providing keys and values type check, and also check the
    list is sorted in ascending order.
    """


class OneOfMeta(MetaBase):
    def getitem(cls, allowed):
        class NewClass(cls):
            _allowed = allowed

        NewClass.__qualname__ = f'{cls.__qualname__}[{", ".join(map(repr, allowed))}]'
        return NewClass

    def instancecheck(cls, instance):
        allowed = cls._allowed
        if instance not in allowed:
            raise ValueError(f'Value {repr(instance)} is not allowed. It must be one of: {", ".join(map(repr, allowed))}')


class OneOf(GenericBase, metaclass=OneOfMeta):
    """
    Check that the provided value is part of a specific set of allowed values.
    """
    def __new__(cls, obj):
        cls.instancecheck(obj)
        return obj
