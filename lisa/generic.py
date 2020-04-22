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

class GenericContainerMetaBase(type):
    """
    Base class for the metaclass of generic containers.

    They are parameterized with the ``type_`` class attribute, and classes can
    also be created by indexing on classes with :class:`GenericContainerBase`
    metaclass. The ``type_`` class attribute will be set with what is passed as
    the key.
    """
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
    def __getitem__(self, type_):
        class NewClass(self):
            _type = type_

        types = type_ if isinstance(type_, Sequence) else [type_]

        def make_name(self_getter, sub_getter):
            return '{}[{}]'.format(
                self_getter(self),
                ','.join(sub_getter(type_) for type_ in types)
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
                return '{}.{}'.format(t.__module__, t.__qualname__)

        NewClass.__qualname__ = make_name(
            attrgetter('__qualname__'),
            type_param_name,
        )
        NewClass.__module__ = self.__module__

        # Since this type name is not resolvable, avoid cross reference
        # warnings from Sphinx
        sphinx_register_nitpick_ignore(NewClass)

        return NewClass


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

class GenericSortedSequenceMeta(GenericSequenceMeta):
    def instancecheck(cls, instance):
        super().instancecheck(instance)
        for i, (x, y) in enumerate(zip(instance, instance[1:])):
            if x > y:
                raise TypeError('Item #{i} "{x}" is higher than the next item "{y}", but the list must be sorted'.format(
                    i=i,
                    x=x,
                    y=y
                ))


class TypedList(GenericContainerBase, list, metaclass=GenericSequenceMeta):
    """
    Subclass of list providing keys and values type check.
    """
    pass


class SortedTypedList(GenericContainerBase, list, metaclass=GenericSortedSequenceMeta):
    """
    Subclass of list providing keys and values type check, and also check the
    list is sorted in ascending order.
    """
    pass
