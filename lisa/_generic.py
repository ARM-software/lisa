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
import inspect
import typing
from typing import Any, Union, Generic, TypeVar
import typeguard
from collections.abc import Iterable

from lisa.utils import get_obj_name, _is_typing_hint

class _TypeguardCustom:
    _HINT = Any

    @classmethod
    def _instancecheck(cls, value):
        return

    @classmethod
    def _typeguard_checker(cls, value, origin_type, args, memo):
        typeguard.check_type_internal(value, cls._HINT, memo)

        try:
            cls._instancecheck(value)
        except TypeError as e:
            raise typeguard.TypeCheckError(str(e))


def _typeguard_lookup(origin_type, args, extras):
    try:
        issub =  issubclass(origin_type, _TypeguardCustom)
    except Exception:
        issub = False

    if issub:
        return origin_type._typeguard_checker
    else:
        return None

typeguard.checker_lookup_functions.append(_typeguard_lookup)


def check_type(x, classinfo):
    """
    Equivalent of ``isinstance()`` that will also work with typing hints.
    """
    if isinstance(classinfo, Iterable):
        typ = Union[tuple(classinfo)]
    else:
        typ = classinfo

    try:
        typeguard.check_type(
            value=x,
            expected_type=typ,
            forward_ref_policy=typeguard.ForwardRefPolicy.ERROR,
            collection_check_strategy=typeguard.CollectionCheckStrategy.ALL_ITEMS,
        )
    except typeguard.TypeCheckError as e:
        raise TypeError(str(e))


def is_instance(obj, classinfo):
    """
    Same as builtin ``isinstance()`` but works with type hints.
    """
    try:
        check_type(obj, classinfo)
    except TypeError:
        return False
    else:
        return True


def is_hint(obj):
    if isinstance(obj, type) and issubclass(obj, _TypeguardCustom):
        return True
    else:
        return _is_typing_hint(obj)


@functools.lru_cache(maxsize=None, typed=True)
def hint_to_class(hint):
    """
    Convert a typing hint to a class that will do a runtime check against the
    hint when ``isinstance()`` is used.
    """
    class Meta(type):
        def __instancecheck__(cls, instance):
            return is_instance(instance, hint)

    class Stub(metaclass=Meta):
        pass

    name = get_obj_name(hint).split('.', 1)
    try:
        name = name[1]
    except IndexError:
        name = name[0]

    Stub.__qualname__ = name
    Stub.__name__ = name.split('.')[-1]

    return Stub


T = TypeVar('T')
class SortedSequence(Generic[T], _TypeguardCustom):
    """
    Same as :class:`typing.List` but enforces sorted values when runtime
    checked using :mod:`typeguard`.
    """
    _HINT = typing.Sequence[T]

    @classmethod
    def _instancecheck(cls, value):
        for i, (x, y) in enumerate(zip(value, value[1:])):
            if x > y:
                raise TypeError(f'Item #{i} "{x}" is higher than the next item "{y}", but the list must be sorted')
