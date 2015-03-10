#    Copyright 2014-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""
Routines for doing various type conversions. These usually embody some higher-level
semantics than are present in standard Python types (e.g. ``boolean`` will convert the
string ``"false"`` to ``False``, where as non-empty strings are usually considered to be
``True``).

A lot of these are intened to stpecify type conversions declaratively in place like
``Parameter``'s ``kind`` argument. These are basically "hacks" around the fact that Python
is not the best language to use for configuration.

"""
import re
import math
from collections import defaultdict

from wlauto.utils.misc import isiterable, to_identifier


def identifier(text):
    """Converts text to a valid Python identifier by replacing all
    whitespace and punctuation."""
    return to_identifier(text)


def boolean(value):
    """
    Returns bool represented by the value. This is different from
    calling the builtin bool() in that it will interpret string representations.
    e.g. boolean('0') and boolean('false') will both yield False.

    """
    false_strings = ['', '0', 'n', 'no']
    if isinstance(value, basestring):
        value = value.lower()
        if value in false_strings or 'false'.startswith(value):
            return False
    return bool(value)


def numeric(value):
    """
    Returns the value as number (int if possible, or float otherwise), or
    raises ``ValueError`` if the specified ``value`` does not have a straight
    forward numeric conversion.

    """
    if isinstance(value, int):
        return value
    try:
        fvalue = float(value)
    except ValueError:
        raise ValueError('Not numeric: {}'.format(value))
    if not math.isnan(fvalue) and not math.isinf(fvalue):
        ivalue = int(fvalue)
        if ivalue == fvalue:  # yeah, yeah, I know. Whatever. This is best-effort.
            return ivalue
    return fvalue


def list_or_string(value):
    """
    If the value is a string, at will be kept as a string, otherwise it will be interpreted
    as a list. If that is not possible, it will be interpreted as a string.

    """
    if isinstance(value, basestring):
        return value
    else:
        try:
            return list(value)
        except ValueError:
            return str(value)


def list_of_strs(value):
    """
    Value must be iterable. All elements will be converted to strings.

    """
    if not isiterable(value):
        raise ValueError(value)
    return map(str, value)

list_of_strings = list_of_strs


def list_of_ints(value):
    """
    Value must be iterable. All elements will be converted to ``int``\ s.

    """
    if not isiterable(value):
        raise ValueError(value)
    return map(int, value)

list_of_integers = list_of_ints


def list_of_numbers(value):
    """
    Value must be iterable. All elements will be converted to numbers (either ``ints`` or
    ``float``\ s depending on the elements).

    """
    if not isiterable(value):
        raise ValueError(value)
    return map(numeric, value)


def list_of_bools(value, interpret_strings=True):
    """
    Value must be iterable. All elements will be converted to ``bool``\ s.

    .. note:: By default, ``boolean()`` conversion function will be used, which means that
              strings like ``"0"`` or ``"false"`` will be interpreted as ``False``. If this
              is undesirable, set ``interpret_strings`` to ``False``.

    """
    if not isiterable(value):
        raise ValueError(value)
    if interpret_strings:
        return map(boolean, value)
    else:
        return map(bool, value)


regex_type = type(re.compile(''))


def regex(value):
    """
    Regular expression. If value is a string, it will be complied with no flags. If you
    want to specify flags, value must be precompiled.

    """
    if isinstance(value, regex_type):
        return value
    else:
        return re.compile(value)


__counters = defaultdict(int)


def reset_counter(name=None):
    __counters[name] = 0


def counter(name=None):
    """
    An auto incremeting value (kind of like an AUTO INCREMENT field in SQL).
    Optionally, the name of the counter to be used is specified (each counter
    increments separately).

    Counts start at 1, not 0.

    """
    __counters[name] += 1
    value = __counters[name]
    return value
