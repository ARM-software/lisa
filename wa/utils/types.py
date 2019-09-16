#    Copyright 2014-2018 ARM Limited
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
Routines for doing various type conversions. These usually embody some
higher-level semantics than are present in standard Python types (e.g.
``boolean`` will convert the string ``"false"`` to ``False``, where as
non-empty strings are usually considered to be ``True``).

A lot of these are intended to specify type conversions declaratively in place
like ``Parameter``'s ``kind`` argument. These are basically "hacks" around the
fact that Python is not the best language to use for configuration.

"""
import os
import re
import numbers
import shlex
import sys
from bisect import insort
if sys.version_info[0] == 3:
    from urllib.parse import quote, unquote  # pylint: disable=no-name-in-module, import-error
    from past.builtins import basestring  # pylint: disable=redefined-builtin
    long = int  # pylint: disable=redefined-builtin
else:
    from urllib import quote, unquote  # pylint: disable=no-name-in-module
# pylint: disable=wrong-import-position
from collections import defaultdict, MutableMapping
from functools import total_ordering

from future.utils import with_metaclass

from devlib.utils.types import identifier, boolean, integer, numeric, caseless_string

from wa.utils.misc import (isiterable, list_to_ranges, list_to_mask,
                           mask_to_list, ranges_to_list)


def list_of_strs(value):
    """
    Value must be iterable. All elements will be converted to strings.

    """
    if not isiterable(value):
        raise ValueError(value)
    return list(map(str, value))


list_of_strings = list_of_strs


def list_of_ints(value):
    """
    Value must be iterable. All elements will be converted to ``int``\ s.

    """
    if not isiterable(value):
        raise ValueError(value)
    return list(map(int, value))


list_of_integers = list_of_ints


def list_of_numbers(value):
    """
    Value must be iterable. All elements will be converted to numbers (either ``ints`` or
    ``float``\ s depending on the elements).

    """
    if not isiterable(value):
        raise ValueError(value)
    return list(map(numeric, value))


def list_of_bools(value, interpret_strings=True):
    """
    Value must be iterable. All elements will be converted to ``bool``\ s.

    .. note:: By default, ``boolean()`` conversion function will be used, which
              means that strings like ``"0"`` or ``"false"`` will be
              interpreted as ``False``. If this is undesirable, set
              ``interpret_strings`` to ``False``.

    """
    if not isiterable(value):
        raise ValueError(value)
    if interpret_strings:
        return list(map(boolean, value))
    else:
        return list(map(bool, value))


def list_of(type_):
    """Generates a "list of" callable for the specified type. The callable
    attempts to convert all elements in the passed value to the specified
    ``type_``, raising ``ValueError`` on error."""
    def __init__(self, values):
        list.__init__(self, list(map(type_, values)))

    def append(self, value):
        list.append(self, type_(value))

    def extend(self, other):
        list.extend(self, list(map(type_, other)))

    def from_pod(cls, pod):
        return cls(list(map(type_, pod)))

    def _to_pod(self):
        return self

    def __setitem__(self, idx, value):
        list.__setitem__(self, idx, type_(value))

    return type('list_of_{}s'.format(type_.__name__),
                (list, ), {
                    "__init__": __init__,
                    "__setitem__": __setitem__,
                    "append": append,
                    "extend": extend,
                    "to_pod": _to_pod,
                    "from_pod": classmethod(from_pod),
    })


def list_or_string(value):
    """
    Converts the value into a list of strings. If the value is not iterable,
    a one-element list with stringified value will be returned.

    """
    if isinstance(value, str):
        return [value]
    else:
        try:
            return list(value)
        except ValueError:
            return [str(value)]


def list_or_caseless_string(value):
    """
    Converts the value into a list of ``caseless_string``'s. If the value is
    not iterable a one-element list with stringified value will be returned.

    """
    if isinstance(value, str):
        return [caseless_string(value)]
    else:
        try:
            return list(map(caseless_string, value))
        except ValueError:
            return [caseless_string(value)]


def list_or(type_):
    """
    Generator for "list or" types. These take either a single value or a list
    values and return a list of the specified ``type_`` performing the
    conversion on the value (if a single value is specified) or each of the
    elements of the specified list.

    """
    list_type = list_of(type_)

    class list_or_type(list_type):
        def __init__(self, value):
            # pylint: disable=non-parent-init-called,super-init-not-called
            if isiterable(value):
                list_type.__init__(self, value)
            else:
                list_type.__init__(self, [value])
    return list_or_type


list_or_integer = list_or(integer)
list_or_number = list_or(numeric)
list_or_bool = list_or(boolean)


regex_type = type(re.compile(''))
none_type = type(None)


def regex(value):
    """
    Regular expression. If value is a string, it will be complied with no
    flags. If you want to specify flags, value must be precompiled.

    """
    if isinstance(value, regex_type):
        return value
    else:
        return re.compile(value)


def version_tuple(v):
    """
    Converts a version string into a tuple of ints that can be used for natural comparison.
    """
    return tuple(map(int, (v.split("."))))


__counters = defaultdict(int)


def reset_counter(name=None, value=0):
    __counters[name] = value


def reset_all_counters(value=0):
    for k in __counters:
        reset_counter(k, value)


def counter(name=None):
    """
    An auto incrementing value (kind of like an AUTO INCREMENT field in SQL).
    Optionally, the name of the counter to be used is specified (each counter
    increments separately).

    Counts start at 1, not 0.

    """
    __counters[name] += 1
    value = __counters[name]
    return value


class arguments(list):
    """
    Represents command line arguments to be passed to a program.

    """

    def __init__(self, value=None):
        if isiterable(value):
            super(arguments, self).__init__(list(map(str, value)))
        elif isinstance(value, str):
            posix = os.name != 'nt'
            super(arguments, self).__init__(shlex.split(value, posix=posix))
        elif value is None:
            super(arguments, self).__init__()
        else:
            super(arguments, self).__init__([str(value)])

    def append(self, value):
        return super(arguments, self).append(str(value))

    def extend(self, values):
        return super(arguments, self).extend(list(map(str, values)))

    def __str__(self):
        return ' '.join(self)


class prioritylist(object):

    def __init__(self):
        """
        Returns an OrderedReceivers object that externally behaves
        like a list but it maintains the order of its elements
        according to their priority.
        """
        self.elements = defaultdict(list)
        self.is_ordered = True
        self.priorities = []
        self.size = 0
        self._cached_elements = None

    def add(self, new_element, priority=0):
        """
        adds a new item in the list.

        - ``new_element`` the element to be inserted in the prioritylist
        - ``priority`` is the priority of the element which specifies its
        order within the List
        """
        self._add_element(new_element, priority)

    def add_before(self, new_element, element):
        priority, index = self._priority_index(element)
        self._add_element(new_element, priority, index)

    def add_after(self, new_element, element):
        priority, index = self._priority_index(element)
        self._add_element(new_element, priority, index + 1)

    def index(self, element):
        return self._to_list().index(element)

    def remove(self, element):
        index = self.index(element)
        self.__delitem__(index)

    def _priority_index(self, element):
        for priority, elements in self.elements.items():
            if element in elements:
                return (priority, elements.index(element))
        raise IndexError(element)

    def _to_list(self):
        if self._cached_elements is None:
            self._cached_elements = []
            for priority in self.priorities:
                self._cached_elements += self.elements[priority]
        return self._cached_elements

    def _add_element(self, element, priority, index=None):
        if index is None:
            self.elements[priority].append(element)
        else:
            self.elements[priority].insert(index, element)
        self.size += 1
        self._cached_elements = None
        if priority not in self.priorities:
            insort(self.priorities, priority)

    def _delete(self, priority, priority_index):
        del self.elements[priority][priority_index]
        self.size -= 1
        if not self.elements[priority]:
            self.priorities.remove(priority)
        self._cached_elements = None

    def __iter__(self):
        for priority in reversed(self.priorities):  # highest priority first
            for element in self.elements[priority]:
                yield element

    def __getitem__(self, index):
        return self._to_list()[index]

    def __delitem__(self, index):
        if isinstance(index, numbers.Integral):
            index = int(index)
            if index < 0:
                index_range = [len(self) + index]
            else:
                index_range = [index]
        elif isinstance(index, slice):
            index_range = list(range(index.start or 0, index.stop, index.step or 1))
        else:
            raise ValueError('Invalid index {}'.format(index))
        current_global_offset = 0
        priority_counts = {priority: count for (priority, count) in
                           zip(self.priorities, [len(self.elements[p])
                                                 for p in self.priorities])}
        for priority in self.priorities:
            if not index_range:
                break
            priority_offset = 0
            while index_range:
                del_index = index_range[0]
                if priority_counts[priority] + current_global_offset <= del_index:
                    current_global_offset += priority_counts[priority]
                    break
                within_priority_index = del_index - \
                    (current_global_offset + priority_offset)
                self._delete(priority, within_priority_index)
                priority_offset += 1
                index_range.pop(0)

    def __len__(self):
        return self.size


class toggle_set(set):
    """
    A set that contains items to enable or disable something.

    A prefix of ``~`` is used to denote disabling something, for example
    the list ['apples', '~oranges', 'cherries'] enables both ``apples``
    and ``cherries`` but disables ``oranges``.
    """

    @staticmethod
    def from_pod(pod):
        return toggle_set(pod)

    @staticmethod
    def merge(dest, source):
        if '~~' in source:
            return toggle_set(source)

        dest = toggle_set(dest)
        for item in source:
            if item not in dest:
                #Disable previously enabled item
                if item.startswith('~') and item[1:] in dest:
                    dest.remove(item[1:])
                #Enable previously disabled item
                if not item.startswith('~') and ('~' + item) in dest:
                    dest.remove('~' + item)
                dest.add(item)
        return dest

    def __init__(self, *args):
        if args:
            value = args[0]
            if isinstance(value, str):
                msg = 'invalid type for toggle_set: "{}"'
                raise TypeError(msg.format(type(value)))
        set.__init__(self, *args)

    def merge_with(self, other):
        return toggle_set.merge(self, other)

    def merge_into(self, other):
        return toggle_set.merge(other, self)

    def add(self, item):
        if item not in self:
            #Disable previously enabled item
            if item.startswith('~') and item[1:] in self:
                self.remove(item[1:])
            #Enable previously disabled item
            if not item.startswith('~') and ('~' + item) in self:
                self.remove('~' + item)
            super(toggle_set, self).add(item)

    def values(self):
        """
        returns a list of enabled items.
        """
        return set([item for item in self if not item.startswith('~')])

    def conflicts_with(self, other):
        """
        Checks if any items in ``other`` conflict with items already in this list.

        Args:
            other (list): The list to be checked against

        Returns:
            A list of items in ``other`` that conflict with items in this list
        """
        conflicts = []
        for item in other:
            if item.startswith('~') and item[1:] in self:
                conflicts.append(item)
            if not item.startswith('~') and ('~' + item) in self:
                conflicts.append(item)
        return conflicts

    def to_pod(self):
        return list(self.values())


class ID(str):

    def merge_with(self, other):
        return '_'.join([self, other])

    def merge_into(self, other):
        return '_'.join([other, self])


class obj_dict(MutableMapping):
    """
    An object that behaves like a dict but each dict entry can also be accessed
    as an attribute.

    :param not_in_dict: A list of keys that can only be accessed as attributes

    """

    @staticmethod
    def from_pod(pod):
        return obj_dict(pod)

    # pylint: disable=super-init-not-called
    def __init__(self, values=None, not_in_dict=None):
        self.__dict__['dict'] = dict(values or {})
        self.__dict__['not_in_dict'] = not_in_dict if not_in_dict is not None else []

    def to_pod(self):
        return self.__dict__['dict']

    def __getitem__(self, key):
        if key in self.not_in_dict:
            msg = '"{}" is in the list keys that can only be accessed as attributes'
            raise KeyError(msg.format(key))
        return self.__dict__['dict'][key]

    def __setitem__(self, key, value):
        self.__dict__['dict'][key] = value

    def __delitem__(self, key):
        del self.__dict__['dict'][key]

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        for key in self.__dict__['dict']:
            if key not in self.__dict__['not_in_dict']:
                yield key

    def __repr__(self):
        return repr(dict(self))

    def __str__(self):
        return str(dict(self))

    def __setattr__(self, name, value):
        self.__dict__['dict'][name] = value

    def __delattr__(self, name):
        if name in self:
            del self.__dict__['dict'][name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __getattr__(self, name):
        if 'dict' not in self.__dict__:
            raise AttributeError("No such attribute: " + name)
        if name in self.__dict__['dict']:
            return self.__dict__['dict'][name]
        else:
            raise AttributeError("No such attribute: " + name)


@total_ordering
class level(object):
    """
    A level has a name and behaves like a string when printed, however it also
    has a numeric value which is used in ordering comparisons.

    """

    @staticmethod
    def from_pod(pod):
        name, value_part = pod.split('(')
        return level(name, numeric(value_part.rstrip(')')))

    def __init__(self, name, value):
        self.name = caseless_string(name)
        self.value = numeric(value)

    def to_pod(self):
        return repr(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '{}({})'.format(self.name, self.value)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, level):
            return self.value == other.value
        elif isinstance(other, basestring):
            return self.name == other
        else:
            return self.value == other

    def __lt__(self, other):
        if isinstance(other, level):
            return self.value < other.value
        elif isinstance(other, basestring):
            return self.name < other
        else:
            return self.value < other

    def __ne__(self, other):
        if isinstance(other, level):
            return self.value != other.value
        elif isinstance(other, basestring):
            return self.name != other
        else:
            return self.value != other


class _EnumMeta(type):

    def __str__(cls):
        return str(cls.levels)

    def __getattr__(cls, name):
        name = name.lower()
        if name in cls.__dict__:
            return cls.__dict__[name]


def enum(args, start=0, step=1):
    """
    Creates a class with attributes named by the first argument.
    Each attribute is a ``level`` so they behave is integers in comparisons.
    The value of the first attribute is specified by the second argument
    (``0`` if not specified).

    ::
        MyEnum = enum(['A', 'B', 'C'])

    is roughly equivalent of::

        class MyEnum(object):
            A = 0
            B = 1
            C = 2

    however it also implement some specialized behaviors for comparisons and
    instantiation.

    """

    class Enum(with_metaclass(_EnumMeta, object)):

        @classmethod
        def from_pod(cls, pod):
            lv = level.from_pod(pod)
            for enum_level in cls.levels:
                if enum_level == lv:
                    return enum_level
            msg = 'Unexpected value "{}" for enum.'
            raise ValueError(msg.format(pod))

        def __new__(cls, name):
            for attr_name in dir(cls):
                if attr_name.startswith('__'):
                    continue

                attr = getattr(cls, attr_name)
                if name == attr:
                    return attr

            try:
                return Enum.from_pod(name)
            except ValueError:
                raise ValueError('Invalid enum value: {}'.format(repr(name)))

    reserved = ['values', 'levels', 'names']

    levels = []
    n = start
    for v in args:
        id_v = identifier(v)
        if id_v in reserved:
            message = 'Invalid enum level name "{}"; must not be in {}'
            raise ValueError(message.format(v, reserved))
        name = caseless_string(id_v)
        lv = level(v, n)
        setattr(Enum, name, lv)
        levels.append(lv)
        n += step

    setattr(Enum, 'levels', levels)
    setattr(Enum, 'values', [lvl.value for lvl in levels])
    setattr(Enum, 'names', [lvl.name for lvl in levels])

    return Enum


class ParameterDict(dict):
    """
    A dict-like object that automatically encodes various types into a url safe string,
    and enforces a single type for the contents in a list.
    Each value is first prefixed with 2 letters to preserve type when encoding to a string.
    The format used is "value_type, value_dimension" e.g a 'list of floats' would become 'fl'.
    """

    # Function to determine the appropriate prefix based on the parameters type
    @staticmethod
    def _get_prefix(obj):
        if isinstance(obj, str):
            prefix = 's'
        elif isinstance(obj, float):
            prefix = 'f'
        elif isinstance(obj, bool):
            prefix = 'b'
        elif isinstance(obj, long):
            prefix = 'i'
        elif isinstance(obj, int):
            prefix = 'i'
        elif obj is None:
            prefix = 'n'
        else:
            raise ValueError('Unable to encode {} {}'.format(obj, type(obj)))
        return prefix

    # Function to add prefix and urlencode a provided parameter.
    @staticmethod
    def _encode(obj):
        if isinstance(obj, list):
            t = type(obj[0])
            prefix = ParameterDict._get_prefix(obj[0]) + 'l'
            for item in obj:
                if not isinstance(item, t):
                    msg = 'Lists must only contain a single type, contains {} and {}'
                    raise ValueError(msg.format(t, type(item)))
            obj = '0newelement0'.join(str(x) for x in obj)
        else:
            prefix = ParameterDict._get_prefix(obj) + 's'
        return quote(prefix + str(obj))

    # Function to decode a string and return a value of the original parameter type.
    # pylint: disable=too-many-return-statements
    @staticmethod
    def _decode(string):
        value_type = string[:1]
        value_dimension = string[1:2]
        value = unquote(string[2:])
        if value_dimension == 's':
            if value_type == 's':
                return str(value)
            elif value_type == 'b':
                return boolean(value)
            elif value_type == 'd':
                return int(value)
            elif value_type == 'f':
                return float(value)
            elif value_type == 'i':
                return int(value)
            elif value_type == 'n':
                return None
        elif value_dimension == 'l':
            return [ParameterDict._decode(value_type + 's' + x)
                    for x in value.split('0newelement0')]
        else:
            raise ValueError('Unknown {} {}'.format(type(string), string))

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            self.__setitem__(k, v)
        dict.__init__(self, *args)

    def __setitem__(self, name, value):
        dict.__setitem__(self, name, self._encode(value))

    def __getitem__(self, name):
        return self._decode(dict.__getitem__(self, name))

    def __contains__(self, item):
        return dict.__contains__(self, self._encode(item))

    def __iter__(self):
        return iter((k, self._decode(v)) for (k, v) in list(self.items()))

    def iteritems(self):
        return self.__iter__()

    def get(self, name):
        return self._decode(dict.get(self, name))

    def pop(self, key):
        return self._decode(dict.pop(self, key))

    def popitem(self):
        key, value = dict.popitem(self)
        return (key, self._decode(value))

    def iter_encoded_items(self):
        if sys.version_info[0] == 3:
            return dict.items(self)
        else:
            return dict.iteritems(self)

    def get_encoded_value(self, name):
        return dict.__getitem__(self, name)

    def values(self):
        return [self[k] for k in dict.keys(self)]

    def update(self, *args, **kwargs):
        for d in list(args) + [kwargs]:
            if isinstance(d, ParameterDict):
                dict.update(self, d)
            else:
                for k, v in d.items():
                    self[k] = v


class cpu_mask(object):
    """
    A class to allow for a consistent way of representing a cpus mask with
    methods to provide conversions between the various required forms. The
    mask can be specified directly as a mask, as a list of cpus indexes or a
    sysfs-style string.
    """
    @staticmethod
    def from_pod(pod):
        return cpu_mask(int(pod['cpu_mask']))

    def __init__(self, cpus):
        self._mask = 0
        if isinstance(cpus, int):
            self._mask = cpus
        elif isinstance(cpus, str):
            if cpus[:2] == '0x' or cpus[:2] == '0X':
                self._mask = int(cpus, 16)
            else:
                self._mask = list_to_mask(ranges_to_list(cpus))
        elif isinstance(cpus, list):
            self._mask = list_to_mask(cpus)
        elif isinstance(cpus, cpu_mask):
            self._mask = cpus._mask  # pylint: disable=protected-access
        else:
            msg = 'Unknown conversion from {} to cpu mask'
            raise ValueError(msg.format(cpus))

    def __bool__(self):
        """Allow for use in comparisons to check if a mask has been set"""
        return bool(self._mask)

    __nonzero__ = __bool__

    def __repr__(self):
        return 'cpu_mask: {}'.format(self.mask())

    __str__ = __repr__

    def list(self):
        """Returns a list of the indexes of bits that are set in the mask."""
        return list(reversed(mask_to_list(self._mask)))

    def mask(self, prefix=True):
        """Returns a hex representation of the mask with an optional prefix"""
        if prefix:
            return hex(self._mask)
        else:
            return hex(self._mask)[2:]

    def ranges(self):
        """"Returns a sysfs-style ranges string"""
        return list_to_ranges(self.list())

    def to_pod(self):
        return {'cpu_mask': self._mask}
