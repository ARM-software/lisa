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
Routines for doing various type conversions. These usually embody some
higher-level semantics than are present in standard Python types (e.g.
``boolean`` will convert the string ``"false"`` to ``False``, where as
non-empty strings are usually considered to be ``True``).

A lot of these are intened to stpecify type conversions declaratively in place
like ``Parameter``'s ``kind`` argument. These are basically "hacks" around the
fact that Python is not the best language to use for configuration.

"""
import os
import re
import numbers
import shlex
from bisect import insort
from urllib import quote, unquote
from collections import defaultdict, MutableMapping
from copy import copy

from devlib.utils.types import identifier, boolean, integer, numeric, caseless_string

from wa.utils.misc import isiterable


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

    .. note:: By default, ``boolean()`` conversion function will be used, which
              means that strings like ``"0"`` or ``"false"`` will be
              interpreted as ``False``. If this is undesirable, set
              ``interpret_strings`` to ``False``.

    """
    if not isiterable(value):
        raise ValueError(value)
    if interpret_strings:
        return map(boolean, value)
    else:
        return map(bool, value)


def list_of(type_):
    """Generates a "list of" callable for the specified type. The callable
    attempts to convert all elements in the passed value to the specifed
    ``type_``, raising ``ValueError`` on error."""
    def __init__(self, values):
        list.__init__(self, map(type_, values))

    def append(self, value):
        list.append(self, type_(value))

    def extend(self, other):
        list.extend(self, map(type_, other))

    def from_pod(cls, pod):
        return cls(map(type_, pod))

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
    if isinstance(value, basestring):
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
    if isinstance(value, basestring):
        return [caseless_string(value)]
    else:
        try:
            return map(caseless_string, value)
        except ValueError:
            return [caseless_string(value)]


def list_or(type_):
    """
    Generator for "list or" types. These take either a single value or a list
    values and return a list of the specfied ``type_`` performing the
    conversion on the value (if a single value is specified) or each of the
    elemented of the specified list.

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


class arguments(list):
    """
    Represents command line arguments to be passed to a program.

    """

    def __init__(self, value=None):
        if isiterable(value):
            super(arguments, self).__init__(map(str, value))
        elif isinstance(value, basestring):
            posix = os.name != 'nt'
            super(arguments, self).__init__(shlex.split(value, posix=posix))
        elif value is None:
            super(arguments, self).__init__()
        else:
            super(arguments, self).__init__([str(value)])

    def append(self, value):
        return super(arguments, self).append(str(value))

    def extend(self, values):
        return super(arguments, self).extend(map(str, values))

    def __str__(self):
        return ' '.join(self)


class prioritylist(object):

    def __init__(self):
        """
        Returns an OrderedReceivers object that externaly behaves
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
        order withing the List
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
        for priority, elements in self.elements.iteritems():
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
        if len(self.elements[priority]) == 0:
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
            index_range = range(index.start or 0, index.stop, index.step or 1)
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
    def merge(source, dest):
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

    def merge_with(self, other):
        new_self = copy(self)
        return toggle_set.merge(other, new_self)

    def merge_into(self, other):
        other = copy(other)
        return toggle_set.merge(self, other)

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
        return '_'.join(self, other)

    def merge_into(self, other):
        return '_'.join(other, self)


class obj_dict(MutableMapping):
    """
    An object that behaves like a dict but each dict entry can also be accesed
    as an attribute.

    :param not_in_dict: A list of keys that can only be accessed as attributes

    """

    @staticmethod
    def from_pod(pod):
        return obj_dict(pod)

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
        if name in self.__dict__['dict']:
            return self.__dict__['dict'][name]
        else:
            raise AttributeError("No such attribute: " + name)


class level(object):
    """
    A level has a name and behaves like a string when printed, however it also
    has a numeric value which is used in ordering comparisons.

    """

    @staticmethod
    def from_pod(pod):
        name, value_part =  pod.split('(')
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

    def __cmp__(self, other):
        if isinstance(other, level):
            return cmp(self.value, other.value)
        else:
            return cmp(self.value, other)

    def __eq__(self, other):
        if isinstance(other, level):
            return self.value == other.value
        elif isinstance(other, basestring):
            return self.name == other
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, level):
            return self.value != other.value
        elif isinstance(other, basestring):
            return self.name != other
        else:
            return self.value != other


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

    class Enum(object):

        class __metaclass__(type):
            def __str__(cls):
                return str(cls.levels)

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
    setattr(Enum, 'values', [lv.value for lv in levels])
    setattr(Enum, 'names', [lv.name for lv in levels])

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
        if isinstance(obj, basestring):
            prefix = 's'
        elif isinstance(obj, float):
            prefix = 'f'
        elif isinstance(obj, long):
            prefix = 'd'
        elif isinstance(obj, bool):
            prefix = 'b'
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
                return long(value)
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
        for k, v in kwargs.iteritems():
            self.__setitem__(k, v)
        dict.__init__(self, *args)

    def __setitem__(self, name, value):
        dict.__setitem__(self, name, self._encode(value))

    def __getitem__(self, name):
        return self._decode(dict.__getitem__(self, name))

    def __contains__(self, item):
        return dict.__contains__(self, self._encode(item))

    def __iter__(self):
        return iter((k, self._decode(v)) for (k, v) in self.items())

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
                for k, v in d.iteritems():
                    self[k] = v
