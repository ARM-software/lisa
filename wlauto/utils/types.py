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
import os
import re
import math
import shlex
from bisect import insort
from collections import defaultdict

from wlauto.utils.misc import isiterable, to_identifier
from devlib.utils.types import identifier, boolean, integer, numeric, caseless_string


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

    def __setitem__(self, idx, value):
        list.__setitem__(self, idx, type_(value))

    return type('list_of_{}s'.format(type_.__name__),
                (list, ), {
                    "__init__": __init__,
                    "__setitem__": __setitem__,
                    "append": append,
                    "extend": extend,
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
    Converts the value into a list of ``caseless_string``'s. If the value is not iterable
    a one-element list with stringified value will be returned.

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
    Generator for "list or" types. These take either a single value or a list values
    and return a list of the specfied ``type_`` performing the conversion on the value
    (if a single value is specified) or each of the elemented of the specified list.

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
                           zip(self.priorities, [len(self.elements[p]) for p in self.priorities])}
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
