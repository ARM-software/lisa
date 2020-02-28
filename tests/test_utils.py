#    Copyright 2013-2018 ARM Limited
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


# pylint: disable=R0201
from unittest import TestCase

from nose.tools import raises, assert_equal, assert_not_equal, assert_in, assert_not_in
from nose.tools import assert_true, assert_false, assert_raises, assert_is, assert_list_equal

from wa.utils.types import (list_or_integer, list_or_bool, caseless_string,
                            arguments, prioritylist, enum, level, toggle_set)



class TestPriorityList(TestCase):

    def test_insert(self):
        pl = prioritylist()
        elements = {3: "element 3",
                    2: "element 2",
                    1: "element 1",
                    5: "element 5",
                    4: "element 4"
                    }
        for key in elements:
            pl.add(elements[key], priority=key)

        match = list(zip(sorted(elements.values()), pl[:]))
        for pair in match:
            assert(pair[0] == pair[1])

    def test_delete(self):
        pl = prioritylist()
        elements = {2: "element 3",
                    1: "element 2",
                    0: "element 1",
                    4: "element 5",
                    3: "element 4"
                    }
        for key in elements:
            pl.add(elements[key], priority=key)
        del elements[2]
        del pl[2]
        match = list(zip(sorted(elements.values()), pl[:]))
        for pair in match:
            assert(pair[0] == pair[1])

    def test_multiple(self):
        pl = prioritylist()
        pl.add('1', 1)
        pl.add('2.1', 2)
        pl.add('3', 3)
        pl.add('2.2', 2)
        it = iter(pl)
        assert_equal(next(it), '3')
        assert_equal(next(it), '2.1')
        assert_equal(next(it), '2.2')
        assert_equal(next(it), '1')

    def test_iterator_break(self):
        pl = prioritylist()
        pl.add('1', 1)
        pl.add('2.1', 2)
        pl.add('3', 3)
        pl.add('2.2', 2)
        for i in pl:
            if i == '2.1':
                break
        assert_equal(pl.index('3'), 3)

    def test_add_before_after(self):
        pl = prioritylist()
        pl.add('m', 1)
        pl.add('a', 2)
        pl.add('n', 1)
        pl.add('b', 2)
        pl.add_before('x', 'm')
        assert_equal(list(pl), ['a', 'b', 'x', 'm', 'n'])
        pl.add_after('y', 'b')
        assert_equal(list(pl), ['a', 'b','y', 'x', 'm', 'n'])
        pl.add_after('z', 'm')
        assert_equal(list(pl), ['a', 'b', 'y', 'x', 'm', 'z', 'n'])


class TestEnumLevel(TestCase):

    def test_enum_creation(self):
        e = enum(['one', 'two', 'three'])
        assert_list_equal(e.values, [0, 1, 2])

        e = enum(['one', 'two', 'three'], start=10)
        assert_list_equal(e.values, [10, 11, 12])

        e = enum(['one', 'two', 'three'], start=-10, step=10)
        assert_list_equal(e.values, [-10, 0, 10])

    def test_enum_name_conflicts(self):
        assert_raises(ValueError, enum, ['names', 'one', 'two'])

        e = enum(['NAMES', 'one', 'two'])
        assert_in('names', e.levels)
        assert_list_equal(e.names, ['names', 'one', 'two'])
        assert_equal(e.ONE, 'one')
        result = not (e.ONE != 'one')
        assert_true(result)

    def test_enum_behavior(self):
        e = enum(['one', 'two', 'three'])

        # case-insensitive level name and level value may all
        # be used for equality comparisons.
        assert_equal(e.one, 'one')
        assert_equal(e.one, 'ONE')
        assert_equal(e.one, 0)
        assert_not_equal(e.one, '0')

        # ditto for enum membership tests
        assert_in('one', e.levels)
        assert_in(2, e.levels)
        assert_not_in('five', e.levels)

        # The same level object returned, only when
        # passing in a valid level name/value.
        assert_is(e('one'), e('ONE'))
        assert_is(e('one'), e(0))
        assert_raises(ValueError, e, 'five')

    def test_serialize_level(self):
        l = level('test', 1)
        s = l.to_pod()
        l2 = level.from_pod(s)
        assert_equal(l, l2)

    def test_deserialize_enum(self):
        e = enum(['one', 'two', 'three'])
        s = e.one.to_pod()
        l = e.from_pod(s)
        assert_equal(l, e.one)


class  TestToggleSet(TestCase):

    def test_equality(self):
        ts1 = toggle_set(['one', 'two',])
        ts2 = toggle_set(['one', 'two', '~three'])

        assert_not_equal(ts1, ts2)
        assert_equal(ts1.values(), ts2.values())
        assert_equal(ts2, toggle_set(['two', '~three', 'one']))

    def test_merge(self):
        ts1 = toggle_set(['one', 'two', 'three', '~four', '~five'])
        ts2 = toggle_set(['two', '~three', 'four', '~five'])

        ts3 = ts1.merge_with(ts2)
        assert_equal(ts1, toggle_set(['one', 'two', 'three', '~four', '~five']))
        assert_equal(ts2, toggle_set(['two', '~three', 'four', '~five']))
        assert_equal(ts3, toggle_set(['one', 'two', '~three', 'four', '~five']))
        assert_equal(ts3.values(), set(['one', 'two','four']))

        ts4 = ts1.merge_into(ts2)
        assert_equal(ts1, toggle_set(['one', 'two', 'three', '~four', '~five']))
        assert_equal(ts2, toggle_set(['two', '~three', 'four', '~five']))
        assert_equal(ts4, toggle_set(['one', 'two', 'three', '~four', '~five']))
        assert_equal(ts4.values(), set(['one', 'two', 'three']))

    def test_drop_all_previous(self):
        ts1 = toggle_set(['one', 'two', 'three'])
        ts2 = toggle_set(['four', '~~', 'five'])
        ts3 = toggle_set(['six', 'seven', '~three'])

        ts4 = ts1.merge_with(ts2).merge_with(ts3)
        assert_equal(ts4, toggle_set(['four', 'five', 'six', 'seven', '~three', '~~']))

        ts5 = ts2.merge_into(ts3).merge_into(ts1)
        assert_equal(ts5, toggle_set(['four', 'five', '~~']))

        ts6 = ts2.merge_into(ts3).merge_with(ts1)
        assert_equal(ts6, toggle_set(['one', 'two', 'three', 'four', 'five', '~~']))

    def test_order_on_create(self):
        ts1 = toggle_set(['one', 'two', 'three', '~one'])
        assert_equal(ts1, toggle_set(['~one', 'two', 'three']))

        ts1 = toggle_set(['~one', 'two', 'three', 'one'])
        assert_equal(ts1, toggle_set(['one', 'two', 'three']))
