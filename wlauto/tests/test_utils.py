#    Copyright 2013-2015 ARM Limited
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

from nose.tools import raises, assert_equal, assert_not_equal  # pylint: disable=E0611

from wlauto.utils.android import check_output
from wlauto.utils.misc import merge_dicts, merge_lists, TimeoutError
from wlauto.utils.types import list_or_integer, list_or_bool, caseless_string, arguments


class TestCheckOutput(TestCase):

    def test_ok(self):
        check_output("python -c 'import time; time.sleep(0.1)'", timeout=0.5, shell=True)

    @raises(TimeoutError)
    def test_bad(self):
        check_output("python -c 'import time; time.sleep(1)'", timeout=0.5, shell=True)


class TestMerge(TestCase):

    def test_dict_merge(self):
        base = {'a': 1, 'b': {'x': 9, 'z': 10}}
        other = {'b': {'x': 7, 'y': 8}, 'c': [1, 2, 3]}
        result = merge_dicts(base, other)
        assert_equal(result['a'], 1)
        assert_equal(result['b']['x'], 7)
        assert_equal(result['b']['y'], 8)
        assert_equal(result['b']['z'], 10)
        assert_equal(result['c'], [1, 2, 3])

    def test_merge_dict_lists(self):
        base = {'a': [1, 3, 2]}
        other = {'a': [3, 4, 5]}
        result = merge_dicts(base, other)
        assert_equal(result['a'], [1, 3, 2, 3, 4, 5])
        result = merge_dicts(base, other, list_duplicates='first')
        assert_equal(result['a'], [1, 3, 2, 4, 5])
        result = merge_dicts(base, other, list_duplicates='last')
        assert_equal(result['a'], [1, 2, 3, 4, 5])

    def test_merge_lists(self):
        result = merge_lists([1, 2, 3], 7)
        assert_equal(result, [1, 2, 3, 7])
        result = merge_lists([1, 2, 3], 1, duplicates='last')
        assert_equal(result, [2, 3, 1])

    @raises(ValueError)
    def test_type_mismatch(self):
        base = {'a': [1, 2, 3]}
        other = {'a': 'test'}
        merge_dicts(base, other, match_types=True)


class TestTypes(TestCase):

    def test_list_or_conversion(self):
        assert_equal(list_or_integer([1, '2', 3]), [1, 2, 3])
        assert_equal(list_or_integer('0xF'), [15,])
        assert_equal(list_or_bool('False'), [False,])

    def test_caseless_string(self):
        cs1 = caseless_string('TeSt')
        assert_equal(cs1, 'TeSt')
        assert_equal('test', cs1)
        assert_equal(cs1[0], 'T')
        assert_not_equal(cs1[0], 't')
        assert_not_equal(cs1, 'test2')

    def test_arguments(self):
        assert_equal(arguments('--foo 7 --bar "fizz buzz"'),
                     ['--foo', '7', '--bar', 'fizz buzz'])
        assert_equal(arguments(['test', 42]), ['test', '42'])

