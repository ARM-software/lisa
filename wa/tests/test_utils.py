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

from nose.tools import raises, assert_equal, assert_not_equal, assert_in, assert_not_in
from nose.tools import assert_true, assert_false

from wa.utils.types import list_or_integer, list_or_bool, caseless_string, arguments, prioritylist, TreeNode


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

        match = zip(sorted(elements.values()), pl[:])
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
        match = zip(sorted(elements.values()), pl[:])
        for pair in match:
            assert(pair[0] == pair[1])

    def test_multiple(self):
        pl = prioritylist()
        pl.add('1', 1)
        pl.add('2.1', 2)
        pl.add('3', 3)
        pl.add('2.2', 2)
        it = iter(pl)
        assert_equal(it.next(), '3')
        assert_equal(it.next(), '2.1')
        assert_equal(it.next(), '2.2')
        assert_equal(it.next(), '1')

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


class TestTreeNode(TestCase):

    def test_addremove(self):
        n1, n2, n3 = TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2)
        n3.parent = n2
        assert_equal(n2.parent, n1)
        assert_in(n3, n2.children)
        n2.remove_child(n3)
        assert_equal(n3.parent, None)
        assert_not_in(n3, n2.children)
        n1.add_child(n2)  # duplicat add
        assert_equal(n1.children, [n2])

    def test_ancestor_descendant(self):
        n1, n2a, n2b, n3 = TreeNode(), TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2a)
        n1.add_child(n2b)
        n2a.add_child(n3)
        assert_equal(list(n3.iter_ancestors()), [n3, n2a, n1])
        assert_equal(list(n1.iter_descendants()), [n2a, n3, n2b])
        assert_true(n1.has_descendant(n3))
        assert_true(n3.has_ancestor(n1))
        assert_false(n3.has_ancestor(n2b))

    def test_root(self):
        n1, n2, n3 = TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2)
        n2.add_child(n3)
        assert_true(n1.is_root)
        assert_false(n2.is_root)
        assert_equal(n3.get_root(), n1)

    def test_common_ancestor(self):
        n1, n2, n3a, n3b, n4, n5 = TreeNode(), TreeNode(), TreeNode(), TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2)
        n2.add_child(n3a)
        n2.add_child(n3b)
        n3b.add_child(n4)
        n3a.add_child(n5)
        assert_equal(n4.get_common_ancestor(n3a), n2)
        assert_equal(n3a.get_common_ancestor(n4), n2)
        assert_equal(n3b.get_common_ancestor(n4), n3b)
        assert_equal(n4.get_common_ancestor(n3b), n3b)
        assert_equal(n4.get_common_ancestor(n5), n2)

    def test_iteration(self):
        n1, n2, n3, n4, n5 = TreeNode(), TreeNode(), TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2)
        n2.add_child(n3)
        n3.add_child(n4)
        n4.add_child(n5)
        ancestors = [a for a in n5.iter_ancestors(upto=n2)]
        assert_equal(ancestors, [n5, n4, n3])
        ancestors = [a for a in n5.iter_ancestors(after=n2)]
        assert_equal(ancestors, [n2, n1])

    @raises(ValueError)
    def test_trivial_loop(self):
        n1, n2, n3 = TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2)
        n2.add_child(n3)
        n3.add_child(n1)

    @raises(ValueError)
    def test_tree_violation(self):
        n1, n2a, n2b, n3 = TreeNode(), TreeNode(), TreeNode(), TreeNode()
        n1.add_child(n2a)
        n1.add_child(n2b)
        n2a.add_child(n3)
        n2b.add_child(n3)

    @raises(ValueError)
    def test_self_parent(self):
        n = TreeNode()
        n.add_child(n)
