import unittest

import louie.prioritylist
from louie.prioritylist import PriorityList

#def populate_list(plist):

class TestPriorityList(unittest.TestCase):

    def test_Insert(self):
        pl = PriorityList()
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
            assert(pair[0]==pair[1])

    def test_Delete(self):
        pl = PriorityList()
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
        match = zip(sorted(elements.values()) , pl[:])
        for pair in match:
            assert(pair[0]==pair[1])

    def test_Multiple(self):
        pl = PriorityList()
        pl.add('1', 1)
        pl.add('2.1', 2)
        pl.add('3', 3)
        pl.add('2.2', 2)
        it = iter(pl)
        assert(it.next() == '1')
        assert(it.next() == '2.1')
        assert(it.next() == '2.2')
        assert(it.next() == '3')

    def test_IteratorBreak(self):
        pl = PriorityList()
        pl.add('1', 1)
        pl.add('2.1', 2)
        pl.add('3', 3)
        pl.add('2.2', 2)
        for i in pl:
            if i == '2.1':
                break
        assert(pl.index('3') == 3)
