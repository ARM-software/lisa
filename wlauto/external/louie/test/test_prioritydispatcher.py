import unittest

import louie
from louie import dispatcher

class Callable(object):

    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val


one = Callable(1)
two = Callable(2)
three = Callable(3)

class TestPriorityDispatcher(unittest.TestCase):

    def test_ConnectNotify(self):
        louie.connect(
            two,
            'one',
            priority=200
            )
        louie.connect(
            one,
            'one',
            priority=100
            )
        louie.connect(
            three,
            'one',
            priority=300
            )
        result = [ i[1] for i in louie.send('one')]
        if not result == [1, 2, 3]:
            print result
            assert(False)

