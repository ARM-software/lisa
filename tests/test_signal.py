import unittest

from nose.tools import assert_equal, assert_true, assert_false

import wa.framework.signal as signal


class Callable(object):

    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val


class TestPriorityDispatcher(unittest.TestCase):

    def test_ConnectNotify(self):
        one = Callable(1)
        two = Callable(2)
        three = Callable(3)
        signal.connect(
            two,
            'test',
            priority=200
        )
        signal.connect(
            one,
            'test',
            priority=100
        )
        signal.connect(
            three,
            'test',
            priority=300
        )
        result = [i[1] for i in signal.send('test')]
        assert_equal(result, [3, 2, 1])

    def test_wrap_propagate(self):
        d = {'before': False, 'after': False, 'success': False}
        def before():
            d['before'] = True
        def after():
            d['after'] = True
        def success():
            d['success'] = True
        signal.connect(before, signal.BEFORE_WORKLOAD_SETUP)
        signal.connect(after, signal.AFTER_WORKLOAD_SETUP)
        signal.connect(success, signal.SUCCESSFUL_WORKLOAD_SETUP)

        caught = False
        try:
            with signal.wrap('WORKLOAD_SETUP'):
                raise RuntimeError()
        except RuntimeError:
            caught=True

        assert_true(d['before'])
        assert_true(d['after'])
        assert_true(caught)
        assert_false(d['success'])
