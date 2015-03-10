import unittest

import louie
from louie import dispatcher


def x(a):
    return a


class Dummy(object):
    pass


class Callable(object):

    def __call__(self, a):
        return a

    def a(self, a):
        return a


class TestDispatcher(unittest.TestCase):

    def setUp(self):
        louie.reset()

    def _isclean(self):
        """Assert that everything has been cleaned up automatically"""
        assert len(dispatcher.senders_back) == 0, dispatcher.senders_back
        assert len(dispatcher.connections) == 0, dispatcher.connections
        assert len(dispatcher.senders) == 0, dispatcher.senders
    
    def test_Exact(self):
        a = Dummy()
        signal = 'this'
        louie.connect(x, signal, a)
        expected = [(x, a)]
        result = louie.send('this', a, a=a)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        louie.disconnect(x, signal, a)
        assert len(list(louie.get_all_receivers(a, signal))) == 0
        self._isclean()
        
    def test_AnonymousSend(self):
        a = Dummy()
        signal = 'this'
        louie.connect(x, signal)
        expected = [(x, a)]
        result = louie.send(signal, None, a=a)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        louie.disconnect(x, signal)
        assert len(list(louie.get_all_receivers(None, signal))) == 0
        self._isclean()
        
    def test_AnyRegistration(self):
        a = Dummy()
        signal = 'this'
        louie.connect(x, signal, louie.Any)
        expected = [(x, a)]
        result = louie.send('this', object(), a=a)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        louie.disconnect(x, signal, louie.Any)
        expected = []
        result = louie.send('this', object(), a=a)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        assert len(list(louie.get_all_receivers(louie.Any, signal))) == 0
        self._isclean()
        
    def test_AllRegistration(self):
        a = Dummy()
        signal = 'this'
        louie.connect(x, louie.All, a)
        expected = [(x, a)]
        result = louie.send('this', a, a=a)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        louie.disconnect(x, louie.All, a)
        assert len(list(louie.get_all_receivers(a, louie.All))) == 0
        self._isclean()
        
    def test_GarbageCollected(self):
        a = Callable()
        b = Dummy()
        signal = 'this'
        louie.connect(a.a, signal, b)
        expected = []
        del a
        result = louie.send('this', b, a=b)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        assert len(list(louie.get_all_receivers(b, signal))) == 0, (
            "Remaining handlers: %s" % (louie.get_all_receivers(b, signal),))
        self._isclean()
        
    def test_GarbageCollectedObj(self):
        class x:
            def __call__(self, a):
                return a
        a = Callable()
        b = Dummy()
        signal = 'this'
        louie.connect(a, signal, b)
        expected = []
        del a
        result = louie.send('this', b, a=b)
        assert result == expected, (
            "Send didn't return expected result:\n\texpected:%s\n\tgot:%s"
            % (expected, result))
        assert len(list(louie.get_all_receivers(b, signal))) == 0, (
            "Remaining handlers: %s" % (louie.get_all_receivers(b, signal),))
        self._isclean()

    def test_MultipleRegistration(self):
        a = Callable()
        b = Dummy()
        signal = 'this'
        louie.connect(a, signal, b)
        louie.connect(a, signal, b)
        louie.connect(a, signal, b)
        louie.connect(a, signal, b)
        louie.connect(a, signal, b)
        louie.connect(a, signal, b)
        result = louie.send('this', b, a=b)
        assert len(result) == 1, result
        assert len(list(louie.get_all_receivers(b, signal))) == 1, (
            "Remaining handlers: %s" % (louie.get_all_receivers(b, signal),))
        del a
        del b
        del result
        self._isclean()

    def test_robust(self):
        """Test the sendRobust function."""
        def fails():
            raise ValueError('this')
        a = object()
        signal = 'this'
        louie.connect(fails, louie.All, a)
        result = louie.send_robust('this', a, a=a)
        err = result[0][1]
        assert isinstance(err, ValueError)
        assert err.args == ('this', )
