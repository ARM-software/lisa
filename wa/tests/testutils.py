from mock import Mock
from nose.tools import assert_true

from wa.framework import signal
from wa.framework.plugin import Plugin
from wa.utils.types import identifier


class SignalWatcher(object):

    signals = []

    def __init__(self):
        for sig in self.signals:
            name = identifier(sig.name)
            callback = Mock()
            callback.im_func.__name__ = name
            setattr(self, name, callback)
            signal.connect(getattr(self, name), sig)

    def assert_all_called(self):
        for m in self.__dict__.itervalues():
            assert_true(m.called)


class MockContainerActor(Plugin):

    name = 'mock-container'
    kind = 'container-actor'

    def __init__(self, owner=None, *args, **kwargs):
        super(MockContainerActor, self).__init__(*args, **kwargs)
        self.owner=owner
        self.initialize = Mock()
        self.finalize = Mock()
        self.enter = Mock()
        self.exit = Mock()
        self.job_started = Mock()
        self.job_completed = Mock()
