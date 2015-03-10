"""Louie plugin tests."""

import unittest

import louie

try:
    import qt
    if not hasattr(qt.qApp, 'for_testing'):
        _app = qt.QApplication([])
        _app.for_testing = True
        qt.qApp = _app
except ImportError:
    qt = None


class ReceiverBase(object):

    def __init__(self):
        self.args = []
        self.live = True

    def __call__(self, arg):
        self.args.append(arg)

class Receiver1(ReceiverBase):
    pass

class Receiver2(ReceiverBase):
    pass


class Plugin1(louie.Plugin):

    def is_live(self, receiver):
        """ReceiverBase instances are only live if their `live`
        attribute is True"""
        if isinstance(receiver, ReceiverBase):
            return receiver.live
        return True


class Plugin2(louie.Plugin):

    def is_live(self, receiver):
        """Pretend all Receiver2 instances are not live."""
        if isinstance(receiver, Receiver2):
            return False
        return True


def test_only_one_instance():
    louie.reset()
    plugin1a = Plugin1()
    plugin1b = Plugin1()
    louie.install_plugin(plugin1a)
    # XXX: Move these tests into test cases so we can use unittest's
    # 'assertRaises' method.
    try:
        louie.install_plugin(plugin1b)
    except louie.error.PluginTypeError:
        pass
    else:
        raise Exception('PluginTypeError not raised')


def test_is_live():
    louie.reset()
    # Create receivers.
    receiver1a = Receiver1()
    receiver1b = Receiver1()
    receiver2a = Receiver2()
    receiver2b = Receiver2()
    # Connect signals.
    louie.connect(receiver1a, 'sig')
    louie.connect(receiver1b, 'sig')
    louie.connect(receiver2a, 'sig')
    louie.connect(receiver2b, 'sig')
    # Check reception without plugins.
    louie.send('sig', arg='foo')
    assert receiver1a.args == ['foo']
    assert receiver1b.args == ['foo']
    assert receiver2a.args == ['foo']
    assert receiver2b.args == ['foo']
    # Install plugin 1.
    plugin1 = Plugin1()
    louie.install_plugin(plugin1)
    # Make some receivers not live.
    receiver1a.live = False
    receiver2b.live = False
    # Check reception.
    louie.send('sig', arg='bar')
    assert receiver1a.args == ['foo']
    assert receiver1b.args == ['foo', 'bar']
    assert receiver2a.args == ['foo', 'bar']
    assert receiver2b.args == ['foo']
    # Remove plugin 1, install plugin 2.
    plugin2 = Plugin2()
    louie.remove_plugin(plugin1)
    louie.install_plugin(plugin2)
    # Check reception.
    louie.send('sig', arg='baz')
    assert receiver1a.args == ['foo', 'baz']
    assert receiver1b.args == ['foo', 'bar', 'baz']
    assert receiver2a.args == ['foo', 'bar']
    assert receiver2b.args == ['foo']
    # Install plugin 1 alongside plugin 2.
    louie.install_plugin(plugin1)
    # Check reception.
    louie.send('sig', arg='fob')
    assert receiver1a.args == ['foo', 'baz']
    assert receiver1b.args == ['foo', 'bar', 'baz', 'fob']
    assert receiver2a.args == ['foo', 'bar']
    assert receiver2b.args == ['foo']
    

if qt is not None:
    def test_qt_plugin():
        louie.reset()
        # Create receivers.
        class Receiver(qt.QWidget):
            def __init__(self):
                qt.QObject.__init__(self)
                self.args = []
            def receive(self, arg):
                self.args.append(arg)
        receiver1 = Receiver()
        receiver2 = Receiver()
        # Connect signals.
        louie.connect(receiver1.receive, 'sig')
        louie.connect(receiver2.receive, 'sig')
        # Destroy receiver2 so only a shell is left.
        receiver2.close(True)
        # Check reception without plugins.
        louie.send('sig', arg='foo')
        assert receiver1.args == ['foo']
        assert receiver2.args == ['foo']
        # Install plugin.
        plugin = louie.QtWidgetPlugin()
        louie.install_plugin(plugin)
        # Check reception with plugins.
        louie.send('sig', arg='bar')
        assert receiver1.args == ['foo', 'bar']
        assert receiver2.args == ['foo']

