"""Common plugins for Louie."""

from louie import dispatcher
from louie import error


def install_plugin(plugin):
    cls = plugin.__class__
    for p in dispatcher.plugins:
        if p.__class__ is cls:
            raise error.PluginTypeError(
                'Plugin of type %r already installed.' % cls)
    dispatcher.plugins.append(plugin)

def remove_plugin(plugin):
    dispatcher.plugins.remove(plugin)


class Plugin(object):
    """Base class for Louie plugins.

    Plugins are used to extend or alter the behavior of Louie
    in a uniform way without having to modify the Louie code
    itself.
    """

    def is_live(self, receiver):
        """Return True if the receiver is still live.

        Only called for receivers who have already been determined to
        be live by default Louie semantics.
        """
        return True

    def wrap_receiver(self, receiver):
        """Return a callable that passes arguments to the receiver.

        Useful when you want to change the behavior of all receivers.
        """
        return receiver


class QtWidgetPlugin(Plugin):
    """A Plugin for Louie that knows how to handle Qt widgets
    when using PyQt built with SIP 4 or higher.

    Weak references are not useful when dealing with QWidget
    instances, because even after a QWidget is closed and destroyed,
    only the C++ object is destroyed.  The Python 'shell' object
    remains, but raises a RuntimeError when an attempt is made to call
    an underlying QWidget method.

    This plugin alleviates this behavior, and if a QWidget instance is
    found that is just an empty shell, it prevents Louie from
    dispatching to any methods on those objects.
    """

    def __init__(self):
        try:
            import qt
        except ImportError:
            self.is_live = self._is_live_no_qt
        else:
            self.qt = qt

    def is_live(self, receiver):
        """If receiver is a method on a QWidget, only return True if
        it hasn't been destroyed."""
        if (hasattr(receiver, 'im_self') and
            isinstance(receiver.im_self, self.qt.QWidget)
            ):
            try:
                receiver.im_self.x()
            except RuntimeError:
                return False
        return True

    def _is_live_no_qt(self, receiver):
        return True


class TwistedDispatchPlugin(Plugin):
    """Plugin for Louie that wraps all receivers in callables
    that return Twisted Deferred objects.

    When the wrapped receiver is called, it adds a call to the actual
    receiver to the reactor event loop, and returns a Deferred that is
    called back with the result.
    """

    def __init__(self):
        # Don't import reactor ourselves, but make access to it
        # easier.
        from twisted import internet
        from twisted.internet.defer import Deferred
        self._internet = internet
        self._Deferred = Deferred

    def wrap_receiver(self, receiver):
        def wrapper(*args, **kw):
            d = self._Deferred()
            def called(dummy):
                return receiver(*args, **kw)
            d.addCallback(called)
            self._internet.reactor.callLater(0, d.callback, None)
            return d
        return wrapper

