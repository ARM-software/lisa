"""Multiple-producer-multiple-consumer signal-dispatching.

``dispatcher`` is the core of Louie, providing the primary API and the
core logic for the system.

Internal attributes:

- ``WEAKREF_TYPES``: Tuple of types/classes which represent weak
  references to receivers, and thus must be dereferenced on retrieval
  to retrieve the callable object
        
- ``connections``::

    { senderkey (id) : { signal : [receivers...] } }
    
- ``senders``: Used for cleaning up sender references on sender
  deletion::

    { senderkey (id) : weakref(sender) }
    
- ``senders_back``: Used for cleaning up receiver references on receiver
  deletion::

    { receiverkey (id) : [senderkey (id)...] }
"""

import os
import weakref

try:
    set
except NameError:
    from sets import Set as set, ImmutableSet as frozenset

from louie import error
from louie import robustapply
from louie import saferef
from louie.sender import Any, Anonymous
from louie.signal import All
from prioritylist import PriorityList


# Support for statistics.
if __debug__:
    connects = 0
    disconnects = 0
    sends = 0

    def print_stats():
        print ('\n'
               'Louie connects: %i\n'
               'Louie disconnects: %i\n'
               'Louie sends: %i\n'
               '\n') % (connects, disconnects, sends)

    if 'PYDISPATCH_STATS' in os.environ:
        import atexit
        atexit.register(print_stats)



WEAKREF_TYPES = (weakref.ReferenceType, saferef.BoundMethodWeakref)


connections = {}
senders = {}
senders_back = {}
plugins = []

def reset():
    """Reset the state of Louie.

    Useful during unit testing.  Should be avoided otherwise.
    """
    global connections, senders, senders_back, plugins
    connections = {}
    senders = {}
    senders_back = {}
    plugins = []


def connect(receiver, signal=All, sender=Any, weak=True, priority=0):
    """Connect ``receiver`` to ``sender`` for ``signal``.

    - ``receiver``: A callable Python object which is to receive
      messages/signals/events.  Receivers must be hashable objects.

      If weak is ``True``, then receiver must be weak-referencable (more
      precisely ``saferef.safe_ref()`` must be able to create a
      reference to the receiver).
    
      Receivers are fairly flexible in their specification, as the
      machinery in the ``robustapply`` module takes care of most of the
      details regarding figuring out appropriate subsets of the sent
      arguments to apply to a given receiver.

      Note: If ``receiver`` is itself a weak reference (a callable), it
      will be de-referenced by the system's machinery, so *generally*
      weak references are not suitable as receivers, though some use
      might be found for the facility whereby a higher-level library
      passes in pre-weakrefed receiver references.

    - ``signal``: The signal to which the receiver should respond.
    
      If ``All``, receiver will receive all signals from the indicated
      sender (which might also be ``All``, but is not necessarily
      ``All``).
        
      Otherwise must be a hashable Python object other than ``None``
      (``DispatcherError`` raised on ``None``).
        
    - ``sender``: The sender to which the receiver should respond.
    
      If ``Any``, receiver will receive the indicated signals from any
      sender.
        
      If ``Anonymous``, receiver will only receive indicated signals
      from ``send``/``send_exact`` which do not specify a sender, or
      specify ``Anonymous`` explicitly as the sender.

      Otherwise can be any python object.
        
    - ``weak``: Whether to use weak references to the receiver.
      
      By default, the module will attempt to use weak references to
      the receiver objects.  If this parameter is ``False``, then strong
      references will be used.

    - ``priority``: specifies the priority by which a reciever should
      get notified

    Returns ``None``, may raise ``DispatcherTypeError``.
    """
    if signal is None:
        raise error.DispatcherTypeError(
            'Signal cannot be None (receiver=%r sender=%r)'
            % (receiver, sender))
    if weak:
        receiver = saferef.safe_ref(receiver, on_delete=_remove_receiver)
    senderkey = id(sender)
    if connections.has_key(senderkey):
        signals = connections[senderkey]
    else:
        connections[senderkey] = signals = {}
    # Keep track of senders for cleanup.
    # Is Anonymous something we want to clean up?
    if sender not in (None, Anonymous, Any):
        def remove(object, senderkey=senderkey):
            _remove_sender(senderkey=senderkey)
        # Skip objects that can not be weakly referenced, which means
        # they won't be automatically cleaned up, but that's too bad.
        try:
            weak_sender = weakref.ref(sender, remove)
            senders[senderkey] = weak_sender
        except:
            pass
    receiver_id = id(receiver)
    # get current set, remove any current references to
    # this receiver in the set, including back-references
    if signals.has_key(signal):
        receivers = signals[signal]
        _remove_old_back_refs(senderkey, signal, receiver, receivers)
    else:
        receivers = signals[signal] = PriorityList()
    try:
        current = senders_back.get(receiver_id)
        if current is None:
            senders_back[receiver_id] = current = []
        if senderkey not in current:
            current.append(senderkey)
    except:
        pass
    receivers.add(receiver, priority)
    # Update stats.
    if __debug__:
        global connects
        connects += 1


def disconnect(receiver, signal=All, sender=Any, weak=True):
    """Disconnect ``receiver`` from ``sender`` for ``signal``.

    - ``receiver``: The registered receiver to disconnect.
    
    - ``signal``: The registered signal to disconnect.
    
    - ``sender``: The registered sender to disconnect.
    
    - ``weak``: The weakref state to disconnect.

    ``disconnect`` reverses the process of ``connect``, the semantics for
    the individual elements are logically equivalent to a tuple of
    ``(receiver, signal, sender, weak)`` used as a key to be deleted
    from the internal routing tables.  (The actual process is slightly
    more complex but the semantics are basically the same).

    Note: Using ``disconnect`` is not required to cleanup routing when
    an object is deleted; the framework will remove routes for deleted
    objects automatically.  It's only necessary to disconnect if you
    want to stop routing to a live object.
        
    Returns ``None``, may raise ``DispatcherTypeError`` or
    ``DispatcherKeyError``.
    """
    if signal is None:
        raise error.DispatcherTypeError(
            'Signal cannot be None (receiver=%r sender=%r)'
            % (receiver, sender))
    if weak:
        receiver = saferef.safe_ref(receiver)
    senderkey = id(sender)
    try:
        signals = connections[senderkey]
        receivers = signals[signal]
    except KeyError:
        raise error.DispatcherKeyError(
            'No receivers found for signal %r from sender %r' 
            % (signal, sender)
            )
    try:
        # also removes from receivers
        _remove_old_back_refs(senderkey, signal, receiver, receivers)
    except ValueError:
        raise error.DispatcherKeyError(
            'No connection to receiver %s for signal %s from sender %s'
            % (receiver, signal, sender)
            )
    _cleanup_connections(senderkey, signal)
    # Update stats.
    if __debug__:
        global disconnects
        disconnects += 1


def get_receivers(sender=Any, signal=All):
    """Get list of receivers from global tables.

    This function allows you to retrieve the raw list of receivers
    from the connections table for the given sender and signal pair.

    Note: There is no guarantee that this is the actual list stored in
    the connections table, so the value should be treated as a simple
    iterable/truth value rather than, for instance a list to which you
    might append new records.

    Normally you would use ``live_receivers(get_receivers(...))`` to
    retrieve the actual receiver objects as an iterable object.
    """
    try:
        return connections[id(sender)][signal]
    except KeyError:
        return []


def live_receivers(receivers):
    """Filter sequence of receivers to get resolved, live receivers.

    This is a generator which will iterate over the passed sequence,
    checking for weak references and resolving them, then returning
    all live receivers.
    """
    for receiver in receivers:
        if isinstance(receiver, WEAKREF_TYPES):
            # Dereference the weak reference.
            receiver = receiver()
        if receiver is not None:
            # Check installed plugins to make sure this receiver is
            # live.
            live = True
            for plugin in plugins:
                if not plugin.is_live(receiver):
                    live = False
                    break
            if live:
                yield receiver
            

def get_all_receivers(sender=Any, signal=All):
    """Get list of all receivers from global tables.

    This gets all receivers which should receive the given signal from
    sender, each receiver should be produced only once by the
    resulting generator.
    """
    yielded = set()
    for receivers in (
        # Get receivers that receive *this* signal from *this* sender.
        get_receivers(sender, signal),
        # Add receivers that receive *all* signals from *this* sender.
        get_receivers(sender, All),
        # Add receivers that receive *this* signal from *any* sender.
        get_receivers(Any, signal),
        # Add receivers that receive *all* signals from *any* sender.
        get_receivers(Any, All),
        ):
        for receiver in receivers:
            if receiver: # filter out dead instance-method weakrefs
                try:
                    if not receiver in yielded:
                        yielded.add(receiver)
                        yield receiver
                except TypeError:
                    # dead weakrefs raise TypeError on hash...
                    pass


def send(signal=All, sender=Anonymous, *arguments, **named):
    """Send ``signal`` from ``sender`` to all connected receivers.
    
    - ``signal``: (Hashable) signal value; see ``connect`` for details.

    - ``sender``: The sender of the signal.
    
      If ``Any``, only receivers registered for ``Any`` will receive the
      message.

      If ``Anonymous``, only receivers registered to receive messages
      from ``Anonymous`` or ``Any`` will receive the message.

      Otherwise can be any Python object (normally one registered with
      a connect if you actually want something to occur).

    - ``arguments``: Positional arguments which will be passed to *all*
      receivers. Note that this may raise ``TypeError`` if the receivers
      do not allow the particular arguments.  Note also that arguments
      are applied before named arguments, so they should be used with
      care.

    - ``named``: Named arguments which will be filtered according to the
      parameters of the receivers to only provide those acceptable to
      the receiver.

    Return a list of tuple pairs ``[(receiver, response), ...]``

    If any receiver raises an error, the error propagates back through
    send, terminating the dispatch loop, so it is quite possible to
    not have all receivers called if a raises an error.
    """
    # Call each receiver with whatever arguments it can accept.
    # Return a list of tuple pairs [(receiver, response), ... ].
    responses = []
    for receiver in live_receivers(get_all_receivers(sender, signal)):
        # Wrap receiver using installed plugins.
        original = receiver
        for plugin in plugins:
            receiver = plugin.wrap_receiver(receiver)
        response = robustapply.robust_apply(
            receiver, original,
            signal=signal,
            sender=sender,
            *arguments,
            **named
            )
        responses.append((receiver, response))
    # Update stats.
    if __debug__:
        global sends
        sends += 1
    return responses


def send_minimal(signal=All, sender=Anonymous, *arguments, **named):
    """Like ``send``, but does not attach ``signal`` and ``sender``
    arguments to the call to the receiver."""
    # Call each receiver with whatever arguments it can accept.
    # Return a list of tuple pairs [(receiver, response), ... ].
    responses = []
    for receiver in live_receivers(get_all_receivers(sender, signal)):
        # Wrap receiver using installed plugins.
        original = receiver
        for plugin in plugins:
            receiver = plugin.wrap_receiver(receiver)
        response = robustapply.robust_apply(
            receiver, original,
            *arguments,
            **named
            )
        responses.append((receiver, response))
    # Update stats.
    if __debug__:
        global sends
        sends += 1
    return responses


def send_exact(signal=All, sender=Anonymous, *arguments, **named):
    """Send ``signal`` only to receivers registered for exact message.

    ``send_exact`` allows for avoiding ``Any``/``Anonymous`` registered
    handlers, sending only to those receivers explicitly registered
    for a particular signal on a particular sender.
    """
    responses = []
    for receiver in live_receivers(get_receivers(sender, signal)):
        # Wrap receiver using installed plugins.
        original = receiver
        for plugin in plugins:
            receiver = plugin.wrap_receiver(receiver)
        response = robustapply.robust_apply(
            receiver, original,
            signal=signal,
            sender=sender,
            *arguments,
            **named
            )
        responses.append((receiver, response))
    return responses
    

def send_robust(signal=All, sender=Anonymous, *arguments, **named):
    """Send ``signal`` from ``sender`` to all connected receivers catching
    errors

    - ``signal``: (Hashable) signal value, see connect for details

    - ``sender``: The sender of the signal.
    
      If ``Any``, only receivers registered for ``Any`` will receive the
      message.

      If ``Anonymous``, only receivers registered to receive messages
      from ``Anonymous`` or ``Any`` will receive the message.

      Otherwise can be any Python object (normally one registered with
      a connect if you actually want something to occur).

    - ``arguments``: Positional arguments which will be passed to *all*
      receivers. Note that this may raise ``TypeError`` if the receivers
      do not allow the particular arguments.  Note also that arguments
      are applied before named arguments, so they should be used with
      care.

    - ``named``: Named arguments which will be filtered according to the
      parameters of the receivers to only provide those acceptable to
      the receiver.

    Return a list of tuple pairs ``[(receiver, response), ... ]``

    If any receiver raises an error (specifically, any subclass of
    ``Exception``), the error instance is returned as the result for
    that receiver.
    """
    # Call each receiver with whatever arguments it can accept.
    # Return a list of tuple pairs [(receiver, response), ... ].
    responses = []
    for receiver in live_receivers(get_all_receivers(sender, signal)):
        original = receiver
        for plugin in plugins:
            receiver = plugin.wrap_receiver(receiver)
        try:
            response = robustapply.robust_apply(
                receiver, original,
                signal=signal,
                sender=sender,
                *arguments,
                **named
                )
        except Exception, err:
            responses.append((receiver, err))
        else:
            responses.append((receiver, response))
    return responses


def _remove_receiver(receiver):
    """Remove ``receiver`` from connections."""
    if not senders_back:
        # During module cleanup the mapping will be replaced with None.
        return False
    backKey = id(receiver)
    for senderkey in senders_back.get(backKey, ()):
        try:
            signals = connections[senderkey].keys()
        except KeyError:
            pass
        else:
            for signal in signals:
                try:
                    receivers = connections[senderkey][signal]
                except KeyError:
                    pass
                else:
                    try:
                        receivers.remove(receiver)
                    except Exception:
                        pass
                _cleanup_connections(senderkey, signal)
    try:
        del senders_back[backKey]
    except KeyError:
        pass

            
def _cleanup_connections(senderkey, signal):
    """Delete empty signals for ``senderkey``. Delete ``senderkey`` if
    empty."""
    try:
        receivers = connections[senderkey][signal]
    except:
        pass
    else:
        if not receivers:
            # No more connected receivers. Therefore, remove the signal.
            try:
                signals = connections[senderkey]
            except KeyError:
                pass
            else:
                del signals[signal]
                if not signals:
                    # No more signal connections. Therefore, remove the sender.
                    _remove_sender(senderkey)


def _remove_sender(senderkey):
    """Remove ``senderkey`` from connections."""
    _remove_back_refs(senderkey)
    try:
        del connections[senderkey]
    except KeyError:
        pass
    # Senderkey will only be in senders dictionary if sender 
    # could be weakly referenced.
    try:
        del senders[senderkey]
    except:
        pass


def _remove_back_refs(senderkey):
    """Remove all back-references to this ``senderkey``."""
    try:
        signals = connections[senderkey]
    except KeyError:
        signals = None
    else:
        for signal, receivers in signals.iteritems():
            for receiver in receivers:
                _kill_back_ref(receiver, senderkey)


def _remove_old_back_refs(senderkey, signal, receiver, receivers):
    """Kill old ``senders_back`` references from ``receiver``.

    This guards against multiple registration of the same receiver for
    a given signal and sender leaking memory as old back reference
    records build up.

    Also removes old receiver instance from receivers.
    """
    try:
        index = receivers.index(receiver)
        # need to scan back references here and remove senderkey
    except ValueError:
        return False
    else:
        old_receiver = receivers[index]
        del receivers[index]
        found = 0
        signals = connections.get(signal)
        if signals is not None:
            for sig, recs in connections.get(signal, {}).iteritems():
                if sig != signal:
                    for rec in recs:
                        if rec is old_receiver:
                            found = 1
                            break
        if not found:
            _kill_back_ref(old_receiver, senderkey)
            return True
        return False
        
        
def _kill_back_ref(receiver, senderkey):
    """Do actual removal of back reference from ``receiver`` to
    ``senderkey``."""
    receiverkey = id(receiver)
    senders = senders_back.get(receiverkey, ())
    while senderkey in senders:
        try:
            senders.remove(senderkey)
        except:
            break
    if not senders:
        try:
            del senders_back[receiverkey]
        except KeyError:
            pass
    return True

    
