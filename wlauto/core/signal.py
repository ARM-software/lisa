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


"""
This module wraps louie signalling mechanism. It relies on modified version of loiue
that has prioritization added to handler invocation.

"""
import logging
from contextlib import contextmanager

from louie import dispatcher

from wlauto.utils.types import prioritylist


logger = logging.getLogger('dispatcher')


class Signal(object):
    """
    This class implements the signals to be used for notifiying callbacks
    registered to respond to different states and stages of the execution of workload
    automation.

    """

    def __init__(self, name, description='no description', invert_priority=False):
        """
        Instantiates a Signal.

            :param name: name is the identifier of the Signal object. Signal instances with
                        the same name refer to the same execution stage/stage.
            :param invert_priority: boolean parameter that determines whether multiple
                                    callbacks for the same signal should be ordered with
                                    ascending or descending priorities. Typically this flag
                                    should be set to True if the Signal is triggered AFTER an
                                    a state/stage has been reached. That way callbacks with high
                                    priorities will be called right after the event has occured.
        """
        self.name = name
        self.description = description
        self.invert_priority = invert_priority

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __hash__(self):
        return id(self.name)


# These are paired events -- if the before_event is sent, the after_ signal is
# guaranteed to also be sent. In particular, the after_ signals will be sent
# even if there is an error, so you cannot assume in the handler that the
# device has booted successfully. In most cases, you should instead use the
# non-paired signals below.
BEFORE_FLASHING = Signal('before-flashing-signal', invert_priority=True)
SUCCESSFUL_FLASHING = Signal('successful-flashing-signal')
AFTER_FLASHING = Signal('after-flashing-signal')

BEFORE_BOOT = Signal('before-boot-signal', invert_priority=True)
SUCCESSFUL_BOOT = Signal('successful-boot-signal')
AFTER_BOOT = Signal('after-boot-signal')

BEFORE_INITIAL_BOOT = Signal('before-initial-boot-signal', invert_priority=True)
SUCCESSFUL_INITIAL_BOOT = Signal('successful-initial-boot-signal')
AFTER_INITIAL_BOOT = Signal('after-initial-boot-signal')

BEFORE_FIRST_ITERATION_BOOT = Signal('before-first-iteration-boot-signal', invert_priority=True)
SUCCESSFUL_FIRST_ITERATION_BOOT = Signal('successful-first-iteration-boot-signal')
AFTER_FIRST_ITERATION_BOOT = Signal('after-first-iteration-boot-signal')

BEFORE_WORKLOAD_SETUP = Signal('before-workload-setup-signal', invert_priority=True)
SUCCESSFUL_WORKLOAD_SETUP = Signal('successful-workload-setup-signal')
AFTER_WORKLOAD_SETUP = Signal('after-workload-setup-signal')

BEFORE_WORKLOAD_EXECUTION = Signal('before-workload-execution-signal', invert_priority=True)
SUCCESSFUL_WORKLOAD_EXECUTION = Signal('successful-workload-execution-signal')
AFTER_WORKLOAD_EXECUTION = Signal('after-workload-execution-signal')

BEFORE_WORKLOAD_RESULT_UPDATE = Signal('before-iteration-result-update-signal', invert_priority=True)
SUCCESSFUL_WORKLOAD_RESULT_UPDATE = Signal('successful-iteration-result-update-signal')
AFTER_WORKLOAD_RESULT_UPDATE = Signal('after-iteration-result-update-signal')

BEFORE_WORKLOAD_TEARDOWN = Signal('before-workload-teardown-signal', invert_priority=True)
SUCCESSFUL_WORKLOAD_TEARDOWN = Signal('successful-workload-teardown-signal')
AFTER_WORKLOAD_TEARDOWN = Signal('after-workload-teardown-signal')

BEFORE_OVERALL_RESULTS_PROCESSING = Signal('before-overall-results-process-signal', invert_priority=True)
SUCCESSFUL_OVERALL_RESULTS_PROCESSING = Signal('successful-overall-results-process-signal')
AFTER_OVERALL_RESULTS_PROCESSING = Signal('after-overall-results-process-signal')

# These are the not-paired signals; they are emitted independently. E.g. the
# fact that RUN_START was emitted does not mean run end will be.
RUN_START = Signal('start-signal', invert_priority=True)
RUN_END = Signal('end-signal')
WORKLOAD_SPEC_START = Signal('workload-spec-start-signal', invert_priority=True)
WORKLOAD_SPEC_END = Signal('workload-spec-end-signal')
ITERATION_START = Signal('iteration-start-signal', invert_priority=True)
ITERATION_END = Signal('iteration-end-signal')

RUN_INIT = Signal('run-init-signal')
SPEC_INIT = Signal('spec-init-signal')
ITERATION_INIT = Signal('iteration-init-signal')

RUN_FIN = Signal('run-fin-signal')

# These signals are used by the LoggerFilter to tell about logging events
ERROR_LOGGED = Signal('error_logged')
WARNING_LOGGED = Signal('warning_logged')


class CallbackPriority(object):

    EXTREMELY_HIGH = 30
    VERY_HIGH = 20
    HIGH = 10
    NORMAL = 0
    LOW = -10
    VERY_LOW = -20
    EXTREMELY_LOW = -30

    def __init__(self):
        raise ValueError('Cannot instantiate')


class _prioritylist_wrapper(prioritylist):
    """
    This adds a NOP append() method so that when louie invokes it to add the
    handler to receivers, nothing will happen; the handler is actually added inside
    the connect() below according to priority, before louie's connect() gets invoked.

    """

    def append(self, *args, **kwargs):
        pass


def connect(handler, signal, sender=dispatcher.Any, priority=0):
    """
    Connects a callback to a signal, so that the callback will be automatically invoked
    when that signal is sent.

    Parameters:

        :handler: This can be any callable that that takes the right arguments for
                  the signal. For most signals this means a single argument that
                  will be an ``ExecutionContext`` instance. But please see documentation
                  for individual signals in the :ref:`signals reference <instrumentation_method_map>`.
        :signal: The signal to which the handler will be subscribed. Please see
                 :ref:`signals reference <instrumentation_method_map>` for the list of standard WA
                 signals.

                 .. note:: There is nothing that prevents instrumentation from sending their
                           own signals that are not part of the standard set. However the signal
                           must always be an :class:`wlauto.core.signal.Signal` instance.

        :sender: The handler will be invoked only for the signals emitted by this sender. By
                 default, this is set to :class:`louie.dispatcher.Any`, so the handler will
                 be invoked for signals from any sender.
        :priority: An integer (positive or negative) the specifies the priority of the handler.
                   Handlers with higher priority will be called before handlers with lower
                   priority. The  call order of handlers with the same priority is not specified.
                   Defaults to 0.

                   .. note:: Priorities for some signals are inverted (so highest priority
                             handlers get executed last). Please see :ref:`signals reference <instrumentation_method_map>`
                             for details.

    """
    if getattr(signal, 'invert_priority', False):
        priority = -priority
    senderkey = id(sender)
    if senderkey in dispatcher.connections:
        signals = dispatcher.connections[senderkey]
    else:
        dispatcher.connections[senderkey] = signals = {}
    if signal in signals:
        receivers = signals[signal]
    else:
        receivers = signals[signal] = _prioritylist_wrapper()
    receivers.add(handler, priority)
    dispatcher.connect(handler, signal, sender)


def disconnect(handler, signal, sender=dispatcher.Any):
    """
    Disconnect a previously connected handler form the specified signal, optionally, only
    for the specified sender.

    Parameters:

        :handler: The callback to be disconnected.
        :signal: The signal the handler is to be disconnected form. It will
                 be an :class:`wlauto.core.signal.Signal` instance.
        :sender: If specified, the handler will only be disconnected from the signal
                sent by this sender.

    """
    dispatcher.disconnect(handler, signal, sender)


def send(signal, sender=dispatcher.Anonymous, *args, **kwargs):
    """
    Sends a signal, causing connected handlers to be invoked.

    Paramters:

        :signal: Signal to be sent. This must be an instance of :class:`wlauto.core.signal.Signal`
                 or its subclasses.
        :sender: The sender of the signal (typically, this would be ``self``). Some handlers may only
                 be subscribed to signals from a particular sender.

        The rest of the parameters will be passed on as aruments to the handler.

    """
    return dispatcher.send(signal, sender, *args, **kwargs)


# This will normally be set to log_error() by init_logging(); see wa.framework/log.py.
# Done this way to prevent a circular import dependency.
log_error_func = logger.error


def safe_send(signal, sender=dispatcher.Anonymous,
              propagate=[KeyboardInterrupt], *args, **kwargs):
    """
    Same as ``send``, except this will catch and log all exceptions raised
    by handlers, except those specified in ``propagate`` argument (defaults
    to just ``[KeyboardInterrupt]``).
    """
    try:
        send(singnal, sender, *args, **kwargs)
    except Exception as e:
        if any(isinstance(e, p) for p in propagate):
            raise e
        log_error_func(e)


@contextmanager
def wrap(signal_name, sender=dispatcher.Anonymous, safe=False, *args, **kwargs):
    """Wraps the suite in before/after signals, ensuring
    that after signal is always sent."""
    signal_name = signal_name.upper().replace('-', '_')
    send_func = safe_send if safe else send
    try:
        before_signal = globals()['BEFORE_' + signal_name]
        success_signal = globals()['SUCCESSFUL_' + signal_name]
        after_signal = globals()['AFTER_' + signal_name]
    except KeyError:
        raise ValueError('Invalid wrapped signal name: {}'.format(signal_name))
    try:
        send_func(before_signal, sender, *args, **kwargs)
        yield
        send_func(success_signal, sender, *args, **kwargs)
    finally:
        send_func(after_signal, sender, *args, **kwargs)
