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

import wrapt
from louie import dispatcher

from wa.utils.types import prioritylist


logger = logging.getLogger('signal')


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
                                    callbacks for the same signal should be
                                    ordered with ascending or descending
                                    priorities. Typically this flag should be
                                    set to True if the Signal is triggered
                                    AFTER an a state/stage has been reached.
                                    That way callbacks with high priorities
                                    will be called right after the event has
                                    occured.
        """
        self.name = name
        self.description = description
        self.invert_priority = invert_priority

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __hash__(self):
        return id(self.name)


# Signals associated with run-related events
RUN_STARTED = Signal('run-started', 'sent at the beginning of the run')
RUN_INITIALIZED = Signal('run-initialized', 'set after the run has been initialized')
RUN_ABORTED = Signal('run-aborted', 'set when the run has been aborted due to a keyboard interrupt')
RUN_FAILED = Signal('run-failed', 'set if the run has failed to complete all jobs.' )
RUN_FINALIZED = Signal('run-finalized', 'set after the run has been finalized')
RUN_COMPLETED = Signal('run-completed', 'set upon completion of the run (regardless of whether or not it has failed')


# Signals associated with job-related events
JOB_STARTED = Signal('job-started', 'set when a a new job has been started')
JOB_ABORTED = Signal('job-aborted',
                     description='''
                     sent if a job has been aborted due to a keyboard interrupt.

                     .. note:: While the status of every job that has not had a
                               chance to run due to being interrupted will be
                               set to "ABORTED", this signal will only be sent
                               for the job that was actually running at the
                               time.

                     ''')
JOB_FAILED = Signal('job-failed', description='set if the job has failed')
JOB_RESTARTED = Signal('job-restarted')
JOB_COMPLETED = Signal('job-completed')
JOB_FINALIZED = Signal('job-finalized')


# Signals associated with particular stages of workload execution
BEFORE_WORKLOAD_INITIALIZED = Signal('before-workload-initialized',
                                     invert_priority=True)
SUCCESSFUL_WORKLOAD_INITIALIZED = Signal('successful-workload-initialized')
AFTER_WORKLOAD_INITIALIZED = Signal('after-workload-initialized')

BEFORE_WORKLOAD_SETUP = Signal('before-workload-setup', invert_priority=True)
SUCCESSFUL_WORKLOAD_SETUP = Signal('successful-workload-setup')
AFTER_WORKLOAD_SETUP = Signal('after-workload-setup')

BEFORE_WORKLOAD_EXECUTION = Signal('before-workload-execution', invert_priority=True)
SUCCESSFUL_WORKLOAD_EXECUTION = Signal('successful-workload-execution')
AFTER_WORKLOAD_EXECUTION = Signal('after-workload-execution')

BEFORE_WORKLOAD_RESULT_EXTRACTION = Signal('before-workload-result-extracton',
                                       invert_priority=True)
SUCCESSFUL_WORKLOAD_RESULT_EXTRACTION = Signal('successful-workload-result-extracton')
AFTER_WORKLOAD_RESULT_EXTRACTION = Signal('after-workload-result-extracton')

BEFORE_WORKLOAD_OUTPUT_UPDATE = Signal('before-workload-output-update',
                                       invert_priority=True)
SUCCESSFUL_WORKLOAD_OUTPUT_UPDATE = Signal('successful-workload-output-update')
AFTER_WORKLOAD_OUTPUT_UPDATE = Signal('after-workload-output-update')

BEFORE_WORKLOAD_TEARDOWN = Signal('before-workload-teardown', invert_priority=True)
SUCCESSFUL_WORKLOAD_TEARDOWN = Signal('successful-workload-teardown')
AFTER_WORKLOAD_TEARDOWN = Signal('after-workload-teardown')

BEFORE_WORKLOAD_FINALIZED = Signal('before-workload-finalized', invert_priority=True)
SUCCESSFUL_WORKLOAD_FINALIZED = Signal('successful-workload-finalized')
AFTER_WORKLOAD_FINALIZED = Signal('after-workload-finalized')


# Signals indicating exceptional conditions
ERROR_LOGGED = Signal('error-logged')
WARNING_LOGGED = Signal('warning-logged')

# These are paired events -- if the before_event is sent, the after_ signal is
# guaranteed to also be sent. In particular, the after_ signals will be sent
# even if there is an error, so you cannot assume in the handler that the
# device has booted successfully. In most cases, you should instead use the
# non-paired signals below.
BEFORE_RUN_INIT = Signal('before-run-init', invert_priority=True)
SUCCESSFUL_RUN_INIT = Signal('successful-run-init')
AFTER_RUN_INIT = Signal('after-run-init')

BEFORE_JOB_TARGET_CONFIG = Signal('before-job-target-config', invert_priority=True)
SUCCESSFUL_JOB_TARGET_CONFIG = Signal('successful-job-target-config')
AFTER_JOB_TARGET_CONFIG = Signal('after-job-target-config')

BEFORE_JOB_SETUP = Signal('before-job-setup', invert_priority=True)
SUCCESSFUL_JOB_SETUP = Signal('successful-job-setup')
AFTER_JOB_SETUP = Signal('after-job-setup')

BEFORE_JOB_EXECUTION = Signal('before-job-execution', invert_priority=True)
SUCCESSFUL_JOB_EXECUTION = Signal('successful-job-execution')
AFTER_JOB_EXECUTION = Signal('after-job-execution')

BEFORE_JOB_OUTPUT_PROCESSED = Signal('before-job-output-processed',
                                     invert_priority=True)
SUCCESSFUL_JOB_OUTPUT_PROCESSED = Signal('successful-job-output-processed')
AFTER_JOB_OUTPUT_PROCESSED = Signal('after-job-output-processed')

BEFORE_JOB_TEARDOWN = Signal('before-job-teardown', invert_priority=True)
SUCCESSFUL_JOB_TEARDOWN = Signal('successful-job-teardown')
AFTER_JOB_TEARDOWN = Signal('after-job-teardown')

BEFORE_FLASHING = Signal('before-flashing', invert_priority=True)
SUCCESSFUL_FLASHING = Signal('successful-flashing')
AFTER_FLASHING = Signal('after-flashing')

BEFORE_BOOT = Signal('before-boot', invert_priority=True)
SUCCESSFUL_BOOT = Signal('successful-boot')
AFTER_BOOT = Signal('after-boot')

BEFORE_TARGET_CONNECT = Signal('before-target-connect', invert_priority=True)
SUCCESSFUL_TARGET_CONNECT = Signal('successful-target-connect')
AFTER_TARGET_CONNECT = Signal('after-target-connect')

BEFORE_TARGET_DISCONNECT = Signal('before-target-disconnect', invert_priority=True)
SUCCESSFUL_TARGET_DISCONNECT = Signal('successful-target-disconnect')
AFTER_TARGET_DISCONNECT = Signal('after-target-disconnect')


BEFORE_OVERALL_RESULTS_PROCESSING = Signal(
    'before-overall-results-process', invert_priority=True)
SUCCESSFUL_OVERALL_RESULTS_PROCESSING = Signal(
    'successful-overall-results-process')
AFTER_OVERALL_RESULTS_PROCESSING = Signal(
    'after-overall-results-process')


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
                  for individual signals in the :ref:`signals reference <instruments_method_map>`.
        :signal: The signal to which the handler will be subscribed. Please see
                 :ref:`signals reference <instruments_method_map>` for the list of standard WA
                 signals.

                 .. note:: There is nothing that prevents instruments from sending their
                           own signals that are not part of the standard set. However the signal
                           must always be an :class:`wa.core.signal.Signal` instance.

        :sender: The handler will be invoked only for the signals emitted by this sender. By
                 default, this is set to :class:`louie.dispatcher.Any`, so the handler will
                 be invoked for signals from any sender.
        :priority: An integer (positive or negative) the specifies the priority of the handler.
                   Handlers with higher priority will be called before handlers with lower
                   priority. The  call order of handlers with the same priority is not specified.
                   Defaults to 0.

                   .. note:: Priorities for some signals are inverted (so highest priority
                             handlers get executed last). Please see :ref:`signals reference <instruments_method_map>`
                             for details.

    """
    logger.debug('Connecting {} to {}({}) with priority {}'.format(handler, signal, sender, priority))
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
                 be an :class:`wa.core.signal.Signal` instance.
        :sender: If specified, the handler will only be disconnected from the signal
                sent by this sender.

    """
    logger.debug('Disconnecting {} from {}({})'.format(handler, signal, sender))
    dispatcher.disconnect(handler, signal, sender)


def send(signal, sender=dispatcher.Anonymous, *args, **kwargs):
    """
    Sends a signal, causing connected handlers to be invoked.

    Paramters:

        :signal: Signal to be sent. This must be an instance of :class:`wa.core.signal.Signal`
                 or its subclasses.
        :sender: The sender of the signal (typically, this would be ``self``). Some handlers may only
                 be subscribed to signals from a particular sender.

        The rest of the parameters will be passed on as aruments to the handler.

    """
    logger.debug('Sending {} from {}'.format(signal, sender))
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
        logger.debug('Safe-sending {} from {}'.format(signal, sender))
        send(signal, sender, *args, **kwargs)
    except Exception as e:
        if any(isinstance(e, p) for p in propagate):
            raise e
        log_error_func(e)


@contextmanager
def wrap(signal_name, sender=dispatcher.Anonymous,*args, **kwargs):
    """Wraps the suite in before/after signals, ensuring
    that after signal is always sent."""
    safe = kwargs.pop('safe', False)
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


def wrapped(signal_name, sender=dispatcher.Anonymous, safe=False):
    """A decorator for wrapping function in signal dispatch."""
    @wrapt.decorator
    def signal_wrapped(wrapped, instance, args, kwargs):
        def signal_wrapper(*args, **kwargs):
            with wrap(signal_name, sender, safe):
                return wrapped(*args, **kwargs)

        return signal_wrapper(*args, **kwargs)

    return signal_wrapped
