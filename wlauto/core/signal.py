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
from louie import dispatcher  # pylint: disable=F0401


class Signal(object):
    """
    This class implements the signals to be used for notifiying callbacks
    registered to respond to different states and stages of the execution of workload
    automation.

    """

    def __init__(self, name, invert_priority=False):
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


def connect(handler, signal, sender=dispatcher.Any, priority=0):
    """
    Connects a callback to a signal, so that the callback will be automatically invoked
    when that signal is sent.

    Parameters:

        :handler: This can be any callable that that takes the right arguments for
                  the signal. For most siginals this means a single argument that
                  will be an ``ExecutionContext`` instance. But please see documentaion
                  for individual signals in the :ref:`signals reference <instrumentation_method_map>`.
        :signal: The signal to which the hanlder will be subscribed. Please see
                 :ref:`signals reference <instrumentation_method_map>` for the list of standard WA
                 signals.

                 .. note:: There is nothing that prevents instrumentation from sending their
                           own signals that are not part of the standard set. However the signal
                           must always be an :class:`wlauto.core.signal.Signal` instance.

        :sender: The handler will be invoked only for the signals emitted by this sender. By
                 default, this is set to :class:`louie.dispatcher.Any`, so the handler will
                 be invoked for signals from any sentder.
        :priority: An integer (positive or negative) the specifies the priority of the handler.
                   Handlers with higher priority will be called before handlers with lower
                   priority. The  call order of handlers with the same priority is not specified.
                   Defaults to 0.

                   .. note:: Priorities for some signals are inverted (so highest priority
                             handlers get executed last). Please see :ref:`signals reference <instrumentation_method_map>`
                             for details.

    """
    if signal.invert_priority:
        dispatcher.connect(handler, signal, sender, priority=-priority)  # pylint: disable=E1123
    else:
        dispatcher.connect(handler, signal, sender, priority=priority)  # pylint: disable=E1123


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


def send(signal, sender, *args, **kwargs):
    """
    Sends a signal, causing connected handlers to be invoked.

    Paramters:

        :signal: Signal to be sent. This must be an instance of :class:`wlauto.core.signal.Signal`
                 or its subclasses.
        :sender: The sender of the signal (typically, this would be ``self``). Some handlers may only
                 be subscribed to signals from a particular sender.

        The rest of the parameters will be passed on as aruments to the handler.

    """
    dispatcher.send(signal, sender, *args, **kwargs)

