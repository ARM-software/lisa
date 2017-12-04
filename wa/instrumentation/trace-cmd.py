#    Copyright 2013-2017 ARM Limited
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


# pylint: disable=W0613,E1101
from __future__ import division
import os

from devlib import FtraceCollector

from wa import Instrument, Parameter
from wa.framework import signal
from wa.framework.instrumentation import very_slow
from wa.framework.exception import InstrumentError
from wa.utils.types import list_of_strings
from wa.utils.misc import which


OUTPUT_TRACE_FILE = 'trace.dat'
OUTPUT_TEXT_FILE = '{}.txt'.format(os.path.splitext(OUTPUT_TRACE_FILE)[0])
TIMEOUT = 180


class TraceCmdInstrument(Instrument):

    name = 'trace-cmd'
    description = """
    trace-cmd is an instrument which interacts with ftrace Linux kernel internal
    tracer

    From trace-cmd man page:

    trace-cmd command interacts with the ftrace tracer that is built inside the
    Linux kernel. It interfaces with the ftrace specific files found in the
    debugfs file system under the tracing directory.

    trace-cmd reads a list of events it will trace, which can be specified in
    the config file as follows ::

        trace_events = ['irq*', 'power*']

    If no event is specified, a default set of events that are generally considered useful
    for debugging/profiling purposes will be enabled.

    The list of available events can be obtained by rooting and running the
    following command line on the device ::

       trace-cmd list

    You may also specify ``trace_buffer_size`` setting which must be an integer
    that will be used to set the ftrace buffer size. It will be interpreted as
    KB::

        trace_cmd_buffer_size = 8000

    The maximum buffer size varies from device to device, but there is a
    maximum and trying to set buffer size beyond that will fail. If you plan
    on collecting a lot of trace over long periods of time, the buffer size
    will not be enough and you will only get trace for the last portion of your
    run. To deal with this you can set the ``trace_mode`` setting to
    ``'record'`` (the default is ``'start'``)::

        trace_cmd_mode = 'record'

    This will cause trace-cmd to trace into file(s) on disk, rather than the
    buffer, and so the limit for the max size of the trace is set by the
    storage available on device. Bear in mind that ``'record'`` mode *is* more
    intrusive than the default, so if you do not plan on generating a lot of
    trace, it is best to use the default ``'start'`` mode.

    .. note:: Mode names correspond to the underlying trace-cmd executable's
              command used to implement them. You can find out more about what
              is happening in each case from trace-cmd documentation:
              https://lwn.net/Articles/341902/.

    This instrument comes with an trace-cmd binary that will be copied and used
    on the device, however post-processing will be, by default, done on-host and you must
    have trace-cmd installed and in your path. On Ubuntu systems, this may be
    done with::

        sudo apt-get install trace-cmd

    Alternatively, you may set ``report_on_target`` parameter to ``True`` to enable on-target
    processing (this is useful when running on non-Linux hosts, but is likely to take longer
    and may fail on particularly resource-constrained targets).

    """

    parameters = [
        Parameter('events', kind=list_of_strings,
                  default=['sched*', 'irq*', 'power*', 'thermal*'],
                  global_alias='trace_events',
                  description="""
                  Specifies the list of events to be traced. Each event in the
                  list will be passed to trace-cmd with -e parameter and must
                  be in the format accepted by trace-cmd.
                  """),
        Parameter('functions', kind=list_of_strings,
                  global_alias='trace_functions',
                  description="""
                  Specifies the list of functions to be traced.
                  """),
        Parameter('buffer_size', kind=int, default=None,
                  global_alias='trace_buffer_size',
                  description="""
                  Attempt to set ftrace buffer size to the specified value (in
                  KB). Default buffer size may need to be increased for
                  long-running workloads, or if a large number of events have
                  been enabled. Note: there is a maximum size that the buffer
                  can be set, and that varies from device to device. Attempting
                  to set buffer size higher than this will fail. In that case,
                  this instrument will set the size to the highest possible
                  value by going down from the specified size in
                  ``buffer_size_step`` intervals.
                  """),
        Parameter('buffer_size_step', kind=int, default=1000,
                  global_alias='trace_buffer_size_step',
                  description="""
                  Defines the decremental step used if the specified
                  ``buffer_size`` could not be set.  This will be subtracted
                  form the buffer size until set succeeds or size is reduced to
                  1MB.
                  """),
        Parameter('report', kind=bool, default=True,
                  description="""
                  Specifies whether reporting should be performed once the
                  binary trace has been generated.
                  """),
        Parameter('no_install', kind=bool, default=False,
                  description="""
                  Do not install the bundled trace-cmd  and use the one on the
                  device instead. If there is not already a trace-cmd on the
                  device, an error is raised.
                  """),
        Parameter('report_on_target', kind=bool, default=False,
                  description="""
                  When enabled generation of reports will be done host-side
                  because the generated file is very large. If trace-cmd is not
                  available on the host device this setting can be disabled and
                  the report will be generated on the target device.

                  .. note:: This requires the latest version of trace-cmd to be
                            installed on the host (the one in your
                            distribution's repos may be too old).
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(TraceCmdInstrument, self).__init__(target, **kwargs)
        self.collector = None

    def initialize(self, context):
        if not self.target.is_rooted:
            raise InstrumentError('trace-cmd instrument cannot be used on an unrooted device.')
        collector_params = dict(
                 events=self.events,
                 functions=self.functions,
                 buffer_size=self.buffer_size,
                 buffer_size_step=1000,
                 automark=False,
                 autoreport=True,
                 autoview=False,
                 no_install=self.no_install,
                 strict=False,
                 report_on_target=False,
        )
        if self.report and self.report_on_target:
            collector_params['autoreport'] = True
            collector_params['report_on_target'] = True
        else:
            collector_params['autoreport'] = False
            collector_params['report_on_target'] = False
        self.collector = FtraceCollector(self.target, **collector_params)

        # Register ourselves as absolute last event before and
        #   first after so we can mark the trace at the right time
        signal.connect(self.mark_start, signal.BEFORE_WORKLOAD_EXECUTION, priority=11)
        signal.connect(self.mark_stop, signal.AFTER_WORKLOAD_EXECUTION, priority=11)

    def setup(self, context):
        self.collector.reset()

    @very_slow
    def start(self, context):
        self.collector.start()

    @very_slow
    def stop(self, context):
        self.collector.stop()

    def update_result(self, context):  # NOQA pylint: disable=R0912
        outfile = os.path.join(context.output_directory, 'trace.dat')
        self.collector.get_trace(outfile)
        context.add_artifact('trace-cmd-bin', outfile, 'data')
        if self.report:
            if not self.report_on_target:
                textfile = os.path.join(context.output_directory, 'trace.txt')
                self.collector.report(outfile, textfile)
            context.add_artifact('trace-cmd-txt', textfile, 'export')

    def teardown(self, context):
        path = self.target.path.join(self.target.working_directory, OUTPUT_TRACE_FILE)
        self.target.remove(path)
        if self.report_on_target:
            path = self.target.path.join(self.target.working_directory, OUTPUT_TEXT_FILE)
            self.target.remove(path)

    def validate(self):
        if self.report and not self.report_on_target and not which('trace-cmd'):
            raise InstrumentError('trace-cmd is not in PATH; is it installed?')

    def mark_start(self, context):
        self.collector.mark_start()

    def mark_stop(self, context):
        self.collector.mark_stop()
