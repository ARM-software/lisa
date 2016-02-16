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


# pylint: disable=W0613,E1101
from __future__ import division
import os
import time
import subprocess
from collections import defaultdict

from wlauto import Instrument, Parameter, Executable
from wlauto.exceptions import InstrumentError, ConfigError, DeviceError
from wlauto.core import signal
from wlauto.utils.types import boolean

OUTPUT_TRACE_FILE = 'trace.dat'
OUTPUT_TEXT_FILE = '{}.txt'.format(os.path.splitext(OUTPUT_TRACE_FILE)[0])
TIMEOUT = 180


class TraceCmdInstrument(Instrument):

    name = 'trace-cmd'
    description = """
    trace-cmd is an instrument which interacts with Ftrace Linux kernel internal
    tracer

    From trace-cmd man page:

    trace-cmd command interacts with the Ftrace tracer that is built inside the
    Linux kernel. It interfaces with the Ftrace specific files found in the
    debugfs file system under the tracing directory.

    trace-cmd reads a list of events it will trace, which can be specified in
    the config file as follows ::

        trace_events = ['irq*', 'power*']

    If no event is specified in the config file, trace-cmd traces the following events:

        - sched*
        - irq*
        - power*
        - cpufreq_interactive*

    The list of available events can be obtained by rooting and running the following
    command line on the device ::

       trace-cmd list

    You may also specify ``trace_buffer_size`` setting which must be an integer that will
    be used to set the ftrace buffer size. It will be interpreted as KB::

        trace_cmd_buffer_size = 8000

    The maximum buffer size varies from device to device, but there is a maximum and trying
    to set buffer size beyound that will fail. If you plan on collecting a lot of trace over
    long periods of time, the buffer size will not be enough and you will only get trace for
    the last portion of your run. To deal with this you can set the ``trace_mode`` setting to
    ``'record'`` (the default is ``'start'``)::

        trace_cmd_mode = 'record'

    This will cause trace-cmd to trace into file(s) on disk, rather than the buffer, and so the
    limit for the max size of the trace is set by the storage available on device. Bear in mind
    that ``'record'`` mode *is* more instrusive than the default, so if you do not plan on
    generating a lot of trace, it is best to use the default ``'start'`` mode.

    .. note:: Mode names correspend to the underlying trace-cmd exectuable's command used to
              implement them. You can find out more about what is happening in each case from
              trace-cmd documentation: https://lwn.net/Articles/341902/.

    This instrument comes with an Android trace-cmd binary that will be copied and used on the
    device, however post-processing will be done on-host and you must have trace-cmd installed and
    in your path. On Ubuntu systems, this may be done with::

        sudo apt-get install trace-cmd

    """

    parameters = [
        Parameter('events', kind=list, default=['sched*', 'irq*', 'power*', 'cpufreq_interactive*'],
                  global_alias='trace_events',
                  description="""
                  Specifies the list of events to be traced. Each event in the list will be passed to
                  trace-cmd with -e parameter and must be in the format accepted by trace-cmd.
                  """),
        Parameter('mode', default='start', allowed_values=['start', 'record'],
                  global_alias='trace_mode',
                  description="""
                  Trace can be collected using either 'start' or 'record' trace-cmd
                  commands. In 'start' mode, trace will be collected into the ftrace buffer;
                  in 'record' mode, trace will be written into a file on the device's file
                  system. 'start' mode is (in theory) less intrusive than 'record' mode, however
                  it is limited by the size of the ftrace buffer (which is configurable --
                  see ``buffer_size`` -- but only up to a point) and that may overflow
                  for long-running workloads, which will result in dropped events.
                  """),
        Parameter('buffer_size', kind=int, default=None,
                  global_alias='trace_buffer_size',
                  description="""
                  Attempt to set ftrace buffer size to the specified value (in KB). Default buffer size
                  may need to be increased for long-running workloads, or if a large number
                  of events have been enabled. Note: there is a maximum size that the buffer can
                  be set, and that varies from device to device. Attempting to set buffer size higher
                  than this will fail. In that case, this instrument will set the size to the highest
                  possible value by going down from the specified size in ``buffer_size_step`` intervals.
                  """),
        Parameter('buffer_size_step', kind=int, default=1000,
                  global_alias='trace_buffer_size_step',
                  description="""
                  Defines the decremental step used if the specified ``buffer_size`` could not be set.
                  This will be subtracted form the buffer size until set succeeds or size is reduced to
                  1MB.
                  """),
        Parameter('buffer_size_file', default='/sys/kernel/debug/tracing/buffer_size_kb',
                  description="""
                  Path to the debugs file that may be used to set ftrace buffer size. This should need
                  to be modified for the vast majority devices.
                  """),
        Parameter('report', kind=boolean, default=True,
                  description="""
                  Specifies whether reporting should be performed once the binary trace has been generated.
                  """),
        Parameter('no_install', kind=boolean, default=False,
                  description="""
                  Do not install the bundled trace-cmd  and use the one on the device instead. If there is
                  not already a trace-cmd on the device, an error is raised.

                  """),
        Parameter('report_on_target', kind=boolean, default=False,
                  description="""
                  When enabled generation of reports will be done host-side because the generated file is
                  very large. If trace-cmd is not available on the host device this setting and be disabled
                  and the report will be generated on the target device.

                  .. note:: This requires the latest version of trace-cmd to be installed on the host (the
                            one in your distribution's repos may be too old).
                  """),
    ]

    def __init__(self, device, **kwargs):
        super(TraceCmdInstrument, self).__init__(device, **kwargs)
        self.trace_cmd = None
        self.event_string = _build_trace_events(self.events)
        self.output_file = os.path.join(self.device.working_directory, OUTPUT_TRACE_FILE)
        self.temp_trace_file = self.device.path.join(self.device.working_directory, OUTPUT_TRACE_FILE)

    def on_run_init(self, context):
        if not self.device.is_rooted:
            raise InstrumentError('trace-cmd instrument cannot be used on an unrooted device.')
        if not self.no_install:
            host_file = context.resolver.get(Executable(self, self.device.abi, 'trace-cmd'))
            self.trace_cmd = self.device.install(host_file)
        else:
            self.trace_cmd = self.device.get_installed("trace-cmd")
            if not self.trace_cmd:
                raise ConfigError('No trace-cmd found on device and no_install=True is specified.')

        # Register ourselves as absolute last event before and
        #   first after so we can mark the trace at the right time
        signal.connect(self.insert_start_mark, signal.BEFORE_WORKLOAD_EXECUTION, priority=11)
        signal.connect(self.insert_end_mark, signal.AFTER_WORKLOAD_EXECUTION, priority=11)

    def setup(self, context):
        if self.mode == 'start':
            if self.buffer_size:
                self._set_buffer_size()
            self.device.execute('{} reset'.format(self.trace_cmd), as_root=True, timeout=180)
        elif self.mode == 'record':
            pass
        else:
            raise ValueError('Bad mode: {}'.format(self.mode))  # should never get here

    def very_slow_start(self, context):
        self.start_time = time.time()  # pylint: disable=attribute-defined-outside-init
        if self.mode == 'start':
            self.device.execute('{} start {}'.format(self.trace_cmd, self.event_string), as_root=True)
        elif self.mode == 'record':
            self.device.kick_off('{} record -o {} {}'.format(self.trace_cmd, self.output_file, self.event_string))
        else:
            raise ValueError('Bad mode: {}'.format(self.mode))  # should never get here

    def stop(self, context):
        self.stop_time = time.time()  # pylint: disable=attribute-defined-outside-init
        if self.mode == 'start':
            self.device.execute('{} stop'.format(self.trace_cmd), timeout=60, as_root=True)
        elif self.mode == 'record':
            # There will be a trace-cmd worker process per CPU core plus a main
            # control trace-cmd process. Interrupting the control process will
            # trigger the generation of the single binary trace file.
            trace_cmds = self.device.ps(name=self.trace_cmd)
            if not trace_cmds:
                raise InstrumentError('Could not find running trace-cmd on device.')
            # The workers will have their PPID set to the PID of control.
            parent_map = defaultdict(list)
            for entry in trace_cmds:
                parent_map[entry.ppid].append(entry.pid)
            controls = [v[0] for _, v in parent_map.iteritems()
                        if len(v) == 1 and v[0] in parent_map]
            if len(controls) > 1:
                self.logger.warning('More than one trace-cmd instance found; stopping all of them.')
            for c in controls:
                self.device.kill(c, signal='INT', as_root=True)
        else:
            raise ValueError('Bad mode: {}'.format(self.mode))  # should never get here

    def update_result(self, context):  # NOQA pylint: disable=R0912
        if self.mode == 'start':
            self.device.execute('{} extract -o {}'.format(self.trace_cmd, self.output_file),
                                timeout=TIMEOUT, as_root=True)
        elif self.mode == 'record':
            self.logger.debug('Waiting for trace.dat to be generated.')
            while self.device.ps(name=self.trace_cmd):
                time.sleep(2)
        else:
            raise ValueError('Bad mode: {}'.format(self.mode))  # should never get here

        # The size of trace.dat will depend on how long trace-cmd was running.
        # Therefore timout for the pull command must also be adjusted
        # accordingly.
        self._pull_timeout = (self.stop_time - self.start_time)  # pylint: disable=attribute-defined-outside-init
        self.device.pull(self.output_file, context.output_directory, timeout=self._pull_timeout)
        context.add_iteration_artifact('bintrace', OUTPUT_TRACE_FILE, kind='data',
                                       description='trace-cmd generated ftrace dump.')

        local_txt_trace_file = os.path.join(context.output_directory, OUTPUT_TEXT_FILE)

        if self.report:
            # To get the output of trace.dat, trace-cmd must be installed
            # By default this is done host-side because the generated file is
            # very large
            if self.report_on_target:
                self._generate_report_on_target(context)
            else:
                self._generate_report_on_host(context)

            if os.path.isfile(local_txt_trace_file):
                context.add_iteration_artifact('txttrace', OUTPUT_TEXT_FILE, kind='export',
                                               description='trace-cmd generated ftrace dump.')
                self.logger.debug('Verifying traces.')
                with open(local_txt_trace_file) as fh:
                    for line in fh:
                        if 'EVENTS DROPPED' in line:
                            self.logger.warning('Dropped events detected.')
                            break
                    else:
                        self.logger.debug('Trace verified.')
            else:
                self.logger.warning('Could not generate trace.txt.')

    def teardown(self, context):
        self.device.remove(os.path.join(self.device.working_directory, OUTPUT_TRACE_FILE))

    def on_run_end(self, context):
        pass

    def validate(self):
        if self.report and not self.report_on_target and os.system('which trace-cmd > /dev/null'):
            raise InstrumentError('trace-cmd is not in PATH; is it installed?')
        if self.buffer_size:
            if self.mode == 'record':
                self.logger.debug('trace_buffer_size specified with record mode; it will be ignored.')
            else:
                try:
                    int(self.buffer_size)
                except ValueError:
                    raise ConfigError('trace_buffer_size must be an int.')

    def insert_start_mark(self, context):
        # trace marker appears in ftrace as an ftrace/print event with TRACE_MARKER_START in info field
        self.device.write_value("/sys/kernel/debug/tracing/trace_marker", "TRACE_MARKER_START", verify=False)

    def insert_end_mark(self, context):
        # trace marker appears in ftrace as an ftrace/print event with TRACE_MARKER_STOP in info field
        self.device.write_value("/sys/kernel/debug/tracing/trace_marker", "TRACE_MARKER_STOP", verify=False)

    def _set_buffer_size(self):
        target_buffer_size = self.buffer_size
        attempt_buffer_size = target_buffer_size
        buffer_size = 0
        floor = 1000 if target_buffer_size > 1000 else target_buffer_size
        while attempt_buffer_size >= floor:
            self.device.write_value(self.buffer_size_file, attempt_buffer_size, verify=False)
            buffer_size = self.device.get_sysfile_value(self.buffer_size_file, kind=int)
            if buffer_size == attempt_buffer_size:
                break
            else:
                attempt_buffer_size -= self.buffer_size_step
        if buffer_size == target_buffer_size:
            return
        while attempt_buffer_size < target_buffer_size:
            attempt_buffer_size += self.buffer_size_step
            self.device.write_value(self.buffer_size_file, attempt_buffer_size, verify=False)
            buffer_size = self.device.get_sysfile_value(self.buffer_size_file, kind=int)
            if attempt_buffer_size != buffer_size:
                self.logger.warning('Failed to set trace buffer size to {}, value set was {}'.format(target_buffer_size, buffer_size))
                break

    def _generate_report_on_target(self, context):
        try:
            trace_file = self.output_file
            txt_trace_file = os.path.join(self.device.working_directory, OUTPUT_TEXT_FILE)
            command = 'trace-cmd report {} > {}'.format(trace_file, txt_trace_file)
            self.device.execute(command)
            self.device.pull(txt_trace_file, context.output_directory, timeout=self._pull_timeout)
        except DeviceError:
            raise InstrumentError('Could not generate TXT report on target.')

    def _generate_report_on_host(self, context):
        local_trace_file = os.path.join(context.output_directory, OUTPUT_TRACE_FILE)
        local_txt_trace_file = os.path.join(context.output_directory, OUTPUT_TEXT_FILE)
        command = 'trace-cmd report {} > {}'.format(local_trace_file, local_txt_trace_file)
        self.logger.debug(command)
        if not os.path.isfile(local_trace_file):
            self.logger.warning('Not generating trace.txt, as {} does not exist.'.format(OUTPUT_TRACE_FILE))
        else:
            try:
                process = subprocess.Popen(command, stderr=subprocess.PIPE, shell=True)
                _, error = process.communicate()
                if process.returncode:
                    raise InstrumentError('trace-cmd returned non-zero exit code {}'.format(process.returncode))
                if error:
                    # logged at debug level, as trace-cmd always outputs some
                    # errors that seem benign.
                    self.logger.debug(error)
            except OSError:
                raise InstrumentError('Could not find trace-cmd. Please make sure it is installed and is in PATH.')


def _build_trace_events(events):
    event_string = ' '.join(['-e {}'.format(e) for e in events])
    return event_string
