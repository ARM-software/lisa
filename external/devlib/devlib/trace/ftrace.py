#    Copyright 2015-2018 ARM Limited
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

from __future__ import division
import os
import json
import time
import re
import subprocess
import sys
from pipes import quote

from devlib.trace import TraceCollector
from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.exception import TargetStableError, HostError
from devlib.utils.misc import check_output, which, memoized


TRACE_MARKER_START = 'TRACE_MARKER_START'
TRACE_MARKER_STOP = 'TRACE_MARKER_STOP'
OUTPUT_TRACE_FILE = 'trace.dat'
OUTPUT_PROFILE_FILE = 'trace_stat.dat'
DEFAULT_EVENTS = [
    'cpu_frequency',
    'cpu_idle',
    'sched_migrate_task',
    'sched_process_exec',
    'sched_process_fork',
    'sched_stat_iowait',
    'sched_switch',
    'sched_wakeup',
    'sched_wakeup_new',
]
TIMEOUT = 180

# Regexps for parsing of function profiling data
CPU_RE = re.compile(r'  Function \(CPU([0-9]+)\)')
STATS_RE = re.compile(r'([^ ]*) +([0-9]+) +([0-9.]+) us +([0-9.]+) us +([0-9.]+) us')

class FtraceCollector(TraceCollector):

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def __init__(self, target,
                 events=None,
                 functions=None,
                 tracer=None,
                 trace_children_functions=False,
                 buffer_size=None,
                 buffer_size_step=1000,
                 tracing_path='/sys/kernel/debug/tracing',
                 automark=True,
                 autoreport=True,
                 autoview=False,
                 no_install=False,
                 strict=False,
                 report_on_target=False,
                 trace_clock='local',
                 saved_cmdlines_nr=4096,
                 ):
        super(FtraceCollector, self).__init__(target)
        self.events = events if events is not None else DEFAULT_EVENTS
        self.functions = functions
        self.tracer = tracer
        self.trace_children_functions = trace_children_functions
        self.buffer_size = buffer_size
        self.buffer_size_step = buffer_size_step
        self.tracing_path = tracing_path
        self.automark = automark
        self.autoreport = autoreport
        self.autoview = autoview
        self.strict = strict
        self.report_on_target = report_on_target
        self.target_output_file = target.path.join(self.target.working_directory, OUTPUT_TRACE_FILE)
        text_file_name = target.path.splitext(OUTPUT_TRACE_FILE)[0] + '.txt'
        self.target_text_file = target.path.join(self.target.working_directory, text_file_name)
        self.target_binary = None
        self.host_binary = None
        self.start_time = None
        self.stop_time = None
        self.event_string = None
        self.function_string = None
        self.trace_clock = trace_clock
        self.saved_cmdlines_nr = saved_cmdlines_nr
        self._reset_needed = True

        # pylint: disable=bad-whitespace
        # Setup tracing paths
        self.available_events_file    = self.target.path.join(self.tracing_path, 'available_events')
        self.available_functions_file = self.target.path.join(self.tracing_path, 'available_filter_functions')
        self.buffer_size_file         = self.target.path.join(self.tracing_path, 'buffer_size_kb')
        self.current_tracer_file      = self.target.path.join(self.tracing_path, 'current_tracer')
        self.function_profile_file    = self.target.path.join(self.tracing_path, 'function_profile_enabled')
        self.marker_file              = self.target.path.join(self.tracing_path, 'trace_marker')
        self.ftrace_filter_file       = self.target.path.join(self.tracing_path, 'set_ftrace_filter')
        self.trace_clock_file         = self.target.path.join(self.tracing_path, 'trace_clock')
        self.save_cmdlines_size_file  = self.target.path.join(self.tracing_path, 'saved_cmdlines_size')
        self.available_tracers_file  = self.target.path.join(self.tracing_path, 'available_tracers')

        self.host_binary = which('trace-cmd')
        self.kernelshark = which('kernelshark')

        if not self.target.is_rooted:
            raise TargetStableError('trace-cmd instrument cannot be used on an unrooted device.')
        if self.autoreport and not self.report_on_target and self.host_binary is None:
            raise HostError('trace-cmd binary must be installed on the host if autoreport=True.')
        if self.autoview and self.kernelshark is None:
            raise HostError('kernelshark binary must be installed on the host if autoview=True.')
        if not no_install:
            host_file = os.path.join(PACKAGE_BIN_DIRECTORY, self.target.abi, 'trace-cmd')
            self.target_binary = self.target.install(host_file)
        else:
            if not self.target.is_installed('trace-cmd'):
                raise TargetStableError('No trace-cmd found on device and no_install=True is specified.')
            self.target_binary = 'trace-cmd'

        # Validate required events to be traced
        def event_to_regex(event):
            if not event.startswith('*'):
                event = '*' + event

            return re.compile(event.replace('*', '.*'))

        def event_is_in_list(event, events):
            return any(
                event_to_regex(event).match(_event)
                for _event in events
            )

        unavailable_events = [
            event
            for event in self.events
            if not event_is_in_list(event, self.available_events)
        ]
        if unavailable_events:
            message = 'Events not available for tracing: {}'.format(
                ', '.join(unavailable_events)
            )
            if self.strict:
                raise TargetStableError(message)
            else:
                self.target.logger.warning(message)

        selected_events = sorted(set(self.events) - set(unavailable_events))

        if self.tracer and self.tracer not in self.available_tracers:
            raise TargetStableError('Unsupported tracer "{}". Available tracers: {}'.format(
                self.tracer, ', '.join(self.available_tracers)))

        # Check for function tracing support
        if self.functions:
            if not self.target.file_exists(self.function_profile_file):
                raise TargetStableError('Function profiling not supported. '\
                        'A kernel build with CONFIG_FUNCTION_PROFILER enable is required')
            # Validate required functions to be traced
            available_functions = self.target.execute(
                    'cat {}'.format(self.available_functions_file),
                    as_root=True).splitlines()
            selected_functions = []
            for function in self.functions:
                if function not in available_functions:
                    message = 'Function [{}] not available for profiling'.format(function)
                    if self.strict:
                        raise TargetStableError(message)
                    self.target.logger.warning(message)
                else:
                    selected_functions.append(function)

            if self.tracer is None:
                self.function_string = _build_trace_functions(selected_functions)
                # If function profiling is enabled we always need at least one event.
                # Thus, if not other events have been specified, try to add at least
                # a tracepoint which is always available and possibly triggered few
                # times.
                if not selected_events:
                    selected_events = ['sched_wakeup_new']
            elif self.tracer == 'function_graph':
                self.function_string = _build_graph_functions(selected_functions, trace_children_functions)

        self.event_string = _build_trace_events(selected_events)


    @property
    @memoized
    def available_tracers(self):
        """
        List of ftrace tracers supported by the target's kernel.
        """
        return self.target.read_value(self.available_tracers_file).split(' ')

    @property
    @memoized
    def available_events(self):
        """
        List of ftrace events supported by the target's kernel.
        """
        return self.target.read_value(self.available_events_file).splitlines()

    def reset(self):
        if self.buffer_size:
            self._set_buffer_size()
        self.target.execute('{} reset'.format(self.target_binary),
                            as_root=True, timeout=TIMEOUT)
        self._reset_needed = False

    def start(self):
        self.start_time = time.time()
        if self._reset_needed:
            self.reset()

        if self.tracer is not None and 'function' in self.tracer:
            tracecmd_functions = self.function_string
        else:
            tracecmd_functions = ''

        tracer_string = '-p {}'.format(self.tracer) if self.tracer else ''

        self.target.write_value(self.trace_clock_file, self.trace_clock, verify=False)
        try:
            self.target.write_value(self.save_cmdlines_size_file, self.saved_cmdlines_nr)
        except TargetStableError as e:
            message = 'Could not set "save_cmdlines_size"'
            if self.strict:
                self.logger.error(message)
                raise e
            else:
                self.logger.warning(message)
                self.logger.debug(e)

        self.target.execute(
            '{} start {events} {tracer} {functions}'.format(
                self.target_binary,
                events=self.event_string,
                tracer=tracer_string,
                functions=tracecmd_functions,
            ),
            as_root=True,
        )
        if self.automark:
            self.mark_start()
        if 'cpufreq' in self.target.modules:
            self.logger.debug('Trace CPUFreq frequencies')
            self.target.cpufreq.trace_frequencies()
        if 'cpuidle' in self.target.modules:
            self.logger.debug('Trace CPUIdle states')
            self.target.cpuidle.perturb_cpus()
        # Enable kernel function profiling
        if self.functions and self.tracer is None:
            self.target.execute('echo nop > {}'.format(self.current_tracer_file),
                                as_root=True)
            self.target.execute('echo 0 > {}'.format(self.function_profile_file),
                                as_root=True)
            self.target.execute('echo {} > {}'.format(self.function_string, self.ftrace_filter_file),
                                as_root=True)
            self.target.execute('echo 1 > {}'.format(self.function_profile_file),
                                as_root=True)


    def stop(self):
        # Disable kernel function profiling
        if self.functions and self.tracer is None:
            self.target.execute('echo 1 > {}'.format(self.function_profile_file),
                                as_root=True)
        if 'cpufreq' in self.target.modules:
            self.logger.debug('Trace CPUFreq frequencies')
            self.target.cpufreq.trace_frequencies()
        self.stop_time = time.time()
        if self.automark:
            self.mark_stop()
        self.target.execute('{} stop'.format(self.target_binary),
                            timeout=TIMEOUT, as_root=True)
        self._reset_needed = True

    def get_trace(self, outfile):
        if os.path.isdir(outfile):
            outfile = os.path.join(outfile, os.path.basename(self.target_output_file))
        self.target.execute('{0} extract -o {1}; chmod 666 {1}'.format(self.target_binary,
                                                                       self.target_output_file),
                            timeout=TIMEOUT, as_root=True)

        # The size of trace.dat will depend on how long trace-cmd was running.
        # Therefore timout for the pull command must also be adjusted
        # accordingly.
        pull_timeout = 10 * (self.stop_time - self.start_time)
        self.target.pull(self.target_output_file, outfile, timeout=pull_timeout)
        if not os.path.isfile(outfile):
            self.logger.warning('Binary trace not pulled from device.')
        else:
            if self.autoreport:
                textfile = os.path.splitext(outfile)[0] + '.txt'
                if self.report_on_target:
                    self.generate_report_on_target()
                    self.target.pull(self.target_text_file,
                                     textfile, timeout=pull_timeout)
                else:
                    self.report(outfile, textfile)
            if self.autoview:
                self.view(outfile)

    def get_stats(self, outfile):
        if not (self.functions and self.tracer is None):
            return

        if os.path.isdir(outfile):
            outfile = os.path.join(outfile, OUTPUT_PROFILE_FILE)
        # pylint: disable=protected-access
        output = self.target._execute_util('ftrace_get_function_stats',
                                            as_root=True)

        function_stats = {}
        for line in output.splitlines():
            # Match a new CPU dataset
            match = CPU_RE.search(line)
            if match:
                cpu_id = int(match.group(1))
                function_stats[cpu_id] = {}
                self.logger.debug("Processing stats for CPU%d...", cpu_id)
                continue
            # Match a new function dataset
            match = STATS_RE.search(line)
            if match:
                fname = match.group(1)
                function_stats[cpu_id][fname] = {
                        'hits' : int(match.group(2)),
                        'time' : float(match.group(3)),
                        'avg'  : float(match.group(4)),
                        's_2'  : float(match.group(5)),
                    }
                self.logger.debug(" %s: %s",
                             fname, function_stats[cpu_id][fname])

        self.logger.debug("FTrace stats output [%s]...", outfile)
        with open(outfile, 'w') as fh:
            json.dump(function_stats, fh, indent=4)
        self.logger.debug("FTrace function stats save in [%s]", outfile)

        return function_stats

    def report(self, binfile, destfile):
        # To get the output of trace.dat, trace-cmd must be installed
        # This is done host-side because the generated file is very large
        try:
            command = '{} report {} > {}'.format(self.host_binary, binfile, destfile)
            self.logger.debug(command)
            process = subprocess.Popen(command, stderr=subprocess.PIPE, shell=True)
            _, error = process.communicate()
            if sys.version_info[0] == 3:
                error = error.decode(sys.stdout.encoding or 'utf-8', 'replace')
            if process.returncode:
                raise TargetStableError('trace-cmd returned non-zero exit code {}'.format(process.returncode))
            if error:
                # logged at debug level, as trace-cmd always outputs some
                # errors that seem benign.
                self.logger.debug(error)
            if os.path.isfile(destfile):
                self.logger.debug('Verifying traces.')
                with open(destfile) as fh:
                    for line in fh:
                        if 'EVENTS DROPPED' in line:
                            self.logger.warning('Dropped events detected.')
                            break
                    else:
                        self.logger.debug('Trace verified.')
            else:
                self.logger.warning('Could not generate trace.txt.')
        except OSError:
            raise HostError('Could not find trace-cmd. Please make sure it is installed and is in PATH.')

    def generate_report_on_target(self):
        command = '{} report {} > {}'.format(self.target_binary,
                                             self.target_output_file,
                                             self.target_text_file)
        self.target.execute(command, timeout=TIMEOUT)

    def view(self, binfile):
        check_output('{} {}'.format(self.kernelshark, binfile), shell=True)

    def teardown(self):
        self.target.remove(self.target.path.join(self.target.working_directory, OUTPUT_TRACE_FILE))

    def mark_start(self):
        self.target.write_value(self.marker_file, TRACE_MARKER_START, verify=False)

    def mark_stop(self):
        self.target.write_value(self.marker_file, TRACE_MARKER_STOP, verify=False)

    def _set_buffer_size(self):
        target_buffer_size = self.buffer_size
        attempt_buffer_size = target_buffer_size
        buffer_size = 0
        floor = 1000 if target_buffer_size > 1000 else target_buffer_size
        while attempt_buffer_size >= floor:
            self.target.write_value(self.buffer_size_file, attempt_buffer_size, verify=False)
            buffer_size = self.target.read_int(self.buffer_size_file)
            if buffer_size == attempt_buffer_size:
                break
            else:
                attempt_buffer_size -= self.buffer_size_step
        if buffer_size == target_buffer_size:
            return
        while attempt_buffer_size < target_buffer_size:
            attempt_buffer_size += self.buffer_size_step
            self.target.write_value(self.buffer_size_file, attempt_buffer_size, verify=False)
            buffer_size = self.target.read_int(self.buffer_size_file)
            if attempt_buffer_size != buffer_size:
                message = 'Failed to set trace buffer size to {}, value set was {}'
                self.logger.warning(message.format(target_buffer_size, buffer_size))
                break


def _build_trace_events(events):
    event_string = ' '.join(['-e {}'.format(e) for e in events])
    return event_string

def _build_trace_functions(functions):
    function_string = " ".join(functions)
    return function_string

def _build_graph_functions(functions, trace_children_functions):
    opt = 'g' if trace_children_functions else 'l'
    return ' '.join(
        '-{} {}'.format(opt, quote(f))
        for f in functions
    )
