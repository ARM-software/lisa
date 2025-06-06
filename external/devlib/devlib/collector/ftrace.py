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

import os
import json
import time
import re
import subprocess
import sys
import contextlib
from shlex import quote
import signal

from devlib.collector import (CollectorBase, CollectorOutput,
                              CollectorOutputEntry)
from devlib.host import PACKAGE_BIN_DIRECTORY
from devlib.exception import TargetStableError, HostError
from devlib.utils.misc import check_output, which, memoized
from devlib.utils.asyn import asyncf


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

class FtraceCollector(CollectorBase):

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def __init__(self, target,
                 events=None,
                 functions=None,
                 tracer=None,
                 trace_children_functions=False,
                 buffer_size=None,
                 top_buffer_size=None,
                 buffer_size_step=1000,
                 tracing_path=None,
                 automark=True,
                 autoreport=True,
                 autoview=False,
                 no_install=False,
                 strict=False,
                 report_on_target=False,
                 trace_clock='local',
                 saved_cmdlines_nr=4096,
                 mode='write-to-memory',
                 ):
        super(FtraceCollector, self).__init__(target)
        self.events = events if events is not None else DEFAULT_EVENTS
        self.functions = functions
        self.tracer = tracer
        self.trace_children_functions = trace_children_functions
        self.buffer_size = buffer_size
        self.top_buffer_size = top_buffer_size
        self.tracing_path = self._resolve_tracing_path(target, tracing_path)
        self.automark = automark
        self.autoreport = autoreport
        self.autoview = autoview
        self.strict = strict
        self.report_on_target = report_on_target
        self.target_output_file = target.path.join(self.target.working_directory, OUTPUT_TRACE_FILE)
        text_file_name = target.path.splitext(OUTPUT_TRACE_FILE)[0] + '.txt'
        self.target_text_file = target.path.join(self.target.working_directory, text_file_name)
        self.output_path = None
        self.target_binary = None
        self.host_binary = None
        self.start_time = None
        self.stop_time = None
        self.function_string = None
        self.trace_clock = trace_clock
        self.saved_cmdlines_nr = saved_cmdlines_nr
        self._reset_needed = True
        self.mode = mode
        self._bg_cmd = None

        # pylint: disable=bad-whitespace
        # Setup tracing paths
        self.available_events_file    = self.target.path.join(self.tracing_path, 'available_events')
        self.available_functions_file = self.target.path.join(self.tracing_path, 'available_filter_functions')
        self.current_tracer_file      = self.target.path.join(self.tracing_path, 'current_tracer')
        self.function_profile_file    = self.target.path.join(self.tracing_path, 'function_profile_enabled')
        self.marker_file              = self.target.path.join(self.tracing_path, 'trace_marker')
        self.ftrace_filter_file       = self.target.path.join(self.tracing_path, 'set_ftrace_filter')
        self.available_tracers_file   = self.target.path.join(self.tracing_path, 'available_tracers')
        self.kprobe_events_file       = self.target.path.join(self.tracing_path, 'kprobe_events')

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

        available_events = self.available_events
        unavailable_events = [
            event
            for event in self.events
            if not event_is_in_list(event, available_events)
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
            # Validate required functions to be traced
            selected_functions = []
            for function in self.functions:
                if function not in self.available_functions:
                    message = 'Function [{}] not available for tracing/profiling'.format(function)
                    if self.strict:
                        raise TargetStableError(message)
                    self.target.logger.warning(message)
                else:
                    selected_functions.append(function)

            # Function profiling
            if self.tracer is None:
                if not self.target.file_exists(self.function_profile_file):
                    raise TargetStableError('Function profiling not supported. '\
                                            'A kernel build with CONFIG_FUNCTION_PROFILER enable is required')
                self.function_string = _build_trace_functions(selected_functions)
                # If function profiling is enabled we always need at least one event.
                # Thus, if not other events have been specified, try to add at least
                # a tracepoint which is always available and possibly triggered few
                # times.
                if not selected_events:
                    selected_events = ['sched_wakeup_new']

            # Function tracing
            elif self.tracer == 'function':
                self.function_string = _build_graph_functions(selected_functions, False)

            # Function graphing
            elif self.tracer == 'function_graph':
                self.function_string = _build_graph_functions(selected_functions, trace_children_functions)

        self._selected_events = selected_events

    @property
    def _buffer_size(self):
        top = self.top_buffer_size
        nontop = self.buffer_size
        if top is None and nontop is None:
            return None
        elif top is None:
            return nontop
        elif nontop is None:
            return top
        else:
            return max(top, nontop)

    @property
    def event_string(self):
        return _build_trace_events(self._selected_events)

    @classmethod
    def _resolve_tracing_path(cls, target, path):
        if path is None:
            return cls.find_tracing_path(target)
        else:
            return path

    @classmethod
    def find_tracing_path(cls, target):
        fs_list = [
            fs.mount_point
            for fs in target.list_file_systems()
            if fs.fs_type == 'tracefs'
        ]
        try:
            return fs_list[0]
        except IndexError:
            # Default legacy value, when the kernel did not have a tracefs yet
            return '/sys/kernel/debug/tracing'

    @property
    @memoized
    def available_tracers(self):
        """
        List of ftrace tracers supported by the target's kernel.
        """
        return self.target.read_value(self.available_tracers_file).split(' ')

    @property
    def available_events(self):
        """
        List of ftrace events supported by the target's kernel.
        """
        return self.target.read_value(self.available_events_file).splitlines()

    @property
    @memoized
    def available_functions(self):
        """
        List of functions whose tracing/profiling is supported by the target's kernel.
        """
        return self.target.read_value(self.available_functions_file).splitlines()

    def reset(self):
        # Save kprobe events
        try:
            kprobe_events = self.target.read_value(self.kprobe_events_file)
        except TargetStableError:
            kprobe_events = None

        self.target.execute('{} reset'.format(self.target_binary),
                            as_root=True, timeout=TIMEOUT)


        # This code is currently not necessary as we are not using alternate
        # instances (not using -B parameter). If we end up using it again, it
        # may very well be that trace-cmd at that point takes care of that
        # problem itself somehow, so this should be re-evaluated.

        # trace-cmd start will not set the top-level buffer size if passed -B
        # parameter, but unfortunately some events still end up there (e.g.
        # print event). So we still need to set that size, otherwise the buffer
        # might be too small and some event lost.
        #  top_buffer_size = self.top_buffer_size or self.buffer_size
        #  if top_buffer_size:
            #  self.target.write_value(
                #  self.target.path.join(self.tracing_path, 'buffer_size_kb'),
                #  top_buffer_size, verify=False
            #  )

        if self.functions:
            self.target.write_value(self.function_profile_file, 0, verify=False)

        # Restore kprobe events
        if kprobe_events:
            self.target.write_value(self.kprobe_events_file, kprobe_events)

        self._reset_needed = False

    def _trace_frequencies(self):
        if 'cpu_frequency' in self._selected_events:
            self.logger.debug('Trace CPUFreq frequencies')
            try:
                mod = self.target.cpufreq
            except TargetStableError as e:
                self.logger.error(f'Could not trace CPUFreq frequencies as the cpufreq module cannot be loaded: {e}')
            else:
                mod.trace_frequencies()

    def _trace_idle(self):
        if 'cpu_idle' in self._selected_events:
            self.logger.debug('Trace CPUIdle states')
            try:
                mod = self.target.cpuidle
            except TargetStableError as e:
                self.logger.error(f'Could not trace CPUIdle states as the cpuidle module cannot be loaded: {e}')
            else:
                mod.perturb_cpus()

    @asyncf
    async def start(self):
        self.start_time = time.time()
        if self._reset_needed:
            self.reset()

        if self.tracer is not None and 'function' in self.tracer:
            tracecmd_functions = self.function_string
        else:
            tracecmd_functions = ''

        tracer_string = '-p {}'.format(self.tracer) if self.tracer else ''

        # Ensure kallsyms contains addresses if possible, so that function the
        # collected trace contains enough data for pretty printing
        with contextlib.suppress(TargetStableError):
            self.target.write_value('/proc/sys/kernel/kptr_restrict', 0)

        params = '{buffer_size} {cmdlines_size} {clock} {events} {tracer} {functions}'.format(
            events=self.event_string,
            tracer=tracer_string,
            functions=tracecmd_functions,
            buffer_size='-b {}'.format(self._buffer_size) if self._buffer_size is not None else '',
            clock='-C {}'.format(self.trace_clock) if self.trace_clock else '',
            cmdlines_size='--cmdlines-size {}'.format(self.saved_cmdlines_nr) if self.saved_cmdlines_nr is not None else '',
        )

        mode = self.mode
        if mode == 'write-to-disk':
            bg_cmd = self.target.background(
                # cd into the working_directory first to workaround this issue:
                # https://lore.kernel.org/linux-trace-devel/20240119162743.1a107fa9@gandalf.local.home/
                f'cd {self.target.working_directory} && devlib-signal-target {self.target_binary} record -o {quote(self.target_output_file)} {params}',
                as_root=True,
            )
            assert self._bg_cmd is None
            self._bg_cmd = bg_cmd.__enter__()
        elif mode == 'write-to-memory':
            self.target.execute(
                f'{self.target_binary} start {params}',
                as_root=True,
            )
        else:
            raise ValueError(f'Unknown mode {mode}')

        if self.automark:
            self.mark_start()

        self._trace_frequencies()
        self._trace_idle()

        # Enable kernel function profiling
        if self.functions and self.tracer is None:
            target = self.target
            await target.async_manager.concurrently(
                execute.asyn('echo nop > {}'.format(self.current_tracer_file),
                                    as_root=True),
                execute.asyn('echo 0 > {}'.format(self.function_profile_file),
                                    as_root=True),
                execute.asyn('echo {} > {}'.format(self.function_string, self.ftrace_filter_file),
                                    as_root=True),
                execute.asyn('echo 1 > {}'.format(self.function_profile_file),
                                    as_root=True),
            )


    def stop(self):
        # Disable kernel function profiling
        if self.functions and self.tracer is None:
            self.target.execute('echo 0 > {}'.format(self.function_profile_file),
                                as_root=True)
        self.stop_time = time.time()
        if self.automark:
            self.mark_stop()

        mode = self.mode
        if mode == 'write-to-disk':
            bg_cmd = self._bg_cmd
            self._bg_cmd = None
            assert bg_cmd is not None
            bg_cmd.send_signal(signal.SIGINT)
            bg_cmd.communicate()
            bg_cmd.__exit__(None, None, None)
        elif mode == 'write-to-memory':
            self.target.execute('{} stop'.format(self.target_binary),
                                timeout=TIMEOUT, as_root=True)
        else:
            raise ValueError(f'Unknown mode {mode}')

        self._reset_needed = True

    def set_output(self, output_path):
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, os.path.basename(self.target_output_file))
        self.output_path = output_path

    def get_data(self):
        if self.output_path is None:
            raise RuntimeError("Output path was not set.")

        busybox = quote(self.target.busybox)

        mode = self.mode
        if mode == 'write-to-disk':
            # Interrupting trace-cmd record will make it create the file
            pass
        elif mode == 'write-to-memory':
            cmd = f'{self.target_binary} extract -o {self.target_output_file} && {busybox} chmod 666 {self.target_output_file}'
            self.target.execute(cmd, timeout=TIMEOUT, as_root=True)
        else:
            raise ValueError(f'Unknown mode {mode}')

        # The size of trace.dat will depend on how long trace-cmd was running.
        # Therefore timout for the pull command must also be adjusted
        # accordingly.
        pull_timeout = 10 * (self.stop_time - self.start_time)
        self.target.pull(self.target_output_file, self.output_path, timeout=pull_timeout)
        output = CollectorOutput()
        if not os.path.isfile(self.output_path):
            self.logger.warning('Binary trace not pulled from device.')
        else:
            output.append(CollectorOutputEntry(self.output_path, 'file'))
            if self.autoreport:
                textfile = os.path.splitext(self.output_path)[0] + '.txt'
                if self.report_on_target:
                    self.generate_report_on_target()
                    self.target.pull(self.target_text_file,
                                     textfile, timeout=pull_timeout)
                else:
                    self.report(self.output_path, textfile)
                output.append(CollectorOutputEntry(textfile, 'file'))
            if self.autoview:
                self.view(self.output_path)
        return output

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
