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
import re
import logging
from ctypes import c_int32
from collections import defaultdict

from devlib.utils.csvutil import create_writer, csvwriter

from wa.utils.trace_cmd import TraceCmdParser, trace_has_marker, TRACE_MARKER_START, TRACE_MARKER_STOP


logger = logging.getLogger('cpustates')

INIT_CPU_FREQ_REGEX = re.compile(r'CPU (?P<cpu>\d+) FREQUENCY: (?P<freq>\d+) kHZ')
DEVLIB_CPU_FREQ_REGEX = re.compile(r'cpu_frequency(?:_devlib):\s+state=(?P<freq>\d+)\s+cpu_id=(?P<cpu>\d+)')


class CorePowerTransitionEvent(object):

    kind = 'transition'
    __slots__ = ['timestamp', 'cpu_id', 'frequency', 'idle_state']

    def __init__(self, timestamp, cpu_id, frequency=None, idle_state=None):
        if (frequency is None) == (idle_state is None):
            raise ValueError('Power transition must specify a frequency or an idle_state, but not both.')
        self.timestamp = timestamp
        self.cpu_id = cpu_id
        self.frequency = frequency
        self.idle_state = idle_state

    def __str__(self):
        return 'cpu {} @ {} -> freq: {} idle: {}'.format(self.cpu_id, self.timestamp,
                                                         self.frequency, self.idle_state)

    def __repr__(self):
        return 'CPTE(c:{} t:{} f:{} i:{})'.format(self.cpu_id, self.timestamp,
                                                  self.frequency, self.idle_state)


class CorePowerDroppedEvents(object):

    kind = 'dropped_events'
    __slots__ = ['cpu_id']

    def __init__(self, cpu_id):
        self.cpu_id = cpu_id

    def __str__(self):
        return 'DROPPED EVENTS on CPU{}'.format(self.cpu_id)

    __repr__ = __str__


class TraceMarkerEvent(object):

    kind = 'marker'
    __slots__ = ['name']

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'MARKER: {}'.format(self.name)


class CpuPowerState(object):

    __slots__ = ['frequency', 'idle_state']

    @property
    def is_idling(self):
        return self.idle_state is not None and self.idle_state >= 0

    @property
    def is_active(self):
        return self.idle_state == -1

    def __init__(self, frequency=None, idle_state=None):
        self.frequency = frequency
        self.idle_state = idle_state

    def __str__(self):
        return 'CP(f:{} i:{})'.format(self.frequency, self.idle_state)

    __repr__ = __str__


class SystemPowerState(object):

    __slots__ = ['timestamp', 'cpus']

    @property
    def num_cores(self):
        return len(self.cpus)

    def __init__(self, num_cores, no_idle=False):
        self.timestamp = None
        self.cpus = []
        idle_state = -1 if no_idle else None
        for _ in range(num_cores):
            self.cpus.append(CpuPowerState(idle_state=idle_state))

    def copy(self):
        new = SystemPowerState(self.num_cores)
        new.timestamp = self.timestamp
        for i, c in enumerate(self.cpus):
            new.cpus[i].frequency = c.frequency
            new.cpus[i].idle_state = c.idle_state
        return new

    def __str__(self):
        return 'SP(t:{} Cs:{})'.format(self.timestamp, self.cpus)

    __repr__ = __str__


class PowerStateProcessor(object):
    """
    This takes a stream of power transition events and yields a timeline stream
    of system power states.

    """

    @property
    def cpu_states(self):
        return self.power_state.cpus

    @property
    def current_time(self):
        return self.power_state.timestamp

    @current_time.setter
    def current_time(self, value):
        self.power_state.timestamp = value

    def __init__(self, cpus, wait_for_marker=True, no_idle=None):
        if no_idle is None:
            no_idle = not (cpus[0].cpuidle and cpus[0].cpuidle.states)
        self.power_state = SystemPowerState(len(cpus), no_idle=no_idle)
        self.requested_states = {}  # cpu_id -> requeseted state
        self.wait_for_marker = wait_for_marker
        self._saw_start_marker = False
        self._saw_stop_marker = False
        self.exceptions = []

        self.idle_related_cpus = build_idle_state_map(cpus)

    def process(self, event_stream):
        for event in event_stream:
            try:
                next_state = self.update_power_state(event)
                if self._saw_start_marker or not self.wait_for_marker:
                    yield next_state
                if self._saw_stop_marker:
                    break
            except Exception as e:  # pylint: disable=broad-except
                self.exceptions.append(e)
        else:
            if self.wait_for_marker:
                logger.warning("Did not see a STOP marker in the trace")

    def update_power_state(self, event):
        """
        Update the tracked power state based on the specified event and
        return updated power state.

        """
        if event.kind == 'transition':
            self._process_transition(event)
        elif event.kind == 'dropped_events':
            self._process_dropped_events(event)
        elif event.kind == 'marker':
            if event.name == 'START':
                self._saw_start_marker = True
            elif event.name == 'STOP':
                self._saw_stop_marker = True
        else:
            raise ValueError('Unexpected event type: {}'.format(event.kind))
        return self.power_state.copy()

    def _process_transition(self, event):
        self.current_time = event.timestamp
        if event.idle_state is None:
            self.cpu_states[event.cpu_id].frequency = event.frequency
        else:
            if event.idle_state == -1:
                self._process_idle_exit(event)
            else:
                self._process_idle_entry(event)

    def _process_dropped_events(self, event):
        self.cpu_states[event.cpu_id].frequency = None
        old_idle_state = self.cpu_states[event.cpu_id].idle_state
        self.cpu_states[event.cpu_id].idle_state = None

        related_ids = self.idle_related_cpus[(event.cpu_id, old_idle_state)]
        for rid in related_ids:
            self.cpu_states[rid].idle_state = None

    def _process_idle_entry(self, event):
        if self.cpu_states[event.cpu_id].is_idling:
            raise ValueError('Got idle state entry event for an idling core: {}'.format(event))
        self.requested_states[event.cpu_id] = event.idle_state
        self._try_transition_to_idle_state(event.cpu_id, event.idle_state)

    def _process_idle_exit(self, event):
        if self.cpu_states[event.cpu_id].is_active:
            raise ValueError('Got idle state exit event for an active core: {}'.format(event))
        self.requested_states.pop(event.cpu_id, None)  # remove outstanding request if there is one
        old_state = self.cpu_states[event.cpu_id].idle_state
        self.cpu_states[event.cpu_id].idle_state = -1

        related_ids = self.idle_related_cpus[(event.cpu_id, old_state)]
        if old_state is not None:
            new_state = old_state - 1
            for rid in related_ids:
                if self.cpu_states[rid].idle_state > new_state:
                    self._try_transition_to_idle_state(rid, new_state)

    def _try_transition_to_idle_state(self, cpu_id, idle_state):
        related_ids = self.idle_related_cpus[(cpu_id, idle_state)]

        # Tristate: True - can transition, False - can't transition,
        #           None - unknown idle state on at least one related cpu
        transition_check = self._can_enter_state(related_ids, idle_state)

        if transition_check is None:
            # Unknown state on a related cpu means we're not sure whether we're
            # entering requested state or a shallower one
            self.cpu_states[cpu_id].idle_state = None
            return

        # Keep trying shallower states until all related
        while not self._can_enter_state(related_ids, idle_state):
            idle_state -= 1
            related_ids = self.idle_related_cpus[(cpu_id, idle_state)]

        self.cpu_states[cpu_id].idle_state = idle_state
        for rid in related_ids:
            self.cpu_states[rid].idle_state = idle_state

    def _can_enter_state(self, related_ids, state):
        """
        This is a tri-state check. Returns ``True`` if related cpu states allow transition
        into this state, ``False`` if related cpu states don't allow transition into this
        state, and ``None`` if at least one of the related cpus is in an unknown state
        (so the decision of whether a transition is possible cannot be made).

        """
        for rid in related_ids:
            rid_requested_state = self.requested_states.get(rid, None)
            rid_current_state = self.cpu_states[rid].idle_state
            if rid_current_state is None:
                return None
            if rid_current_state < state:
                if rid_requested_state is None or rid_requested_state < state:
                    return False
        return True


def stream_cpu_power_transitions(events):
    for event in events:
        if event.name == 'cpu_idle':
            state = c_int32(event.state).value
            yield CorePowerTransitionEvent(event.timestamp, event.cpu_id, idle_state=state)
        elif event.name == 'cpu_frequency':
            yield CorePowerTransitionEvent(event.timestamp, event.cpu_id, frequency=event.state)
        elif event.name == 'DROPPED EVENTS DETECTED':
            yield CorePowerDroppedEvents(event.cpu_id)
        elif event.name == 'print':
            if TRACE_MARKER_START in event.text:
                yield TraceMarkerEvent('START')
            elif TRACE_MARKER_STOP in event.text:
                yield TraceMarkerEvent('STOP')
            else:
                if 'cpu_frequency' in event.text:
                    match = DEVLIB_CPU_FREQ_REGEX.search(event.text)
                else:
                    match = INIT_CPU_FREQ_REGEX.search(event.text)
                if match:
                    yield CorePowerTransitionEvent(event.timestamp,
                                                   int(match.group('cpu')),
                                                   frequency=int(match.group('freq')))


def gather_core_states(system_state_stream, freq_dependent_idle_states=None):  # NOQA
    if freq_dependent_idle_states is None:
        freq_dependent_idle_states = []
    for system_state in system_state_stream:
        core_states = []
        for cpu in system_state.cpus:
            if cpu.idle_state == -1:
                core_states.append((-1, cpu.frequency))
            elif cpu.idle_state in freq_dependent_idle_states:
                if cpu.frequency is not None:
                    core_states.append((cpu.idle_state, cpu.frequency))
                else:
                    core_states.append((None, None))
            else:
                core_states.append((cpu.idle_state, None))
        yield (system_state.timestamp, core_states)


def record_state_transitions(reporter, stream):
    for event in stream:
        if event.kind == 'transition':
            reporter.record_transition(event)
        yield event


class PowerStateTransitions(object):

    name = 'transitions-timeline'

    def __init__(self, output_directory):
        self.filepath = os.path.join(output_directory, 'state-transitions-timeline.csv')
        self.writer, self._wfh = create_writer(self.filepath)
        headers = ['timestamp', 'cpu_id', 'frequency', 'idle_state']
        self.writer.writerow(headers)

    def update(self, timestamp, core_states):  # NOQA
        # Just recording transitions, not doing anything
        # with states.
        pass

    def record_transition(self, transition):
        row = [transition.timestamp, transition.cpu_id,
               transition.frequency, transition.idle_state]
        self.writer.writerow(row)

    def report(self):
        return self

    def write(self):
        self._wfh.close()


class PowerStateTimeline(object):

    name = 'state-timeline'

    def __init__(self, output_directory, cpus):
        self.filepath = os.path.join(output_directory, 'power-state-timeline.csv')
        self.idle_state_names = {cpu.id: [s.name for s in cpu.cpuidle.states] for cpu in cpus}
        self.writer, self._wfh = create_writer(self.filepath)
        headers = ['ts'] + ['{} CPU{}'.format(cpu.name, cpu.id) for cpu in cpus]
        self.writer.writerow(headers)

    def update(self, timestamp, core_states):  # NOQA
        row = [timestamp]
        for cpu_idx, (idle_state, frequency) in enumerate(core_states):
            if frequency is None:
                if idle_state == -1:
                    row.append('Running (unknown kHz)')
                elif idle_state is None:
                    row.append('unknown')
                elif not self.idle_state_names[cpu_idx]:
                    row.append('idle[{}]'.format(idle_state))
                else:
                    row.append(self.idle_state_names[cpu_idx][idle_state])
            else:  # frequency is not None
                if idle_state == -1:
                    row.append(frequency)
                elif idle_state is None:
                    row.append('unknown')
                else:
                    row.append('{} ({})'.format(self.idle_state_names[cpu_idx][idle_state],
                                                frequency))
        self.writer.writerow(row)

    def report(self):
        return self

    def write(self):
        self._wfh.close()


class ParallelStats(object):

    def __init__(self, output_directory, cpus, use_ratios=False):
        self.filepath = os.path.join(output_directory, 'parallel-stats.csv')
        self.clusters = defaultdict(set)
        self.use_ratios = use_ratios

        clusters = []
        for cpu in cpus:
            if cpu.cpufreq.related_cpus not in clusters:
                clusters.append(cpu.cpufreq.related_cpus)

        for i, clust in enumerate(clusters):
            self.clusters[str(i)] = set(clust)
        self.clusters['all'] = {cpu.id for cpu in cpus}

        self.first_timestamp = None
        self.last_timestamp = None
        self.previous_states = None
        self.parallel_times = defaultdict(lambda: defaultdict(int))
        self.running_times = defaultdict(int)

    def update(self, timestamp, core_states):
        if self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp
            active_cores = [i for i, c in enumerate(self.previous_states)
                            if c and c[0] == -1]
            for cluster, cluster_cores in self.clusters.items():
                clust_active_cores = len(cluster_cores.intersection(active_cores))
                self.parallel_times[cluster][clust_active_cores] += delta
                if clust_active_cores:
                    self.running_times[cluster] += delta
        else:  # initial update
            self.first_timestamp = timestamp

        self.last_timestamp = timestamp
        self.previous_states = core_states

    def report(self):  # NOQA
        if self.last_timestamp is None:
            return None

        report = ParallelReport(self.filepath)
        total_time = self.last_timestamp - self.first_timestamp
        for cluster in sorted(self.parallel_times):
            running_time = self.running_times[cluster]
            for n in range(len(self.clusters[cluster]) + 1):
                time = self.parallel_times[cluster][n]
                time_pc = time / total_time
                if not self.use_ratios:
                    time_pc *= 100
                if n:
                    if running_time:
                        running_time_pc = time / running_time
                    else:
                        running_time_pc = 0
                    if not self.use_ratios:
                        running_time_pc *= 100
                else:
                    running_time_pc = 0
                precision = 3 if self.use_ratios else 1
                fmt = '{{:.{}f}}'.format(precision)
                report.add([cluster, n,
                            fmt.format(time),
                            fmt.format(time_pc),
                            fmt.format(running_time_pc),
                            ])
        return report


class ParallelReport(object):

    name = 'parallel-stats'

    def __init__(self, filepath):
        self.filepath = filepath
        self.values = []

    def add(self, value):
        self.values.append(value)

    def write(self):
        with csvwriter(self.filepath) as writer:
            writer.writerow(['cluster', 'number_of_cores', 'total_time', '%time', '%running_time'])
            writer.writerows(self.values)


class PowerStateStats(object):

    def __init__(self, output_directory, cpus, use_ratios=False):
        self.filepath = os.path.join(output_directory, 'power-state-stats.csv')
        self.core_names = [cpu.name for cpu in cpus]
        self.idle_state_names = {cpu.id: [s.name for s in cpu.cpuidle.states] for cpu in cpus}
        self.use_ratios = use_ratios
        self.first_timestamp = None
        self.last_timestamp = None
        self.previous_states = None
        self.cpu_states = defaultdict(lambda: defaultdict(int))

    def update(self, timestamp, core_states):  # NOQA
        if self.last_timestamp is not None:
            delta = timestamp - self.last_timestamp
            for cpu, (idle, freq) in enumerate(self.previous_states):
                if idle == -1:
                    if freq is not None:
                        state = '{:07}KHz'.format(freq)
                    else:
                        state = 'Running (unknown KHz)'
                elif freq:
                    state = '{}-{:07}KHz'.format(self.idle_state_names[cpu][idle], freq)
                elif idle is not None and self.idle_state_names[cpu]:
                    state = self.idle_state_names[cpu][idle]
                else:
                    state = 'unknown'
                self.cpu_states[cpu][state] += delta
        else:  # initial update
            self.first_timestamp = timestamp

        self.last_timestamp = timestamp
        self.previous_states = core_states

    def report(self):
        if self.last_timestamp is None:
            return None
        total_time = self.last_timestamp - self.first_timestamp
        state_stats = defaultdict(lambda: [None] * len(self.core_names))

        for cpu, states in self.cpu_states.items():
            for state in states:
                time = states[state]
                time_pc = time / total_time
                if not self.use_ratios:
                    time_pc *= 100
                state_stats[state][cpu] = time_pc

        precision = 3 if self.use_ratios else 1
        return PowerStateStatsReport(self.filepath, state_stats, self.core_names, precision)


class PowerStateStatsReport(object):

    name = 'power-state-stats'

    def __init__(self, filepath, state_stats, core_names, precision=2):
        self.filepath = filepath
        self.state_stats = state_stats
        self.core_names = core_names
        self.precision = precision

    def write(self):
        with csvwriter(self.filepath) as writer:
            headers = ['state'] + ['{} CPU{}'.format(c, i)
                                   for i, c in enumerate(self.core_names)]
            writer.writerow(headers)
            for state in sorted(self.state_stats):
                stats = self.state_stats[state]
                fmt = '{{:.{}f}}'.format(self.precision)
                writer.writerow([state] + [fmt.format(s if s is not None else 0)
                                           for s in stats])


class CpuUtilizationTimeline(object):

    name = 'utilization-timeline'

    def __init__(self, output_directory, cpus):
        self.filepath = os.path.join(output_directory, 'utilization-timeline.csv')
        self.writer, self._wfh = create_writer(self.filepath)

        headers = ['ts'] + ['{} CPU{}'.format(cpu.name, cpu.id) for cpu in cpus]
        self.writer.writerow(headers)
        self._max_freq_list = [cpu.cpufreq.available_frequencies[-1] for cpu in cpus if cpu.cpufreq.available_frequencies]

    def update(self, timestamp, core_states):  # NOQA
        row = [timestamp]
        for core, [_, frequency] in enumerate(core_states):
            if frequency is not None and core in self._max_freq_list:
                frequency /= float(self._max_freq_list[core])
                row.append(frequency)
            else:
                row.append(None)
        self.writer.writerow(row)

    def report(self):
        return self

    def write(self):
        self._wfh.close()


def build_idle_state_map(cpus):
    idle_state_map = defaultdict(list)
    for cpu_idx, cpu in enumerate(cpus):
        related_cpus = set(cpu.cpufreq.related_cpus) - set([cpu_idx])
        first_cluster_state = cpu.cpuidle.num_states - 1
        for state_idx, _ in enumerate(cpu.cpuidle.states):
            if state_idx < first_cluster_state:
                idle_state_map[(cpu_idx, state_idx)] = []
            else:
                idle_state_map[(cpu_idx, state_idx)] = list(related_cpus)
    return idle_state_map


def report_power_stats(trace_file, cpus, output_basedir, use_ratios=False, no_idle=None,  # pylint: disable=too-many-locals
                       split_wfi_states=False):
    """
    Process trace-cmd output to generate timelines and statistics of CPU power
    state (a.k.a P- and C-state) transitions in the trace.

    The results will be written into a subdirectory called "power-stats" under
    the specified ``output_basedir``.

    :param trace_file: trace-cmd's text trace to process.
    :param cpus: A list of ``CpuInfo`` objects describing a target's CPUs.
                 These are typically reported as part of ``TargetInfo`` in
                 WA output.
    :param output_basedir: Base location for the output. This directory must
                        exist and must not contain a directory of file
                        named ``"power-states"``.
    :param use_rations: By default, stats will be reported as percentages. Set
                        this to ``True`` to report stats as decimals in the
                        ``0 <= value <= 1`` instead.
    :param no_idle: ``False`` if cpuidle and at least one idle state per CPU are
                    enabled, should be ``True`` otherwise. This influences the
                    assumptions about CPU's initial states. If not explicitly
                    set, the value for this will be guessed based on whether
                    cpuidle states are present in the first ``CpuInfo``.


    The output directory will contain the following files:

    power-state-stats.csv
        Power state residency statistics for each CPU. Shows the percentage of
        time a CPU has spent in each of its available power states.

    parallel-stats.csv
        Parallel execution stats for each CPU cluster, and combined stats for
        the whole system.

    power-state-timeline.csv
        Timeline of CPU power states. Shows which power state each CPU is in at
        a point in time.

    state-transitions-timeline.csv
        Timeline of CPU power state transitions. Each entry shows a CPU's
        transition from one power state to another.

    utilzation-timeline.csv
        Timeline of CPU utilizations.

    .. note:: Timeline entries aren't at regular intervals, but at times of
              power transition events.

    Stats are generated by assembling a pipeline consisting of the following
    stages:

        1. Parse trace into trace events
        2. Filter trace events into power state transition events
        3. Record power state transitions
        4. Convert transitions into a power states.
        5. Collapse the power states into timestamped ``(C state, P state)``
           tuples for each cpu.
        6. Update reporters/stats generators with cpu states.

    """
    output_directory = os.path.join(output_basedir, 'power-states')
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    freq_dependent_idle_states = []
    if split_wfi_states:
        freq_dependent_idle_states = [0]

    # init trace, processor, and reporters
    # note: filter_markers is False here, even though we *will* filter by them. The
    #       reason for this is that we want to observe events before the start
    #       marker in order to establish the intial power states.
    parser = TraceCmdParser(filter_markers=False,
                            events=['cpu_idle', 'cpu_frequency', 'print'])
    ps_processor = PowerStateProcessor(cpus, wait_for_marker=trace_has_marker(trace_file),
                                       no_idle=no_idle)
    transitions_reporter = PowerStateTransitions(output_directory)
    reporters = [
        ParallelStats(output_directory, cpus, use_ratios),
        PowerStateStats(output_directory, cpus, use_ratios),
        PowerStateTimeline(output_directory, cpus),
        CpuUtilizationTimeline(output_directory, cpus),
        transitions_reporter,
    ]

    # assemble the pipeline
    event_stream = parser.parse(trace_file)
    transition_stream = stream_cpu_power_transitions(event_stream)
    recorded_trans_stream = record_state_transitions(transitions_reporter, transition_stream)
    power_state_stream = ps_processor.process(recorded_trans_stream)
    core_state_stream = gather_core_states(power_state_stream, freq_dependent_idle_states)

    # execute the pipeline
    for timestamp, states in core_state_stream:
        for reporter in reporters:
            reporter.update(timestamp, states)

    # report any issues encountered while executing the pipeline
    if ps_processor.exceptions:
        logger.warning('There were errors while processing trace:')
        for e in ps_processor.exceptions:
            logger.warning(str(e))

    # generate reports
    reports = {}
    for reporter in reporters:
        report = reporter.report()
        report.write()
        reports[report.name] = report
    return reports
