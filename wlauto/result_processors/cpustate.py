#    Copyright 2015 ARM Limited
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
import csv
from collections import OrderedDict

from wlauto import ResultProcessor, Parameter
from wlauto.exceptions import ConfigError
from wlauto.instrumentation import instrument_is_installed
from wlauto.utils.power import report_power_stats


class CpuStatesProcessor(ResultProcessor):

    name = 'cpustates'
    description = '''
    Process power ftrace to produce CPU state and parallelism stats.

    Parses trace-cmd output to extract power events and uses those to generate
    statistics about parallelism and frequency/idle core residency.

    .. note:: trace-cmd instrument must be enabled and configured to collect
              at least ``power:cpu_idle`` and ``power:cpu_frequency`` events.
              Reporting should also be enabled (it is by default) as
              ``cpustate`` parses the text version of the trace.
              Finally, the device should have ``cpuidle`` module installed.

    This generates two reports for the run:

    *parallel.csv*

    Shows what percentage of time was spent with N cores active (for N
    from 0 to the total number of cores), for a cluster or for a system as
    a whole. It contain the following columns:

        :workload: The workload label
        :iteration: iteration that was run
        :cluster: The cluster for which statics are reported. The value of
                  ``"all"`` indicates that this row reports statistics for
                  the whole system.
        :number_of_cores: number of cores active. ``0`` indicates the cluster
                          was idle.
        :total_time: Total time spent in this state during workload execution
        :%time: Percentage of total workload execution time spent in this state
        :%running_time: Percentage of the time the cluster was active (i.e.
                        ignoring time the cluster was idling) spent in this
                        state.

    *cpustate.csv*

    Shows percentage of the time a core spent in a particular power state. The first
    column names the state is followed by a column for each core. Power states include
    available DVFS frequencies (for heterogeneous systems, this is the union of
    frequencies supported by different core types) and idle states. Some shallow
    states (e.g. ARM WFI) will consume different amount of power depending on the
    current OPP. For such states, there will be an entry for each opp. ``"unknown"``
    indicates the percentage of time for which a state could not be established from the
    trace. This is usually due to core state being unknown at the beginning of the trace,
    but may also be caused by dropped events in the middle of the trace.

    '''

    parameters = [
        Parameter('first_cluster_state', kind=int, default=2,
                  description="""
                  The first idle state which is common to a cluster.
                  """),
        Parameter('first_system_state', kind=int, default=3,
                  description="""
                  The first idle state which is common to all cores.
                  """),
        Parameter('write_iteration_reports', kind=bool, default=False,
                  description="""
                  By default, this instrument will generate reports for the entire run
                  in the overall output directory. Enabling this option will, in addition,
                  create reports in each iteration's output directory. The formats of these
                  reports will be similar to the overall report, except they won't mention
                  the workload name or iteration number (as that is implied by their location).
                  """),
        Parameter('use_ratios', kind=bool, default=False,
                  description="""
                  By default proportional values will be reported as percentages, if this
                  flag is enabled, they will be reported as ratios instead.
                  """),
        Parameter('create_timeline', kind=bool, default=True,
                  description="""
                  Create a CSV with the timeline of core power states over the course of the run
                  as well as the usual stats reports.
                  """),

    ]

    def validate(self):
        if not instrument_is_installed('trace-cmd'):
            message = '''
            {} requires "trace-cmd" instrument to be installed and the collection of at
            least "power:cpu_frequency" and "power:cpu_idle" events to be enabled during worklad
            execution.
            '''
            raise ConfigError(message.format(self.name).strip())

    def initialize(self, context):
        # pylint: disable=attribute-defined-outside-init
        device = context.device
        if not device.has('cpuidle'):
            raise ConfigError('Device does not appear to have cpuidle capability; is the right module installed?')
        if not device.core_names:
            message = '{} requires"core_names" and "core_clusters" to be specified for the device.'
            raise ConfigError(message.format(self.name))
        self.core_names = device.core_names
        self.core_clusters = device.core_clusters
        idle_states = {s.id: s.desc for s in device.get_cpuidle_states()}
        self.idle_state_names = [idle_states[i] for i in sorted(idle_states.keys())]
        self.num_idle_states = len(self.idle_state_names)
        self.iteration_reports = OrderedDict()

    def process_iteration_result(self, result, context):
        trace = context.get_artifact('txttrace')
        if not trace:
            self.logger.debug('Text trace does not appear to have been generated; skipping this iteration.')
            return
        self.logger.debug('Generating power state reports from trace...')
        if self.create_timeline:
            timeline_csv_file = os.path.join(context.output_directory, 'power_states.csv')
        else:
            timeline_csv_file = None
        parallel_report, powerstate_report = report_power_stats(  # pylint: disable=unbalanced-tuple-unpacking
            trace_file=trace.path,
            idle_state_names=self.idle_state_names,
            core_names=self.core_names,
            core_clusters=self.core_clusters,
            num_idle_states=self.num_idle_states,
            first_cluster_state=self.first_cluster_state,
            first_system_state=self.first_system_state,
            use_ratios=self.use_ratios,
            timeline_csv_file=timeline_csv_file,
        )
        if parallel_report is None:
            self.logger.warning('No power state reports generated; are power '
                                'events enabled in the trace?')
            return
        else:
            self.logger.debug('Reports generated.')

        iteration_id = (context.spec.id, context.spec.label, context.current_iteration)
        self.iteration_reports[iteration_id] = (parallel_report, powerstate_report)
        if self.write_iteration_reports:
            self.logger.debug('Writing iteration reports')
            parallel_report.write(os.path.join(context.output_directory, 'parallel.csv'))
            powerstate_report.write(os.path.join(context.output_directory, 'cpustates.csv'))

    def process_run_result(self, result, context):  # pylint: disable=too-many-locals
        if not self.iteration_reports:
            self.logger.warning('No power state reports generated.')
            return

        parallel_rows = []
        powerstate_rows = []
        for iteration_id, reports in self.iteration_reports.iteritems():
            spec_id, workload, iteration = iteration_id
            parallel_report, powerstate_report = reports
            for record in parallel_report.values:
                parallel_rows.append([spec_id, workload, iteration] + record)
            for state in sorted(powerstate_report.state_stats):
                stats = powerstate_report.state_stats[state]
                powerstate_rows.append([spec_id, workload, iteration, state] +
                                       ['{:.3f}'.format(s if s is not None else 0)
                                           for s in stats])

        with open(os.path.join(context.output_directory, 'parallel.csv'), 'w') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(['id', 'workload', 'iteration', 'cluster',
                             'number_of_cores', 'total_time',
                             '%time', '%running_time'])
            writer.writerows(parallel_rows)

        with open(os.path.join(context.output_directory, 'cpustate.csv'), 'w') as wfh:
            writer = csv.writer(wfh)
            headers = ['id', 'workload', 'iteration', 'state']
            headers += ['{} CPU{}'.format(c, i)
                        for i, c in enumerate(powerstate_report.core_names)]
            writer.writerow(headers)
            writer.writerows(powerstate_rows)

