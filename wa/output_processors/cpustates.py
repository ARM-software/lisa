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

from collections import OrderedDict

from devlib.utils.csvutil import csvwriter

from wa import OutputProcessor, Parameter
from wa.utils.cpustates import report_power_stats


def _get_cpustates_description():
    """
    Reuse the description for report_power_stats() but strip away it's
    parameter docs, as they are not relevant to the OuputProcessor.
    """
    output_lines = []
    lines = iter(report_power_stats.__doc__.split('\n'))
    line = next(lines)
    while True:
        try:
            if line.strip().startswith(':param'):
                while line.strip():
                    line = next(lines)
            output_lines.append(line)
            line = next(lines)
        except StopIteration:
            break
    return '\n'.join(output_lines)


class CpuStatesProcessor(OutputProcessor):

    name = 'cpustates'

    description = _get_cpustates_description()

    parameters = [
        Parameter('use_ratios', kind=bool, default=False,
                  description="""
                  By default proportional values will be reported as
                  percentages, if this flag is enabled, they will be reported
                  as ratios instead.
                  """),
        Parameter('no_idle', kind=bool, default=False,
                  description="""
                  Indicate that there will be no idle transitions in the trace.
                  By default, a core will be reported as being in an "unknown"
                  state until the first idle transtion for that core. Normally,
                  this is not an issue, as cores are "nudged" as part of the
                  setup to ensure that there is an idle transtion before the
                  meassured region. However, if all idle states for the core
                  have been disabled, or if the kernel does not have cpuidle,
                  the nudge will not result in an idle transition, which would
                  cause the cores to be reported to be in "unknown" state for
                  the entire execution.

                  If this parameter is set to ``True``, the processor will
                  assume that cores are running prior to the begining of the
                  issue, and they will leave unknown state on the first
                  frequency transition.
                  """),
        Parameter('split_wfi_states', kind=bool, default=False,
                  description="""
                  WFI is a very shallow idle state. The core remains powered on
                  when in this state, which means the power usage while in this
                  state will depend on the current voltage, and therefore current
                  frequency.

                  Setting this to ``True`` will track time spent in WFI at
                  each frequency separately, allowing to gain the most accurate
                  picture of energy usage.
                  """),
    ]

    def __init__(self, *args, **kwargs):
        super(CpuStatesProcessor, self).__init__(*args, **kwargs)
        self.iteration_reports = OrderedDict()

    def process_job_output(self, output, target_info, run_output):  # pylint: disable=unused-argument
        trace_file = output.get_artifact_path('trace-cmd-txt')
        if not trace_file:
            self.logger.warning('Text trace does not appear to have been generated; skipping this iteration.')
            return
        if 'cpufreq' not in target_info.modules:
            msg = '"cpufreq" module not detected on target, cpu frequency information may be missing.'
            self.logger.warning(msg)
        if 'cpuidle' not in target_info.modules:
            msg = '"cpuidle" module not detected on target, cpu idle information may be missing.'
            self.logger.debug(msg)

        self.logger.info('Generating power state reports from trace...')
        reports = report_power_stats(  # pylint: disable=unbalanced-tuple-unpacking
            trace_file=trace_file,
            output_basedir=output.basepath,
            cpus=target_info.cpus,
            use_ratios=self.use_ratios,
            no_idle=self.no_idle,
            split_wfi_states=self.split_wfi_states,
        )

        for report in reports.values():
            output.add_artifact(report.name, report.filepath, kind='data')

        iteration_id = (output.id, output.label, output.iteration)
        self.iteration_reports[iteration_id] = reports

    # pylint: disable=too-many-locals,unused-argument
    def process_run_output(self, output, target_info):
        if not self.iteration_reports:
            self.logger.warning('No power state reports generated.')
            return

        parallel_rows = []
        powerstate_rows = []
        for iteration_id, reports in self.iteration_reports.items():
            job_id, workload, iteration = iteration_id
            parallel_report = reports['parallel-stats']
            powerstate_report = reports['power-state-stats']

            for record in parallel_report.values:
                parallel_rows.append([job_id, workload, iteration] + record)
            for state in sorted(powerstate_report.state_stats):
                stats = powerstate_report.state_stats[state]
                powerstate_rows.append([job_id, workload, iteration, state]
                                       + ['{:.3f}'.format(s if s is not None else 0)
                                           for s in stats])

        outpath = output.get_path('parallel-stats.csv')
        with csvwriter(outpath) as writer:
            writer.writerow(['id', 'workload', 'iteration', 'cluster',
                             'number_of_cores', 'total_time',
                             '%time', '%running_time'])
            writer.writerows(parallel_rows)
        output.add_artifact('run-parallel-stats', outpath, kind='export')

        outpath = output.get_path('power-state-stats.csv')
        with csvwriter(outpath) as writer:
            headers = ['id', 'workload', 'iteration', 'state']
            headers += ['{} CPU{}'.format(c, i)
                        for i, c in enumerate(powerstate_report.core_names)]
            writer.writerow(headers)
            writer.writerows(powerstate_rows)
        output.add_artifact('run-power-state-stats', outpath, kind='export')
