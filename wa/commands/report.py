from collections import Counter
from datetime import datetime, timedelta
import logging
import os

from wa import Command, settings
from wa.framework.configuration.core import Status
from wa.framework.output import RunOutput, discover_wa_outputs
from wa.utils.doc import underline
from wa.utils.log import COLOR_MAP, RESET_COLOR
from wa.utils.terminalsize import get_terminal_size


class ReportCommand(Command):

    name = 'report'
    description = '''
    Monitor an ongoing run and provide information on its progress.

    Specify the output directory of the run you would like the monitor;
    alternatively report will attempt to discover wa output directories
    within the current directory. The output includes run information such as
    the UUID, start time, duration, project name and a short summary of the
    run's progress (number of completed jobs, the number of jobs in each
    different status).

    If verbose output is specified, the output includes a list of all events
    labelled as not specific to any job, followed by a list of the jobs in the
    order executed, with their retries (if any), current status and, if the job
    is finished, a list of events that occurred during that job's execution.

    This is an example of a job status line:

        wk1 (exoplayer) [1] - 2, PARTIAL

    It contains two entries delimited by a comma: the job's descriptor followed
    by its completion status (``PARTIAL``, in this case). The descriptor
    consists of the following elements:

        - the job ID (``wk1``)
        - the job label (which defaults to the workload name) in parentheses
        - job iteration number in square brakets (``1`` in this case)
        - a hyphen followed by the retry attempt number.
            (note: this will only be shown if the job has been retried as least
            once. If the job has not yet run, or if it completed on the first
            attempt, the hyphen and retry count -- which in that case would be
            zero -- will not appear).
    '''

    def initialize(self, context):
        self.parser.add_argument('-d', '--directory',
                                 help='''
                                 Specify the WA output path. report will
                                 otherwise attempt to discover output
                                 directories in the current directory.
                                 ''')

    def execute(self, state, args):
        if args.directory:
            output_path = args.directory
            run_output = RunOutput(output_path)
        else:
            possible_outputs = list(discover_wa_outputs(os.getcwd()))
            num_paths = len(possible_outputs)

            if num_paths > 1:
                print('More than one possible output directory found,'
                      ' please choose a path from the following:'
                      )

                for i in range(num_paths):
                    print("{}: {}".format(i, possible_outputs[i].basepath))

                while True:
                    try:
                        select = int(input())
                    except ValueError:
                        print("Please select a valid path number")
                        continue

                    if select not in range(num_paths):
                        print("Please select a valid path number")
                        continue
                    break

                run_output = possible_outputs[select]

            else:
                run_output = possible_outputs[0]

        rm = RunMonitor(run_output)
        print(rm.generate_output(args.verbose))


class RunMonitor:

    @property
    def elapsed_time(self):
        if self._elapsed is None:
            if self.ro.info.duration is None:
                self._elapsed = datetime.utcnow() - self.ro.info.start_time
            else:
                self._elapsed = self.ro.info.duration
        return self._elapsed

    @property
    def job_outputs(self):
        if self._job_outputs is None:
            self._job_outputs = {
                (j_o.id, j_o.label, j_o.iteration): j_o for j_o in self.ro.jobs
            }
        return self._job_outputs

    @property
    def projected_duration(self):
        elapsed = self.elapsed_time.total_seconds()
        proj = timedelta(seconds=elapsed * (len(self.jobs) / len(self.segmented['finished'])))
        return proj - self.elapsed_time

    def __init__(self, ro):
        self.ro = ro
        self._elapsed = None
        self._p_duration = None
        self._job_outputs = None
        self._termwidth = None
        self._fmt = _simple_formatter()
        self.get_data()

    def get_data(self):
        self.jobs = [state for label_id, state in self.ro.state.jobs.items()]
        if self.jobs:
            rc = self.ro.run_config
            self.segmented = segment_jobs_by_state(self.jobs,
                                                   rc.max_retries,
                                                   rc.retry_on_status
                                                   )

    def generate_run_header(self):
        info = self.ro.info

        header = underline('Run Info')
        header += "UUID: {}\n".format(info.uuid)
        if info.run_name:
            header += "Run name: {}\n".format(info.run_name)
        if info.project:
            header += "Project: {}\n".format(info.project)
        if info.project_stage:
            header += "Project stage: {}\n".format(info.project_stage)

        if info.start_time:
            duration = _seconds_as_smh(self.elapsed_time.total_seconds())
            header += ("Start time: {}\n"
                       "Duration: {:02}:{:02}:{:02}\n"
                       ).format(info.start_time,
                                duration[2], duration[1], duration[0],
                                )
            if self.segmented['finished'] and not info.end_time:
                p_duration = _seconds_as_smh(self.projected_duration.total_seconds())
                header += "Projected time remaining: {:02}:{:02}:{:02}\n".format(
                    p_duration[2], p_duration[1], p_duration[0]
                )

            elif self.ro.info.end_time:
                header += "End time: {}\n".format(info.end_time)

        return header + '\n'

    def generate_job_summary(self):
        total = len(self.jobs)
        num_fin = len(self.segmented['finished'])

        summary = underline('Job Summary')
        summary += 'Total: {}, Completed: {} ({}%)\n'.format(
            total, num_fin, (num_fin / total) * 100
        ) if total > 0 else 'No jobs created\n'

        ctr = Counter()
        for run_state, jobs in ((k, v) for k, v in self.segmented.items() if v):
            if run_state == 'finished':
                ctr.update([job.status.name.lower() for job in jobs])
            else:
                ctr[run_state] += len(jobs)

        return summary + ', '.join(
            [str(count) + ' ' + self._fmt.highlight_keyword(status) for status, count in ctr.items()]
        ) + '\n\n'

    def generate_job_detail(self):
        detail = underline('Job Detail')
        for job in self.jobs:
            detail += ('{} ({}) [{}]{}, {}\n').format(
                job.id,
                job.label,
                job.iteration,
                ' - ' + str(job.retries)if job.retries else '',
                self._fmt.highlight_keyword(str(job.status))
            )

            job_output = self.job_outputs[(job.id, job.label, job.iteration)]
            for event in job_output.events:
                detail += self._fmt.fit_term_width(
                    '\t{}\n'.format(event.summary)
                )
        return detail

    def generate_run_detail(self):
        detail = underline('Run Events') if self.ro.events else ''

        for event in self.ro.events:
            detail += '{}\n'.format(event.summary)

        return detail + '\n'

    def generate_output(self, verbose):
        if not self.jobs:
            return 'No jobs found in output directory\n'

        output = self.generate_run_header()
        output += self.generate_job_summary()

        if verbose:
            output += self.generate_run_detail()
            output += self.generate_job_detail()

        return output


def _seconds_as_smh(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return seconds, minutes, hours


def segment_jobs_by_state(jobstates, max_retries, retry_status):
    finished_states = [
        Status.PARTIAL, Status.FAILED,
        Status.ABORTED, Status.OK, Status.SKIPPED
    ]

    segmented = {
        'finished': [], 'other': [], 'running': [],
        'pending': [], 'uninitialized': []
    }

    for jobstate in jobstates:
        if (jobstate.status in retry_status) and jobstate.retries < max_retries:
            segmented['running'].append(jobstate)
        elif jobstate.status in finished_states:
            segmented['finished'].append(jobstate)
        elif jobstate.status == Status.RUNNING:
            segmented['running'].append(jobstate)
        elif jobstate.status == Status.PENDING:
            segmented['pending'].append(jobstate)
        elif jobstate.status == Status.NEW:
            segmented['uninitialized'].append(jobstate)
        else:
            segmented['other'].append(jobstate)

    return segmented


class _simple_formatter:
    color_map = {
        'running': COLOR_MAP[logging.INFO],
        'partial': COLOR_MAP[logging.WARNING],
        'failed': COLOR_MAP[logging.CRITICAL],
        'aborted': COLOR_MAP[logging.ERROR]
    }

    def __init__(self):
        self.termwidth = get_terminal_size()[0]
        self.color = settings.logging['color']

    def fit_term_width(self, text):
        text = text.expandtabs()
        if len(text) <= self.termwidth:
            return text
        else:
            return text[0:self.termwidth - 4] + " ...\n"

    def highlight_keyword(self, kw):
        if not self.color or kw not in _simple_formatter.color_map:
            return kw

        color = _simple_formatter.color_map[kw.lower()]
        return '{}{}{}'.format(color, kw, RESET_COLOR)
