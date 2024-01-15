#    Copyright 2018 ARM Limited
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

# pylint: disable=E1101,W0201,E0203


import os
import re
import select
import json
import threading
import sqlite3
import subprocess
import sys
from copy import copy

import pandas as pd

from wa import ApkWorkload, Parameter, WorkloadError, ConfigError
from wa.utils.types import list_or_string, numeric


DELAY = 2


class Jankbench(ApkWorkload):

    name = 'jankbench'
    description = """
    Internal Google benchmark for evaluating jank on Android.

    """
    package_names = ['com.android.benchmark']
    activity = '.app.RunLocalBenchmarksActivity'

    results_db_file = 'BenchmarkResults'

    iteration_regex = re.compile(r'System.out: iteration: (?P<iteration>[0-9]+)')
    metrics_regex = re.compile(
        r'System.out: Mean: (?P<mean>[0-9\.]+)\s+JankP: (?P<junk_p>[0-9\.]+)\s+'
        r'StdDev: (?P<std_dev>[0-9\.]+)\s+Count Bad: (?P<count_bad>[0-9]+)\s+'
        r'Count Jank: (?P<count_junk>[0-9]+)'
    )

    valid_test_ids = [
        # Order matters -- the index of the id must match what is expected by
        # the App.
        'list_view',
        'image_list_view',
        'shadow_grid',
        'low_hitrate_text',
        'high_hitrate_text',
        'edit_text',
        'overdraw_test',
    ]

    parameters = [
        Parameter('test_ids', kind=list_or_string,
                  allowed_values=valid_test_ids,
                  description='ID of the jankbench test to be run.'),
        Parameter('loops', kind=int, default=1, constraint=lambda x: x > 0, aliases=['reps'],
                  description='''
                  Specifies the number of times the benchmark will be run in a "tight loop",
                  i.e. without performaing setup/teardown inbetween.
                  '''),
        Parameter('pull_results_db', kind=bool,
                  description='''
                  Secifies whether an sqlite database with detailed results should be pulled
                  from benchmark app's data. This requires the device to be rooted.

                  This defaults to ``True`` for rooted devices and ``False`` otherwise.
                  '''),
        Parameter('timeout', kind=int, default=10 * 60, aliases=['run_timeout'],
                  description="""
                  Time out for workload execution. The workload will be killed if it hasn't completed
                  within this period.
                  """),
    ]

    def setup(self, context):
        super(Jankbench, self).setup(context)

        if self.pull_results_db is None:
            self.pull_results_db = self.target.is_rooted
        elif self.pull_results_db and not self.target.is_rooted:
            raise ConfigError('pull_results_db set for an unrooted device')

        if not self.target.is_container:
            self.target.ensure_screen_is_on()

        self.command = self._build_command()
        self.monitor = JankbenchRunMonitor(self.target)
        self.monitor.start()

    def run(self, context):
        result = self.target.execute(self.command, timeout=self.timeout)
        if 'FAILURE' in result:
            raise WorkloadError(result)
        else:
            self.logger.debug(result)
        self.target.sleep(DELAY)
        self.monitor.wait_for_run_end(self.timeout)

    def extract_results(self, context):
        self.monitor.stop()
        if self.pull_results_db:
            target_file = self.target.path.join(self.target.package_data_directory,
                                                self.package, 'databases', self.results_db_file)
            host_file = os.path.join(context.output_directory, self.results_db_file)
            self.target.pull(target_file, host_file, as_root=True)
            context.add_artifact('jankbench-results', host_file, 'data')

    def update_output(self, context):  # NOQA
        super(Jankbench, self).update_output(context)
        if self.pull_results_db:
            self.extract_metrics_from_db(context)
        else:
            self.extract_metrics_from_logcat(context)

    def extract_metrics_from_db(self, context):  # pylint: disable=no-self-use
        dbfile = context.get_artifact_path('jankbench-results')
        with sqlite3.connect(dbfile) as conn:
            df = pd.read_sql('select name, iteration, total_duration, jank_frame from ui_results', conn)
            g = df.groupby(['name', 'iteration'])
            janks = g.jank_frame.sum()
            janks_pc = janks / g.jank_frame.count() * 100
            results = pd.concat([
                g.total_duration.mean(),
                g.total_duration.std(),
                janks,
                janks_pc,
            ], axis=1)
            results.columns = ['mean', 'std_dev', 'count_jank', 'jank_p']

            for test_name, rep in results.index:
                test_results = results.loc[test_name, rep]
                for metric, value in test_results.items():
                    context.add_metric(metric, value, units=None, lower_is_better=True,
                                       classifiers={'test_name': test_name, 'rep': rep})

    def extract_metrics_from_logcat(self, context):
        metric_names = ['mean', 'junk_p', 'std_dev', 'count_bad', 'count_junk']
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file, errors='replace') as fh:
            run_tests = copy(self.test_ids or self.valid_test_ids)
            current_iter = None
            current_test = None
            for line in fh:

                match = self.iteration_regex.search(line)
                if match:
                    if current_iter is not None:
                        msg = 'Did not see results for iteration {} of {}'
                        self.logger.warning(msg.format(current_iter, current_test))
                    current_iter = int(match.group('iteration'))
                    if current_iter == 0:
                        try:
                            current_test = run_tests.pop(0)
                        except IndexError:
                            self.logger.warning('Encountered an iteration for an unknown test.')
                            current_test = 'unknown'
                    continue

                match = self.metrics_regex.search(line)
                if match:
                    if current_iter is None:
                        self.logger.warning('Encountered unexpected metrics (no iteration)')
                        continue

                    for name in metric_names:
                        value = numeric(match.group(name))
                        context.add_metric(name, value, units=None, lower_is_better=True,
                                           classifiers={'test_id': current_test, 'rep': current_iter})

                    current_iter = None

    def _build_command(self):
        command_params = []
        if self.test_ids:
            test_idxs = [str(self.valid_test_ids.index(i)) for i in self.test_ids]
            command_params.append('--eia com.android.benchmark.EXTRA_ENABLED_BENCHMARK_IDS {}'.format(','.join(test_idxs)))
        if self.loops:
            command_params.append('--ei com.android.benchmark.EXTRA_RUN_COUNT {}'.format(self.loops))
        return 'am start -W -S -n {}/{} {}'.format(self.package,
                                                   self.activity,
                                                   ' '.join(command_params))


class JankbenchRunMonitor(threading.Thread):

    regex = re.compile(r'I BENCH\s+:\s+BenchmarkDone!')

    def __init__(self, device):
        super(JankbenchRunMonitor, self).__init__()
        self.target = device
        self.daemon = True
        self.run_ended = threading.Event()
        self.stop_event = threading.Event()
        self.target.clear_logcat()
        if self.target.adb_name:
            self.command = ['adb', '-s', self.target.adb_name, 'logcat']
        else:
            self.command = ['adb', 'logcat']

    def run(self):
        proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while not self.stop_event.is_set():
            if self.run_ended.is_set():
                self.target.sleep(DELAY)
            else:
                ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 2)
                if ready:
                    line = ready[0].readline()
                    line = line.decode(sys.stdout.encoding, 'replace')
                    if self.regex.search(line):
                        self.run_ended.set()
        proc.terminate()

    def stop(self):
        self.stop_event.set()
        self.join()

    def wait_for_run_end(self, timeout):
        self.run_ended.wait(timeout)
        self.run_ended.clear()
