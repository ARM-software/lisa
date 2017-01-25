# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import os
import logging

from subprocess import Popen, PIPE

from android import Screen, System, Workload

# Available test workloads
_jankbench = {
    'list_view'         : 0,
    'image_list_view'   : 1,
    'shadow_grid'       : 2,
    'low_hitrate_text'  : 3,
    'high_hitrate_text' : 4,
    'edit_text'         : 5,
}

# Regexps for benchmark synchronization
JANKBENCH_BENCHMARK_START_RE = re.compile(
    r'ActivityManager: START.*'
    '(cmp=com.android.benchmark/.app.RunLocalBenchmarksActivity)'
)
JANKBENCH_ITERATION_COUNT_RE = re.compile(
    r'System.out: iteration: (?P<iteration>[0-9]+)'
)
JANKBENCH_ITERATION_METRICS_RE = re.compile(
    r'System.out: Mean: (?P<mean>[0-9\.]+)\s+JankP: (?P<junk_p>[0-9\.]+)\s+'
    'StdDev: (?P<std_dev>[0-9\.]+)\s+Count Bad: (?P<count_bad>[0-9]+)\s+'
    'Count Jank: (?P<count_junk>[0-9]+)'
)
JANKBENCH_BENCHMARK_DONE_RE = re.compile(
    r'I BENCH\s+:\s+BenchmarkDone!'
)

JANKBENCH_DB_PATH = '/data/data/com.android.benchmark/databases/'
JANKBENCH_DB_NAME = 'BenchmarkResults'

class Jankbench(Workload):
    """
    Android Jankbench workload
    """

    # Package required by this workload
    package = 'com.android.benchmark'

    def __init__(self, test_env):
        super(Jankbench, self).__init__(test_env)
        self._log = logging.getLogger('Jankbench')
        self._log.debug('Workload created')

        # Set of output data reported by Jankbench
        self.db_file = None

    def run(self, out_dir, collect,
            test_name, iterations):

        # Keep track of mandatory parameters
        self.out_dir = out_dir
        self.collect = collect

        # Setup test id
        try:
            test_id = _jankbench[test_name]
        except KeyError:
            raise ValueError('Jankbench test [%s] not supported', test_name)

        # Initialize energy meter results
        nrg_report = None

        # Make sure we exit the app if already open
        System.menu(self.target)
        System.back(self.target)

        # Close and clear application
        System.force_stop(self.target, self.package, clear=True)

        # Set airplane mode
        System.set_airplane_mode(self.target, on=True)

        # Force screen in PORTRAIT mode
        Screen.set_orientation(self.target, portrait=True)

        # Clear logcat
        os.system(self._adb('logcat -c'));

        self._log.debug('Start Jank Benchmark [%d:%s]', test_id, test_name)
        test_cmd = 'am start -n "com.android.benchmark/.app.RunLocalBenchmarksActivity" '\
                    '--eia "com.android.benchmark.EXTRA_ENABLED_BENCHMARK_IDS" {0} '\
                    '--ei "com.android.benchmark.EXTRA_RUN_COUNT" {1}'\
                    .format(test_id, iterations)
        self._log.info(test_cmd)
        self.target.execute(test_cmd);

        # Parse logcat output lines
        logcat_cmd = self._adb(
                'logcat ActivityManager:* System.out:I *:S BENCH:*'\
                .format(self.target.adb_name))
        self._log.info(logcat_cmd)

        self._log.debug('Iterations:')
        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)
        while True:

            # read next logcat line (up to max 1024 chars)
            message = logcat.stdout.readline(1024)

            # Benchmark start trigger
            match = JANKBENCH_BENCHMARK_START_RE.search(message)
            if match:
                if 'energy' in collect and self.te.emeter:
                    self.te.emeter.reset()
                self._log.debug('Benchmark started!')

            # Benchmark completed trigger
            match = JANKBENCH_BENCHMARK_DONE_RE.search(message)
            if match:
                if 'energy' in collect and self.te.emeter:
                    nrg_report = self.te.emeter.report(out_dir)
                self._log.debug('Benchmark done!')
                break

            # Iteration completd
            match = JANKBENCH_ITERATION_COUNT_RE.search(message)
            if match:
                self._log.debug('Iteration %2d:',
                                int(match.group('iteration'))+1)
            # Iteration metrics
            match = JANKBENCH_ITERATION_METRICS_RE.search(message)
            if match:
                self._log.info('   Mean: %7.3f JankP: %7.3f StdDev: %7.3f Count Bad: %4d Count Jank: %4d',
                               float(match.group('mean')),
                               float(match.group('junk_p')),
                               float(match.group('std_dev')),
                               int(match.group('count_bad')),
                               int(match.group('count_junk')))

        # get results
        self.db_file = os.path.join(out_dir, JANKBENCH_DB_NAME)
        self.target.pull(JANKBENCH_DB_PATH + JANKBENCH_DB_NAME, self.db_file)

        System.force_stop(self.target, self.package, clear=True)

        # Go back to home screen
        System.home(self.target)

        # Reset initial setup
        Screen.set_orientation(self.target, auto=True)
        System.set_airplane_mode(self.target, on=False)

        return self.db_file, nrg_report

# vim :set tabstop=4 shiftwidth=4 expandtab
