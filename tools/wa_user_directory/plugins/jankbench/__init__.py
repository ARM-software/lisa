#    Copyright 2017 ARM Limited
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

import csv
import os
import re
import subprocess
import threading
import select
import sqlite3

from wa import Parameter, ApkWorkload
from wa.framework.exception import WorkloadError

REGEXPS = {
    'start': (r'.*START.*'
              'cmp=com.android.benchmark/.app.RunLocalBenchmarksActivity.*'),
    'count': '.*iteration: (?P<iteration>[0-9]+).*',
    'metrics': (r'.*Mean: (?P<mean>[0-9\.]+)\s+JankP: (?P<junk_p>[0-9\.]+)\s+'
                'StdDev: (?P<std_dev>[0-9\.]+)\s+Count Bad: (?P<count_bad>[0-9]+)\s+'
                'Count Jank: (?P<count_junk>[0-9]+).*'),
    'done': r'.*BenchmarkDone!.*',
}

class Jankbench(ApkWorkload):

    name = 'jankbench'
    description = """
    Google's Jankbench benchmark.

    Jankbench simulates user interaction with Android UI components and records
    frame rendering times and 'jank' (rendering discontinuity) in an SQLite
    database. This  is believed to be a good proxy for the smoothness of user
    experience.

    Dumps a JankbenchResults.sqlite file in the output directory. This database
    contains a table 'ui_results' with a row for each frame, showing its
    rendering time in ms in the 'total_duration' column, and whether or not it
    was a jank frame in the 'jank_frame' column.

    This information is also extracted from the SQLite file and dumped as
    jankbench_frames.csv. This is _not_ necessarily the same information as
    provided by gfxinfo (fps instrument).
    """

    versions = ['1.0']
    activity = '.app.RunLocalBenchmarksActivity'
    package = 'com.android.benchmark'
    package_names = [package]

    target_db_path = '/data/data/{}/databases/BenchmarkResults'.format(package)

    test_ids = {
        'list_view'         : 0,
        'image_list_view'   : 1,
        'shadow_grid'       : 2,
        'low_hitrate_text'  : 3,
        'high_hitrate_text' : 4,
        'edit_text'         : 5,
    }

    parameters = [
        Parameter('test',
                  default=test_ids.keys()[0], allowed_values=test_ids.keys(),
                  description='Which Jankbench sub-benchmark to run'),
        Parameter('run_timeout', kind=int, default=10 * 60,
                  description="""
                  Timeout for workload execution. The workload will be killed if it hasn't completed
                  within this period. In seconds.
                  """),
        Parameter('times', kind=int, default=1, constraint=lambda x: x > 0,
                  description=('Specifies the number of times the benchmark will be run in a "tight '
                               'loop", i.e. without performing setup/teardown in between.')),
    ]

    def initialize(self, context):
        super(Jankbench, self).initialize(context)

        # Need root to get results database
        if not self.target.is_rooted:
            raise WorkloadError('Jankbench workload requires device to be rooted')

    def setup(self, context):
        super(Jankbench, self).setup(context)
        self.monitor = self.target.get_logcat_monitor(REGEXPS.values())
        self.monitor.start()

        self.command = (
            'am start -n com.android.benchmark/.app.RunLocalBenchmarksActivity '
            '--eia com.android.benchmark.EXTRA_ENABLED_BENCHMARK_IDS {0} '
            '--ei com.android.benchmark.EXTRA_RUN_COUNT {1}'
        ).format(self.test_ids[self.test], self.times)


    def run(self, context):
        # All we need to do is
        # - start the activity,
        # - then use the JbRunMonitor to wait until the benchmark reports on
        #   logcat that it is finished,
        # - pull the result database file.

        result = self.target.execute(self.command)
        if 'FAILURE' in result:
            raise WorkloadError(result)
        else:
            self.logger.debug(result)

        self.monitor.wait_for(REGEXPS['start'], timeout=30)
        self.logger.info('Detected Jankbench start')

        self.monitor.wait_for(REGEXPS['done'], timeout=300*self.times)

    def extract_results(self, context):
        # TODO make these artifacts where they should be
        super(Jankbench, self).extract_results(context)
        host_db_path =  os.path.join(context.output_directory,
                                     'BenchmarkResults.sqlite')
        self.target.pull(self.target_db_path, host_db_path, as_root=True)
        context.add_artifact('jankbench_results_db', host_db_path, 'data')

        columns = ['_id', 'name', 'run_id', 'iteration', 'total_duration', 'jank_frame']
        jank_frame_idx = columns.index('jank_frame')
        query = 'SELECT {} FROM ui_results'.format(','.join(columns))
        conn = sqlite3.connect(os.path.join(host_db_path))

        csv_path = os.path.join(context.output_directory, 'jankbench_frames.csv')
        jank_frames = 0
        with open(csv_path, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for db_row in conn.execute(query):
                writer.writerow(db_row)
                if int(db_row[jank_frame_idx]):
                    jank_frames += 1
        context.add_artifact('jankbench_results_csv', csv_path, 'data')

        context.add_metric('jankbench_jank_frames', jank_frames,
                           lower_is_better=True)

    def teardown(self, context):
        self.monitor.stop()
