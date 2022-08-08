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

import json
import os
from collections import defaultdict

import pandas as pd

from wa import Workload, Parameter, ConfigError, TargetError, WorkloadError
from wa.utils.exec_control import once


class Mongoperf(Workload):

    name = 'mongoperf'
    description = """
    A utility to check disk I/O performance independently of MongoDB.

    It times tests of random disk I/O and presents the results. You can use
    mongoperf for any case apart from MongoDB. The mmf true mode is completely
    generic.

    .. note:: ``mongoperf`` seems to ramp up threads in powers of two over a
              period of tens of seconds (there doesn't appear to be a way to
              change that). Bear this in mind when setting the ``duration``.
    """

    parameters = [
        Parameter('duration', kind=int, default=300,
                  description="""
                  Duration of of the workload.
                  """),
        Parameter('threads', kind=int, default=16,
                  description="""
                  Defines the number of threads mongoperf will use in the test.
                  To saturate the system storage system you will need
                  multiple threads.
                  """),
        Parameter('file_size_mb', kind=int, default=1,
                  description="""
                  Test file size in MB.
                  """),
        Parameter('sleep_micros', kind=int, default=0,
                  description="""
                  mongoperf will pause for this number of microseconds  divided
                  by the the number of threads between each operation.
                  """),
        Parameter('mmf', kind=bool, default=True,
                  description="""
                  When ``True``,  use memory mapped files for the tests.
                  Generally:

                  - when mmf is ``False``, mongoperf tests direct, physical, I/O,
                    without caching. Use a large file size to test heavy random
                    I/O load and to avoid I/O coalescing.
                  - when mmf is ``True``, mongoperf runs tests of the caching
                    system, and can use normal file system cache. Use mmf in
                    this mode to test file system cache behavior with memory
                    mapped files.
                  """),
        Parameter('read', kind=bool, default=True,
                  aliases=['r'],
                  description="""
                  When ``True``,  perform reads as part of the test. Either
                  ``read`` or ``write`` must be ``True``.
                  """),
        Parameter('write', kind=bool, default=True,
                  aliases=['w'],
                  description="""
                  When ``True``,  perform writes as part of the test. Either
                  ``read`` or ``write`` must be ``True``.
                  """),
        Parameter('rec_size_kb', kind=int, default=4,
                  description="""
                  The size of each write operation
                  """),
        Parameter('sync_delay', kind=int, default=0,
                  description="""
                  Seconds between disk flushes. Only use this if ``mmf`` is set
                  to ``True``.
                  """),
    ]

    def validate(self):
        if not self.read and not self.write:
            raise ConfigError('Either "read" or "write" must be True.')
        if not self.mmf and self.sync_delay:
            raise ConfigError('sync_delay can only be set if mmf is True')

    @once
    def initialize(self, context):
        try:
            self.target.execute('mongoperf -h')
        except TargetError:
            raise WorkloadError('Mongoperf must be installed and in $PATH on the target.')

    def setup(self, context):
        config = {}
        config['nThreads'] = self.threads
        config['fileSizeMB'] = self.file_size_mb
        config['sleepMicros'] = self.sleep_micros
        config['mmf'] = self.mmf
        config['r'] = self.read
        config['w'] = self.write
        config['recSizeKB'] = self.rec_size_kb
        config['syncDelay'] = self.sync_delay

        config_text = json.dumps(config)
        self.outfile = self.target.get_workpath('mongperf.out')
        self.command = 'echo "{}" | mongoperf > {}'.format(config_text, self.outfile)

    def run(self, context):
        self.target.kick_off(self.command)
        self.target.sleep(self.duration)
        self.target.killall('mongoperf', signal='SIGTERM')

    def extract_results(self, context):
        host_outfile = os.path.join(context.output_directory, 'mongoperf.out')
        self.target.pull(self.outfile, host_outfile)
        context.add_artifact('mongoperf-output', host_outfile, kind='raw')

    def update_output(self, context):
        host_file = context.get_artifact_path('mongoperf-output')
        results = defaultdict(list)
        threads = None
        with open(host_file) as fh:
            for line in fh:
                if 'new thread,' in line:
                    threads = int(line.split()[-1])
                elif 'ops/sec' in line:
                    results[threads].append(int(line.split()[0]))

        if not results:
            raise WorkloadError('No mongoperf results found in the output.')

        for threads, values in results.items():
            rs = pd.Series(values)
            context.add_metric('ops_per_sec', rs.mean(),
                               classifiers={'threads': threads})
            context.add_metric('ops_per_sec_std', rs.std(), lower_is_better=True,
                               classifiers={'threads': threads})

    def teardown(self, context):
        if self.cleanup_assets:
            self.target.remove(self.outfile)
