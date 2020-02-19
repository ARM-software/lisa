#    Copyright 2020 ARM Limited
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
import time
from datetime import datetime, timedelta

import pandas as pd

from wa import Instrument, Parameter, File, InstrumentError


class ProcStatCollector(Instrument):

    name = 'proc_stat'
    description = '''
    Collect CPU load information from /proc/stat.
    '''

    parameters = [
        Parameter('period', int, default=5,
                  constraint=lambda x: x > 0,
                  description='''
                  Time (in seconds) between collections.
                  '''),
    ]

    def initialize(self, context):  # pylint: disable=unused-argument
        self.host_script = context.get_resource(File(self, 'gather-load.sh'))
        self.target_script = self.target.install(self.host_script)
        self.target_output = self.target.get_workpath('proc-stat-raw.csv')
        self.stop_file = self.target.get_workpath('proc-stat-stop.signal')

    def setup(self, context):  # pylint: disable=unused-argument
        self.command = '{} sh {} {} {} {} {}'.format(
            self.target.busybox,
            self.target_script,
            self.target.busybox,
            self.target_output,
            self.period,
            self.stop_file,
        )
        self.target.remove(self.target_output)
        self.target.remove(self.stop_file)

    def start(self, context):  # pylint: disable=unused-argument
        self.target.kick_off(self.command)

    def stop(self, context):  # pylint: disable=unused-argument
        self.target.execute('{} touch {}'.format(self.target.busybox, self.stop_file))

    def update_output(self, context):
        self.logger.debug('Waiting for collector script to terminate...')
        self._wait_for_script()
        self.logger.debug('Waiting for collector script to terminate...')
        host_output = os.path.join(context.output_directory, 'proc-stat-raw.csv')
        self.target.pull(self.target_output, host_output)
        context.add_artifact('proc-stat-raw', host_output, kind='raw')

        df = pd.read_csv(host_output)
        no_ts = df[df.columns[1:]]
        deltas = (no_ts - no_ts.shift())
        total = deltas.sum(axis=1)
        util = (total - deltas.idle) / total * 100
        out_df = pd.concat([df.timestamp, util], axis=1).dropna()
        out_df.columns = ['timestamp', 'cpu_util']

        util_file = os.path.join(context.output_directory, 'proc-stat.csv')
        out_df.to_csv(util_file, index=False)
        context.add_artifact('proc-stat', util_file, kind='data')

    def finalize(self, context):  # pylint: disable=unused-argument
        if self.cleanup_assets and getattr(self, 'target_output'):
            self.target.remove(self.target_output)
            self.target.remove(self.target_script)

    def _wait_for_script(self):
        start_time = datetime.utcnow()
        timeout = timedelta(seconds=300)
        while self.target.file_exists(self.stop_file):
            delta = datetime.utcnow() - start_time
            if delta > timeout:
                raise InstrumentError('Timed out wating for /proc/stat collector to terminate..')
