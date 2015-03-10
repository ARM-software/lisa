#    Copyright 2013-2015 ARM Limited
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


import re

from wlauto import AndroidUiAutoBenchmark


class Sqlite(AndroidUiAutoBenchmark):

    name = 'sqlitebm'
    description = """
    Measures the performance of the sqlite database. It determines within
    what time the target device processes a number of SQL queries.

    """
    package = 'com.redlicense.benchmark.sqlite'
    activity = '.Main'
    summary_metrics = ['Overall']

    score_regex = re.compile(r'V/sqlite.*:\s+([\w ]+) = ([\d\.]+) sec')

    def update_result(self, context):
        super(Sqlite, self).update_result(context)
        with open(self.logcat_log) as fh:
            text = fh.read()
            for match in self.score_regex.finditer(text):
                metric = match.group(1)
                value = match.group(2)
                try:
                    value = float(value)
                except ValueError:
                    self.logger.warn("Reported results do not match expected format (seconds)")
                context.result.add_metric(metric, value, 'Seconds', lower_is_better=True)

