#    Copyright 2014-2018 ARM Limited
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

from wa import ApkUiautoWorkload
from wa.framework.exception import WorkloadError


class Androbench(ApkUiautoWorkload):

    name = 'androbench'
    package_names = ['com.andromeda.androbench2']
    regex_matches = [re.compile(r'Sequential Read Score ([\d.]+)'),
                     re.compile(r'Sequential Write Score ([\d.]+)'),
                     re.compile(r'Random Read Score ([\d.]+)'),
                     re.compile(r'Random Write Score ([\d.]+)'),
                     re.compile(r'SQL Insert Score ([\d.]+)'),
                     re.compile(r'SQL Update Score ([\d.]+)'),
                     re.compile(r'SQL Delete Score ([\d.]+)')]
    description = '''
    Executes storage performance benchmarks

    The Androbench workflow carries out the following typical productivity tasks.
    1. Open Androbench application
    2. Execute all memory benchmarks

    Known working APK version: 5.0.1
    '''

    def update_output(self, context):
        super(Androbench, self).update_output(context)
        expected_results = len(self.regex_matches)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file, errors='replace') as fh:
            for line in fh:
                for regex in self.regex_matches:
                    match = regex.search(line)
                    if match:
                        result = float(match.group(1))
                        entry = regex.pattern.rsplit(None, 1)[0]
                        context.add_metric(entry, result, 'MB/s', lower_is_better=False)
                        expected_results -= 1
        if expected_results > 0:
            msg = "The Androbench workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(self.regex_matches), expected_results))
