#    Copyright 2014-2016 ARM Limited
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

from wa import ApkUiautoWorkload, WorkloadError

class Antutu(ApkUiautoWorkload):

    name = 'antutu'
    package_names = ['com.antutu.ABenchMark']
    regex_matches = [re.compile(r'CPU Maths Score (.+)'),
                     re.compile(r'CPU Common Score (.+)'),
                     re.compile(r'CPU Multi Score (.+)'),
                     re.compile(r'GPU Marooned Score (.+)'),
                     re.compile(r'GPU Coastline Score (.+)'),
                     re.compile(r'GPU Refinery Score (.+)'),
                     re.compile(r'Data Security Score (.+)'),
                     re.compile(r'Data Processing Score (.+)'),
                     re.compile(r'Image Processing Score (.+)'),
                     re.compile(r'User Experience Score (.+)'),
                     re.compile(r'RAM Score (.+)'),
                     re.compile(r'ROM Score (.+)')]
    description = '''
    Executes Antutu 3D, UX, CPU and Memory tests

    Test description:
    1. Open Antutu application
    2. Execute Antutu benchmark

    Known working APK version: 7.0.4
    '''

    def update_output(self, context):
        super(Antutu, self).update_output(context)
        expected_results = len(self.regex_matches)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file) as fh:
            for line in fh:
                for regex in self.regex_matches:
                    match = regex.search(line)
                    if match:
                        try:
                            result = float(match.group(1))
                        except ValueError:
                            result = 'NaN'
                        entry = regex.pattern.rsplit(None, 1)[0]
                        context.add_metric(entry, result, lower_is_better=False)
                        expected_results -= 1
        if expected_results > 0:
            msg = "The Antutu workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(self.regex_matches), expected_results))
