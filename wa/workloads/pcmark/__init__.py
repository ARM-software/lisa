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
import os
import re
import zipfile

from wa import ApkUiautoWorkload

class PcMark(ApkUiautoWorkload):

    name = 'pcmark'
    package_names = ['com.futuremark.pcmark.android.benchmark']
    regex_matches = [re.compile(r'PcmaWebV2Score>([\d.]+)'), 
                     re.compile(r'PcmaVideoEditingScore>([\d.]+)'), 
                     re.compile(r'PcmaDataManipulationScore>([\d.]+)'),
                     re.compile(r'PcmaPhotoEditingV2Score>([\d.]+)'),
                     re.compile(r'PcmaWorkv2Score>([\d.]+)'),
                     re.compile(r'PcmaWritingV2Score>([\d.]+)')]
    description = '''
    A workload to execute the Work 2.0 benchmarks within PCMark - https://www.futuremark.com/benchmarks/pcmark-android

    Test description:
    1. Open PCMark application
    2. Swipe right to the Benchmarks screen
    3. Select the Work 2.0 benchmark
    4. Install the Work 2.0 benchmark
    5. Execute the Work 2.0 benchmark

    Known working APK version: 2.0.3716
    '''

    def __init__(self, target, **kwargs):
        super(PcMark, self).__init__(target, **kwargs)
        self.gui.timeout = 1500

    def extract_results(self, context):
        results_path = self.target.path.join(self.target.external_storage, "PCMark for Android")
        result_file = self.target.list_directory(results_path)[-1]
        self.result_file = result_file.rstrip()
        result = self.target.path.join(results_path, result_file)
        self.target.pull(result, context.output_directory)
        context.add_artifact('pcmark-result', self.result_file, kind='raw')

    def update_output(self, context):
        expected_results = len(self.regex_matches)
        zf = zipfile.ZipFile(os.path.join(context.output_directory, self.result_file), 'r').read('Result.xml')
        for line in zf.split('\n'):
            for regex in self.regex_matches:
                match = regex.search(line)
                if match:
                    scores = float(match.group(1))
                    entry = regex.pattern
                    entry = entry[:-9]
                    context.add_metric(entry, scores, lower_is_better=False)
                    expected_results -= 1
        if expected_results > 0:
            msg = "The PCMark workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(self.regex_matches), expected_results))
