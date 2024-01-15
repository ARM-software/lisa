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
import os
import re
import sys
import zipfile

from wa import ApkUiautoWorkload, Parameter
from wa.framework.exception import WorkloadError


class PcMark(ApkUiautoWorkload):

    name = 'pcmark'

    supported_versions = ['3', '2']
    package_names = ['com.futuremark.pcmark.android.benchmark']
    regex_matches = {
        '2': [re.compile(r'PcmaWebV2Score>([\d.]+)'),
              re.compile(r'PcmaVideoEditingScore>([\d.]+)'),
              re.compile(r'PcmaDataManipulationScore>([\d.]+)'),
              re.compile(r'PcmaPhotoEditingV2Score>([\d.]+)'),
              re.compile(r'PcmaWorkv2Score>([\d.]+)'),
              re.compile(r'PcmaWritingV2Score>([\d.]+)')],

        '3': [re.compile(r'PcmaWebV3Score>([\d.]+)'),
              re.compile(r'PcmaVideoEditingV3Score>([\d.]+)'),
              re.compile(r'PcmaDataManipulationV3Score>([\d.]+)'),
              re.compile(r'PcmaPhotoEditingV3Score>([\d.]+)'),
              re.compile(r'PcmaWorkv3Score>([\d.]+)'),
              re.compile(r'PcmaWritingV3Score>([\d.]+)')]
    }

    description = '''
    A workload to execute the Work x.0 benchmarks within PCMark - https://www.futuremark.com/benchmarks/pcmark-android

    Test description:
    1. Open PCMark application
    2. Swipe right to the Benchmarks screen
    3. Select the Work x.0 benchmark
    4. If needed, install the Work x.0 benchmark (requires an internet connection)
    5. Execute the Work x.0 benchmark

    Known working APK versions: 3.0.4061, 2.0.3716
    '''

    # Do not delete Work x.0 data-set before each run
    clear_data_on_reset = False

    parameters = [
        Parameter('version', allowed_values=supported_versions,
                  description='Specifies which version of the workload should be run.',
                  override=True)
    ]

    def __init__(self, target, **kwargs):
        super(PcMark, self).__init__(target, **kwargs)
        self.gui.timeout = 1500

    def initialize(self, context):
        super(PcMark, self).initialize(context)
        self.major_version = self.version.strip()[0]

    def extract_results(self, context):
        if self.version.startswith('3'):
            results_path = self.target.path.join(self.target.package_data_directory, self.package, 'files')
            result_file = [f for f in self.target.list_directory(results_path, as_root=self.target.is_rooted) if f.endswith(".zip")][-1]
        elif self.version.startswith('2'):
            results_path = self.target.path.join(self.target.external_storage, "PCMark for Android")
            result_file = self.target.list_directory(results_path)[-1]

        self.result_file = result_file.rstrip()
        result = self.target.path.join(results_path, result_file)
        self.target.pull(result, context.output_directory, as_root=self.target.is_rooted)
        context.add_artifact('pcmark-result', self.result_file, kind='raw')

    def update_output(self, context):
        expected_results = len(self.regex_matches[self.major_version])
        zf = zipfile.ZipFile(os.path.join(context.output_directory, self.result_file), 'r').read('Result.xml')
        zf = zf.decode(sys.stdout.encoding)
        for line in zf.split('\n'):
            for regex in self.regex_matches[self.major_version]:
                match = regex.search(line)
                if match:
                    scores = float(match.group(1))
                    entry = regex.pattern
                    entry = entry[:-9]
                    context.add_metric(entry, scores, lower_is_better=False)
                    expected_results -= 1
        if expected_results > 0:
            msg = "The PCMark workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(self.regex_matches[self.major_version]), expected_results))
