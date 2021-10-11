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

from wa import ApkUiautoWorkload, WorkloadError, Parameter


class Antutu(ApkUiautoWorkload):

    name = 'antutu'
    package_names = ['com.antutu.ABenchMark']
    regex_matches_v7 = [re.compile(r'CPU Maths Score (.+)'),
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
    regex_matches_v8 = [re.compile(r'CPU Mathematical Operations Score (.+)'),
                        re.compile(r'CPU Common Algorithms Score (.+)'),
                        re.compile(r'CPU Multi-Core Score (.+)'),
                        re.compile(r'GPU Terracotta Score (.+)'),
                        re.compile(r'GPU Coastline Score (.+)'),
                        re.compile(r'GPU Refinery Score (.+)'),
                        re.compile(r'Data Security Score (.+)'),
                        re.compile(r'Data Processing Score (.+)'),
                        re.compile(r'Image Processing Score (.+)'),
                        re.compile(r'User Experience Score (.+)'),
                        re.compile(r'RAM Access Score (.+)'),
                        re.compile(r'ROM APP IO Score (.+)'),
                        re.compile(r'ROM Sequential Read Score (.+)'),
                        re.compile(r'ROM Sequential Write Score (.+)'),
                        re.compile(r'ROM Random Access Score (.+)')]
    regex_matches_v9 = [re.compile(r'CPU Mathematical Operations Score (.+)'),
                        re.compile(r'CPU Common Algorithms Score (.+)'),
                        re.compile(r'CPU Multi-Core Score (.+)'),
                        re.compile(r'GPU Terracotta Score (.+)'),
                        re.compile(r'GPU Swordsman Score (.+)'),
                        re.compile(r'GPU Refinery Score (.+)'),
                        re.compile(r'Data Security Score (.+)'),
                        re.compile(r'Data Processing Score (.+)'),
                        re.compile(r'Image Processing Score (.+)'),
                        re.compile(r'User Experience Score (.+)'),
                        re.compile(r'Video CTS Score (.+)'),
                        re.compile(r'Video Decode Score (.+)'),
                        re.compile(r'RAM Access Score (.+)'),
                        re.compile(r'ROM APP IO Score (.+)'),
                        re.compile(r'ROM Sequential Read Score (.+)'),
                        re.compile(r'ROM Sequential Write Score (.+)'),
                        re.compile(r'ROM Random Access Score (.+)')]
    description = '''
    Executes Antutu 3D, UX, CPU and Memory tests

    Test description:
    1. Open Antutu application
    2. Execute Antutu benchmark

    Known working APK version: 8.0.4
    '''

    supported_versions = ['7.0.4', '7.2.0', '8.0.4', '8.1.9', '9.1.6']

    parameters = [
        Parameter('version', kind=str, allowed_values=supported_versions, override=True,
                  description=(
                      '''Specify the version of Antutu to be run.
                      If not specified, the latest available version will be used.
                      ''')
                  )
    ]

    def __init__(self, device, **kwargs):
        super(Antutu, self).__init__(device, **kwargs)
        self.gui.timeout = 1200

    def setup(self, context):
        self.gui.uiauto_params['version'] = self.version
        super(Antutu, self).setup(context)

    def extract_scores(self, context, regex_version):
        #pylint: disable=no-self-use
        expected_results = len(regex_version)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file, errors='replace') as fh:
            for line in fh:
                for regex in regex_version:
                    match = regex.search(line)
                    if match:
                        try:
                            result = float(match.group(1))
                        except ValueError:
                            result = float('NaN')
                        entry = regex.pattern.rsplit(None, 1)[0]
                        context.add_metric(entry, result, lower_is_better=False)
                        expected_results -= 1
        if expected_results > 0:
            msg = "The Antutu workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(regex_version), expected_results))

    def update_output(self, context):
        super(Antutu, self).update_output(context)
        if self.version.startswith('9'):
            self.extract_scores(context, self.regex_matches_v9)
        if self.version.startswith('8'):
            self.extract_scores(context, self.regex_matches_v8)
        if self.version.startswith('7'):
            self.extract_scores(context, self.regex_matches_v7)
