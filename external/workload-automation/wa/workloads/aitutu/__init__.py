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


class Aitutu(ApkUiautoWorkload):

    name = 'aitutu'
    package_names = ['com.antutu.aibenchmark']
    regex_matches = [re.compile(r'Overall Score ([\d.]+)'),
                     re.compile(r'Image Total Score ([\d.]+) ([\w]+) ([\w]+)'),
                     re.compile(r'Image Speed Score ([\d.]+) ([\w]+) ([\w]+)'),
                     re.compile(r'Image Accuracy Score ([\d.]+) ([\w]+) ([\w]+)'),
                     re.compile(r'Object Total Score ([\d.]+) ([\w]+) ([\w]+)'),
                     re.compile(r'Object Speed Score ([\d.]+) ([\w]+) ([\w]+)'),
                     re.compile(r'Object Accuracy Score ([\d.]+) ([\w]+) ([\w]+)')]
    description = '''
    Executes Aitutu Image Speed/Accuracy and Object Speed/Accuracy tests

    The Aitutu workflow carries out the following tasks.
    1. Open Aitutu application
    2. Download the resources for the test
    3. Execute the tests

    Known working APK version: 1.0.3
    '''

    requires_network = True

    def __init__(self, target, **kwargs):
        super(Aitutu, self).__init__(target, **kwargs)
        self.gui.timeout = 1200000

    def update_output(self, context):
        super(Aitutu, self).update_output(context)
        expected_results = len(self.regex_matches)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file) as fh:
            for line in fh:
                for regex in self.regex_matches:
                    match = regex.search(line)
                    if match:
                        classifiers = {}
                        result = match.group(1)
                        if (len(match.groups())) > 1:
                            entry = regex.pattern.rsplit(None, 3)[0]
                            classifiers = {'model': match.group(3)}
                        else:
                            entry = regex.pattern.rsplit(None, 1)[0]
                        context.add_metric(entry, result, '', lower_is_better=False, classifiers=classifiers)
                        expected_results -= 1
        if expected_results > 0:
            msg = "The Aitutu workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(self.regex_matches), expected_results))
