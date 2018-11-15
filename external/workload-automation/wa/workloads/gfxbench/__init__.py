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

from wa import ApkUiautoWorkload, WorkloadError, Parameter


class Gfxbench(ApkUiautoWorkload):

    name = 'gfxbench-corporate'
    package_names = ['net.kishonti.gfxbench.gl.v50000.corporate']
    clear_data_on_reset = False
    regex_matches = [re.compile(r'Car Chase score (.+)'),
                     re.compile(r'Car Chase Offscreen score (.+)'),
                     re.compile(r'Manhattan 3.1 score (.+)'),
                     re.compile(r'1080p Manhattan 3.1 Offscreen score (.+)'),
                     re.compile(r'1440p Manhattan 3.1 Offscreen score (.+)'),
                     re.compile(r'Tessellation score (.+)'),
                     re.compile(r'Tessellation Offscreen score (.+)')]
    description = '''
    Execute a subset of graphical performance benchmarks

    Test description:
    1. Open the gfxbench application
    2. Execute Car Chase, Manhattan and Tessellation benchmarks

    '''
    parameters = [
        Parameter('timeout', kind=int, default=3600,
                  description=('Timeout for a single iteration of the benchmark. This value is '
                               'multiplied by ``times`` to calculate the overall run timeout. ')),
    ]

    def __init__(self, target, **kwargs):
        super(Gfxbench, self).__init__(target, **kwargs)
        self.gui.timeout = self.timeout

    def update_output(self, context):
        super(Gfxbench, self).update_output(context)
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
                        context.add_metric(entry, result, 'FPS', lower_is_better=False)
                        expected_results -= 1
        if expected_results > 0:
            msg = "The GFXBench workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(self.regex_matches), expected_results))
