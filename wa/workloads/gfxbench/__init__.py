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
from wa.utils.types import list_or_string


class Gfxbench(ApkUiautoWorkload):

    name = 'gfxbench'
    package_names = ['com.glbenchmark.glbenchmark27']
    supported_versions = ['4', '5']
    is_corporate = False
    clear_data_on_reset = False
    regex_template = 'name: \((?P<test_name>.*)\).*result: \((?P<result>.*)?\).* sub_result:.*\((?P<sub_result>.*?)?\).*'
    description = '''
    Execute a subset of graphical performance benchmarks

    Test description:
    1. Open the gfxbench application
    2. Execute Car Chase, Manhattan and Tessellation benchmarks

    Note: Some of the default tests are unavailable on devices running
          with a smaller resolution than 1080p.

    '''

    default_test_list = [
        "Car Chase",
        "1080p Car Chase Offscreen",
        "Manhattan 3.1",
        "1080p Manhattan 3.1 Offscreen",
        "1440p Manhattan 3.1.1 Offscreen",
        "Tessellation",
        "1080p Tessellation Offscreen",
    ]

    parameters = [
        Parameter('timeout', kind=int, default=3600,
                  description=('Timeout for an iteration of the benchmark.')),
        Parameter('tests', kind=list_or_string, default=default_test_list,
                  description=('List of tests to be executed.')),
    ]

    def __init__(self, target, **kwargs):
        super(Gfxbench, self).__init__(target, **kwargs)
        self.gui.timeout = self.timeout
        self.gui.uiauto_params['tests'] = self.tests

    # pylint: disable=too-many-locals
    def update_output(self, context):
        super(Gfxbench, self).update_output(context)
        regex_matches = [re.compile(self.regex_template.format(t)) for t in self.tests]
        logcat_file = context.get_artifact_path('logcat')
        found = []
        detected_results = 0
        failed = False
        with open(logcat_file, errors='replace') as fh:
            for line in fh:
                for regex in regex_matches:
                    match = regex.search(line)
                    # Check if we have matched the score string in logcat and not already found.
                    if match and match.group('test_name') not in found:
                        found.append(match.group('test_name'))
                        # Set Default values
                        result = 'NaN'
                        unit = 'FPS'

                        # For most tests we usually want the `sub_result`
                        # as this is our FPS value
                        try:
                            result = float(match.group('sub_result').split()[0].replace(',', ''))
                        except (ValueError, TypeError):
                            # However for some tests the value is stored in `result`
                            # and the unit is saved in the `sub_result`.
                            try:
                                result = float(match.group('result').replace(',', ''))
                                if match.group('sub_result'):
                                    unit = match.group('sub_result').upper()
                            except (ValueError, TypeError):
                                failed = True

                        entry = match.group('test_name')
                        context.add_metric(entry, result, unit, lower_is_better=False)
                        detected_results += 1

        if failed or detected_results < len(regex_matches):
            msg = "The workload has failed to process all scores. Expected >={} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(regex_matches), detected_results))


class GfxbenchCorporate(Gfxbench):  # pylint: disable=too-many-ancestors

    name = 'gfxbench-corporate'
    package_names = ['net.kishonti.gfxbench.gl.v50000.corporate']
    is_corporate = True
