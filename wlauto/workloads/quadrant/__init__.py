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
from collections import defaultdict

from wlauto import AndroidUiAutoBenchmark


TEST_TYPES = {
    'benchmark_cpu_branching_logic': 'time',
    'benchmark_cpu_matrix_int': 'time',
    'benchmark_cpu_matrix_long': 'time',
    'benchmark_cpu_matrix_short': 'time',
    'benchmark_cpu_matrix_byte': 'time',
    'benchmark_cpu_matrix_float': 'time',
    'benchmark_cpu_matrix_double': 'time',
    'benchmark_cpu_checksum': 'time',
    'benchmark_cpu': 'aggregate',
    'benchmark_memory_transfer': 'time',
    'benchmark_memory': 'aggregate',
    'benchmark_io_fs_write': 'time',
    'benchmark_io_fs_read': 'time',
    'benchmark_io_db_write': 'time',
    'benchmark_io_db_read': 'time',
    'benchmark_io': 'aggregate',
    'benchmark_g2d_fractal': 'rate',
    'benchmark_g2d': 'aggregate',
    'benchmark_g3d_corridor': 'rate',
    'benchmark_g3d_planet': 'rate',
    'benchmark_g3d_dna': 'rate',
    'benchmark_g3d': 'aggregate',
    'benchmark': 'aggregate',
}

TYPE_TESTS = defaultdict(list)
for k, v in TEST_TYPES.iteritems():
    TYPE_TESTS[v].append(k)

TYPE_UNITS = {
    'time': 'ms',
    'rate': 'Hz',
}

REGEX_TEMPLATES = {
    'aggregate': r'(?P<metric>{}) aggregate score is (?P<score>\d+)',
    'time': r'(?P<metric>{}) executed in (?P<time>\d+) ms, '
            r'reference time: (?P<reference>\d+) ms, '
            r'score: (?P<score>\d+)',
    'rate': r'(?P<metric>{}) executed with a rate of (?P<rate>[0-9.]+)/sec, '
            r'reference rate: (?P<reference>[0-9.]+)/sec, '
            r'score: (?P<score>\d+)',
}

TEST_REGEXES = {}
for test_, type_ in TEST_TYPES.items():
    TEST_REGEXES[test_] = re.compile(REGEX_TEMPLATES[type_].format(test_))


class Quadrant(AndroidUiAutoBenchmark):

    name = 'quadrant'
    description = """
    Quadrant is a benchmark for mobile devices, capable of measuring CPU, memory,
    I/O and 3D graphics performance.

    http://www.aurorasoftworks.com/products/quadrant

    From the website:
    Quadrant outputs a score for the following categories: 2D, 3D, Mem, I/O, CPU
    , Total.
    """
    package = 'com.aurorasoftworks.quadrant.ui.professional'
    activity = '.QuadrantProfessionalLauncherActivity'
    summary_metrics = ['benchmark_score']

    run_timeout = 10 * 60

    def __init__(self, device, **kwargs):
        super(Quadrant, self).__init__(device, **kwargs)
        self.uiauto_params['has_gpu'] = self.device.has_gpu
        self.regex = {}

    def update_result(self, context):
        super(Quadrant, self).update_result(context)
        with open(self.logcat_log) as fh:
            for line in fh:
                for test, regex in TEST_REGEXES.items():
                    match = regex.search(line)
                    if match:
                        test_type = TEST_TYPES[test]
                        data = match.groupdict()
                        if test_type != 'aggregate':
                            context.result.add_metric(data['metric'] + '_' + test_type,
                                                      data[test_type],
                                                      TYPE_UNITS[test_type])
                        context.result.add_metric(data['metric'] + '_score', data['score'])
                        break

