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

import os
from collections import defaultdict, OrderedDict

from wlauto import AndroidUiAutoBenchmark, Parameter


class Antutu(AndroidUiAutoBenchmark):

    name = 'antutu'
    description = """
    AnTuTu Benchmark is an benchmarking tool for Android Mobile Phone/Pad. It
    can run a full test of a key project, through the "Memory Performance","CPU
    Integer Performance","CPU Floating point Performance","2D 3D Graphics
    Performance","SD card reading/writing speed","Database IO" performance
    testing, and gives accurate analysis for Andriod smart phones.

    http://www.antutulabs.com/AnTuTu-Benchmark

    From the website:

    AnTuTu Benchmark can support the latest quad-core cpu. In reaching the
    overall and individual scores of the hardware, AnTuTu Benchmark could judge
    your phone by the scores of the performance of the hardware. By uploading
    the scores, Benchmark can view your device in the world rankings, allowing
    points to let you know the level of hardware performance equipment.

    """
    #pylint: disable=E1101

    package = "com.antutu.ABenchMark"
    activity = ".ABenchMarkStart"
    summary_metrics = ['score', 'Overall_Score']

    valid_versions = ['3.3.2', '4.0.3', '5.3.0']

    device_prefs_directory = '/data/data/com.antutu.ABenchMark/shared_prefs'
    device_prefs_file = '/'.join([device_prefs_directory, 'com.antutu.ABenchMark_preferences.xml'])
    local_prefs_directory = os.path.join(os.path.dirname(__file__), 'shared_prefs')

    parameters = [
        Parameter('version', allowed_values=valid_versions, default=sorted(valid_versions, reverse=True)[0],
                  description=('Specify the version of AnTuTu to be run. If not specified, the latest available '
                               'version will be used.')),
        Parameter('times', kind=int, default=1,
                  description=('The number of times the benchmark will be executed in a row (i.e. '
                               'without going through the full setup/teardown process). Note: this does '
                               'not work with versions prior to 4.0.3.')),
        Parameter('enable_sd_tests', kind=bool, default=False,
                  description=('If ``True`` enables SD card tests in pre version 4 AnTuTu. These tests '
                               'were know to cause problems on platforms without an SD card. This parameter '
                               'will be ignored on AnTuTu version 4 and higher.')),
    ]

    def __init__(self, device, **kwargs):  # pylint: disable=W0613
        super(Antutu, self).__init__(device, **kwargs)
        self.run_timeout = 6 * 60 * self.times
        self.uiauto_params['version'] = self.version
        self.uiauto_params['times'] = self.times
        self.uiauto_params['enable_sd_tests'] = self.enable_sd_tests

    def update_result(self, context):
        super(Antutu, self).update_result(context)
        with open(self.logcat_log) as fh:
            if self.version == '4.0.3':
                metrics = extract_version4_metrics(fh)
            else:
                metrics = extract_older_version_metrics(fh)
        for key, value in metrics.iteritems():
            key = key.replace(' ', '_')
            context.result.add_metric(key, value)


# Utility functions

def extract_version4_metrics(fh):
    metrics = OrderedDict()
    metric_counts = defaultdict(int)
    for line in fh:
        if 'ANTUTU RESULT:' in line:
            result = line.split('ANTUTU RESULT:')[1]
            metric, value_string = [v.strip() for v in result.split(':', 1)]
            # If times prameter > 1 the same metric will appear
            # multiple times in logcat -- we want to collet all of
            # them as they're from different iterations.
            metric_counts[metric] += 1
            if metric_counts[metric] > 1:
                metric += '_' + str(metric_counts[metric])

            # Grahics results report resolution in square brackets
            # as part of value string.
            if ']' in value_string:
                value = int(value_string.split(']')[1].strip())
            else:
                value = int(value_string)

            metrics[metric] = value
    return metrics


def extract_older_version_metrics(fh):
    metrics = {}
    metric_counts = defaultdict(int)
    for line in fh:
        if 'i/antutu' in line.lower():
            parts = line.split(':')
            if not len(parts) == 3:
                continue
            metric = parts[1].strip()
            value = int(parts[2].strip())

            # If times prameter > 1 the same metric will appear
            # multiple times in logcat -- we want to collet all of
            # them as they're from different iterations.
            metric_counts[metric] += 1
            if metric_counts[metric] > 1:
                metric += ' ' + str(metric_counts[metric])

            metrics[metric] = value
    return metrics

