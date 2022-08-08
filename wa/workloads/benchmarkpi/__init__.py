#    Copyright 2013-2018 ARM Limited
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


class BenchmarkPi(ApkUiautoWorkload):

    name = 'benchmarkpi'
    description = """
    Measures the time the target device takes to run and complete the Pi
    calculation algorithm.

    http://androidbenchmark.com/howitworks.php

    from the website:

    The whole idea behind this application is to use the same Pi calculation
    algorithm on every Android Device and check how fast that process is.
    Better calculation times, conclude to faster Android devices. This way you
    can also check how lightweight your custom made Android build is. Or not.

    As Pi is an irrational number, Benchmark Pi does not calculate the actual Pi
    number, but an approximation near the first digits of Pi over the same
    calculation circles the algorithms needs.

    So, the number you are getting in milliseconds is the time your mobile device
    takes to run and complete the Pi calculation algorithm resulting in a
    approximation of the first Pi digits.
    """
    package_names = ['gr.androiddev.BenchmarkPi']
    activity = '.BenchmarkPi'

    regex = re.compile('You calculated Pi in ([0-9]+)')

    def update_output(self, context):
        super(BenchmarkPi, self).update_output(context)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file, errors='replace') as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    result = int(match.group(1))

        if result is not None:
            context.add_metric('pi calculation', result,
                               'milliseconds', lower_is_better=True)
