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

# pylint: disable=E1101,W0201,E0203

import re

from wlauto import AndroidUiAutoBenchmark, Parameter


class RealLinpack(AndroidUiAutoBenchmark):

    name = 'real-linpack'
    description = """
    This version of `Linpack <http://en.wikipedia.org/wiki/LINPACK_benchmarks>`
    was developed by Dave Butcher. RealLinpack tries to find the number of threads
    that give you the maximum linpack score.

    RealLinpack runs 20 runs of linpack for each number of threads and
    calculates the mean and confidence.  It stops when the
    score's confidence interval drops below the current best score
    interval.  That is, when (current_score + confidence) < (best_score -
    best_score_confidence)

    """
    package = 'com.arm.RealLinpack'
    activity = '.RealLinpackActivity'

    parameters = [
        Parameter('max_threads', kind=int, default=16, constraint=lambda x: x > 0,
                  description='The maximum number of threads that real linpack will try.'),
    ]

    def __init__(self, device, **kwargs):
        super(RealLinpack, self).__init__(device, **kwargs)
        self.uiauto_params['max_threads'] = self.max_threads
        self.run_timeout = 120 + 120 * self.max_threads  # a base of 2 minutes plus 2 minutes for each thread

    def update_result(self, context):
        super(RealLinpack, self).update_result(context)
        score_regex = re.compile(r'Optimum.*threads:\s*([0-9])+.*score:\s*([0-9]+\.[0-9]+).*MFLOPS')
        match_found = False
        with open(self.logcat_log) as logcat_file:
            for line in logcat_file:
                match = re.search(score_regex, line)
                if match:
                    number_of_threads = match.group(1)
                    score = match.group(2)
                    context.result.add_metric('optimal number of threads', number_of_threads, None)
                    context.result.add_metric('score', score, 'MFLOPS')
                    match_found = True
                    break
            if not match_found:
                self.logger.warning('Failed To collect results for real linpack')
