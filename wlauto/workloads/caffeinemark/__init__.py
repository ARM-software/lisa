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

from wlauto import AndroidUiAutoBenchmark


class Caffeinemark(AndroidUiAutoBenchmark):

    name = 'caffeinemark'
    description = """
    CaffeineMark is a series of tests that measure the speed of Java
    programs running in various hardware and software configurations.

    http://www.benchmarkhq.ru/cm30/info.html

    From the website:

    CaffeineMark scores roughly correlate with the number of Java instructions
    executed per second, and do not depend significantly on the the amount of
    memory in the system or on the speed of a computers disk drives or internet
    connection.

    The following is a brief description of what each test does:

        - Sieve: The classic sieve of eratosthenes finds prime numbers.
        - Loop: The loop test uses sorting and sequence generation as to measure
                compiler optimization of loops.
        - Logic: Tests the speed with which the virtual machine executes
                 decision-making instructions.
        - Method: The Method test executes recursive function calls to see how
                  well the VM handles method calls.
        - Float: Simulates a 3D rotation of objects around a point.
        - Graphics: Draws random rectangles and lines.
        - Image: Draws a sequence of three graphics repeatedly.
        - Dialog: Writes a set of values into labels and editboxes on a form.

    The overall CaffeineMark score is the geometric mean of the individual
    scores, i.e., it is the 9th root of the product of all the scores.
    """
    package = "com.flexycore.caffeinemark"
    activity = ".Application"
    summary_metrics = ['OverallScore']

    regex = re.compile(r'CAFFEINEMARK RESULT: (?P<type>\w+) (?P<value>\S+)')

    def update_result(self, context):
        super(Caffeinemark, self).update_result(context)
        with open(self.logcat_log) as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    metric = match.group('type')
                    value = float(match.group('value'))
                    context.result.add_metric(metric, value)
