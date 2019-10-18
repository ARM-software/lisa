#    Copyright 2014-2019 ARM Limited
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

from wa import UiautoWorkload, Parameter
from wa.framework.exception import ValidationError, WorkloadError
from wa.utils.types import list_of_strs
from wa.utils.misc import unique


class Motionmark(UiautoWorkload):

    name = 'motionmark'

    description = '''
    A workload to execute the motionmark web based benchmark

    MotionMark is a graphics benchmark that measures a browser capability to animate complex scenes at a target frame rate

    Test description:
    1. Open browser application
    2. Navigate to the motionmark website - http://browserbench.org/MotionMark/
    3. Execute the benchmark
    '''

    requires_network = True

    regex = [re.compile(r'Multiply Score (.+)'),
             re.compile(r'Canvas Score (.+)'),
             re.compile(r'Leaves Score (.+)'),
             re.compile(r'Paths Score (.+)'),
             re.compile(r'Canvas Lines Score (.+)'),
             re.compile(r'Focus Score (.+)'),
             re.compile(r'Images Score (.+)'),
             re.compile(r'Design Score (.+)'),
             re.compile(r'Suits Score (.+)')]
    score_regex = re.compile(r'.*?([\d.]+).*')

    def __init__(self, target, **kwargs):
        super(Motionmark, self).__init__(target, **kwargs)
        self.gui.timeout = 1500

    def setup(self, context):
        super(Motionmark, self).setup(context)
        self.target.open_url('https://browserbench.org/MotionMark/')

    def update_output(self, context):
        super(Motionmark, self).update_output(context)
        num_unprocessed_results = len(self.regex)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file) as fh:
            for line in fh:
                for regex in self.regex:
                    match = regex.search(line)
                    # Check if we have matched the score string in logcat
                    if match:
                        score_match = self.score_regex.search(match.group(1))
                        # Check if there is valid number found for the score.
                        if score_match:
                            result = float(score_match.group(1))
                        else:
                            result = float('NaN')
                        entry = regex.pattern.rsplit(None, 1)[0]
                        context.add_metric(entry, result, 'Score', lower_is_better=False)
                        num_unprocessed_results -= 1
        if num_unprocessed_results > 0:
            msg = "The Motionmark workload has failed. Expected {} scores, Missing {} scores."
            raise WorkloadError(msg.format(len(self.regex), num_unprocessed_results))
