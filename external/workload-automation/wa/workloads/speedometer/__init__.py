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
import os
import re

from wa import UiautoWorkload, Parameter
from wa.framework.exception import ValidationError, WorkloadError
from wa.utils.types import list_of_strs
from wa.utils.misc import unique


class Speedometer(UiautoWorkload):

    name = 'speedometer'
    regex = re.compile(r'Speedometer Score ([\d.]+)')
    versions = ['1.0', '2.0']
    description = '''
    A workload to execute the speedometer web based benchmark

    Test description:
    1. Open browser application
    2. Navigate to the speedometer website - http://browserbench.org/Speedometer/
    3. Execute the benchmark

    '''

    parameters = [
        Parameter('version', allowed_values=versions, kind=str, default='2.0',
                  description='''
                  The speedometer version to be used.
                  ''')
    ]

    requires_network = True

    def __init__(self, target, **kwargs):
        super(Speedometer, self).__init__(target, **kwargs)
        self.gui.timeout = 1500
        self.gui.uiauto_params['version'] = self.version

    def setup(self, context):
        super(Speedometer, self).setup(context)
        url = 'am start -a android.intent.action.VIEW -d http://browserbench.org/Speedometer' + self.version
        if self.version == '1.0':
            url = 'am start -a android.intent.action.VIEW -d http://browserbench.org/Speedometer'
        self.target.execute(url)

    def update_output(self, context):
        super(Speedometer, self).update_output(context)
        result = None
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file) as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    result = float(match.group(1))

        if result is not None:
            context.add_metric('Speedometer Score', result, 'Runs per minute', lower_is_better=False)
        else:
            raise WorkloadError("The Speedometer workload has failed. No score was obtainable.")
