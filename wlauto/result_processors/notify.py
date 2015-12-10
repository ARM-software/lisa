#    Copyright 2015 ARM Limited
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

import collections
import sys


try:
    import notify2
except ImportError:
    notify2 = None


from wlauto import ResultProcessor
from wlauto.core.result import IterationResult
from wlauto.exceptions import ResultProcessorError


class NotifyProcessor(ResultProcessor):

    name = 'notify'
    description = '''Display a desktop notification when the run finishes

    Notifications only work in linux systems. It uses the generic
    freedesktop notification specification. For this results processor
    to work, you need to have python-notify installed in your system.

    '''

    def initialize(self, context):
        if sys.platform != 'linux2':
            raise ResultProcessorError('Notifications are only supported in linux')

        if not notify2:
            raise ResultProcessorError('notify2 not installed.  Please install the notify2 package')

        notify2.init("Workload Automation")

    def process_run_result(self, result, context):
        num_iterations = sum(context.job_iteration_counts.values())

        counter = collections.Counter()
        for result in result.iteration_results:
            counter[result.status] += 1

        score_board = []
        for status in IterationResult.values:
            if status in counter:
                score_board.append('{} {}'.format(counter[status], status))

        summary = 'Workload Automation run finised'
        body = 'Ran a total of {} iterations: '.format(num_iterations)
        body += ', '.join(score_board)
        notification = notify2.Notification(summary, body)

        if not notification.show():
            self.logger.warning('Notification failed to show')
