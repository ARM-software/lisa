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


# pylint: disable=R0201
import os
import time
from collections import Counter
from wlauto import ResultProcessor
from wlauto.utils.misc import write_table


class StatusTxtReporter(ResultProcessor):
    name = 'status'
    description = """
    Outputs a txt file containing general status information about which runs
    failed and which were successful

    """

    def process_run_result(self, result, context):
        counter = Counter()
        for ir in result.iteration_results:
            counter[ir.status] += 1

        outfile = os.path.join(context.run_output_directory, 'status.txt')
        self.logger.info('Status available in {}'.format(outfile))
        with open(outfile, 'w') as wfh:
            wfh.write('Run name: {}\n'.format(context.run_info.run_name))
            wfh.write('Run status: {}\n'.format(context.run_result.status))
            wfh.write('Date: {}\n'.format(time.strftime("%c")))
            wfh.write('{}/{} iterations completed without error\n'.format(counter['OK'], len(result.iteration_results)))
            wfh.write('\n')
            status_lines = [map(str, [ir.id, ir.spec.label, ir.iteration, ir.status,
                                      ir.events and ir.events[0].message.split('\n')[0] or ''])
                            for ir in result.iteration_results]
            write_table(status_lines, wfh, align='<<>><')
        context.add_artifact('run_status_summary', 'status.txt', 'export')

