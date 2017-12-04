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
import time
from collections import Counter

from wa import ResultProcessor, Status
from wa.utils.misc import write_table


class StatusTxtReporter(ResultProcessor):
    name = 'status'
    description = """
    Outputs a txt file containing general status information about which runs
    failed and which were successful

    """

    def process_run_output(self, output, target_info):
        counter = Counter()
        for jo in output.jobs:
            counter[jo.status] += 1

        outfile = output.get_path('status.txt')
        self.logger.info('Status available in {}'.format(outfile))
        with open(outfile, 'w') as wfh:
            wfh.write('Run name: {}\n'.format(output.info.run_name))
            wfh.write('Run status: {}\n'.format(output.status))
            wfh.write('Date: {}\n'.format(time.strftime("%c")))
            if output.events:
                wfh.write('Events:\n')
                for event in output.events:
                    wfh.write('\t{}\n'.format(event.summary))

            txt = '{}/{} iterations completed without error\n'
            wfh.write(txt.format(counter[Status.OK], len(output.jobs)))
            wfh.write('\n')
            status_lines = [map(str, [o.id, o.label, o.iteration, o.status,
                                      o.event_summary])
                            for o in output.jobs]
            write_table(status_lines, wfh, align='<<>><')

        output.add_artifact('run_status_summary', 'status.txt', 'export')

