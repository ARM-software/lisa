#    Copyright 2012-2015 ARM Limited
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
# pylint: disable=attribute-defined-outside-init
from collections import OrderedDict
from itertools import izip_longest
import os
import re
import csv


from wlauto import Workload, Parameter, Executable
from wlauto.exceptions import ConfigError
from wlauto.utils.types import list_of_ints


iozone_results_txt = 'iozone_results.txt'


class Iozone(Workload):
    name = 'iozone'
    description = """
    Iozone is a filesystem benchmark that runs a series of disk
    I/O performance tests.

    Here is a list of tests that you can run in the iozone
    workload. The descriptions are from the official iozone
    document.

    0  - Write Test
         Measure performance of writing a new file. Other
         tests rely on the file written by this, so it must
         always be enabled (WA will automatically neable this
         if not specified).

    1  - Rewrite Test
         Measure performance of writing an existing file.

    2  - Read Test
         Measure performance of reading an existing file.

    3  - Reread Test
         Measure performance of rereading an existing file.

    4  - Random Read Test
         Measure performance of reading a file by accessing
         random locations within the file.

    5  - Random Write Test
         Measure performance of writing a file by accessing
         random locations within the file.

    6  - Backwards Read Test
         Measure performance of reading a file backwards.

    7  - Record Rewrite Test
         Measure performance of writing and rewriting a
         particular spot within the file.

    8  - Strided Read Test
         Measure performance of reading a file with strided
         access behavior.

    9  - Fwrite Test
         Measure performance of writing a file using the
         library function fwrite() that performances
         buffered write operations.

    10 - Frewrite Test
         Measure performance of writing a file using the
         the library function fwrite() that performs
         buffered and blocked write operations.

    11 - Fread Test
         Measure performance of reading a file using the
         library function fread() that performs buffered
         and blocked read operations.

    12 - Freread Test
         Same as the Fread Test except the current file
         being read was read previously sometime in the
         past.

    By default, iozone will run all tests in auto mode. To run
    specific tests, they must be written in the form of:

    [0,1,4,5]

    Please enable classifiers in your agenda or config file
    in order to display the results properly in the results.csv
    file.

    The official website for iozone is at www.iozone.org.
    """

    parameters = [
        Parameter('tests', kind=list_of_ints, allowed_values=range(13),
                  description='List of performance tests to run.'),
        Parameter('auto_mode', kind=bool, default=True,
                  description='Run tests in auto mode.'),
        Parameter('timeout', kind=int, default=14400,
                  description='Timeout for the workload.'),
        Parameter('file_size', kind=int,
                  description='Fixed file size.'),
        Parameter('record_length', kind=int,
                  description='Fixed record length.'),
        Parameter('threads', kind=int,
                  description='Number of threads'),
        Parameter('other_params', kind=str, default='',
                  description='Other parameter. Run iozone -h to see'
                              ' list of options.')
    ]

    def initialize(self, context):
        Iozone.host_binary = context.resolver.get(Executable(self,
                                                             self.device.abi,
                                                             'iozone'))
        Iozone.device_binary = self.device.install(Iozone.host_binary)

    def setup(self, context):
        self.results = os.path.join(self.device.working_directory,
                                    iozone_results_txt)
        self.command = self._build_command()

        if self.threads and self.auto_mode:
            raise ConfigError("You cannot set the number of threads and enable"
                              " auto mode at the same time.")

    def _build_command(self):
        # pylint: disable=access-member-before-definition
        iozone_command = 'cd {} && {}'.format(self.device.working_directory,
                                              self.device_binary)

        if self.auto_mode:
            iozone_command += ' -a'

        if self.tests:
            if 0 not in self.tests:
                self.tests = [0] + self.tests
            iozone_command += ''.join([' -i {}'.format(t) for t in self.tests])

        if self.record_length > 0:
            iozone_command += ' -r {}'.format(self.record_length)

        if self.threads > 0:
            iozone_command += ' -t {}'.format(self.threads)

        if self.file_size > 0:
            iozone_command += ' -s {}'.format(self.file_size)

        if self.other_params:
            iozone_command += ' ' + self.other_params

        # enable reporting mode for parsing non-thread results
        iozone_command += ' -R > {}'.format(self.results)

        # check if -b option is used
        match = re.search(r'-b (.?\w+.?\w+?\s)', iozone_command)
        if match:
            self.user_file = match.group(1)
            self.device_output_file = os.path.join(self.device.working_directory,
                                                   self.user_file)

        return iozone_command

    def run(self, context):
        self.device.execute(self.command, timeout=self.timeout)

    def update_result(self, context):
        self.device.pull_file(self.results, context.output_directory)
        self.outfile = os.path.join(context.output_directory,
                                    iozone_results_txt)

        if '-b' in self.other_params:
            self.device.pull_file(self.device_output_file,
                                  context.output_directory)

        # if running in thread mode
        if self.threads:
            thread_results = self.parse_thread_results()

            for name, value, units in thread_results:
                context.add_metric(name, value, units)

        # for non-thread mode results
        else:
            with open(self.outfile, 'r') as iozone_file:
                iozone_file = (line.replace('\"', '') for line in iozone_file)
                table_list = []

                # begin parsing results
                for line in iozone_file:
                    if 'Writer report' in line:
                        table_list.append(line.split())
                        break

                for line in iozone_file:
                    if 'exiting' in line or 'completed' in line:
                        break
                    else:
                        table_list.append(line.split())

                # create csv file
                self.write_to_csv(context, table_list)

                # parse metrics
                self.parse_metrics(context, table_list)

    def write_to_csv(self, context, csv_table_list):
        self.test_file = os.path.join(context.output_directory,
                                      'table_results.csv')

        # create csv file for writing
        csv_file = open(self.test_file, 'w')
        wr = csv.writer(csv_file, delimiter=',')

        # shift second row by adding extra element
        # for "prettier" formatting
        index = 0
        for element in csv_table_list:
            if element:
                if index == 1:
                    element.insert(0, '0')
                index += 1
            else:
                index = 0

        # write to csv file
        for item in csv_table_list:
            wr.writerow(item)

        csv_file.close()

    # break list of results into smaller groups based on
    # test name
    def parse_metrics(self, context, plist):  # pylint: disable=no-self-use
        subvalue_list = []
        value_list = []
        for values in plist:
            if values:
                subvalue_list.append(values)
            else:
                value_list.append(subvalue_list)
                subvalue_list = []

        # If users run a list of specific tests, make
        # sure that the results for the last test
        # executed are appended.
        if subvalue_list:
            value_list.append(subvalue_list)

        for reports in value_list:
            # grab report name and convert it to a string
            report_name = reports[0]
            report_name = report_name[:-1]
            report_name = '_'.join(report_name).lower()

            record_sizes = reports[1]
            values = reports[2:]

            for v in values:
                templist = OrderedDict(izip_longest(record_sizes, v))

                for reclen, value in templist.items():
                    if reclen is '0':
                        fs = value

                    if value is None:
                        value = '0'

                    classifiers = {'reclen': reclen, 'file_size': fs}
                    if reclen != '0':
                        context.add_metric(report_name, int(value), 'kb/s',
                                           classifiers=classifiers)

    # parse thread-mode results
    def parse_thread_results(self):
        results = []
        with open(self.outfile, 'r') as iozone_file:
            for line in iozone_file:
                # grab section of data we care about
                if 'Throughput report' in line:
                    break
                else:
                    if '=' in line:
                        if 'Time Resolution' not in line:
                            line = line.replace('=', '')
                            line = line.split()

                            # grab headers
                            if len(line) >= 8:
                                header = line[0]
                                subheader = ' '.join(line[-5:-2])
                                header += ' ' + subheader
                            else:
                                header = ' '.join(line[0:2])

                            units = line[-1]
                            value = line[-2]
                            tup = (header, value, units)
                            results.append(tup)

        return results

    def finalize(self, context):
        self.device.uninstall_executable(self.device_binary)
