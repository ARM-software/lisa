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
import sys
import re
import time
import shutil
import logging
import threading
import subprocess
import tempfile
import csv

from wlauto import Instrument, Parameter
from wlauto.core.execution import ExecutionContext
from wlauto.exceptions import InstrumentError, WorkerThreadError
from wlauto.core import signal


class CoreUtilization(Instrument):

    name = 'coreutil'
    description = """
    Measures CPU core activity during workload execution in terms of the percentage of time a number
    of cores were utilized above the specfied threshold.

    This workload generates ``coreutil.csv`` report in the workload's output directory. The report is
    formatted as follows::

        <threshold,1core,2core,3core,4core
        18.098132,38.650248000000005,10.736180000000001,3.6809760000000002,28.834312000000001

    Interpretation of the result:

     - 38.65% of total time only single core is running above or equal to threshold value
     - 10.736% of total time two cores are running simultaneously above or equal to threshold value
     - 3.6809% of total time three cores are running simultaneously above or equal to threshold value
     - 28.8314% of total time four cores are running simultaneously above or equal to threshold value
     - 18.098% of time all core are running below threshold value.

    ..note : This instrument doesn't work on ARM big.LITTLE IKS implementation

    """

    parameters = [
        Parameter('threshold', kind=int, default=50,
                  constraint=lambda x: 0 < x <= 100,
                  description='Cores with percentage utilization above this value will be considered '
                              'as "utilized". This value may need to be adjusted based on the background '
                              'activity and the intensity of the workload being instrumented (e.g. it may '
                              'need to be lowered for low-intensity workloads such as video playback).'
                  )
    ]

    def __init__(self, device, **kwargs):
        super(CoreUtilization, self).__init__(device, **kwargs)
        self.collector = None
        self.output_dir = None
        self.cores = None
        self.output_artifact_registered = False

    def setup(self, context):
        ''' Calls ProcCollect class '''
        self.output_dir = context.output_directory
        self.collector = ProcCollect(self.device, self.logger, self.output_dir)
        self.cores = self.device.number_of_cores

    def start(self, context):  # pylint: disable=W0613
        ''' Starts collecting data once the workload starts '''
        self.logger.debug('Starting to collect /proc/stat data')
        self.collector.start()

    def stop(self, context):  # pylint: disable=W0613
        ''' Stops collecting data once the workload stops '''
        self.logger.debug('Stopping /proc/stat data collection')
        self.collector.stop()

    def update_result(self, context):
        ''' updates result into coreutil.csv '''
        self.collector.join()    # wait for "proc.txt" to generate.
        context.add_artifact('proctxt', 'proc.txt', 'raw')
        calc = Calculator(self.cores, self.threshold, context)  # pylint: disable=E1101
        calc.calculate()
        if not self.output_artifact_registered:
            context.add_run_artifact('cpuutil', 'coreutil.csv', 'data')
            self.output_artifact_registered = True


class ProcCollect(threading.Thread):
    ''' Dumps data into proc.txt '''

    def __init__(self, device, logger, out_dir):
        super(ProcCollect, self).__init__()
        self.device = device
        self.logger = logger
        self.dire = out_dir
        self.stop_signal = threading.Event()
        self.command = 'cat /proc/stat'
        self.exc = None

    def run(self):
        try:
            self.stop_signal.clear()
            _, temp_file = tempfile.mkstemp()
            self.logger.debug('temp file : {}'.format(temp_file))
            with open(temp_file, 'wb') as tempfp:
                while not self.stop_signal.is_set():
                    tempfp.write(self.device.execute(self.command))
                    tempfp.write('\n')
                    time.sleep(0.5)
            raw_file = os.path.join(self.dire, 'proc.txt')
            shutil.copy(temp_file, raw_file)
            os.unlink(temp_file)
        except Exception, error:  # pylint: disable=W0703
            self.logger.warning('Exception on collector thread : {}({})'.format(error.__class__.__name__, error))
            self.exc = WorkerThreadError(self.name, sys.exc_info())

    def stop(self):
        '''Executed once the workload stops'''
        self.stop_signal.set()
        if self.exc is not None:
            raise self.exc  # pylint: disable=E0702


class Calculator(object):
    """
    Read /proc/stat and dump data into ``proc.txt`` which is parsed to generate ``coreutil.csv``
    Sample output from 'proc.txt' ::

        ----------------------------------------------------------------------
        cpu  9853753 51448 3248855 12403398 4241 111 14996 0 0 0
        cpu0 1585220 7756 1103883 4977224 552 97 10505 0 0 0
        cpu1 2141168 7243 564347 972273 504 4 1442 0 0 0
        cpu2 1940681 7994 651946 1005534 657 3 1424 0 0 0
        cpu3 1918013 8833 667782 1012249 643 3 1326 0 0 0
        cpu4 165429 5363 50289 1118910 474 0 148 0 0 0
        cpu5 1661299 4910 126654 1104018 480 0 53 0 0 0
        cpu6 333642 4657 48296 1102531 482 2 55 0 0 0
        cpu7 108299 4691 35656 1110658 448 0 41 0 0 0
        ----------------------------------------------------------------------
        Description:

        1st column  : cpu_id( cpu0, cpu1, cpu2,......)
        Next all column represents the amount of time, measured in units of USER_HZ
        2nd column  : Time spent in user mode
        3rd column  : Time spent in user mode with low priority
        4th column  : Time spent in system mode
        5th column  : Time spent in idle task
        6th column  : Time waiting for i/o to compelete
        7th column  : Time servicing interrupts
        8th column  : Time servicing softirqs
        9th column  : Stolen time is the time spent in other operating systems
        10th column : Time spent running a virtual CPU
        11th column : Time spent running a niced guest

        ----------------------------------------------------------------------------

    Procedure to calculate instantaneous CPU utilization:

    1) Subtract two consecutive samples for every column( except 1st )
    2) Sum all the values except "Time spent in idle task"
    3) CPU utilization(%) = ( value obtained in 2 )/sum of all the values)*100

    """

    idle_time_index = 3

    def __init__(self, cores, threshold, context):
        self.cores = cores
        self.threshold = threshold
        self.context = context
        self.cpu_util = None  # Store CPU utilization for each core
        self.active = None  # Store active time(total time - idle)
        self.total = None   # Store the total amount of time (in USER_HZ)
        self.output = None
        self.cpuid_regex = re.compile(r'cpu(\d+)')
        self.outfile = os.path.join(context.run_output_directory, 'coreutil.csv')
        self.infile = os.path.join(context.output_directory, 'proc.txt')

    def calculate(self):
        self.calculate_total_active()
        self.calculate_core_utilization()
        self.generate_csv(self.context)

    def calculate_total_active(self):
        """ Read proc.txt file and calculate 'self.active' and 'self.total' """
        all_cores = set(xrange(self.cores))
        self.total = [[] for _ in all_cores]
        self.active = [[] for _ in all_cores]
        with open(self.infile, "r") as fh:
            # parsing logic:
            # - keep spinning through lines until see the cpu summary line
            #   (taken to indicate start of new record).
            # - extract values for individual cores after the summary line,
            #   keeping  track of seen cores until no more lines match 'cpu\d+'
            #   pattern.
            # - For every core not seen in this record, pad zeros.
            # - Loop
            try:
                while True:
                    line = fh.next()
                    if not line.startswith('cpu '):
                        continue

                    seen_cores = set([])
                    line = fh.next()
                    match = self.cpuid_regex.match(line)
                    while match:
                        cpu_id = int(match.group(1))
                        seen_cores.add(cpu_id)
                        times = map(int, line.split()[1:])  # first column is the cpu_id
                        self.total[cpu_id].append(sum(times))
                        self.active[cpu_id].append(sum(times) - times[self.idle_time_index])
                        line = fh.next()
                        match = self.cpuid_regex.match(line)

                    for unseen_core in all_cores - seen_cores:
                        self.total[unseen_core].append(0)
                        self.active[unseen_core].append(0)
            except StopIteration:  # EOF
                pass

    def calculate_core_utilization(self):
        """Calculates CPU utilization"""
        diff_active = [[] for _ in xrange(self.cores)]
        diff_total = [[] for _ in xrange(self.cores)]
        self.cpu_util = [[] for _ in xrange(self.cores)]
        for i in xrange(self.cores):
            for j in xrange(len(self.active[i]) - 1):
                temp = self.active[i][j + 1] - self.active[i][j]
                diff_active[i].append(temp)
                diff_total[i].append(self.total[i][j + 1] - self.total[i][j])
                if diff_total[i][j] == 0:
                    self.cpu_util[i].append(0)
                else:
                    temp = float(diff_active[i][j]) / diff_total[i][j]
                    self.cpu_util[i].append(round((float(temp)) * 100, 2))

    def generate_csv(self, context):
        """ generates ``coreutil.csv``"""
        self.output = [0 for _ in xrange(self.cores + 1)]
        for i in range(len(self.cpu_util[0])):
            count = 0
            for j in xrange(len(self.cpu_util)):
                if self.cpu_util[j][i] > round(float(self.threshold), 2):
                    count = count + 1
            self.output[count] += 1
        if self.cpu_util[0]:
            scale_factor = round((float(1) / len(self.cpu_util[0])) * 100, 6)
        else:
            scale_factor = 0
        for i in xrange(len(self.output)):
            self.output[i] = self.output[i] * scale_factor
        with open(self.outfile, 'a+') as tem:
            writer = csv.writer(tem)
            reader = csv.reader(tem)
            if sum(1 for row in reader) == 0:
                row = ['workload', 'iteration', '<threshold']
                for i in xrange(1, self.cores + 1):
                    row.append('{}core'.format(i))
                writer.writerow(row)
            row = [context.result.workload.name, context.result.iteration]
            row.extend(self.output)
            writer.writerow(row)
