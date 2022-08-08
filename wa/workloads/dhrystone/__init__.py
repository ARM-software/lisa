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

# pylint: disable=E1101,W0201

import os
import re

from wa import Workload, Parameter, ConfigError, Executable
from wa.utils.exec_control import once
from wa.utils.types import cpu_mask


class Dhrystone(Workload):

    name = 'dhrystone'
    description = """
    Runs the Dhrystone benchmark.

    Original source from::

        http://classes.soe.ucsc.edu/cmpe202/benchmarks/standard/dhrystone.c

    This version has been modified to configure duration and the number of
    threads used.

    """

    bm_regex = re.compile(r'This machine benchmarks at (?P<score>\d+)')
    dmips_regex = re.compile(r'(?P<score>\d+) DMIPS')
    time_regex = re.compile(r'Total dhrystone run time: (?P<time>[0-9.]+)')

    default_mloops = 100

    parameters = [
        Parameter('duration', kind=int, default=0,
                  description='''
                  The duration, in seconds, for which dhrystone will be
                  executed. Either this or ``mloops`` should be specified but
                  not both.
                  '''),
        Parameter('mloops', kind=int, default=0,
                  description='''
                  Millions of loops to run. Either this or ``duration`` should
                  be specified, but not both. If neither is specified, this
                  will default ' to ``{}``
                  '''.format(default_mloops)),
        Parameter('threads', kind=int, default=4,
                  description='''
                  The number of separate dhrystone "threads" that will be forked.
                  '''),
        Parameter('delay', kind=int, default=0,
                  description=('''
                  The delay, in seconds, between kicking off of dhrystone
                  threads (if ``threads`` > 1).
                  ''')),
        Parameter('cpus', kind=cpu_mask, default=0, aliases=['taskset_mask'],
                  description=''' The processes spawned by dhrystone will be
                  pinned to cores as specified by this parameter. The mask can
                  be specified directly as a mask, as a list of cpus or a sysfs-
                  style string '''),
    ]

    @once
    def initialize(self, context):
        resource = Executable(self, self.target.abi, 'dhrystone')
        host_exe = context.get_resource(resource)
        Dhrystone.target_exe = self.target.install(host_exe)

    def setup(self, context):
        if self.mloops:
            execution_mode = '-l {}'.format(self.mloops)
        else:
            execution_mode = '-r {}'.format(self.duration)
        if self.cpus:
            taskset_string = '{} taskset {} '.format(self.target.busybox,
                                                     self.cpus.mask())
        else:
            taskset_string = ''
        self.command = '{}{} {} -t {} -d {}'.format(taskset_string,
                                                    self.target_exe,
                                                    execution_mode,
                                                    self.threads, self.delay)
        if self.duration:
            self.timeout = self.duration + self.delay * self.threads + 10
        else:
            self.timeout = 300

        self.target.killall('dhrystone')

    def run(self, context):
        self.output = None
        try:
            self.output = self.target.execute(self.command,
                                              timeout=self.timeout,
                                              check_exit_code=False)
        except KeyboardInterrupt:
            self.target.killall('dhrystone')
            raise

    def extract_results(self, context):
        if self.output:
            outfile = os.path.join(context.output_directory, 'dhrystone.output')
            with open(outfile, 'w') as wfh:
                wfh.write(self.output)
            context.add_artifact('dhrystone-output', outfile, 'raw', "dhrystone's stdout")

    def update_output(self, context):
        if not self.output:
            return

        score_count = 0
        dmips_count = 0
        total_score = 0
        total_dmips = 0

        for line in self.output.split('\n'):
            match = self.time_regex.search(line)
            if match:
                context.add_metric('time', float(match.group('time')), 'seconds',
                                   lower_is_better=True)
            else:
                match = self.bm_regex.search(line)
                if match:
                    metric = 'thread {} score'.format(score_count)
                    value = int(match.group('score'))
                    context.add_metric(metric, value)
                    score_count += 1
                    total_score += value
                else:
                    match = self.dmips_regex.search(line)
                    if match:
                        metric = 'thread {} DMIPS'.format(dmips_count)
                        value = int(match.group('score'))
                        context.add_metric(metric, value)
                        dmips_count += 1
                        total_dmips += value

        context.add_metric('total DMIPS', total_dmips)
        context.add_metric('total score', total_score)

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall('dhrystone')

    def validate(self):
        if self.mloops and self.duration:  # pylint: disable=E0203
            msg = 'mloops and duration cannot be both specified at the '\
                  'same time for dhrystone.'
            raise ConfigError(msg)
        if not self.mloops and not self.duration:  # pylint: disable=E0203
            self.mloops = self.default_mloops
