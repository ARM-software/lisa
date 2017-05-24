# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import fileinput
import json
import logging
import os
import re

from wlgen import Workload

class PerfMessaging(Workload):

    def __init__(self,
                 target,
                 name):

        # Setup logging
        self.logger = logging.getLogger('perf_bench')

        # TODO: Assume perf is pre-installed on target
        #target.setup('perf')

        super(PerfMessaging, self).__init__(target, name)

        # perf "sched" executor
        self.wtype = 'perf_bench_messaging'
        self.executor = 'perf bench sched messaging'

        # Setup post-processing callback
        self.setCallback('postrun', self.__postrun)

    def conf(self,
             group = 1,
             loop = 500,
             pipe = '',
             thread = '',
             cpus=None,
             cgroup=None,
             exc_id=0,
             run_dir=None):

        if pipe is not '':
            pipe = '--pipe'
        if thread is not '':
            thread = '--thread'

        super(PerfMessaging, self).conf(
            'custom',
            params={'group': str(group),
                    'loop': str(loop),
                    'pipe': pipe,
                    'thread': thread},
            duration=0, cpus=cpus, exc_id=exc_id, run_dir=run_dir)

        self.command = '{0:s}/perf bench sched messaging {1:s} {2:s} --group {3:s} --loop {4:s}'\
                .format(self.target.executables_directory,
                        self.params['custom']['pipe'],
                        self.params['custom']['thread'],
                        self.params['custom']['group'],
                        self.params['custom']['loop'])

        self.logger.debug('%14s - Command line: %s', 'PerfBench', self.command)

        # Set and return the test label
        self.test_label = '{0:s}_{1:02d}'.format(self.name, self.exc_id)
        return self.test_label

    def getCompletionTime(self):
        results = self.getOutput()
        match = re.search('Total time: ([0-9\.]+) \[sec\]', results)
        return match.group(1)

    def __postrun(self, params):
        destdir = params['destdir']
        if destdir is None:
            return

        logfile = '{}/output.log'.format(destdir)
        self.logger.debug('%14s - Saving output on [%s]...',
                          'PerfBench', logfile)
        with open(logfile, 'w') as ofile:
            for line in self.getOutput().split('\n'):
                ofile.write(line+'\n')

        # Computing performance metric
        ctime = float(self.getCompletionTime())
        perf = 1.0 / ctime
        results = {
                "ctime" : ctime,
                "performance" : perf
        }

        self.logger.info('%14s - Completion time: %.6f, Performance %.6f',
                         'PerfBench', ctime, perf)

        perfile = '{}/performance.json'.format(destdir)
        self.logger.debug('%14s - Saving performance into [%s]...',
                          'PerfBench', perfile)
        with open(perfile, 'w') as ofile:
            json.dump(results, ofile, sort_keys=True, indent=4)


class PerfPipe(Workload):

    def __init__(self,
                 target,
                 name):

        # TODO: Assume perf is pre-installed on target
        #target.setup('perf')

        super(PerfPipe, self).__init__(target, name)

        # Setup logging
        self.logger = logging.getLogger('perf_bench')

        # perf "sched" executor
        self.wtype = 'perf_bench_pipe'
        self.executor = 'perf bench sched pipe'

        # Setup post-processing callback
        self.setCallback('postrun', self.__postrun)

    def conf(self,
             loop = 10,
             cpus=None,
             cgroup=None,
             exc_id=0):

        super(PerfPipe, self).conf('custom',
                {'loop': str(loop)},
                0, cpus, cgroup, exc_id)

        self.command = '{0:s}/perf bench sched pipe --loop {1:s}'\
                .format(self.target.executables_directory,
                        self.params['custom']['loop'])

        self.logger.debug('%14s - Command line: %s',
                          'PerfBench', self.command)

        # Set and return the test label
        self.test_label = '{0:s}_{1:02d}'.format(self.name, self.exc_id)
        return self.test_label

    def getCompletionTime(self):
        results = self.getOutput()
        match = re.search('Total time: ([0-9\.]+) \[sec\]', results)
        return match.group(1)

    def getUsecPerOp(self):
        results = self.getOutput()
        match = re.search('([0-9\.]+) usecs/op', results)
        return match.group(1)

    def getOpPerSec(self):
        results = self.getOutput()
        match = re.search('([0-9]+) ops/sec', results)
        return match.group(1)

    def __postrun(self, params):
        destdir = params['destdir']
        if destdir is None:
            return

        logfile = '{}/output.log'.format(destdir)
        self.logger.debug('%14s - Saving output on [%s]...',
                          'PerfBench', logfile)
        with open(logfile, 'w') as ofile:
            for line in self.getOutput().split('\n'):
                ofile.write(line+'\n')

        # Computing performance metric
        ctime = float(self.getCompletionTime())
        uspo = float(self.getUsecPerOp())
        ops = float(self.getOpPerSec())

        perf = 1.0 / ctime
        results = {
                "ctime" : ctime,
                "performance" : perf,
                "usec/op" : uspo,
                "ops/sec" : ops
        }

        self.logger.info('%14s - Completion time: %.6f, Performance %.6f',
                         'PerfBench', ctime, perf)

        # Reporting performance metric
        perfile = '{}/performance.json'.format(destdir)
        self.logger.debug('%14s - Saving performance into [%s]...',
                          'PerfBench', perfile)
        with open(perfile, 'w') as ofile:
            json.dump(results, ofile, sort_keys=True, indent=4)


