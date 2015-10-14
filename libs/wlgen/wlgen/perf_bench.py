
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

        # TODO: Assume perf is pre-installed on target
        #target.setup('perf')

        super(PerfMessaging, self).__init__(target, name, None)

        # perf "sched" executor
        self.wtype = 'perf_bench_messaging'
        self.executor = 'perf bench sched messaging'


    def conf(self,
             group = 1,
             loop = 10,
             pipe = '',
             thread = '',
             cpus=None,
             cgroup=None,
             exc_id=0):

        if pipe is not '':
            pipe = '--pipe'
        if thread is not '':
            thread = '--thread'

        super(PerfMessaging, self).conf('custom',
                {'group': str(group), 'loop': str(loop), 'pipe': pipe, 'thread': thread},
                0, cpus, cgroup, exc_id)


        self.command = '{0:s}/perf bench sched messaging {1:s} {2:s} --group {3:s} --loop {4:s}'\
                .format(self.target.executables_directory,
                        self.params['custom']['pipe'],
                        self.params['custom']['thread'],
                        self.params['custom']['group'],
                        self.params['custom']['loop'])

        logging.debug('Command line: {}'.format(self.command))

        # Set and return the test label
        self.test_label = '{0:s}_{1:02d}'.format(self.name, self.exc_id)
        return self.test_label

    def getCompletionTime(self):
        results = self.getOutput()
        match = re.search('Total time: ([0-9\.]+) \[sec\]', results)
        return match.group(1)

class PerfPipe(Workload):

    def __init__(self,
                 target,
                 name):

        # TODO: Assume perf is pre-installed on target
        #target.setup('perf')

        super(PerfPipe, self).__init__(target, name, None)

        # perf "sched" executor
        self.wtype = 'perf_bench_pipe'
        self.executor = 'perf bench sched pipe'


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

        logging.debug('Command line: {}'.format(self.command))

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


