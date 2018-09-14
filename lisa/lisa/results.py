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


import argparse
import fnmatch as fnm
import json
import math
import numpy as np
import os
import re
import sys
import logging
from collections import defaultdict

from lisa.colors import TestColors

class Results(object):

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results_json = results_dir + '/results.json'
        self.results = {}

        # Setup logging
        self._log = logging.getLogger('Results')

        # Do nothing if results have been already parsed
        if os.path.isfile(self.results_json):
            return

        # Parse results
        self.base_wls = defaultdict(list)
        self.test_wls = defaultdict(list)

        self._log.info('Loading energy/perf data...')

        for test_idx in sorted(os.listdir(self.results_dir)):

            test_dir = self.results_dir + '/' + test_idx
            if not os.path.isdir(test_dir):
                continue

            test = TestFactory.get(test_idx, test_dir, self.results)
            test.parse()

        results_json = self.results_dir + '/results.json'
        self._log.info('Dump perf results on JSON file [%s]...',
                       results_json)
        with open(results_json, 'w') as outfile:
            json.dump(self.results, outfile, indent=4, sort_keys=True)

################################################################################
# Tests processing base classes
################################################################################

class Test(object):

    def __init__(self, test_idx, test_dir, res):
        self.test_idx = test_idx
        self.test_dir = test_dir
        self.res = res
        match = TEST_DIR_RE.search(test_dir)
        if not match:
            self._log.error('Results folder not matching naming template')
            self._log.error('Skip parsing of test results [%s]', test_dir)
            return

        # Create required JSON entries
        wtype = match.group(1)
        if wtype not in list(res.keys()):
            res[wtype] = {}
        wload_idx = match.group(3)
        if wload_idx not in list(res[wtype].keys()):
            res[wtype][wload_idx] = {}
        conf_idx = match.group(2)
        if conf_idx not in list(res[wtype][wload_idx].keys()):
            res[wtype][wload_idx][conf_idx] = {}

        # Set the workload type for this test
        self.wtype = wtype
        self.wload_idx = wload_idx
        self.conf_idx = conf_idx

        # Energy metrics collected for all tests
        self.little = []
        self.total = []
        self.big = []

    def parse(self):

        self._log.info('Processing results from wtype [%s]', self.wtype)

        # Parse test's run results
        for run_idx in sorted(os.listdir(self.test_dir)):

            # Skip all files which are not folders
            run_dir = os.path.join(self.test_dir,  run_idx)
            if not os.path.isdir(run_dir):
                continue

            run = self.parse_run(run_idx, run_dir)
            self.collect_energy(run)
            self.collect_performance(run)

        # Report energy/performance stats over all runs
        self.res[self.wtype][self.wload_idx][self.conf_idx]\
                ['energy'] = self.energy()
        self.res[self.wtype][self.wload_idx][self.conf_idx]\
                ['performance'] = self.performance()

    def collect_energy(self, run):
        # Keep track of average energy of each run
        self.little.append(run.little_nrg)
        self.total.append(run.total_nrg)
        self.big.append(run.big_nrg)

    def energy(self):
        # Compute energy stats over all run
        return {
                'LITTLE' : Stats(self.little).get(),
                'big'    : Stats(self.big).get(),
                'Total'  : Stats(self.total).get()
        }

class TestFactory(object):

    @staticmethod
    def get(test_idx, test_dir, res):

        # Retrive workload class from results folder name
        match = TEST_DIR_RE.search(test_dir)
        if not match:
            self._log.error('Results folder not matching naming template')
            self._log.error('Skip parsing of test results [%s]', test_dir)
            return

        # Create workload specifi test class
        wtype = match.group(1)

        if wtype == 'rtapp':
            return RTAppTest(test_idx, test_dir, res)

        # Return a generi test parser
        return DefaultTest(test_idx, test_dir, res)

class Energy(object):

    def __init__(self, nrg_file):

        # Set of exposed attributes
        self.little = 0.0
        self.big = 0.0
        self.total = 0.0

        self._log.debug('Parse [%s]...', nrg_file)

        with open(nrg_file, 'r') as infile:
            nrg = json.load(infile)

        if 'LITTLE' in nrg:
            self.little = float(nrg['LITTLE'])
        if 'big' in nrg:
            self.big = float(nrg['big'])
        self.total = self.little + self.big

        self._log.debug('Energy LITTLE [%s], big [%s], Total [%s]',
                        self.little, self.big, self.total)

class Stats(object):

    def __init__(self, data):
        self.stats = {}
        self.stats['count'] = len(data)
        self.stats['min']   = min(data)
        self.stats['max']   = max(data)
        self.stats['avg']   = sum(data)/len(data)
        std = Stats.stdev(data)
        c99 = Stats.ci99(data, std)
        self.stats['std']   = std
        self.stats['c99']   = c99

    def get(self):
        return self.stats

    @staticmethod
    def stdev(values):
        sum1 = 0
        sum2 = 0
        for value in values:
            sum1 += value
            sum2 += math.pow(value, 2)
        # print 'sum1: {}, sum2: {}'.format(sum1, sum2)
        avg =  sum1 / len(values)
        var = (sum2 / len(values)) - (avg * avg)
        # print 'avg: {} var: {}'.format(avg, var)
        std = math.sqrt(var)
        return float(std)

    @staticmethod
    def ci99(values, std):
        count = len(values)
        ste = std / math.sqrt(count)
        c99 = 2.58 * ste
        return c99


################################################################################
# Run processing base classes
################################################################################

class Run(object):

    def __init__(self, run_idx, run_dir):
        self.run_idx = run_idx
        self.nrg = None

        self._log.debug('Parse [%s]...', 'Run', run_dir)

        # Energy stats
        self.little_nrg = 0
        self.total_nrg = 0
        self.big_nrg = 0

        nrg_file = run_dir + '/energy.json'
        if os.path.isfile(nrg_file):
            self.nrg = Energy(nrg_file)
            self.little_nrg = self.nrg.little
            self.total_nrg = self.nrg.total
            self.big_nrg = self.nrg.big

################################################################################
# RTApp workload parsing classes
################################################################################

class RTAppTest(Test):

    def __init__(self, test_idx, test_dir, res):
        super(RTAppTest, self).__init__(test_idx, test_dir, res)

        # RTApp specific performance metric
        self.slack_pct = []
        self.perf_avg = []
        self.edp1 = []
        self.edp2 = []
        self.edp3 = []

        self.rtapp_run = {}

    def parse_run(self, run_idx, run_dir):
        return RTAppRun(run_idx, run_dir)

    def collect_performance(self, run):
        # Keep track of average performances of each run
        self.slack_pct.extend(run.slack_pct)
        self.perf_avg.extend(run.perf_avg)
        self.edp1.extend(run.edp1)
        self.edp2.extend(run.edp2)
        self.edp3.extend(run.edp3)

        # Keep track of performance stats for each run
        self.rtapp_run[run.run_idx] = {
                'slack_pct' : Stats(run.slack_pct).get(),
                'perf_avg'  : Stats(run.perf_avg).get(),
                'edp1'      : Stats(run.edp1).get(),
                'edp2'      : Stats(run.edp2).get(),
                'edp3'      : Stats(run.edp3).get(),
        }

    def performance(self):

        # Dump per run rtapp stats
        prf_file = os.path.join(self.test_dir, 'performance.json')
        with open(prf_file, 'w') as ofile:
            json.dump(self.rtapp_run, ofile, indent=4, sort_keys=True)

        # Return oveall stats
        return {
                'slack_pct' : Stats(self.slack_pct).get(),
                'perf_avg'  : Stats(self.perf_avg).get(),
                'edp1'      : Stats(self.edp1).get(),
                'edp2'      : Stats(self.edp2).get(),
                'edp3'      : Stats(self.edp3).get(),
        }


class RTAppRun(Run):

    def __init__(self, run_idx, run_dir):
        # Call base class to parse energy data
        super(RTAppRun, self).__init__(run_idx, run_dir)

        # RTApp specific performance stats
        self.slack_pct = []
        self.perf_avg = []
        self.edp1 = []
        self.edp2 = []
        self.edp3 = []

        rta = {}

        # Load run's performance of each task
        for task_idx in sorted(os.listdir(run_dir)):

            if not fnm.fnmatch(task_idx, 'rt-app-*.log'):
                continue

            # Parse run's performance results
            prf_file = run_dir + '/' + task_idx
            task = RTAppPerf(prf_file, self.nrg)

            # Keep track of average performances of each task
            self.slack_pct.append(task.prf['slack_pct'])
            self.perf_avg.append(task.prf['perf_avg'])
            self.edp1.append(task.prf['edp1'])
            self.edp2.append(task.prf['edp2'])
            self.edp3.append(task.prf['edp3'])

            # Keep track of performance stats for each task
            rta[task.name] = task.prf

        # Dump per task rtapp stats
        prf_file = os.path.join(run_dir, 'performance.json')
        with open(prf_file, 'w') as ofile:
            json.dump(rta, ofile, indent=4, sort_keys=True)


class RTAppPerf(object):

    def __init__(self, perf_file, nrg):

        # Set of exposed attibutes
        self.prf = {
                'perf_avg'  : 0,
                'perf_std'  : 0,
                'run_sum'   : 0,
                'slack_sum' : 0,
                'slack_pct' : 0,
                'edp1' : 0,
                'edp2' : 0,
                'edp3' : 0
        }

        self._log.debug('Parse [%s]...', perf_file)

        # Load performance data for each RT-App task
        self.name = perf_file.split('-')[-2]
        self.data = np.loadtxt(perf_file, comments='#', unpack=False)

        # Max Slack (i.e. configured/expected slack): period - run
        max_slack = np.subtract(
                self.data[:,RTAPP_COL_C_PERIOD], self.data[:,RTAPP_COL_C_RUN])

        # Performance Index: 100 * slack / max_slack
        perf = np.divide(self.data[:,RTAPP_COL_SLACK], max_slack)
        perf = np.multiply(perf, 100)
        self.prf['perf_avg'] = np.mean(perf)
        self.prf['perf_std'] = np.std(perf)
        self._log.debug('perf [%s]: %6.2f,%6.2f',
                        self.name, self.prf['perf_avg'],
                        self.prf['perf_std'])

        # Negative slacks
        nslacks = self.data[:,RTAPP_COL_SLACK]
        nslacks = nslacks[nslacks < 0]
        self._log.debug('Negative slacks: %s', nslacks)
        self.prf['slack_sum'] = -nslacks.sum()
        self._log.debug('Negative slack [%s] sum: %6.2f',
                        self.name, self.prf['slack_sum'])

        # Slack over run-time
        self.prf['run_sum'] = np.sum(self.data[:,RTAPP_COL_RUN])
        self.prf['slack_pct'] = 100 * self.prf['slack_sum'] / self.prf['run_sum']
        self._log.debug('SlackPct [%s]: %6.2f %%', self.name, self.slack_pct)

        if nrg is None:
            return

        # Computing EDP
        self.prf['edp1'] = nrg.total * math.pow(self.prf['run_sum'], 1)
        self._log.debug('EDP1 [%s]: {%6.2f}', self.name, self.prf['edp1'])
        self.prf['edp2'] = nrg.total * math.pow(self.prf['run_sum'], 2)
        self._log.debug('EDP2 [%s]: %6.2f', self.name, self.prf['edp2'])
        self.prf['edp3'] = nrg.total * math.pow(self.prf['run_sum'], 3)
        self._log.debug('EDP3 [%s]: %6.2f', self.name, self.prf['edp3'])


# Columns of the per-task rt-app log file
RTAPP_COL_IDX = 0
RTAPP_COL_PERF = 1
RTAPP_COL_RUN = 2
RTAPP_COL_PERIOD = 3
RTAPP_COL_START = 4
RTAPP_COL_END = 5
RTAPP_COL_REL_ST = 6
RTAPP_COL_SLACK = 7
RTAPP_COL_C_RUN = 8
RTAPP_COL_C_PERIOD = 9
RTAPP_COL_WU_LAT = 10

################################################################################
# Generic workload performance parsing class
################################################################################

class DefaultTest(Test):

    def __init__(self, test_idx, test_dir, res):
        super(DefaultTest, self).__init__(test_idx, test_dir, res)

        # Default performance metric
        self.ctime_avg = []
        self.perf_avg = []
        self.edp1 = []
        self.edp2 = []
        self.edp3 = []

    def parse_run(self, run_idx, run_dir):
        return DefaultRun(run_idx, run_dir)

    def collect_performance(self, run):
        # Keep track of average performances of each run
        self.ctime_avg.append(run.ctime_avg)
        self.perf_avg.append(run.perf_avg)
        self.edp1.append(run.edp1)
        self.edp2.append(run.edp2)
        self.edp3.append(run.edp3)

    def performance(self):
        return {
                'ctime_avg' : Stats(self.ctime_avg).get(),
                'perf_avg'  : Stats(self.perf_avg).get(),
                'edp1'      : Stats(self.edp1).get(),
                'edp2'      : Stats(self.edp2).get(),
                'edp3'      : Stats(self.edp3).get(),
        }

class DefaultRun(Run):

    def __init__(self, run_idx, run_dir):
        # Call base class to parse energy data
        super(DefaultRun, self).__init__(run_idx, run_dir)

        # Default specific performance stats
        self.ctime_avg = 0
        self.perf_avg = 0
        self.edp1 = 0
        self.edp2 = 0
        self.edp3 = 0

        # Load default performance.json
        prf_file = os.path.join(run_dir, 'performance.json')
        if not os.path.isfile(prf_file):
            self._log.warning('No performance.json found in %s',
                              run_dir)
            return

        # Load performance report from JSON
        with open(prf_file, 'r') as infile:
            prf = json.load(infile)

        # Keep track of performance value
        self.ctime_avg = prf['ctime']
        self.perf_avg = prf['performance']

        # Compute EDP indexes if energy measurements are available
        if self.nrg is None:
            return

        # Computing EDP
        self.edp1 = self.nrg.total * math.pow(self.ctime_avg, 1)
        self.edp2 = self.nrg.total * math.pow(self.ctime_avg, 2)
        self.edp3 = self.nrg.total * math.pow(self.ctime_avg, 3)


################################################################################
# Globals
################################################################################

# Regexp to match the format of a result folder
TEST_DIR_RE = re.compile(
        r'.*/([^:]*):([^:]*):([^:]*)'
    )

#vim :set tabstop=4 shiftwidth=4 expandtab
