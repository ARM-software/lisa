#!/usr/bin/python

import argparse
import fnmatch as fnm
import json
import math
import numpy as np
import os
import re
import sys

from collections import defaultdict
from colors import TestColors

import logging

class Results(object):

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results_json = results_dir + '/results.json'
        self.results = {}

        # Do nothing if results have been already parsed
        if os.path.isfile(self.results_json):
            return

        # Parse results
        self.base_wls = defaultdict(list)
        self.test_wls = defaultdict(list)

        logging.info('%14s - Loading energy/perf data...', 'Parser')

        for test_idx in sorted(os.listdir(self.results_dir)):

            test_dir = self.results_dir + '/' + test_idx
            if not os.path.isdir(test_dir):
                continue

            test = TestFactory.get(test_idx, test_dir, self.results)
            test.parse()

        results_json = self.results_dir + '/results.json'
        logging.info('%14s - Dump perf results on JSON file [%s]...',
                'Parser', results_json)
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
            logging.error('%14s - Results folder not matching naming template',
                    'TestParser')
            logging.error('%14s - Skip parsing of test results [%s]',
                    'TestParser', test_dir)
            return

        # Create required JSON entries
        wtype = match.group(1)
        if wtype not in res.keys():
            res[wtype] = {}
        wload_idx = match.group(3)
        if wload_idx not in res[wtype].keys():
            res[wtype][wload_idx] = {}
        conf_idx = match.group(2)
        if conf_idx not in res[wtype][wload_idx].keys():
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

        logging.info('%14s - Processing results from wtype [%s]',
                'TestParser', self.wtype)

        # Parse test's run results
        for run_idx in sorted(os.listdir(self.test_dir)):
            run_dir = self.test_dir + '/' + run_idx
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
            logging.error('%14s - Results folder not matching naming template',
                    'TestParser')
            logging.error('%14s - Skip parsing of test results [%s]',
                    'TestParser', test_dir)
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
        self.little = None
        self.big = None
        self.total = None

        logging.debug('%14s - Parse [%s]...', 'Energy', nrg_file)

        with open(nrg_file, 'r') as infile:
            nrg = json.load(infile)

        self.little = float(nrg['LITTLE'])
        self.big = float(nrg['big'])
        self.total = self.little + self.big

        logging.debug('%14s - Energy LITTLE [%s], big [%s], Total [%s]',
                'Energy', self.little, self.big, self.total)

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

        logging.debug('%14s - Parse [%s]...', 'Run', run_dir)

        # Energy stats
        self.little_nrg = None
        self.total_nrg = None
        self.big_nrg = None

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

    def parse_run(self, run_idx, run_dir):
        return RTAppRun(run_idx, run_dir)

    def collect_performance(self, run):
        # Keep track of average performances of each run
        self.slack_pct.extend(run.slack_pct)
        self.perf_avg.extend(run.perf_avg)
        self.edp1.extend(run.edp1)
        self.edp2.extend(run.edp2)
        self.edp3.extend(run.edp3)

    def performance(self):
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

        # Load run's performance of each task
        for perf_idx in sorted(os.listdir(run_dir)):

            if not fnm.fnmatch(perf_idx, 'rt-app-*.log'):
                continue

            # Parse run's performance results
            prf_file = run_dir + '/' + perf_idx
            prf = RTAppPerf(prf_file, self.nrg)

            # Keep track of average performances of each task
            self.slack_pct.append(prf.slack_pct)
            self.perf_avg.append(prf.perf_avg)
            self.edp1.append(prf.edp1)
            self.edp2.append(prf.edp2)
            self.edp3.append(prf.edp3)

class RTAppPerf(object):

    def __init__(self, perf_file, nrg):

        # Set of exposed attibutes
        self.perf_avg = None
        self.perf_std = None
        self.run_sum = None
        self.slack_sum = None
        self.slack_pct = None
        self.edp1 = None
        self.edp2 = None
        self.edp3 = None

        logging.debug('%14s - Parse [%s]...', 'Perf', perf_file)

        # Load performance data for each RT-App task
        self.name = perf_file.split('-')[-2]
        self.data = np.loadtxt(perf_file, comments='#', unpack=False)

        # Max Slack (i.e. configured/expected slack): period - run
        self.max_slack = np.subtract(
                self.data[:,RTAPP_COL_PERIOD], self.data[:,RTAPP_COL_RUN])

        # Performance Index: 100 * slack / max_slack
        perf = np.divide(self.data[:,RTAPP_COL_SLACK], self.max_slack)
        perf = np.multiply(perf, 100)
        self.perf_avg = np.mean(perf)
        self.perf_std  = np.std(perf)
        # logging.debug('perf [%s]: %6.2f,%6.2f',
        #                 self.name, self.perf_mean, self.perf_std)

        # Negative slacks
        slacks = self.data[:,RTAPP_COL_SLACK]
        slacks = slacks[slacks < 0]
        # logging.debug('Negative Slacks: %s', self.slacks)
        self.slack_sum = slacks.sum()
        # logging.debug('slack [%s]: %6.2f', self.name, self.slack_sum)

        # Slack over run-time
        self.run_sum = np.sum(self.data[:,RTAPP_COL_RUN])
        self.slack_pct = 100 * self.slack_sum / self.run_sum
        # logging.debug('SlackPct [%s]: %6.2f %%', self.name, self.slack_pct)

        if nrg is None:
            return

        # Computing EDP
        self.edp1 = nrg.total * math.pow(self.run_sum, 1)
        # logging.debug('EDP1 [%s]: {%6.2f}', self.name, self.edp1)
        self.edp2 = nrg.total * math.pow(self.run_sum, 2)
        # logging.debug('EDP2 [%s]: %6.2f', self.name, self.edp2)
        self.edp3 = nrg.total * math.pow(self.run_sum, 3)
        # logging.debug('EDP3 [%s]: %6.2f', self.name, self.edp3)


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


################################################################################
# Globals
################################################################################

# Regexp to match the format of a result folder
TEST_DIR_RE = re.compile(
        r'.*/([^:]*):([^:]*):([^:]*)'
    )

#vim :set tabstop=4 shiftwidth=4 expandtab
