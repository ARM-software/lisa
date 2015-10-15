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
from results import Results

# Configure logging
import logging
reload(logging)
logging.basicConfig(
    format='%(asctime)-9s %(levelname)-8s: %(message)s',
    level=logging.DEBUG,
    # level=logging.INFO,
    datefmt='%I:%M:%S')

# By default compare all the possible combinations
DEFAULT_COMPARE = [(r'.*', r'.*')]

class Report(object):


    def __init__(self, results_dir, compare=None, numbers=False):
        self.results_json = results_dir + '/results.json'
        self.results = {}

        self.compare = []

        # Parse results (if required)
        if not os.path.isfile(self.results_json):
            Results(results_dir)

        # Load results from file (if already parsed)
        logging.info('%14s - Load results from [%s]...',
                'Results', self.results_json)
        with open(self.results_json) as infile:
           self.results = json.load(infile)

        # Setup configuration comparisons
        if compare is None:
            compare = DEFAULT_COMPARE
            logging.warning('%14s - Comparing all the possible combination',
                    'Results')
        for (base_rexp, test_rexp) in compare:
            base_rexp = re.compile(base_rexp, re.DOTALL)
            test_rexp = re.compile(test_rexp, re.DOTALL)
            self.compare.append((base_rexp, test_rexp))

        # Report all supported workload classes
        self.__rtapp_report(numbers)
        self.__default_report(numbers)

    ############################### REPORT RTAPP ###############################

    def __rtapp_report(self, numbers):

        if 'rtapp' not in self.results.keys():
            logging.debug('%14s - No RTApp workloads to report', 'ReportRTApp')
            return

        logging.debug('%14s - Reporting RTApp workloads', 'ReportRTApp')

        # Setup lables depending on requested report
        if numbers:
            nrg_lable = 'Energy Indexes (Absolute)'
            prf_lable = 'Performance Indexes (Absolute)'
            logging.info('')
            logging.info('%14s - Absolute comparisions:', 'Report')
            print ''
        else:
            nrg_lable = 'Energy Indexes (Relative)'
            prf_lable = 'Performance Indexes (Relative)'
            logging.info('')
            logging.info('%14s - Relative comparisions:', 'Report')
            print ''

        # Dump headers
        print '{:9s}   {:15s} |'\
                ' {:33s} | {:54s} |'\
                .format('Test Id', 'Comparision',
                        nrg_lable, prf_lable)
        print '{:9s}   {:15s} |'\
                ' {:>10s} {:>10s} {:>10s}  |'\
                ' {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} |'\
                .format('', '',
                        'LITTLE', 'big', 'Total',
                        'PerfIndex', 'NegSlacks', 'EDP1', 'EDP2', 'EDP3')

        # For each test
        _results = self.results['rtapp']
        for tid in sorted(_results.keys()):
            new_test = True
            # For each configuration...
            for i, base_idx in enumerate(sorted(_results[tid].keys())):
                # Which matches at least on base regexp
                for (base_rexp, test_rexp) in self.compare:
                    if not base_rexp.match(base_idx):
                        continue
                    # Look for a configuration which matches the test regexp
                    for test_idx in sorted(_results[tid].keys())[i+1:]:
                        if test_idx == base_idx:
                            continue
                        if new_test:
                            print '{:-<28s}+{:-<35s}+{:-<56s}+'\
                                    .format('','', '')
                            new_test = False
                        if not test_rexp.match(test_idx):
                            continue
                        self.__rtapp_compare(tid, base_idx, test_idx, numbers)

        print ''

    def __rtapp_compare(self, tid, base_idx, test_idx, numbers):
        _results = self.results['rtapp']

        logging.debug('Test %s: compare %s with %s',
                tid, base_idx, test_idx)
        res_comp = '{0:s} vs {1:s}'.format(test_idx, base_idx)
        res_line = '{0:8s}: {1:15s} | '.format(tid, res_comp)

        # Dump all energy metrics
        for cpus in ['LITTLE', 'big', 'Total']:
            res_base = _results[tid][base_idx]['energy'][cpus]['avg']
            res_test = _results[tid][test_idx]['energy'][cpus]['avg']
            speedup_cnt =  res_test - res_base
            if numbers:
                res_line += ' {0:10.2f}'.format(speedup_cnt)
            else:
                speedup_pct =  100.0 * speedup_cnt / res_base
                res_line += ' {0:s}'\
                        .format(TestColors.rate(
                            speedup_pct,
                            positive_is_good = False))
        res_line += ' |'

        # If available, dump also performance results
        if 'performance' not in _results[tid][base_idx].keys():
            print res_line
            return

        for pidx in ['perf_avg', 'slack_pct', 'edp1', 'edp2', 'edp3']:
            res_base = _results[tid][base_idx]['performance'][pidx]['avg']
            res_test = _results[tid][test_idx]['performance'][pidx]['avg']

            logging.debug('idx: %s, base: %s, test: %s',
                    pidx, res_base, res_test)

            # Compute difference base-vs-test
            speedup_cnt = 0
            if res_base != 0:
                if pidx in ['perf_avg']:
                    speedup_cnt =  res_test - res_base
                else:
                    speedup_cnt =  res_base - res_test

            # Compute speedup if required
            speedup_pct = 0
            if numbers:
                if 'edp' in pidx:
                    res_line += ' {0:10.2e}'.format(speedup_cnt)
                else:
                    res_line += ' {0:10.2f}'.format(speedup_cnt)
            else:
                if res_base != 0:
                    if pidx in ['perf_avg']:
                        # speedup_pct =  100.0 * speedup_cnt / res_base
                        speedup_pct =  speedup_cnt
                    else:
                        speedup_pct =  100.0 * speedup_cnt / res_base
                res_line += ' {0:s}'.format(TestColors.rate(speedup_pct))
        res_line += ' |'
        print res_line

    ############################### REPORT DEFAULT #############################

    def __default_report(self, numbers):

        # Build list of workload types which can be rendered using the default parser
        wtypes = []
        for supported_wtype in DEFAULT_WTYPES:
            if supported_wtype in self.results.keys():
                wtypes.append(supported_wtype)

        if len(wtypes) == 0:
            logging.debug('%14s - No Default workloads to report', 'ReportDefault')
            return

        logging.debug('%14s - Reporting Default workloads', 'ReportDefault')

        # Setup lables depending on requested report
        if numbers:
            nrg_lable = 'Energy Indexes (Absolute)'
            prf_lable = 'Performance Indexes (Absolute)'
            logging.info('')
            logging.info('%14s - Absolute comparisions:', 'Report')
            print ''
        else:
            nrg_lable = 'Energy Indexes (Relative)'
            prf_lable = 'Performance Indexes (Relative)'
            logging.info('')
            logging.info('%14s - Relative comparisions:', 'Report')
            print ''

        # Dump headers
        print '{:9s}   {:15s} |'\
                ' {:33s} | {:54s} |'\
                .format('Test Id', 'Comparision',
                        nrg_lable, prf_lable)
        print '{:9s}   {:15s} |'\
                ' {:>10s} {:>10s} {:>10s}  |'\
                ' {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} |'\
                .format('', '',
                        'LITTLE', 'big', 'Total',
                        'Perf', 'CTime', 'EDP1', 'EDP2', 'EDP3')

        # For each default test
        for wtype in wtypes:
            _results = self.results[wtype]
            for tid in sorted(_results.keys()):
                new_test = True
                # For each configuration...
                for i, base_idx in enumerate(sorted(_results[tid].keys())):
                    # Which matches at least on base regexp
                    for (base_rexp, test_rexp) in self.compare:
                        if not base_rexp.match(base_idx):
                            continue
                        # Look for a configuration which matches the test regexp
                        for test_idx in sorted(_results[tid].keys())[i+1:]:
                            if test_idx == base_idx:
                                continue
                            if new_test:
                                print '{:-<28s}+{:-<35s}+{:-<56s}+'\
                                        .format('','', '')
                                new_test = False
                            if not test_rexp.match(test_idx):
                                continue
                            self.__default_compare(wtype, tid, base_idx, test_idx, numbers)

        print ''

    def __default_compare(self, wtype, tid, base_idx, test_idx, numbers):
        _results = self.results[wtype]

        logging.debug('Test %s: compare %s with %s',
                tid, base_idx, test_idx)
        res_comp = '{0:s} vs {1:s}'.format(test_idx, base_idx)
        res_line = '{0:8s}: {1:15s} | '.format(tid, res_comp)

        # Dump all energy metrics
        for cpus in ['LITTLE', 'big', 'Total']:

            # If either base of test have a 0 MAX energy, this measn that
            # energy has not been collected
            base_max = _results[tid][base_idx]['energy'][cpus]['max']
            test_max = _results[tid][test_idx]['energy'][cpus]['max']
            if base_max == 0 or test_max == 0:
                res_line += ' {0:10s}'.format('NA')
                continue

            # Otherwise, report energy values
            res_base = _results[tid][base_idx]['energy'][cpus]['avg']
            res_test = _results[tid][test_idx]['energy'][cpus]['avg']

            speedup_cnt =  res_test - res_base
            if numbers:
                res_line += ' {0:10.2f}'.format(speedup_cnt)
            else:
                speedup_pct =  100.0 * speedup_cnt / res_base
                res_line += ' {0:s}'\
                        .format(TestColors.rate(
                            speedup_pct,
                            positive_is_good = False))
        res_line += ' |'

        # If available, dump also performance results
        if 'performance' not in _results[tid][base_idx].keys():
            print res_line
            return

        for pidx in ['perf_avg', 'ctime_avg', 'edp1', 'edp2', 'edp3']:
            res_base = _results[tid][base_idx]['performance'][pidx]['avg']
            res_test = _results[tid][test_idx]['performance'][pidx]['avg']

            logging.debug('idx: %s, base: %s, test: %s',
                    pidx, res_base, res_test)

            # Compute difference base-vs-test
            speedup_cnt = 0
            if res_base != 0:
                if pidx in ['perf_avg']:
                    speedup_cnt =  res_test - res_base
                else:
                    speedup_cnt =  res_base - res_test

            # Compute speedup if required
            speedup_pct = 0
            if numbers:
                if 'edp' in pidx:
                    res_line += ' {0:10.2e}'.format(speedup_cnt)
                else:
                    res_line += ' {0:10.2f}'.format(speedup_cnt)
            else:
                if res_base != 0:
                    if pidx in ['perf_avg']:
                        # speedup_pct =  100.0 * speedup_cnt / res_base
                        speedup_pct =  speedup_cnt
                    else:
                        speedup_pct =  100.0 * speedup_cnt / res_base
                res_line += ' {0:s}'.format(TestColors.rate(speedup_pct))
        res_line += ' |'
        print res_line

# List of workload types which can be parsed using the default test parser
DEFAULT_WTYPES = ['perf_bench_messaging', 'perf_bench_pipe']

#vim :set tabstop=4 shiftwidth=4 expandtab
