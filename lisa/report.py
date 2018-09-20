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
from lisa.results import Results
from lisa.utils import Loggable


# By default compare all the possible combinations
DEFAULT_COMPARE = [(r'base_', r'test_')]

class Report(Loggable):


    def __init__(self, results_dir, compare=None, formats=['relative']):
        logger = self.get_logger()
        self.results_json = results_dir + '/results.json'
        self.results = {}

        self.compare = []

        # Parse results (if required)
        if not os.path.isfile(self.results_json):
            Results(results_dir)

        # Load results from file (if already parsed)
        logger.info('Load results from [%s]...',
                       self.results_json)
        with open(self.results_json) as infile:
           self.results = json.load(infile)

        # Setup configuration comparisons
        if compare is None:
            compare = DEFAULT_COMPARE
            logger.warning('Comparing all the possible combination')
        for (base_rexp, test_rexp) in compare:
            logger.info('Configured regexps for comparisions '
                           '(bases , tests): (%s, %s)',
                           base_rexp, test_rexp)
            base_rexp = re.compile(base_rexp, re.DOTALL)
            test_rexp = re.compile(test_rexp, re.DOTALL)
            self.compare.append((base_rexp, test_rexp))

        # Report all supported workload classes
        self.__rtapp_report(formats)
        self.__default_report(formats)

    ############################### REPORT RTAPP ###############################

    def __rtapp_report(self, formats):
        logger = self.get_logger()

        if 'rtapp' not in list(self.results.keys()):
            logger.debug('No RTApp workloads to report')
            return

        logger.debug('Reporting RTApp workloads')

        # Setup lables depending on requested report
        if 'absolute' in formats:
            nrg_lable = 'Energy Indexes (Absolute)'
            prf_lable = 'Performance Indexes (Absolute)'
            logger.info('')
            logger.info('Absolute comparisions:')
            print('')
        else:
            nrg_lable = 'Energy Indexes (Relative)'
            prf_lable = 'Performance Indexes (Relative)'
            logger.info('')
            logger.info('Relative comparisions:')
            print('')

        # Dump headers
        print('{:13s}   {:20s} |'\
                ' {:33s} | {:54s} |'\
                .format('Test Id', 'Comparision',
                        nrg_lable, prf_lable))
        print('{:13s}   {:20s} |'\
                ' {:>10s} {:>10s} {:>10s}  |'\
                ' {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} |'\
                .format('', '',
                        'LITTLE', 'big', 'Total',
                        'PerfIndex', 'NegSlacks', 'EDP1', 'EDP2', 'EDP3'))

        # For each test
        _results = self.results['rtapp']
        for tid in sorted(_results.keys()):
            new_test = True
            # For each configuration...
            for base_idx in sorted(_results[tid].keys()):
                # Which matches at least on base regexp
                for (base_rexp, test_rexp) in self.compare:
                    if not base_rexp.match(base_idx):
                        continue
                    # Look for a configuration which matches the test regexp
                    for test_idx in sorted(_results[tid].keys()):
                        if test_idx == base_idx:
                            continue
                        if new_test:
                            print('{:-<37s}+{:-<35s}+{:-<56s}+'\
                                    .format('','', ''))
                            self.__rtapp_reference(tid, base_idx)
                            new_test = False
                        if test_rexp.match(test_idx) == None:
                            continue
                        self.__rtapp_compare(tid, base_idx, test_idx, formats)

        print('')

    def __rtapp_reference(self, tid, base_idx):
        logger = self.get_logger()
        _results = self.results['rtapp']

        logger.debug('Test %s: compare against [%s] base',
                        tid, base_idx)
        res_line = '{0:12s}: {1:22s} | '.format(tid, base_idx)

        # Dump all energy metrics
        for cpus in ['LITTLE', 'big', 'Total']:
            res_base = _results[tid][base_idx]['energy'][cpus]['avg']
            # Dump absolute values
            res_line += ' {0:10.3f}'.format(res_base)
        res_line += ' |'

        # If available, dump also performance results
        if 'performance' not in list(_results[tid][base_idx].keys()):
            print(res_line)
            return

        for pidx in ['perf_avg', 'slack_pct', 'edp1', 'edp2', 'edp3']:
            res_base = _results[tid][base_idx]['performance'][pidx]['avg']

            logger.debug('idx: %s, base: %s', pidx, res_base)

            if pidx in ['perf_avg']:
                res_line += ' {0:s}'.format(TestColors.rate(res_base))
                continue
            if pidx in ['slack_pct']:
                res_line += ' {0:s}'.format(
                        TestColors.rate(res_base, positive_is_good = False))
                continue
            if 'edp' in pidx:
                res_line += ' {0:10.2e}'.format(res_base)
                continue
        res_line += ' |'
        print(res_line)

    def __rtapp_compare(self, tid, base_idx, test_idx, formats):
        logger = self.get_logger()
        _results = self.results['rtapp']

        logger.debug('Test %s: compare %s with %s',
                        tid, base_idx, test_idx)
        res_line = '{0:12s}:   {1:20s} | '.format(tid, test_idx)

        # Dump all energy metrics
        for cpus in ['LITTLE', 'big', 'Total']:
            res_base = _results[tid][base_idx]['energy'][cpus]['avg']
            res_test = _results[tid][test_idx]['energy'][cpus]['avg']
            speedup_cnt =  res_test - res_base
            if 'absolute' in formats:
                res_line += ' {0:10.2f}'.format(speedup_cnt)
            else:
                speedup_pct = 0
                if res_base != 0:
                    speedup_pct =  100.0 * speedup_cnt / res_base
                res_line += ' {0:s}'\
                        .format(TestColors.rate(
                            speedup_pct,
                            positive_is_good = False))
        res_line += ' |'

        # If available, dump also performance results
        if 'performance' not in list(_results[tid][base_idx].keys()):
            print(res_line)
            return

        for pidx in ['perf_avg', 'slack_pct', 'edp1', 'edp2', 'edp3']:
            res_base = _results[tid][base_idx]['performance'][pidx]['avg']
            res_test = _results[tid][test_idx]['performance'][pidx]['avg']

            logger.debug('idx: %s, base: %s, test: %s',
                            pidx, res_base, res_test)

            if pidx in ['perf_avg']:
                res_line += ' {0:s}'.format(TestColors.rate(res_test))
                continue

            if pidx in ['slack_pct']:
                res_line += ' {0:s}'.format(
                        TestColors.rate(res_test, positive_is_good = False))
                continue

            # Compute difference base-vs-test
            if 'edp' in pidx:
                speedup_cnt = res_base - res_test
                if 'absolute':
                    res_line += ' {0:10.2e}'.format(speedup_cnt)
                else:
                    res_line += ' {0:s}'.format(TestColors.rate(speedup_pct))

        res_line += ' |'
        print(res_line)

    ############################### REPORT DEFAULT #############################

    def __default_report(self, formats):
        logger = self.get_logger()

        # Build list of workload types which can be rendered using the default parser
        wtypes = []
        for supported_wtype in DEFAULT_WTYPES:
            if supported_wtype in list(self.results.keys()):
                wtypes.append(supported_wtype)

        if len(wtypes) == 0:
            logger.debug('No Default workloads to report')
            return

        logger.debug('Reporting Default workloads')

        # Setup lables depending on requested report
        if 'absolute' in formats:
            nrg_lable = 'Energy Indexes (Absolute)'
            prf_lable = 'Performance Indexes (Absolute)'
            logger.info('')
            logger.info('Absolute comparisions:')
            print('')
        else:
            nrg_lable = 'Energy Indexes (Relative)'
            prf_lable = 'Performance Indexes (Relative)'
            logger.info('')
            logger.info('Relative comparisions:')
            print('')

        # Dump headers
        print('{:9s}   {:20s} |'\
                ' {:33s} | {:54s} |'\
                .format('Test Id', 'Comparision',
                        nrg_lable, prf_lable))
        print('{:9s}   {:20s} |'\
                ' {:>10s} {:>10s} {:>10s}  |'\
                ' {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} |'\
                .format('', '',
                        'LITTLE', 'big', 'Total',
                        'Perf', 'CTime', 'EDP1', 'EDP2', 'EDP3'))

        # For each default test
        for wtype in wtypes:
            _results = self.results[wtype]
            for tid in sorted(_results.keys()):
                new_test = True
                # For each configuration...
                for base_idx in sorted(_results[tid].keys()):
                    # Which matches at least on base regexp
                    for (base_rexp, test_rexp) in self.compare:
                        if not base_rexp.match(base_idx):
                            continue
                        # Look for a configuration which matches the test regexp
                        for test_idx in sorted(_results[tid].keys()):
                            if test_idx == base_idx:
                                continue
                            if new_test:
                                print('{:-<37s}+{:-<35s}+{:-<56s}+'\
                                        .format('','', ''))
                                new_test = False
                            if not test_rexp.match(test_idx):
                                continue
                            self.__default_compare(wtype, tid, base_idx, test_idx, formats)

        print('')

    def __default_compare(self, wtype, tid, base_idx, test_idx, formats):
        logger = self.get_logger()
        _results = self.results[wtype]

        logger.debug('Test %s: compare %s with %s',
                        tid, base_idx, test_idx)
        res_comp = '{0:s} vs {1:s}'.format(test_idx, base_idx)
        res_line = '{0:8s}: {1:22s} | '.format(tid, res_comp)

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
            if 'absolute' in formats:
                res_line += ' {0:10.2f}'.format(speedup_cnt)
            else:
                speedup_pct =  100.0 * speedup_cnt / res_base
                res_line += ' {0:s}'\
                        .format(TestColors.rate(
                            speedup_pct,
                            positive_is_good = False))
        res_line += ' |'

        # If available, dump also performance results
        if 'performance' not in list(_results[tid][base_idx].keys()):
            print(res_line)
            return

        for pidx in ['perf_avg', 'ctime_avg', 'edp1', 'edp2', 'edp3']:
            res_base = _results[tid][base_idx]['performance'][pidx]['avg']
            res_test = _results[tid][test_idx]['performance'][pidx]['avg']

            logger.debug('idx: %s, base: %s, test: %s',
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
            if 'absolute' in formats:
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
        print(res_line)

# List of workload types which can be parsed using the default test parser
DEFAULT_WTYPES = ['perf_bench_messaging', 'perf_bench_pipe']

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
