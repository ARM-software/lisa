#!/usr/bin/python
#
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

import sys
# sys.path.insert(1, "./libs")

from perf_analysis import PerfAnalysis
from trace import Trace

import os
import re
import argparse
import json

# Configure logging
import logging
reload(logging)
logging.basicConfig(
    format='%(asctime)-9s %(levelname)-8s: %(message)s',
    level=logging.DEBUG,
    # level=logging.INFO,
    datefmt='%I:%M:%S')

# Regexp to match the format of a result folder
TEST_DIR_RE = re.compile(
        r'([^:]*):([^:]*):([^:]*)'
    )

parser = argparse.ArgumentParser(
        description='EAS Performance and Trace Plotter')
parser.add_argument('--results', type=str,
        default='./results_latest',
        help='Folder containing experimental results')
parser.add_argument('--outdir', type=str,
        default=None,
        help='A single output folder we want to produce plots for')
parser.add_argument('--tmin', type=float,
        default=None,
        help='Minimum timestamp for all plots')
parser.add_argument('--tmax', type=float,
        default=None,
        help='Maximum timestamp for all plots')
parser.add_argument('--plots', type=str,
        default='all',
        help='List of plots to produce (all,')

args = None

def main():
    global args
    args = parser.parse_args()

    # Setup plots to produce
    if args.plots == 'all':
        args.plots = 'tasks clusters cpus stune ediff edspace'

    # For each rtapp and each run
    if args.outdir is not None:

        # Load platform descriptior
        platform = None
        plt_file = os.path.join(args.outdir, 'platform.json')
        if os.path.isfile(plt_file):
            with open(plt_file, 'r') as ifile:
                platform = json.load(ifile)
        logging.info('Platform description:')
        logging.info('  %s', platform)

        # Plot the specified results folder
        return plotdir(args.outdir, platform)

    for test_idx in sorted(os.listdir(args.results)):

        match = TEST_DIR_RE.search(test_idx)
        if match == None:
            continue
        wtype = match.group(1)
        conf_idx = match.group(2)
        wload_idx = match.group(3)


        # Generate performance plots only for RTApp workloads
        if wtype != 'rtapp':
            continue

        logging.debug('Processing [%s:%s:%s]...',
                wtype, conf_idx, wload_idx)

        # For each run of an rt-app workload
        test_dir = os.path.join(args.results, test_idx)

        # Load platform descriptior
        platform = None
        plt_file = os.path.join(test_dir, 'platform.json')
        if os.path.isfile(plt_file):
            with open(plt_file, 'r') as ifile:
                platform = json.load(ifile)
        logging.info('Platform description:')
        logging.info('  %s', platform)

        for run_idx in sorted(os.listdir(test_dir)):

            run_dir = os.path.join(test_dir, run_idx)
            try:
                run_id = int(run_idx)
            except ValueError:
                continue

            logging.info('Generate plots for [%s]...', run_dir)
            plotdir(run_dir, platform)

def plotdir(run_dir, platform):
    global args
    tasks = None
    pa = None

    # Load RTApp performance data
    try:
        pa = PerfAnalysis(run_dir)

        # Get the list of RTApp tasks
        tasks = pa.tasks()
        logging.info('Tasks: %s', tasks)
    except ValueError:
        pa = None
        logging.info('No performance data found')

    # Load Trace Analysis modules
    trace = Trace(run_dir, platform=platform)

    # Define time ranges for all the temporal plots
    trace.setXTimeRange(args.tmin, args.tmax)

    # Tasks plots
    if 'tasks' in args.plots:
        trace.analysis.tasks.plot_tasks(tasks)
        if pa:
            for task in tasks:
                pa.plotPerf(task)

    # Cluster and CPUs plots
    if 'clusters' in args.plots:
        trace.analysis.frequency.plot_cluster_frequencies()
    if 'cpus' in args.plots:
        trace.analysis.cpus.plot_cpu()

    # SchedTune plots
    if 'stune' in args.plots:
        trace.analysis.eas.plot_sched_tune_conf()
    if 'ediff' in args.plots:
        trace.analysis.eas.plot_e_diff_time();
    if 'edspace' in args.plots:
        trace.analysis.eas.plot_e_diff_space();

if __name__ == "__main__":
    main()
