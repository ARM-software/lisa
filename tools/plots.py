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
parser.add_argument('--platform_file', type=str,
        help='Platform file to use when plotting')

args = None



def load_platform(output_directory):
    plt_file = None
    platform = None

    if (args.platform_file):
        plt_file = args.platform_file
    elif ('platform.json' in os.listdir(output_directory)):
        plt_file = os.path.join(output_directory, 'platform.json')

    if plt_file is not None:
        with open(plt_file, 'r') as ifile:
            platform = json.load(ifile)

    if platform is None:
        logging.warning("could not find platform file!")
    logging.info('Platform description:')
    logging.info('  %s', platform)
    return platform

def main():
    global args
    args = parser.parse_args()

    # Setup plots to produce
    if args.plots == 'all':
        args.plots = 'tasks clusters cpus stune ediff edspace peripherals'

    # For each rtapp and each run
    if args.outdir is not None:
        # Plot the specified results folder
        return plotdir(args.outdir, load_platform(args.outdir))

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

        for run_idx in sorted(os.listdir(test_dir)):

            run_dir = os.path.join(test_dir, run_idx)
            try:
                run_id = int(run_idx)
            except ValueError:
                continue

            logging.info('Generate plots for [%s]...', run_dir)
            plotdir(run_dir, load_platform(args.outdir))

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
    trace_format = 'ftrace'
    for f in os.listdir(run_dir):
        if f.endswith('html'):
            trace_format = 'systrace' 
        run_dir = f
        break;
    trace = Trace(platform, run_dir, [], trace_format=trace_format);

    # Define time ranges for all the temporal plots
    trace.setXTimeRange(args.tmin, args.tmax)

    # Tasks plots
    if 'tasks' in args.plots:
        trace.analysis.tasks.plotTasks(tasks)
        if pa:
            for task in tasks:
                pa.plotPerf(task)

    # Cluster and CPUs plots
    if 'clusters' in args.plots:
        trace.analysis.frequency.plotClusterFrequencies()
        trace.analysis.frequency.plotClusterFrequencyResidency()
    if 'cpus' in args.plots:
        trace.analysis.cpus.plotCPU()
        trace.analysis.frequency.plotCPUFrequencies()
        trace.analysis.frequency.plotCPUFrequencyResidency()

    print platform
    if 'peripherals' in args.plots:
        if 'peripherals' not in platform:
            logging.warning("no peripheral clocks specified, skipping plotting")
        else:
            for name, clock_infra_name in platform['peripherals'].iteritems():
                trace.analysis.frequency.plotPeripheralClock(
                    title="Clock Frequency for {}".format(name), clk=clock_infra_name)

    # SchedTune plots
    if 'stune' in args.plots:
        trace.analysis.eas.plotSchedTuneConf()
    if 'ediff' in args.plots:
        trace.analysis.eas.plotEDiffTime();
    if 'edspace' in args.plots:
        trace.analysis.eas.plotEDiffSpace();

if __name__ == "__main__":
    main()
