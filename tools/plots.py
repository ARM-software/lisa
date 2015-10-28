#!/usr/bin/python

import sys
# sys.path.insert(1, "./libs")
#from utils.perf_analysis import PerfAnalysis
#from utils.trace_analysis import TraceAnalysis
from perf_analysis import PerfAnalysis
from trace_analysis import TraceAnalysis

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
        description='EAS RFC Configuration Comparator.')
parser.add_argument('--results', type=str,
        default='./results_latest',
        help='Folder containing experimental results')
parser.add_argument('--tmin', type=float,
        default=None,
        help='Minimum timestamp for all plots')
parser.add_argument('--tmax', type=float,
        default=None,
        help='Maximum timestamp for all plots')

if __name__ == "__main__":
    args = parser.parse_args()

    # For each rtapp and each run
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

            # Load RTApp performance data
            pa = PerfAnalysis(run_dir)

            # Get the list of RTApp tasks
            tasks = pa.tasks()
            logging.info('Tasks: %s', tasks)

            # Load Trace Analysis modules
            ta = TraceAnalysis(platform, run_dir, tasks)

            # Define time ranges for all the temporal plots
            ta.setXTimeRange(args.tmin, args.tmax)

            # Tasks plots
            ta.plotTasks()
            for task in tasks:
                pa.plotPerf(task)

            # Cluster and CPUs plots
            ta.plotClusterFrequencies()
            ta.plotCPU()

            # SchedTune plots
            ta.plotSchedTuneConf()
            ta.plotEDiffTime();
            ta.plotEDiffSpace();

