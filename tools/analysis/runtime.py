#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited, Google, and contributors.
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

import trappy
import numpy as np
import argparse

class RunData:
    def __init__(self, pid, comm, time):
        self.pid = pid
        self.comm = comm
        self.last_start_time = time
        self.total_time = np.float64(0.0)
        self.start_time = -1
        self.end_time = -1
        self.running = 1
        self.maxrun = -1

parser = argparse.ArgumentParser(description='Analyze runtimes')

parser.add_argument('--trace', dest='trace_file', action='store', required=True,
                    help='trace file')

parser.add_argument('--normalize', dest='normalize', action='store_true', default=False,
                    help='normalize time')

parser.add_argument('--rows', dest='nrows', action='store', default=20, type=int,
                    help='normalize time')

parser.add_argument('--start-time', dest='start_time', action='store', default=0, type=float,
                    help='trace window start time')

parser.add_argument('--end-time', dest='end_time', action='store', default=None, type=float,
                    help='trace window end time')

args = parser.parse_args()

path_to_html = args.trace_file
nrows = args.nrows

# Hash table for runtimes
runpids = {}

starttime = None
endtime = None

def switch_cb(data):
    global starttime, endtime

    prevpid = data['prev_pid']
    nextpid = data['next_pid']
    time = data['Index']

    if not starttime:
        starttime = time
    endtime = time

    if prevpid != 0:
        # prev pid processing (switch out)
        if runpids.has_key(prevpid):
            rp = runpids[prevpid]
            if rp.running == 1:
                rp.running = 0
                runtime = time - rp.last_start_time
                if runtime > rp.maxrun:
                    rp.maxrun = runtime
                    rp.start_time = rp.last_start_time
                    rp.end_time = time
                rp.total_time += runtime

            else:
                print 'switch out seen while no switch in {}'.format(prevpid)
        else:
            print 'switch out seen while no switch in {}'.format(prevpid)

    if nextpid == 0:
        return

    # nextpid processing  (switch in)
    if not runpids.has_key(nextpid):
        # print 'creating data for nextpid {}'.format(nextpid)
        rp = RunData(nextpid, data['next_comm'], time)
        runpids[nextpid] = rp
        return

    rp = runpids[nextpid]
    if rp.running == 1:
        print 'switch in seen for already running task {}'.format(nextpid)
        return

    rp.running = 1
    rp.last_start_time = time


if args.normalize:
    kwargs = { 'window': (args.start_time, args.end_time) }
else:
    kwargs = { 'abs_window': (args.start_time, args.end_time) }

systrace_obj = trappy.SysTrace(name="systrace", path=path_to_html, \
        scope="sched", events=["sched_switch"], normalize_time=args.normalize, **kwargs)

ncpus = systrace_obj.sched_switch.data_frame["__cpu"].max() + 1
print 'cpus found: {}\n'.format(ncpus)

systrace_obj.apply_callbacks({ "sched_switch": switch_cb })

## Print results
testtime = (endtime - starttime) * ncpus              # for 4 cpus
print "total test time (scaled by number of CPUs): {} secs".format(testtime)

# Print the results: PID, latency, start, end, sort
result = sorted(runpids.items(), key=lambda x: x[1].total_time, reverse=True)
print "\t".join([
    "PID".ljust(10),
    "name".ljust(15),
    "max run (secs)".ljust(15),
    "start time".ljust(15),
    "end time".ljust(15),
    "total runtime".ljust(15),
    "percent cpu".ljust(15),
    "totalpc"
])

totalpc = np.float64(0.0)
for r in result[:nrows]:
    rd = r[1] # RunData named tuple
    if rd.pid != r[0]:
        raise RuntimeError("BUG: pid inconsistency found") # Sanity check
    start = rd.start_time
    end = rd.end_time
    cpupc = (rd.total_time / testtime) * 100
    totalpc += cpupc

    print "\t".join([
        str(r[0]).ljust(10),
        str(rd.comm).ljust(15),
        str(rd.maxrun).ljust(15)[:15],
        str(start).ljust(15)[:15],
        str(end).ljust(15)[:15],
        str(rd.total_time).ljust(15),
        str(cpupc),
        str(totalpc)
    ])
