#    Copyright 2015-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Utils module has generic utils that will be used across
objects
"""
import collections
import warnings
from trappy.utils import listify


def normalize_list(val, lst):
    """Normalize a unitary list"""

    if len(lst) != 1:
        raise RuntimeError("Cannot Normalize a non-unitary list")

    return lst * val


def decolonize(val):
    """Remove the colon at the end of the word
    This will be used by the unique word of
    template class to sanitize attr accesses
    """

    return val.strip(":")


def get_trace_event_data(trace, execnames=None, pids=None):
    """Create a list of objects that can be consumed by EventPlot to plot
    task residency like kernelshark

    """

    if execnames:
        execnames = listify(execnames)

    if pids:
        pids = listify(pids)

    data = collections.defaultdict(list)
    pmap = {}

    data_frame = trace.sched_switch.data_frame
    start_idx = data_frame.index.values[0]
    end_idx = data_frame.index.values[-1]

    procs = set()

    for index, row in data_frame.iterrows():
        prev_pid = row["prev_pid"]
        next_pid = row["next_pid"]
        next_comm = row["next_comm"]

        if prev_pid in pmap:
            name = pmap[prev_pid]
            data[name][-1][1] = index
            del pmap[prev_pid]

        name = "{}-{}".format(next_comm, next_pid)

        if next_pid in pmap:
            # Corrupted trace probably due to dropped events.  We
            # don't know when the pid in pmap finished.  We just
            # ignore it and don't plot it
            warn_str = "Corrupted trace (dropped events) for PID {} at time {}". \
                       format(next_pid, index)
            warnings.warn(warn_str)
            del pmap[next_pid]
            del data[name][-1]

        if next_pid != 0 and not next_comm.startswith("migration"):

            if execnames and next_comm not in execnames:
                continue

            if pids and next_pid not in pids:
                continue

            data[name].append([index, end_idx, row["__cpu"]])
            pmap[next_pid] = name
            procs.add(name)

    return data, procs, [start_idx, end_idx]
