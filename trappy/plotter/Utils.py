#    Copyright 2015-2015 ARM Limited
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


def listify(to_select):
    """Utitlity function to handle both single and
    list inputs
    """

    if not isinstance(to_select, list):
        to_select = [to_select]

    return to_select


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


def get_trace_event_data(run):
    """
        Args:
            trappy.Run: A trappy.Run object

        Returns:
            A list of objects that can be
            consumed by EventPlot to plot task
            residency like kernelshark
    """

    data = collections.defaultdict(list)
    pmap = {}

    data_frame = run.sched_switch.data_frame
    start_idx = data_frame.index.values[0]
    end_idx = data_frame.index.values[-1]

    procs = {}

    for index, row in data_frame.iterrows():
        prev_pid = row["prev_pid"]
        next_pid = row["next_pid"]
        next_comm = row["next_comm"]

        if prev_pid in pmap.keys():
            name = pmap[prev_pid]
            data[name][-1][1] = index
            del pmap[prev_pid]

        if next_pid in pmap.keys():
            raise ValueError("Malformed data for PID: {}".format(next_pid))

        if next_pid != 0 and not next_comm.startswith("migration"):
            name = "{}-{}".format(next_comm, next_pid)
            data[name].append([index, end_idx, row["__cpu"]])
            pmap[next_pid] = name
            procs[name] = 1

    return data, procs.keys(), [start_idx, end_idx]
