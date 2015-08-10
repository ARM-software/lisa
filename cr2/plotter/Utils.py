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
import urllib
import os
import shutil
from cr2.plotter import AttrConf
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


def install_http_resource(url, to_path):
    """Install a HTTP Resource (eg. javascript) to
       a destination on the disk

        Args:
            url (str): HTTP URL
            to_path (str): Destintation path on the disk
    """
    urllib.urlretrieve(url, filename=to_path)


def install_local_resource(from_path, to_path):
    """Move a local resource  to the desired
       a destination.

        Args:
            from_path (str): Path relative to this file
            to_path (str): Destintation path on the disk
    """
    base_dir = os.path.dirname(__file__)
    from_path = os.path.join(base_dir, from_path)
    shutil.copy(from_path, to_path)


def install_resource(from_path, to_path):
    """Install a resource to a location on the disk

        Args:
            from_path (str): URL or relative path
            to_path (str): Destintation path on the disk
    """

    if from_path.startswith("http"):
        if not os.path.isfile(to_path):
            install_http_resource(from_path, to_path)
    else:
        install_local_resource(from_path, to_path)


def iplot_install(module_name):
    """Install the resources for the module to the Ipython
       profile directory

        Args:
            module_name (str): Name of the module

        Returns:
            A list than can be consumed by requirejs or
            any relative resource dependency resolver
    """

    resources = AttrConf.IPLOT_RESOURCES[module_name]
    for resource in resources:
        resource_name = os.path.basename(resource)
        resource_dest_dir = os.path.join(
            AttrConf.PLOTTER_SCRIPTS_PATH,
            module_name)

        # Ensure if the directory exists
        if not os.path.isdir(resource_dest_dir):
            os.mkdir(resource_dest_dir)
        resource_dest_path = os.path.join(resource_dest_dir, resource_name)
        install_resource(resource, resource_dest_path)

def get_trace_event_data(run):
    """
        Args:
            cr2.Run: A cr2.Run object

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
