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

"""Init Module for the Plotter Code"""


import pandas as pd
import LinePlot
import AttrConf
try:
    import trappy.plotter.EventPlot
except ImportError:
    pass
import Utils
import trappy
import IPythonConf

def register_forwarding_arg(arg_name):
    """Allows the user to register args to
       be forwarded to matplotlib

    :param arg_name: The arg to register
    :type arg_name: str
    """
    if arg_name not in AttrConf.ARGS_TO_FORWARD:
        AttrConf.ARGS_TO_FORWARD.append(arg_name)

def unregister_forwarding_arg(arg_name):
    """Unregisters arg_name from being passed to
       plotter matplotlib calls

    :param arg_name: The arg to register
    :type arg_name: str
    """
    try:
        AttrConf.ARGS_TO_FORWARD.remove(arg_name)
    except ValueError:
        pass

def plot_trace(trace,
               execnames=None,
               pids=None):
    """Creates a kernelshark like plot of the trace file

    :param trace: The path to the trace or a trace object
    :type trace: str, :mod:`trappy.trace.FTrace`, :mod:`trappy.trace.SysTrace`
        or :mod:`trappy.trace.BareTrace`.

    :param execnames: List of execnames to be filtered. If not
        specified all execnames will be plotted
    :type execnames: list, str

    :param pids: List of pids to be filtered. If not specified
        all pids will be plotted
    :type pids: list, str
    """

    if not IPythonConf.check_ipython():
        raise RuntimeError("plot_trace needs ipython environment")

    if not isinstance(trace, trappy.BareTrace):
        if trace.endswith("html"):
            trace = trappy.SysTrace(trace)
        else:
            trace = trappy.FTrace(trace)

    data, procs, domain = Utils.get_trace_event_data(trace, execnames, pids)
    trace_graph = EventPlot.EventPlot(data, procs, domain,
                                      lane_prefix="CPU :",
                                      num_lanes=int(trace._cpus))
    trace_graph.view()
