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

def plot_trace(trace_dir):
    """Creates a kernelshark like plot of the trace file

    :param trace_dir: The location of the trace file
    :type trace_dir: str
    """

    if not IPythonConf.check_ipython():
        raise RuntimeError("plot_trace needs ipython environment")

    run = trappy.Run(trace_dir)
    data, procs, domain = Utils.get_trace_event_data(run)
    trace_graph = EventPlot.EventPlot(data, procs, "CPU: ", int(run._cpus), domain)
    trace_graph.view()
