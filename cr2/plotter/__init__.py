# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        __init__.py
# ----------------------------------------------------------------
# $
#
"""Init Module for the Plotter Code"""


import pandas as pd
import LinePlot
import AttrConf
try:
    import cr2.plotter.EventPlot
except ImportError:
    pass
import Utils
import cr2


def register_forwarding_arg(arg_name):
    """Allows the user to register args to
       be forwarded to matplotlib
    """
    if arg_name not in AttrConf.ARGS_TO_FORWARD:
        AttrConf.ARGS_TO_FORWARD.append(arg_name)

def unregister_forwarding_arg(arg_name):
    """Unregisters arg_name from being passed to
       plotter matplotlib calls
    """
    try:
        AttrConf.ARGS_TO_FORWARD.remove(arg_name)
    except ValueError:
        pass

def plot_trace(trace_dir):
    """Creates a kernelshark like plot of the trace file"""

    if not AttrConf.PLOTTER_IPYTHON:
        raise RuntimeError("plot_trace needs ipython environment")

    run = cr2.Run(trace_dir)
    data, procs = Utils.get_trace_event_data(run)
    trace_graph = EventPlot.EventPlot(data, procs, "CPU: ", int(run._cpus))
    trace_graph.view()
