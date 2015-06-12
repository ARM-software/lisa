#!/usr/bin/python
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

from run import Run
from plotter.LinePlot import LinePlot
from dynamic import register_dynamic, register_class

# Load all the modules to make sure all classes are registered with Run
import os
for fname in os.listdir(os.path.dirname(__file__)):
    import_name, extension = os.path.splitext(fname)
    if (extension == ".py") and (fname != "__init__.py"):
        __import__("cr2.{}".format(import_name))

del fname, import_name, extension

def summary_plots(actor_order, map_label, **kwords):
    """A summary of plots for a given run

    This is a wrapper around compare_runs().  Use that instead."""

    path = kwords.pop("path", ".")
    title = kwords.pop("title", "")

    return compare_runs(actor_order, map_label, [(title, path)], **kwords)

def compare_runs(actor_order, map_label, runs, **kwords):
    """A side by side comparison of multiple runs

    Plots include temperature, utilisation, frequencies, PID
    controller and power.

    actor_order must be an array showing the order in which the actors
    where registered.  The array values are the labels that will be
    used in the input and output power plots.  E.g. actor_order can be
    ["GPU", "A15", "A7]

    map_label has to be a dict that matches cpumasks (as found in the
    trace) with their proper name.  This "proper name" will be used as
    a label for the load and allfreqs plots.  It's recommended that
    the names of the cpus matches those in actor_order.  map_label can
    be {"0000000f": "A7", "000000f0": "A15"}

    runs is an array of tuples consisting of a name and the path to
    the directory where the trace.dat is.  For example:
    [("experiment1", "wa_output/antutu_antutu_1"),
     ("known good", "good/antutu_antutu_1")]

    """
    import plot_utils

    if type(actor_order) is not list:
        raise TypeError("actor_order has to be an array")

    if type(map_label) is not dict:
        raise TypeError("map_label has to be a dict")

    if "width" not in kwords:
        kwords["width"] = 20
    if "height" not in kwords:
        kwords["height"] = 5

    run_data = []
    for run in runs:
        run_data.append(Run(name=run[0], path=run[1], scope="thermal"))

    plot_utils.plot_temperature(run_data, **kwords)
    plot_utils.plot_load(run_data, map_label, **kwords)
    plot_utils.plot_allfreqs(run_data, map_label, **kwords)
    plot_utils.plot_controller(run_data, **kwords)
    plot_utils.plot_input_power(run_data, actor_order, **kwords)
    plot_utils.plot_output_power(run_data, actor_order, **kwords)
    plot_utils.plot_freq_hists(run_data, map_label)
    plot_utils.plot_temperature_hist(run_data)
