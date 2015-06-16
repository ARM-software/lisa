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
# File:        compare_runs.py
# ----------------------------------------------------------------
# $
#

import cr2.plot_utils
import cr2.run
import cr2.wa

def compare_runs(actor_order, map_label, runs, **kwords):
    """A side by side comparison of multiple runs

    Plots include temperature, utilization, frequencies, PID
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

    if not isinstance(actor_order, list):
        raise TypeError("actor_order has to be an array")

    if not isinstance(map_label, dict):
        raise TypeError("map_label has to be a dict")

    if "width" not in kwords:
        kwords["width"] = 20
    if "height" not in kwords:
        kwords["height"] = 5

    run_data = []
    for name, path in runs:
        run_data.append(cr2.Run(name=name, path=path, scope="thermal"))
        cr2.wa.SysfsExtractor(path).pretty_print_in_ipython()

    cr2.plot_utils.plot_temperature(run_data, **kwords)
    cr2.plot_utils.plot_load(run_data, map_label, **kwords)
    cr2.plot_utils.plot_allfreqs(run_data, map_label, **kwords)
    cr2.plot_utils.plot_controller(run_data, **kwords)
    cr2.plot_utils.plot_input_power(run_data, actor_order, **kwords)
    cr2.plot_utils.plot_output_power(run_data, actor_order, **kwords)
    cr2.plot_utils.plot_freq_hists(run_data, map_label)
    cr2.plot_utils.plot_temperature_hist(run_data)

def summary_plots(actor_order, map_label, **kwords):
    """A summary of plots for a given run

    This is a wrapper around compare_runs().  Use that instead."""

    path = kwords.pop("path", ".")
    title = kwords.pop("title", "")

    return compare_runs(actor_order, map_label, [(title, path)], **kwords)
