#!/usr/bin/python

from pid_controller import PIDController
from power import OutPower, InPower
from thermal import Thermal, ThermalGovernor
from run import Run
from results import CR2, get_results, combine_results

def summary_plots(actor_order, map_label, **kwords):
    """A summary of plots for a given run

    This is a wrapper around compare_runs().  Use that instead."""

    if "path" in kwords:
        path = kwords["path"]
        del kwords["path"]
    else:
        path = None

    if "title" in kwords:
        title = kwords["title"]
        del kwords["title"]
    else:
        title = ""

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
        this_run = Run(name=run[0], path=run[1])
        basetime = this_run.thermal.data_frame.index[0]
        this_run.normalize_time(basetime)

        run_data.append(this_run)

    plot_utils.plot_temperature(run_data, **kwords)
    plot_utils.plot_load(run_data, map_label, **kwords)
    plot_utils.plot_allfreqs(run_data, map_label, **kwords)
    plot_utils.plot_controller(run_data, **kwords)
    plot_utils.plot_input_power(run_data, actor_order, **kwords)
    plot_utils.plot_output_power(run_data, actor_order, **kwords)
    plot_utils.plot_freq_hists(run_data, map_label)
    plot_utils.plot_temperature_hist(run_data)
