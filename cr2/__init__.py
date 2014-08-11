#!/usr/bin/python

from pid_controller import PIDController
from power import OutPower, InPower
from thermal import Thermal, ThermalGovernor
from run import Run
from results import CR2, get_results, combine_results

def summary_plots(actor_order, map_label, **kwords):
    """A summary of plots as similar as possible to what CompareRuns plots

    actor_order must be an array showing the order in which the actors
    where registered.  The array values are the labels that will be
    used in the input and output power plots.  E.g. actor_order can be
    ["GPU", "A15", "A7]

    map_label has to be a dict that matches cpumasks (as found in the
    trace) with their proper name.  This "proper name" will be used as
    a label for the load and allfreqs plots.  It's recommended that
    the names of the cpus matches those in actor_order.  map_label can
    be {"0000000f": "A7", "000000f0": "A15"}

    """
    import plot_utils

    if type(actor_order) is not list:
        raise TypeError("actor_order has to be an array")

    if type(map_label) is not dict:
        raise TypeError("map_label has to be a dict")

    if "path" in kwords:
        path = kwords["path"]
        del kwords["path"]
    else:
        path = None

    run_data = Run(path=path)

    basetime = run_data.thermal.data_frame.index[0]
    run_data.normalize_time(basetime)

    if "width" not in kwords:
        kwords["width"] = 20
    if "height" not in kwords:
        kwords["height"] = 5

    if "title" in kwords:
        title = kwords["title"]
    else:
        title = ""

    plot_temp_kwords = kwords.copy()
    if "title" in plot_temp_kwords:
        del plot_temp_kwords["title"]

    temperature_data = {title: [run_data.thermal, run_data.thermal_governor]}

    plot_utils.plot_temperature(temperature_data, **plot_temp_kwords)
    run_data.in_power.plot_load(map_label, **kwords)
    run_data.plot_allfreqs(map_label, **kwords)
    run_data.pid_controller.plot_controller(**kwords)
    run_data.thermal_governor.plot_input_power(actor_order, **kwords)
    run_data.thermal_governor.plot_output_power(actor_order, **kwords)
    run_data.plot_power_hists(map_label, title)
    plot_utils.plot_temperature_hist(run_data.thermal, title)
