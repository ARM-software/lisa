#!/usr/bin/python

from pid_controller import PIDController
from power import OutPower, InPower
from thermal import Thermal, ThermalGovernor
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

    thermal_data = Thermal(path=path)
    gov_data = ThermalGovernor(path=path)
    inpower_data = InPower(path=path)
    outpower_data = OutPower(path=path)
    pid_data = PIDController(path=path)

    basetime = thermal_data.data_frame.index[0]
    for data_class in [thermal_data, gov_data, inpower_data, outpower_data,
                       pid_data]:
        data_class.normalize_time(basetime)

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

    temperature_data = {title: [thermal_data, gov_data]}

    plot_utils.plot_temperature(temperature_data, **plot_temp_kwords)
    inpower_data.plot_load(map_label, **kwords)
    plot_utils.plot_allfreqs(inpower_data, outpower_data, map_label, **kwords)
    pid_data.plot_controller(**kwords)
    gov_data.plot_input_power(actor_order, **kwords)
    gov_data.plot_output_power(actor_order, **kwords)
    plot_utils.plot_power_hists(inpower_data, outpower_data, map_label, title)
    plot_utils.plot_temperature_hist(thermal_data, title)
