#!/usr/bin/python

from pid_controller import PIDController
from power import OutPower, InPower
from thermal import Thermal, ThermalGovernor
from results import CR2, get_results, combine_results

def summary_plots(**kwords):
    """A summary of plots as similar as possible to what CompareRuns plots"""
    import plot_utils

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

    if "width" not in kwords:
        kwords["width"] = 20
    if "height" not in kwords:
        kwords["height"] = 5

    if "title" in kwords:
        title = kwords["title"]
    else:
        title = ""

    # XXX This needs to be made generic
    map_label = {"0000000f": "A7", "000000f0": "A15"}

    gov_data.plot_temperature(**kwords)
    inpower_data.plot_load(map_label, **kwords)
    plot_utils.plot_allfreqs(inpower_data, outpower_data, map_label, **kwords)
    pid_data.plot_controller(**kwords)
    gov_data.plot_input_power(**kwords)
    gov_data.plot_output_power(**kwords)
    plot_utils.plot_power_hists(inpower_data, outpower_data, map_label, title)
    plot_utils.plot_temperature_hist(thermal_data, title)
