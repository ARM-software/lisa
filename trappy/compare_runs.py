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


import trappy.plot_utils
import trappy.run
import trappy.wa

def compare_runs(actor_order, map_label, runs, **kwords):
    """A side by side comparison of multiple runs

    Plots include temperature, utilization, frequencies, PID
    controller and power.

    :param actor_order: An array showing the order in which the actors
        were registered.  The array values are the labels that
        will be used in the input and output power plots.

        For Example:
        ::

            ["GPU", "A15", "A7"]

    :param map_label: A dict that matches cpumasks (as found in the
        trace) with their proper name.  This "proper name" will be used as
        a label for the load and allfreqs plots.  It's recommended that
        the names of the cpus matches those in actor_order.

        For Example:
        ::

            {"0000000f": "A7", "000000f0": "A15"}

    :param runs: An array of tuples consisting of a name and the path to
        the directory where the trace.dat is.

        For example:
        ::

            [("experiment1", "wa_output/antutu_antutu_1"),
             ("known good", "good/antutu_antutu_1")]

    :type actor_order: list
    :type map_label: dict
    :type runs: list
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
        run_data.append(trappy.Run(name=name, path=path, scope="thermal"))
        trappy.wa.SysfsExtractor(path).pretty_print_in_ipython()

    trappy.plot_utils.plot_temperature(run_data, **kwords)
    trappy.plot_utils.plot_load(run_data, map_label, **kwords)
    trappy.plot_utils.plot_allfreqs(run_data, map_label, **kwords)
    trappy.plot_utils.plot_controller(run_data, **kwords)
    trappy.plot_utils.plot_input_power(run_data, actor_order, **kwords)
    trappy.plot_utils.plot_output_power(run_data, actor_order, **kwords)
    trappy.plot_utils.plot_freq_hists(run_data, map_label)
    trappy.plot_utils.plot_temperature_hist(run_data)

def summary_plots(actor_order, map_label, **kwords):
    """A summary of plots for a given run

    .. warning::

        This is a wrapper around compare_runs().  Use that instead.
    """

    path = kwords.pop("path", ".")
    title = kwords.pop("title", "")

    return compare_runs(actor_order, map_label, [(title, path)], **kwords)
