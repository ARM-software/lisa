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
# File:        thermal.py
# ----------------------------------------------------------------
# $
#
"""Process the output of the power allocator trace in the current
directory's trace.dat"""

from collections import OrderedDict
import pandas as pd
import re
from matplotlib import pyplot as plt

from cr2.base import Base
from cr2.run import Run
from cr2.plot_utils import normalize_title, pre_plot_setup, post_plot_setup, plot_hist

class Thermal(Base):
    """Process the thermal framework data in a ftrace dump"""

    unique_word = "thermal_temperature:"
    name = "thermal"

    def __init__(self):
        super(Thermal, self).__init__(unique_word=self.unique_word)

    def plot_temperature(self, control_temperature=None, title="", width=None,
                         height=None, ylim="range", ax=None, legend_label=""):
        """Plot the temperature.

        If control_temp is a pd.Series() representing the (possible)
        variation of control_temp during the run, draw it using a
        dashed yellow line.  Otherwise, only the temperature is
        plotted.

        """
        title = normalize_title("Temperature", title)

        if len(self.data_frame) == 0:
            raise ValueError("Empty DataFrame")

        setup_plot = False
        if not ax:
            ax = pre_plot_setup(width, height)
            setup_plot = True

        temp_label = normalize_title("Temperature", legend_label)
        (self.data_frame["temp"] / 1000).plot(ax=ax, label=temp_label)
        if control_temperature is not None:
            ct_label = normalize_title("Control", legend_label)
            control_temperature.plot(ax=ax, color="y", linestyle="--",
                           label=ct_label)

        if setup_plot:
            post_plot_setup(ax, title=title, ylim=ylim)
            plt.legend()

    def plot_temperature_hist(self, ax, title):
        """Plot a temperature histogram"""

        temps = self.data_frame["temp"] / 1000
        title = normalize_title("Temperature", title)
        xlim = (0, temps.max())

        plot_hist(temps, ax, title, "C", 30, "Temperature", xlim, "default")

Run.register_class(Thermal, "thermal")

class ThermalGovernor(Base):
    """Process the power allocator data in a ftrace dump"""

    unique_word = "thermal_power_allocator:"
    name = "thermal_governor"
    def __init__(self):
        super(ThermalGovernor, self).__init__(
            unique_word=self.unique_word,
        )

    def plot_temperature(self, title="", width=None, height=None, ylim="range",
                         ax=None, legend_label=""):
        """Plot the temperature"""
        dfr = self.data_frame
        curr_temp = dfr["current_temperature"]
        control_temp_series = (curr_temp + dfr["delta_temperature"]) / 1000
        title = normalize_title("Temperature", title)

        setup_plot = False
        if not ax:
            ax = pre_plot_setup(width, height)
            setup_plot = True

        temp_label = normalize_title("Temperature", legend_label)
        (curr_temp / 1000).plot(ax=ax, label=temp_label)
        control_temp_series.plot(ax=ax, color="y", linestyle="--",
                                 label="control temperature")

        if setup_plot:
            post_plot_setup(ax, title=title, ylim=ylim)
            plt.legend()

    def plot_input_power(self, actor_order, title="", width=None, height=None,
                         ax=None):
        """Plot input power

        actor_order is an array with the order in which the actors
        were registered.

        """

        dfr = self.data_frame
        in_cols = [s for s in dfr.columns if re.match("req_power[0-9]+", s)]

        plot_dfr = dfr[in_cols]
        # Rename the columns from "req_power0" to "A15" or whatever is
        # in actor_order.  Note that we can do it just with an
        # assignment because the columns are already sorted (i.e.:
        # req_power0, req_power1...)
        plot_dfr.columns = actor_order

        title = normalize_title("Input Power", title)

        if not ax:
            ax = pre_plot_setup(width, height)

        plot_dfr.plot(ax=ax)
        post_plot_setup(ax, title=title)

    def plot_weighted_input_power(self, actor_weights, title="", width=None,
                                  height=None, ax=None):
        """Plot weighted input power

        actor_weights is an array of tuples.  First element of the
        tuple is the name of the actor, the second is the weight.  The
        array is in the same order as the req_power appear in the
        trace.

        """

        dfr = self.data_frame
        in_cols = [s for s in dfr.columns if re.match(r"req_power\d+", s)]

        plot_dfr_dict = OrderedDict()
        for in_col, (name, weight) in zip(in_cols, actor_weights):
            plot_dfr_dict[name] = dfr[in_col] * weight / 1024

        plot_dfr = pd.DataFrame(plot_dfr_dict)

        title = normalize_title("Weighted Input Power", title)

        if not ax:
            ax = pre_plot_setup(width, height)

        plot_dfr.plot(ax=ax)
        post_plot_setup(ax, title=title)

    def plot_output_power(self, actor_order, title="", width=None, height=None,
                          ax=None):
        """Plot output power

        actor_order is an array with the order in which the actors
        were registered.

        """

        out_cols = [s for s in self.data_frame.columns
                    if re.match("granted_power[0-9]+", s)]

        # See the note in plot_input_power()
        plot_dfr = self.data_frame[out_cols]
        plot_dfr.columns = actor_order

        title = normalize_title("Output Power", title)

        if not ax:
            ax = pre_plot_setup(width, height)

        plot_dfr.plot(ax=ax)
        post_plot_setup(ax, title=title)

    def plot_inout_power(self, title=""):
        """Make multiple plots showing input and output power for each actor"""
        dfr = self.data_frame

        actors = []
        for col in dfr.columns:
            match = re.match("P(.*)_in", col)
            if match and col != "Ptot_in":
                actors.append(match.group(1))

        for actor in actors:
            cols = ["P" + actor + "_in", "P" + actor + "_out"]
            this_title = normalize_title(actor, title)
            dfr[cols].plot(title=this_title)

Run.register_class(ThermalGovernor, "thermal")
