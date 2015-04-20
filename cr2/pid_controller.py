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
# File:        pid_controller.py
# ----------------------------------------------------------------
# $
#
"""Process the output of the power allocator's PID controller in the
current directory's trace.dat"""

from base import Base
from plot_utils import normalize_title, pre_plot_setup, post_plot_setup

class PIDController(Base):
    """Process the power allocator PID controller data in a ftrace dump"""

    name = "pid_controller"

    def __init__(self):
        super(PIDController, self).__init__(
            unique_word="thermal_power_allocator_pid",
        )

    def plot_controller(self, title="", width=None, height=None, ax=None):
        """Plot a summary of the controller data"""
        title = normalize_title("PID", title)

        if not ax:
            ax = pre_plot_setup(width, height)

        self.data_frame[["output", "p", "i", "d"]].plot(ax=ax)
        post_plot_setup(ax, title=title)
