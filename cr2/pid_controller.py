#!/usr/bin/python
"""Process the output of the power allocator's PID controller in the
current directory's trace.dat"""

from base import Base
from plot_utils import normalize_title, pre_plot_setup, post_plot_setup

class PIDController(Base):
    """Process the power allocator PID controller data in a ftrace dump"""
    def __init__(self, path=None):
        super(PIDController, self).__init__(
            basepath=path,
            unique_word="thermal_power_allocator_pid",
        )

    def plot_controller(self, title="", width=None, height=None, ax=None):
        """Plot a summary of the controller data"""
        title = normalize_title("PID", title)

        if not ax:
            ax = pre_plot_setup(width, height)

        self.data_frame[["output", "p", "i", "d"]].plot(ax=ax)
        post_plot_setup(ax, title=title)
