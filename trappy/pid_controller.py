#    Copyright 2015-2017 ARM Limited
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

"""Process the output of the power allocator's PID controller in the
current directory's trace.dat"""

from trappy.base import Base
from trappy.dynamic import register_ftrace_parser

class PIDController(Base):
    """Process the power allocator PID controller data in a FTrace dump"""

    name = "pid_controller"
    """The name of the :mod:`pandas.DataFrame` member that will be created in a
    :mod:`trappy.ftrace.FTrace` object"""

    pivot = "thermal_zone_id"
    """The Pivot along which the data is orthogonal"""

    unique_word="thermal_power_allocator_pid"
    """The event name in the trace"""

    def plot_controller(self, title="", width=None, height=None, ax=None):
        """Plot a summary of the controller data

        :param ax: Axis instance
        :type ax: :mod:`matplotlib.Axis`

        :param title: The title of the plot
        :type title: str

        :param width: The width of the plot
        :type width: int

        :param height: The height of the plot
        :type int: int
        """
        import trappy.plot_utils

        title = trappy.plot_utils.normalize_title("PID", title)

        if not ax:
            ax = trappy.plot_utils.pre_plot_setup(width, height)

        self.data_frame[["output", "p", "i", "d"]].plot(ax=ax)
        trappy.plot_utils.post_plot_setup(ax, title=title)

register_ftrace_parser(PIDController, "thermal")
