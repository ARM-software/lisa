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


"""Process the output of the devfreq_cooling devices in the current
directory's trace.dat"""

import pandas as pd

from trappy.base import Base
from trappy.dynamic import register_ftrace_parser


class DevfreqInPower(Base):
    """Process de devfreq cooling device data regarding get_power in an
FTrace dump"""

    name = "devfreq_in_power"
    """The name of the :mod:`pandas.DataFrame` member that will be created in a
    :mod:`trappy.ftrace.FTrace` object"""

    unique_word="thermal_power_devfreq_get_power:"
    """The event name in the trace"""

    def get_all_freqs(self):
        """Return a :mod:`pandas.DataFrame` with
        the frequencies for the devfreq device

        The format should be the same as the one for
        :code:`CpuInPower().get_all_freqs()`.

        .. note:: Frequencies are in MHz.
        """

        return pd.DataFrame(self.data_frame["freq"] / 1000000)

register_ftrace_parser(DevfreqInPower, "thermal")


class DevfreqOutPower(Base):
    """Process de devfreq cooling device data regarding power2state in an
ftrace dump"""

    name = "devfreq_out_power"
    """The name of the :mod:`pandas.DataFrame` member that will be created in a
    :mod:`trappy.ftrace.FTrace` object"""

    unique_word="thermal_power_devfreq_limit:"
    """The event name in the trace"""

    def get_all_freqs(self):
        """Return a :mod:`pandas.DataFrame` with
        the output frequencies for the devfreq device

        The format should be the same as the one for
        :code:`CpuOutPower().get_all_freqs()`.

        .. note:: Frequencies are in MHz.
        """

        return pd.DataFrame(self.data_frame["freq"] / 1000000)

register_ftrace_parser(DevfreqOutPower, "thermal")
