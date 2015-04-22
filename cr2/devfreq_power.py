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
# File:        devfreq_power.py
# ----------------------------------------------------------------
# $
#

"""Process the output of the devfreq_cooling devices in the current
directory's trace.dat"""

from cr2.base import Base
from cr2.run import Run


class DevfreqInPower(Base):
    """Process de devfreq cooling device data regarding get_power in an
ftrace dump"""

    name = "devfreq_in_power"

    def __init__(self):
        super(DevfreqInPower, self).__init__(
            unique_word="thermal_power_devfreq_get_power:",
        )

    def get_all_freqs(self):
        """Return a pandas.Series with the frequencies for the devfreq device

        The format should be the same as the one for
        CpuInPower().get_all_freqs().  Frequencies are in MHz.

        """

        return self.data_frame["freq"] / 1000000

Run.register_class(DevfreqInPower, "thermal")


class DevfreqOutPower(Base):
    """Process de devfreq cooling device data regarding power2state in an
ftrace dump"""

    name = "devfreq_out_power"

    def __init__(self):
        super(DevfreqOutPower, self).__init__(
            unique_word="thermal_power_devfreq_limit:",
        )

    def get_all_freqs(self):
        """Return a pandas.Series with the output frequencies for the devfreq
device

        The format should be the same that that of
        CpuOutPower().get_all_freqs().  The frequencies are in MHz.

        """

        return self.data_frame["freq"] / 1000000

Run.register_class(DevfreqOutPower, "thermal")
