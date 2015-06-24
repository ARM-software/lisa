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
# File:        test_devfreq.py
# ----------------------------------------------------------------
# $
#
import pandas as pd

import cr2
from test_thermal import BaseTestThermal

class TestDevfreqPower(BaseTestThermal):
    """Tests for the DevfreqInPower and DevfreqOutPower classes"""

    def test_devfreq_inp_dataframe(self):
        """Test that DevfreqInPower creates proper data frames"""
        devfreq_in_power = cr2.Run().devfreq_in_power

        self.assertTrue("freq" in devfreq_in_power.data_frame.columns)

    def test_devfreq_outp_dataframe(self):
        """Test that DevfreqOutPower creates proper data frames"""
        devfreq_out_power = cr2.Run().devfreq_out_power

        self.assertTrue("freq" in devfreq_out_power.data_frame.columns)

    def test_get_inp_all_freqs(self):
        """Test that DevfreqInPower get_all_freqs() work"""

        all_freqs = cr2.Run().devfreq_in_power.get_all_freqs()
        self.assertTrue(isinstance(all_freqs, pd.DataFrame))

        self.assertEquals(all_freqs["freq"].iloc[0], 525)

    def test_get_outp_all_freqs(self):
        """Test that DevfreqOutPower get_all_freqs() work"""

        all_freqs = cr2.Run().devfreq_out_power.get_all_freqs()
        self.assertTrue(isinstance(all_freqs, pd.DataFrame))

        self.assertEquals(all_freqs["freq"].iloc[0], 525)
