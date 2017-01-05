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

import pandas as pd

import trappy
from test_thermal import BaseTestThermal

class TestDevfreqPower(BaseTestThermal):
    """Tests for the DevfreqInPower and DevfreqOutPower classes"""

    def test_devfreq_inp_dataframe(self):
        """Test that DevfreqInPower creates proper data frames"""
        devfreq_in_power = trappy.FTrace().devfreq_in_power

        self.assertTrue("freq" in devfreq_in_power.data_frame.columns)

    def test_devfreq_outp_dataframe(self):
        """Test that DevfreqOutPower creates proper data frames"""
        devfreq_out_power = trappy.FTrace().devfreq_out_power

        self.assertTrue("freq" in devfreq_out_power.data_frame.columns)

    def test_get_inp_all_freqs(self):
        """Test that DevfreqInPower get_all_freqs() work"""

        all_freqs = trappy.FTrace().devfreq_in_power.get_all_freqs()
        self.assertTrue(isinstance(all_freqs, pd.DataFrame))

        self.assertEquals(all_freqs["freq"].iloc[0], 525)

    def test_get_outp_all_freqs(self):
        """Test that DevfreqOutPower get_all_freqs() work"""

        all_freqs = trappy.FTrace().devfreq_out_power.get_all_freqs()
        self.assertTrue(isinstance(all_freqs, pd.DataFrame))

        self.assertEquals(all_freqs["freq"].iloc[0], 525)
