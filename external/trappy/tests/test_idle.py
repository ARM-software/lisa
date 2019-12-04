#    Copyright 2016-2017 ARM Limited
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
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

import pandas as pd
from pandas.util.testing import assert_series_equal

import utils_tests
import trappy

@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestCpuIdle(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestCpuIdle, self).__init__(
            [("trace_idle.dat", "trace.dat")],
            *args,
            **kwargs)

    def test_get_dataframe(self):
        """Test that CpuIdle creates a proper data_frame"""

        df = trappy.FTrace(normalize_time=False).cpu_idle.data_frame

        exp_index = pd.Float64Index([
            162534.2157642,
            162534.21600068,
            162534.216552,
            162534.21656774,
            162534.21740058,
            162534.2175208,
            162534.21765486,
            162534.2190772,
            162534.21925174,
            162534.21926752,
            162534.21932854,
            162534.21933622,
            162534.21958702,
            162534.2197626,
            162534.21985288,
            162534.22094658,
            162534.22094704,
        ], name="Time")

        exp_states = pd.Series([
            2, -1, 2, -1, -1, -1, 2, -1, 2, -1, 0, 0, 2, -1, 2, -1, -1
        ], index=exp_index, name="state")
        exp_cpus = pd.Series([
            5,  2, 2,  1,  3,  0, 0,  0, 0,  0, 1, 3, 0,  0, 0,  3,  1
        ], index=exp_index, name="cpu_id")

        assert_series_equal(df["state"], exp_states, check_exact=True)
        assert_series_equal(df["cpu_id"], exp_cpus, check_exact=True)
