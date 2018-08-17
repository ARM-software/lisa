from __future__ import unicode_literals
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
            162534.215764,
            162534.216001,
            162534.216552,
            162534.216568,
            162534.217401,
            162534.217521,
            162534.217655,
            162534.219077,
            162534.219252,
            162534.219268,
            162534.219329,
            162534.219336,
            162534.219587,
            162534.219763,
            162534.219853,
            162534.220947,
            162534.220947
        ], name="Time")

        exp_states = pd.Series([
            2, -1, 2, -1, -1, -1, 2, -1, 2, -1, 0, 0, 2, -1, 2, -1, -1
        ], index=exp_index, name="state")
        exp_cpus = pd.Series([
            5,  2, 2,  1,  3,  0, 0,  0, 0,  0, 1, 3, 0,  0, 0,  3,  1
        ], index=exp_index, name="cpu_id")

        assert_series_equal(df["state"], exp_states, check_exact=True)
        assert_series_equal(df["cpu_id"], exp_cpus, check_exact=True)
