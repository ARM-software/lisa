from __future__ import unicode_literals
#    Copyright 2018 Arm Limited
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
            [("trace_idle_unsorted.txt", "trace.txt")],
            *args,
            **kwargs)

    def test_get_dataframe(self):
        """Test that unsorted events are handled correctly"""

        df = trappy.FTrace(normalize_time=False).cpu_idle.data_frame
        self.assertEqual(df.index.size, 8)
