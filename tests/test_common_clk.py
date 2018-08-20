#    Copyright 2017 ARM Limited, Google
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

import os
import sys

import utils_tests
import trappy

sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "trappy"))

class TestCommonClk(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestCommonClk, self).__init__(
             [("trace_common_clk.txt", "trace_common_clk.txt"),],
             *args,
             **kwargs)

    def test_common_clk_set_rate_can_be_parsed(self):
        """TestCommonClk: test that clock__set_rate events can be parsed"""
        trace = trappy.FTrace("trace_common_clk.txt", events=['clock_set_rate'])
        df = trace.clock_set_rate.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "cpu_id", "clk_name", "rate"]))

    def test_common_clk_enable_can_be_parsed(self):
        """TestCommonClk: test that clock_enable events can be parsed"""
        trace = trappy.FTrace("trace_common_clk.txt", events=['clock_enable'])
        df = trace.clock_enable.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "cpu_id", "clk_name", "state"]))

    def test_common_clk_disable_can_be_parsed(self):
        """TestCommonClk: test that clock_disable events can be parsed"""
        trace = trappy.FTrace("trace_common_clk.txt", events=['clock_disable'])
        df = trace.clock_disable.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "cpu_id", "clk_name", "state"]))
