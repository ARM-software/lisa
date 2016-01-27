#    Copyright 2015-2016 ARM Limited
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
from utils_tests import TestBART
from bart.common.signal import SignalCompare
import numpy as np


class TestSignalCompare(TestBART):

    def __init__(self, *args, **kwargs):
        super(TestSignalCompare, self).__init__(*args, **kwargs)

    def test_conditional_compare(self):
        """Test conditional_compare"""

        # Refer to the example in
        # bart.common.signal.SignalCompare.conditional_compare
        # doc-strings which explains the calculation for the
        # data set below
        A = [0, 0, 0, 3, 3, 0, 0, 0]
        B = [0, 0, 2, 2, 2, 2, 1, 1]

        trace = trappy.BareTrace()
        df = pd.DataFrame({"A": A, "B": B})
        trace.add_parsed_event("event", df)

        s = SignalCompare(trace, "event:A", "event:B")
        expected = (1.5, 2.0 / 7)
        self.assertEqual(
            s.conditional_compare(
                "event:A > event:B",
                method="rect"),
            expected)

    def test_get_overshoot(self):
        """Test get_overshoot"""

        A = [0, 0, 0, 3, 3, 0, 0, 0]
        B = [0, 0, 2, 2, 2, 2, 1, 1]

        trace = trappy.BareTrace()
        df = pd.DataFrame({"A": A, "B": B})
        trace.add_parsed_event("event", df)

        s = SignalCompare(trace, "event:A", "event:B")
        expected = (1.5, 2.0 / 7)
        self.assertEqual(
            s.get_overshoot(method="rect"),
            expected)

        A = [0, 0, 0, 1, 1, 0, 0, 0]
        B = [0, 0, 2, 2, 2, 2, 1, 1]

        df = pd.DataFrame({"A": A, "B": B})
        trace.event.data_frame = df
        s = SignalCompare(trace, "event:A", "event:B")

        expected = (float("nan"), 0.0)
        result = s.get_overshoot(method="rect")
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], expected[1])

    def test_get_undershoot(self):
        """Test get_undershoot"""

        A = [0, 0, 0, 1, 1, 1, 1, 1]
        B = [2, 2, 2, 2, 2, 2, 2, 2]

        trace = trappy.BareTrace()
        df = pd.DataFrame({"A": A, "B": B})
        trace.add_parsed_event("event", df)

        s = SignalCompare(trace, "event:A", "event:B")
        expected = (4.0 / 14.0, 1.0)
        self.assertEqual(
            s.get_undershoot(method="rect"),
            expected)

        A = [3, 3, 3, 3, 3, 3, 3, 3]
        B = [2, 2, 2, 2, 2, 2, 1, 1]

        df = pd.DataFrame({"A": A, "B": B})
        trace.event.data_frame = df
        s = SignalCompare(trace, "event:A", "event:B")

        expected = (float("nan"), 0.0)
        result = s.get_undershoot(method="rect")
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], expected[1])
