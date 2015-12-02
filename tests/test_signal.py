#    Copyright 2015-2015 ARM Limited
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


class TestSignalCompare(TestBART):

    def __init__(self, *args, **kwargs):
        super(TestSignalCompare, self).__init__(*args, **kwargs)

    def test_conditional_compare(self):
        """Test conditional_compare"""

        A = [0, 0, 0, 3, 3, 0, 0, 0]
        B = [0, 0, 2, 2, 2, 2, 1, 1]

        run = trappy.Run(".", events=["event"])
        df = pd.DataFrame({"A": A, "B": B})
        run.event.data_frame = df

        s = SignalCompare(run, "event:A", "event:B")
        expected = (1.5, 2.0 / 7)
        self.assertEqual(
            s.conditional_compare(
                "event:A > event:B",
                method="rect"),
            expected)
