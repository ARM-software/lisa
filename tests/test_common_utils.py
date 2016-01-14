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

from bart.common import Utils
from bart.common.Analyzer import Analyzer
import unittest
import pandas as pd
import trappy


class TestCommonUtils(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCommonUtils, self).__init__(*args, **kwargs)

    def test_interval_sum(self):
        """Test Utils Function: interval_sum"""

        # A series with a non uniform index
        # Refer to the example illustrations in the
        # the interval sum docs-strings which explains
        # the difference between step-post and ste-pre
        # calculations
        values = [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1]
        index = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
        series = pd.Series(values, index=index)

        self.assertEqual(Utils.interval_sum(series, 1, step="post"), 8)
        self.assertEqual(Utils.interval_sum(series, 1, step="pre"), 7)

        # check left boundary
        array = [1, 1, 0, 0]
        series = pd.Series(array)

        self.assertEqual(Utils.interval_sum(series, 1, step="post"), 2)
        self.assertEqual(Utils.interval_sum(series, 1, step="pre"), 1)

        # check right boundary
        array = [0, 0, 1, 1]
        series = pd.Series(array)

        self.assertEqual(Utils.interval_sum(series, 1, step="post"), 1)
        self.assertEqual(Utils.interval_sum(series, 1, step="pre"), 2)

        array = [False, False, True, True, True, True, False, False]
        series = pd.Series(array)
        self.assertEqual(Utils.interval_sum(series), 4)

    def test_area_under_curve(self):
        """Test Utils function: area_under_curve"""

        array = [0, 0, 2, 2, 2, 1, 1, 1]
        series = pd.Series(array)

        # Area under curve post stepping
        self.assertEqual(
            Utils.area_under_curve(
                series,
                method="rect",
                step="post"),
            8)

        # Area under curve pre stepping
        self.assertEqual(
            Utils.area_under_curve(
                series,
                method="rect",
                step="pre"),
            9)

        array = [1]
        series = pd.Series(array)

        # Area under curve post stepping, edge case
        self.assertEqual(
            Utils.area_under_curve(
                series,
                method="rect",
                step="post"),
            0)

        # Area under curve pre stepping, edge case
        self.assertEqual(
            Utils.area_under_curve(
                series,
                method="rect",
                step="pre"),
            0)


class TestAnalyzer(unittest.TestCase):

    def test_assert_statement_bool(self):
        """Check that asssertStatement() works with a simple boolean case"""

        rolls_dfr = pd.DataFrame({"results": [1, 3, 2, 6, 2, 4]})
        trace = trappy.BareTrace()
        trace.add_parsed_event("dice_rolls", rolls_dfr)
        config = {"MAX_DICE_NUMBER": 6}

        t = Analyzer(trace, config)
        statement = "numpy.max(dice_rolls:results) <= MAX_DICE_NUMBER"
        self.assertTrue(t.assertStatement(statement, select=0))

    def test_assert_statement_dataframe(self):
        """assertStatement() works if the generated statement creates a pandas.DataFrame of bools"""

        rolls_dfr = pd.DataFrame({"results": [1, 3, 2, 6, 2, 4]})
        trace = trappy.BareTrace()
        trace.add_parsed_event("dice_rolls", rolls_dfr)
        config = {"MIN_DICE_NUMBER": 1, "MAX_DICE_NUMBER": 6}
        t = Analyzer(trace, config)

        statement = "(dice_rolls:results <= MAX_DICE_NUMBER) & (dice_rolls:results >= MIN_DICE_NUMBER)"
        self.assertTrue(t.assertStatement(statement))

        statement = "dice_rolls:results == 3"
        self.assertFalse(t.assertStatement(statement))
