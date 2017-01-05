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


import unittest
from trappy import utils
import pandas
from pandas.util.testing import assert_series_equal


class TestUtils(unittest.TestCase):

    def test_handle_duplicate_index(self):
        """Test Util Function: handle_duplicate_index
        """

        # Refer to the example in the function doc string
        values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 1.0, 6.0, 7.0]
        series = pandas.Series(values, index=index)
        new_index = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0]

        with self.assertRaises(ValueError):
            series.reindex(new_index)

        max_delta = 0.001
        expected_index = [0.0, 1.0, 1 + max_delta, 6.0, 7.0]
        expected_series = pandas.Series(values, index=expected_index)
        series = utils.handle_duplicate_index(series, max_delta)
        assert_series_equal(series, expected_series)

        # Make sure that the reindex doesn't raise ValueError any more
        series.reindex(new_index)

    def test_handle_duplicate_index_duplicate_end(self):
        """handle_duplicate_index copes with duplicates at the end of the series"""

        max_delta = 0.001
        values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 2.0, 6.0, 6.0]
        expected_index = index[:]
        expected_index[-1] += max_delta
        series = pandas.Series(values, index=index)
        expected_series = pandas.Series(values, index=expected_index)

        series = utils.handle_duplicate_index(series, max_delta)
        assert_series_equal(series, expected_series)
