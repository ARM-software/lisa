from bart.common import Utils
import unittest
import pandas as pd


class TestCommonUtils(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCommonUtils, self).__init__(*args, **kwargs)

    def test_interval_sum(self):
        """Test Utils Function: interval_sum"""

        array = [0, 0, 1, 1, 1, 1, 0, 0]
        series = pd.Series(array)
        self.assertEqual(Utils.interval_sum(series, 1), 3)

        array = [False, False, True, True, True, True, False, False]
        series = pd.Series(array)
        self.assertEqual(Utils.interval_sum(series), 3)

        array = [0, 0, 1, 0, 0, 0]
        series = pd.Series(array)
        self.assertEqual(Utils.interval_sum(series, 1), 0)

        array = [0, 0, 1, 0, 1, 1]
        series = pd.Series(array)
        self.assertEqual(Utils.interval_sum(series, 1), 1)
