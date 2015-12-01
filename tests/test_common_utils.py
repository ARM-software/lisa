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
