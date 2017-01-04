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


import pandas as pd
import unittest

from trappy.plotter import AttrConf
from trappy.plotter.Constraint import Constraint, ConstraintManager

class TestConstraintManager(unittest.TestCase):
    """Test trappy.plotter.ConstraintManager"""

    def __init__(self, *args, **kwargs):
        """Init some common data for the tests"""

        self.dfrs = [pd.DataFrame({"load": [1, 2, 2, 3],
                                   "freq": [2, 3, 3, 4],
                                   "cpu": [0, 1, 0, 1]}),
                     pd.DataFrame({"load": [2, 3, 2, 1],
                                   "freq": [1, 2, 2, 1],
                                   "cpu": [1, 0, 1, 0]})]
        self.cols = ["load", "freq"]
        super(TestConstraintManager, self).__init__(*args, **kwargs)

    def test_one_constraint(self):
        """Test that the constraint manager works with one constraint"""

        dfr = self.dfrs[0]

        c_mgr = ConstraintManager(dfr, "load", None, AttrConf.PIVOT, {})

        self.assertEquals(len(c_mgr), 1)

        constraint = iter(c_mgr).next()
        series = constraint.result[AttrConf.PIVOT_VAL]
        self.assertEquals(series.to_dict().values(),
                          dfr["load"].to_dict().values())

    def test_no_pivot_multiple_traces(self):
        """Test that the constraint manager works with multiple traces and no pivots"""

        c_mgr = ConstraintManager(self.dfrs, "load", None, AttrConf.PIVOT, {})

        self.assertEquals(len(c_mgr), 2)

        for constraint, orig_dfr in zip(c_mgr, self.dfrs):
            series = constraint.result[AttrConf.PIVOT_VAL]
            self.assertEquals(series.to_dict().values(),
                              orig_dfr["load"].to_dict().values())

    def test_no_pivot_zipped_columns_and_traces(self):
        """Test the constraint manager with multiple columns and traces zipped"""

        c_mgr = ConstraintManager(self.dfrs, self.cols, None, AttrConf.PIVOT, {})

        self.assertEquals(len(c_mgr), 2)

        for constraint, orig_dfr, col in zip(c_mgr, self.dfrs, self.cols):
            series = constraint.result[AttrConf.PIVOT_VAL]
            self.assertEquals(series.to_dict().values(),
                              orig_dfr[col].to_dict().values())

    def test_no_pivot_multicolumns_multitraces(self):
        """Test the constraint manager with multiple traces that can have each multiple columns"""

        c_mgr = ConstraintManager(self.dfrs, self.cols, None, AttrConf.PIVOT,
                                  {}, zip_constraints=False)

        self.assertEquals(len(c_mgr), 4)

        expected_series = [dfr[col] for dfr in self.dfrs for col in self.cols]
        for constraint, orig_series in zip(c_mgr, expected_series):
            series = constraint.result[AttrConf.PIVOT_VAL]
            self.assertEquals(series.to_dict(), orig_series.to_dict())

    def test_no_pivot_filters(self):
        """Test the constraint manager with filters"""

        simple_filter = {"freq": [2]}

        c_mgr = ConstraintManager(self.dfrs, "load", None, AttrConf.PIVOT,
                                  simple_filter)

        num_constraints = len(c_mgr)
        self.assertEquals(num_constraints, 2)

        constraint_iter = iter(c_mgr)
        constraint = constraint_iter.next()
        self.assertEquals(len(constraint.result), 1)

        constraint = constraint_iter.next()
        series_second_frame = constraint.result[AttrConf.PIVOT_VAL]
        self.assertEquals(series_second_frame.to_dict().values(), [3, 2])

    def test_pivoted_data(self):
        """Test the constraint manager with a pivot and one trace"""

        c_mgr = ConstraintManager(self.dfrs[0], "load", None, "cpu", {})

        self.assertEquals(len(c_mgr), 1)

        constraint = iter(c_mgr).next()
        results = dict([(k, v.to_dict().values()) for k, v in constraint.result.items()])
        expected_results = {0: [1, 2], 1: [2, 3]}

        self.assertEquals(results, expected_results)

    def test_pivoted_multitrace(self):
        """Test the constraint manager with a pivot and multiple traces"""

        c_mgr = ConstraintManager(self.dfrs, "load", None, "cpu", {})

        self.assertEquals(len(c_mgr), 2)

        constraint_iter = iter(c_mgr)
        constraint = constraint_iter.next()
        self.assertEquals(constraint.result[0].to_dict().values(), [1, 2])

        constraint = constraint_iter.next()
        self.assertEquals(constraint.result[1].to_dict().values(), [2, 2])

    def test_pivoted_multitraces_multicolumns(self):
        """Test the constraint manager with multiple traces and columns"""

        c_mgr = ConstraintManager(self.dfrs, ["load", "freq"], None, "cpu", {})
        self.assertEquals(len(c_mgr), 2)

        constraint_iter = iter(c_mgr)
        constraint = constraint_iter.next()
        self.assertEquals(constraint.result[1].to_dict().values(), [2, 3])

        constraint = constraint_iter.next()
        self.assertEquals(constraint.result[0].to_dict().values(), [2, 1])

    def test_pivoted_with_filters(self):
        """Test the constraint manager with pivoted data and filters"""

        simple_filter = {"load": [2]}
        c_mgr = ConstraintManager(self.dfrs[0], "freq", None, "cpu",
                                  simple_filter)

        self.assertEquals(len(c_mgr), 1)

        constraint = iter(c_mgr).next()
        result = constraint.result

        self.assertEquals(result[0].iloc[0], 3)
        self.assertEquals(result[1].iloc[0], 3)

    def test_constraint_with_window(self):
        """Test that the constraint manager can constraint to a window of time"""
        c_mgr = ConstraintManager(self.dfrs[0], "freq", None, AttrConf.PIVOT, {},
                                  window=(1, 3))

        constraint = iter(c_mgr).next()
        series = constraint.result[AttrConf.PIVOT_VAL]
        self.assertEquals(len(series), 3)

        # For the graph to plot a value at 0.75, the resulting series
        # must contain the value before 0.75.  Same for the upper limit.
        c_mgr = ConstraintManager(self.dfrs[0], "freq", None, AttrConf.PIVOT, {},
                                  window=(0.75, 1.5))

        constraint = iter(c_mgr).next()
        series = constraint.result[AttrConf.PIVOT_VAL]
        self.assertEquals(series.index.tolist(), [0, 1, 2])

        c_mgr = ConstraintManager(self.dfrs[0], "freq", None, AttrConf.PIVOT, {},
                                  window=(0, 2))

        constraint = iter(c_mgr).next()
        series = constraint.result[AttrConf.PIVOT_VAL]
        self.assertEquals(len(series), 3)

class TestConstraint(unittest.TestCase):
    def test_str_constraint(self):
        """str(constraint) doesn't fail when the column is not a string"""
        dfr = pd.DataFrame({12: [1, 2, 3], 13: [3, 4, 5]})

        constraint = Constraint(dfr, AttrConf.PIVOT, 12, template=None,
                                trace_index=0, filters={}, window=None)

        self.assertEqual(str(constraint), "DataFrame 0:12")
