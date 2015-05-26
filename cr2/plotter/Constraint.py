# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        Constraint.py
# ----------------------------------------------------------------
# $
#
"""This module provides the Constraint class for handling
filters and pivots in a modular fashion. This enable easy
constrain application

What is a Constraint?
1. It is collection of data based on two rules:
    a. A Pivot
    b. A Set of Filters

For Example:
    for a dataframe

    Time    CPU       Latency
    1       x           <val>
    2       y           <val>
    3       z           <val>
    4       a           <val>

The resultant data will be for each unique pivot value with the filters applied

result["x"] = pd.Series.filtered()
result["y"] = pd.Series.filtered()
result["z"] = pd.Series.filtered()
result["a"] = pd.Series.filtered()

"""
# pylint: disable=R0913
import Utils
import AttrConf


class Constraint(object):

    """The constructor takes a filter and a pivot object,
       The apply method takes a CR2 Run object and a column
       and applies the constraint on input object
    """

    def __init__(
            self, cr2_run, pivot, column, template, run_index, filters):
        self._cr2_run = cr2_run
        self._filters = filters
        self._pivot = pivot
        self._column = column
        self._template = template
        self._dup_resolved = False
        self._data = self.populate_data_frame()

        try:
            self.result = self._apply()
        except ValueError:
            if not self._dup_resolved:
                self._handle_duplicate_index()
                try:
                    self.result = self._apply()
                except:
                    raise ValueError("Unable to handle duplicates")

        self.run_index = run_index

    def _apply(self):
        """This method applies the filter on the resultant data
           on the input column.
           Do we need pivot_val?
        """
        data = self._data
        result = {}

        try:
            values = data[self._column]
        except KeyError:
            return result

        if self._pivot == AttrConf.PIVOT:
            criterion = values.map(lambda x: True)
            for key in self._filters.keys():
                if key in data.columns:
                    criterion = criterion & data[key].map(
                        lambda x: x in self._filters[key])
                    values = values[criterion]
            result[AttrConf.PIVOT_VAL] = values
            return result

        pivot_vals = self.pivot_vals(data)

        for pivot_val in pivot_vals:
            criterion = values.map(lambda x: True)

            for key in self._filters.keys():
                if key != self._pivot and key in data.columns:
                    criterion = criterion & data[key].map(
                        lambda x: x in self._filters[key])
                    values = values[criterion]

            val_series = values[data[self._pivot] == pivot_val]
            if len(val_series) != 0:
               result[pivot_val] = val_series

        return result

    def _handle_duplicate_index(self):
        """Handle duplicate values in index"""
        data = self._data
        self._dup_resolved = True
        index = data.index
        new_index = index.values

        dups = index.get_duplicates()
        for dup in dups:
            # Leave one of the values intact
            dup_index_left = index.searchsorted(dup, side="left")
            dup_index_right = index.searchsorted(dup, side="right") - 1
            num_dups = dup_index_right - dup_index_left + 1
            delta = (index[dup_index_right + 1] - dup) / num_dups

            if delta > AttrConf.DUPLICATE_VALUE_MAX_DELTA:
                delta = AttrConf.DUPLICATE_VALUE_MAX_DELTA

            # Add a delta to the others
            dup_index_left += 1
            while dup_index_left <= dup_index_right:
                new_index[dup_index_left] += delta
                delta += delta
                dup_index_left += 1
        self._data = self._data.reindex(new_index)

    def _uses_cr2_run(self):
        if not self._template:
            return False
        else:
            return True

    def populate_data_frame(self):
        """Return the data frame"""
        if not self._uses_cr2_run():
            return self._cr2_run

        data_container = getattr(
            self._cr2_run,
            Utils.decolonize(
                self._template.name))
        return data_container.data_frame

    def pivot_vals(self, data):
        """This method returns the unique pivot values for the
           Constraint's pivot and the column
        """
        if self._pivot == AttrConf.PIVOT:
            return AttrConf.PIVOT_VAL

        if self._pivot not in data.columns:
            return []

        pivot_vals = set(data[self._pivot])
        if self._pivot in self._filters:
            pivot_vals = pivot_vals & set(self._filters[self._pivot])

        return list(pivot_vals)

    def __repr__(self):
        if not self._uses_cr2_run():
            return "DataFrame " + str(self.run_index) + ":" + self._column

        return "Run " + str(self.run_index) + ":" + \
            self._template.name + ":" + self._column


class ConstraintManager(object):

    """A class responsible for converting inputs
    to constraints and also ensuring sanity
    """

    def __init__(self, runs, columns, templates, pivot, filters):

        self._ip_vec = []
        self._ip_vec.append(Utils.listify(runs))
        self._ip_vec.append(Utils.listify(columns))
        self._ip_vec.append(Utils.listify(templates))

        self._lens = map(len, self._ip_vec)
        self._max_len = max(self._lens)
        self._pivot = pivot
        self._filters = filters
        self.constraints = []

        self._run_expanded = False
        self._expand()
        self._constraints()

    def _expand(self):
        """This is really important. We need to
           meet the following criteria for constraint
           expansion:

           Len[runs] == Len[columns] == Len[templates]
                            OR
           Permute(
               Len[runs] = 1
               Len[columns] = 1
               Len[templates] != 1
            }


           Permute(
               Len[runs] = 1
               Len[columns] != 1
               Len[templates] != 1
            )

        """
        min_len = min(self._lens)
        max_pos_comp = [
            i for i,
            j in enumerate(
                self._lens) if j != self._max_len]

        if self._max_len == 1 and min_len != 1:
            raise RuntimeError("Essential Arg Missing")

        if self._max_len > 1:

            # Are they all equal?
            if len(set(self._lens)) == 1:
                return

            if min_len > 1:
                raise RuntimeError("Cannot Expand a list of Constraints")

            for val in max_pos_comp:
                if val == 0:
                    self._run_expanded = True
                self._ip_vec[val] = Utils.normalize_list(
                    self._max_len,
                    self._ip_vec[val])

    def _constraints(self):
        """Popluate the expanded constraints"""

        for idx in range(self._max_len):
            if self._run_expanded:
                run_idx = 0
            else:
                run_idx = idx

            run = self._ip_vec[0][idx]
            col = self._ip_vec[1][idx]
            template = self._ip_vec[2][idx]
            self.constraints.append(
                Constraint(
                    run,
                    self._pivot,
                    col,
                    template,
                    run_idx,
                    self._filters))

    def get_all_pivots(self):
        """Return a union of the pivot values"""
        pivot_vals = []
        for constraint in self.constraints:
            pivot_vals += constraint.result.keys()

        p_list = list(set(pivot_vals))

        try:
            return sorted(p_list, key=lambda x: int(x))
        except ValueError:
            pass

        try:
            return sorted(p_list, key=lambda x: int(x, 16))
        except ValueError:
            return sorted(p_list)

    def constraint_labels(self):
        """Get the Str representation of the constraints"""
        return map(str, self.constraints)
