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

"""This module provides the Constraint class for handling
filters and pivots in a modular fashion. This enable easy
constraint application.

An implementation of :mod:`trappy.plotter.AbstractDataPlotter`
is expected to use the :mod:`trappy.plotter.Constraint.ConstraintManager`
class to pivot and filter data and handle multiple column,
run and event inputs.

The underlying object that encapsulates a unique set of
a data column, data event and the requisite filters is
:mod:`trappy.plotter.Constraint.Constraint`
"""
# pylint: disable=R0913
from trappy.plotter.Utils import decolonize, listify, normalize_list
from trappy.plotter import AttrConf


class Constraint(object):

    """
    What is a Constraint?
        It is collection of data based on two rules:

        - A Pivot

        - A Set of Filters

        - A Data Column

    For Example a :mod:`pandas.DataFrame`

    =====  ======== =========
    Time    CPU       Latency
    =====  ======== =========
    1       x           <val>
    2       y           <val>
    3       z           <val>
    4       a           <val>
    =====  ======== =========

    The resultant data will be split for each unique pivot value
    with the filters applied
    ::

        result["x"] = pd.Series.filtered()
        result["y"] = pd.Series.filtered()
        result["z"] = pd.Series.filtered()
        result["a"] = pd.Series.filtered()


    :param trappy_run: Input Data
    :type trappy_run: :mod:`pandas.DataFrame`, :mod:`trappy.run.Run`

    :param column: The data column
    :type column: str

    :param template: TRAPpy Event
    :type template: :mod:`trappy.base.Base` event

    :param run_index: The index of the run/data in the overall constraint
        data
    :type run_index: int

    :param filters: A dictionary of filter values
    :type filters: dict
    """

    def __init__(
            self, trappy_run, pivot, column, template, run_index, filters):
        self._trappy_run = trappy_run
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

    def _uses_trappy_run(self):
        if not self._template:
            return False
        else:
            return True

    def populate_data_frame(self):
        """Return the populated :mod:`pandas.DataFrame`"""
        if not self._uses_trappy_run():
            return self._trappy_run

        data_container = getattr(
            self._trappy_run,
            decolonize(self._template.name))
        return data_container.data_frame

    def pivot_vals(self, data):
        """This method returns the unique pivot values for the
        Constraint's pivot and the column

        :param data: Input Data
        :type data: :mod:`pandas.DataFrame`
        """
        if self._pivot == AttrConf.PIVOT:
            return AttrConf.PIVOT_VAL

        if self._pivot not in data.columns:
            return []

        pivot_vals = set(data[self._pivot])
        if self._pivot in self._filters:
            pivot_vals = pivot_vals & set(self._filters[self._pivot])

        return list(pivot_vals)

    def __str__(self):

        name = self.get_data_name()

        if not self._uses_trappy_run():
            return name + ":" + self._column

        return name + ":" + \
            self._template.name + ":" + self._column


    def get_data_name(self):
        """Get name for the data member. This method
        relies on the "name" attribute for the name.
        If the name attribute is absent, it associates
        a numeric name to the respective data element

        :returns: The name of the data member
        """
        if self._uses_trappy_run():
            if self._trappy_run.name != "":
                return self._trappy_run.name
            else:
                return "Run {}".format(self.run_index)
        else:
            return "DataFrame {}".format(self.run_index)

class ConstraintManager(object):

    """A class responsible for converting inputs
    to constraints and also ensuring sanity


    :param runs: Input Run data
    :type runs: :mod:`trappy.run.Run`, list(:mod:`trappy.run.Run`)
    :param columns: The column values from the corresponding
        :mod:`pandas.DataFrame`
    :type columns: str, list(str)
    :param pivot: The column around which the data will be
        pivoted:
    :type pivot: str
    :param filters: A dictionary of values to be applied on the
        respective columns
    :type filters: dict
    :param zip_constraints: Permutes the columns and runs instead
        of a one-to-one correspondence
    :type zip_constraints: bool
    """

    def __init__(self, runs, columns, templates, pivot, filters,
                 zip_constraints=True):

        self._ip_vec = []
        self._ip_vec.append(listify(runs))
        self._ip_vec.append(listify(columns))
        self._ip_vec.append(listify(templates))

        self._lens = map(len, self._ip_vec)
        self._max_len = max(self._lens)
        self._pivot = pivot
        self._filters = filters
        self._constraints = []

        self._run_expanded = False
        self._expand()
        if zip_constraints:
            self._populate_zip_constraints()
        else:
            self._populate_constraints()

    def _expand(self):
        """This is really important. We need to
        meet the following criteria for constraint
        expansion:
        ::

            Len[runs] == Len[columns] == Len[templates]

        Or:
        ::

            Permute(
                Len[runs] = 1
                Len[columns] = 1
                Len[templates] != 1
            )

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
                self._ip_vec[val] = normalize_list(self._max_len,
                                                   self._ip_vec[val])

    def _populate_constraints(self):
        """Populate the constraints creating one for each column in
        each run

        In a multi-run, multicolumn scenario, constraints are created for
        all the columns in each of the runs.  _populate_constraints()
        creates one constraint for the first run and first column, the
        next for the second run and second column,...  This function
        creates a constraint for every combination of runs and columns
        possible.
        """

        for run_idx, run in enumerate(self._ip_vec[0]):
            for col in self._ip_vec[1]:
                template = self._ip_vec[2][run_idx]
                constraint = Constraint(run, self._pivot, col, template,
                                        run_idx, self._filters)
                self._constraints.append(constraint)

    def get_column_index(self, constraint):
        return self._ip_vec[1].index(constraint._column)

    def _populate_zip_constraints(self):
        """Populate the expanded constraints

        In a multirun, multicolumn scenario, create constraints for
        the first run and the first column, second run and second
        column,... that is, as if you run zip(runs, columns)
        """

        for idx in range(self._max_len):
            if self._run_expanded:
                run_idx = 0
            else:
                run_idx = idx

            run = self._ip_vec[0][idx]
            col = self._ip_vec[1][idx]
            template = self._ip_vec[2][idx]
            self._constraints.append(
                Constraint(
                    run,
                    self._pivot,
                    col,
                    template,
                    run_idx,
                    self._filters))

    def generate_pivots(self, permute=False):
        """Return a union of the pivot values

        :param permute: Permute the Runs and Columns
        :type permute: bool
        """
        pivot_vals = []
        for constraint in self._constraints:
            pivot_vals += constraint.result.keys()

        p_list = list(set(pivot_vals))
        runs = range(self._lens[0])

        try:
            sorted_plist = sorted(p_list, key=int)
        except ValueError, TypeError:
            try:
                sorted_plist = sorted(p_list, key=lambda x: int(x, 16))
            except ValueError, TypeError:
                sorted_plist = sorted(p_list)

        if permute:
            pivot_gen = ((run_idx, pivot) for run_idx in runs for pivot in sorted_plist)
            return pivot_gen, len(sorted_plist) * self._lens[0]
        else:
            return sorted_plist, len(sorted_plist)

    def constraint_labels(self):
        """
        :return: string to represent the
            set of Constraints

        """
        return map(str, self._constraints)

    def __len__(self):
        return len(self._constraints)

    def __iter__(self):
        return iter(self._constraints)
