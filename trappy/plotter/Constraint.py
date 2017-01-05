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

"""This module provides the Constraint class for handling
filters and pivots in a modular fashion. This enable easy
constraint application.

An implementation of :mod:`trappy.plotter.AbstractDataPlotter`
is expected to use the :mod:`trappy.plotter.Constraint.ConstraintManager`
class to pivot and filter data and handle multiple column,
trace and event inputs.

The underlying object that encapsulates a unique set of
a data column, data event and the requisite filters is
:mod:`trappy.plotter.Constraint.Constraint`
"""
# pylint: disable=R0913
from trappy.plotter.Utils import decolonize, normalize_list
from trappy.utils import listify
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


    :param trappy_trace: Input Data
    :type trappy_trace: :mod:`pandas.DataFrame` or a class derived from
        :mod:`trappy.trace.BareTrace`

    :param column: The data column
    :type column: str

    :param template: TRAPpy Event
    :type template: :mod:`trappy.base.Base` event

    :param trace_index: The index of the trace/data in the overall constraint
        data
    :type trace_index: int

    :param filters: A dictionary of filter values
    :type filters: dict

    :param window: A time window to apply to the constraint.
    E.g. window=(5, 20) will constraint to events that happened
    between Time=5 to Time=20.
    :type window: tuple of two ints

    """

    def __init__(self, trappy_trace, pivot, column, template, trace_index,
                 filters, window):
        self._trappy_trace = trappy_trace
        self._filters = filters
        self._pivot = pivot
        self.column = column
        self._template = template
        self._dup_resolved = False
        self._data = self.populate_data_frame()

        if window:
            # We want to include the previous value before the window
            # and the next after the window in the dataset
            min_idx = self._data.loc[:window[0]].index.max()
            max_idx = self._data.loc[window[1]:].index.min()
            self._data = self._data.loc[min_idx:max_idx]

        self.result = self._apply()
        self.trace_index = trace_index

    def _apply(self):
        """This method applies the filter on the resultant data
        on the input column.
        """
        data = self._data
        result = {}

        try:
            values = data[self.column]
        except KeyError:
            return result

        if self._pivot == AttrConf.PIVOT:
            pivot_vals = [AttrConf.PIVOT_VAL]
        else:
            pivot_vals = self.pivot_vals(data)

        for pivot_val in pivot_vals:
            criterion = values.map(lambda x: True)

            for key in self._filters.keys():
                if key != self._pivot and key in data.columns:
                    criterion = criterion & data[key].map(
                        lambda x: x in self._filters[key])

            if pivot_val != AttrConf.PIVOT_VAL:
                criterion &= data[self._pivot] == pivot_val

            val_series = values[criterion]
            if len(val_series) != 0:
                result[pivot_val] = val_series

        return result

    def _uses_trappy_trace(self):
        if not self._template:
            return False
        else:
            return True

    def populate_data_frame(self):
        """Return the populated :mod:`pandas.DataFrame`"""
        if not self._uses_trappy_trace():
            return self._trappy_trace

        data_container = getattr(
            self._trappy_trace,
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

        if not self._uses_trappy_trace():
            return name + ":" + str(self.column)

        return name + ":" + \
            self._template.name + ":" + self.column


    def get_data_name(self):
        """Get name for the data member. This method
        relies on the "name" attribute for the name.
        If the name attribute is absent, it associates
        a numeric name to the respective data element

        :returns: The name of the data member
        """
        if self._uses_trappy_trace():
            if self._trappy_trace.name != "":
                return self._trappy_trace.name
            else:
                return "Trace {}".format(self.trace_index)
        else:
            return "DataFrame {}".format(self.trace_index)

class ConstraintManager(object):

    """A class responsible for converting inputs
    to constraints and also ensuring sanity


    :param traces: Input Trace data
    :type traces: :mod:`trappy.trace.BareTrace`, list(:mod:`trappy.trace.BareTrace`)
        (or a class derived from :mod:`trappy.trace.BareTrace`)
    :param columns: The column values from the corresponding
        :mod:`pandas.DataFrame`
    :type columns: str, list(str)
    :param pivot: The column around which the data will be
        pivoted:
    :type pivot: str
    :param templates: TRAPpy events
    :type templates: :mod:`trappy.base.Base`
    :param filters: A dictionary of values to be applied on the
        respective columns
    :type filters: dict
    :param window: A time window to apply to the constraints
    :type window: tuple of ints
    :param zip_constraints: Permutes the columns and traces instead
        of a one-to-one correspondence
    :type zip_constraints: bool
    """

    def __init__(self, traces, columns, templates, pivot, filters,
                 window=None, zip_constraints=True):

        self._ip_vec = []
        self._ip_vec.append(listify(traces))
        self._ip_vec.append(listify(columns))
        self._ip_vec.append(listify(templates))

        self._lens = map(len, self._ip_vec)
        self._max_len = max(self._lens)
        self._pivot = pivot
        self._filters = filters
        self.window = window
        self._constraints = []

        self._trace_expanded = False
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

            Len[traces] == Len[columns] == Len[templates]

        Or:
        ::

            Permute(
                Len[traces] = 1
                Len[columns] = 1
                Len[templates] != 1
            )

            Permute(
                   Len[traces] = 1
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
                    self._trace_expanded = True
                self._ip_vec[val] = normalize_list(self._max_len,
                                                   self._ip_vec[val])

    def _populate_constraints(self):
        """Populate the constraints creating one for each column in
        each trace

        In a multi-trace, multicolumn scenario, constraints are created for
        all the columns in each of the traces.  _populate_constraints()
        creates one constraint for the first trace and first column, the
        next for the second trace and second column,...  This function
        creates a constraint for every combination of traces and columns
        possible.
        """

        for trace_idx, trace in enumerate(self._ip_vec[0]):
            for col in self._ip_vec[1]:
                template = self._ip_vec[2][trace_idx]
                constraint = Constraint(trace, self._pivot, col, template,
                                        trace_idx, self._filters, self.window)
                self._constraints.append(constraint)

    def get_column_index(self, constraint):
        return self._ip_vec[1].index(constraint.column)

    def _populate_zip_constraints(self):
        """Populate the expanded constraints

        In a multitrace, multicolumn scenario, create constraints for
        the first trace and the first column, second trace and second
        column,... that is, as if you run zip(traces, columns)
        """

        for idx in range(self._max_len):
            if self._trace_expanded:
                trace_idx = 0
            else:
                trace_idx = idx

            trace = self._ip_vec[0][idx]
            col = self._ip_vec[1][idx]
            template = self._ip_vec[2][idx]
            self._constraints.append(
                Constraint(trace, self._pivot, col, template, trace_idx,
                           self._filters, self.window))

    def generate_pivots(self, permute=False):
        """Return a union of the pivot values

        :param permute: Permute the Traces and Columns
        :type permute: bool
        """
        pivot_vals = []
        for constraint in self._constraints:
            pivot_vals += constraint.result.keys()

        p_list = list(set(pivot_vals))
        traces = range(self._lens[0])

        try:
            sorted_plist = sorted(p_list, key=int)
        except (ValueError, TypeError):
            try:
                sorted_plist = sorted(p_list, key=lambda x: int(x, 16))
            except (ValueError, TypeError):
                sorted_plist = sorted(p_list)

        if permute:
            pivot_gen = ((trace_idx, pivot) for trace_idx in traces for pivot in sorted_plist)
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
