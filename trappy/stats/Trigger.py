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

"""Trigger is a representation of the following:

    - Event(s) (:mod:`trappy.base.Base`)
    - An associated value
        - scalar
        - vector
    - A set of filters
        - value based
        - function based
"""

import types
from trappy.utils import listify
import pandas as pd


class Trigger(object):
    """Trigger is an event-value relationship which
    accepts a trace object to "generate" qualified data

    :param trace: A trappy FTrace object
    :type trace: :mod:`trappy.trace.FTrace`

    :param template: A trappy Event to act as a trigger
    :type template: trappy.Base

    :param filters: Key value filter pairs
    :type filters: dict

    The filter can either have a function:
    ::

        def function_based_filter(elem):
            if condition:
                return True
            else:
                return False

    or a value/list of values
    ::

        f = {}
        f["data_column_a"] = function_based_filter
        f["data_column_b"] = value

    function_based_filter is anything that behaves like a function,
    i.e. a callable.

    :param value: Value can be a string or a numeric
    :type value: str, int, float

    :param pivot: This is the column around which the data will be
        pivoted
    :type pivot: str
    """

    def __init__(self, trace, template, filters, value, pivot):

        self.template = template
        self._filters = filters
        self._value = value
        self._pivot = pivot
        self.trace = trace

    def generate(self, pivot_val):
        """Generate the trigger data for a given pivot value
        and a trace index

        :param pivot_val: The pivot to generate data for
        :type pivot_val: hashable
        """

        trappy_event = getattr(self.trace, self.template.name)
        data_frame = trappy_event.data_frame
        data_frame = data_frame[data_frame[self._pivot] == pivot_val]

        mask = [True for _ in range(len(data_frame))]

        for key, value in self._filters.iteritems():
            if hasattr(value, "__call__"):
                mask = mask & (data_frame[key].apply(value))
            else:
                mask = apply_filter_kv(key, value, data_frame, mask)

        data_frame = data_frame[mask]

        if isinstance(self._value, str):
            return data_frame[value]
        else:
            return pd.Series(self._value, index=data_frame.index)


def apply_filter_kv(key, value, data_frame, mask):
    """Internal function to apply a key value
    filter to a data_frame and update the initial
    condition provided in mask.

    :param value: The value to checked for

    :param data_frame: The data to be filtered
    :type data_frame: :mod:`pandas.DataFrame`

    :param mask: Initial Condition Mask
    :type mask: :mod:`pandas.Series`

    :return: A **mask** to index the data frame
    """

    value = listify(value)
    if key not in data_frame.columns:
        return mask
    else:
        for val in value:
            mask = mask & (data_frame[key] == val)
        return mask
