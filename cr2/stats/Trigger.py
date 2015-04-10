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
# File:        Trigger.py
# ----------------------------------------------------------------
# $
#
"""Trigger is a representation of the following:

    1. Event
    2. An associated value
    3. A set of filters
"""

import types
from cr2.plotter.Utils import listify


class Trigger(object):
    """The tigger is an event relationship which
       accepts a run object to "generate" qualified data

       The filter can either have a function

       def function_based_filter(elem):
            if condition:
                return True
            else:
                return False

      or value

      f = {}
      f["data_column_a"] = function_based_filter
      f["data_column_b"] = value
    """

    def __init__(self, run, template, filters, value, pivot):
        """
            Args:
                run (cr2.Run): A cr2 Run object
                template (cr2.Base): A cr2 Event to act as a trigger
                filters (dict): Key value filter pairs
                value: Value can be a string or a numeric
                pivot: This is the column around which the data will be
                    pivoted
        """

        self.template = template
        self._filters = filters
        self.value = value
        self._pivot = pivot
        self.run = run

    def generate(self, pivot_val):
        """Generate the trigger data for a given pivot value
           and a run index

            Args:
                pivot_val: The pivot to generate data for
        """


        cr2_event = getattr(self.run, self.template.name)
        data_frame = cr2_event.data_frame

        mask = (data_frame[self._pivot] == pivot_val)
        for key in self._filters:

            operator = self._filters[key]

            if isinstance(operator, types.FunctionType):
                mask = mask & (data_frame[key].apply(operator))
            else:
                value = operator
                mask = apply_filter_kv(key, value, data_frame, mask)

        return data_frame[mask]


def apply_filter_kv(key, value, data_frame, mask):
    """Internal function to apply a key value
       filter to a data_frame and update the initial
       condition provided in mask.

       Returns:
           Mask to index the data frame
    """

    value = listify(value)
    if key not in data_frame.columns:
        return mask
    else:
        for val in value:
            mask = mask & (data_frame[key] == val)
        return mask
