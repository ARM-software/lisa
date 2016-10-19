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

"""Utility functions for sheye"""

import trappy
import numpy as np

# pylint fails to recognize numpy members.
# pylint: disable=no-member

def listify(to_select):
    """Utitlity function to handle both single and
    list inputs
    """

    if not isinstance(to_select, list):
        to_select = [to_select]

    return to_select

def init_ftrace(trace):
    """Initialize the FTrace Object

    :param trace: Path for the trace file
        or a trace object
    :type trace: str, :mod:`trappy.ftrace.FTrace`
    """

    if isinstance(trace, basestring):
        return trappy.FTrace(trace)

    elif isinstance(trace, trappy.BareTrace):
        return trace

    raise ValueError("Invalid trace Object")

def select_window(series, window):
    """Helper Function to select a portion of
    pandas time series

    :param series: Input Time Series data
    :type series: :mod:`pandas.Series`

    :param window: A tuple indicating a time window
    :type window: tuple
    """

    if not window:
        return series

    start, stop = window
    ix = series.index
    selector = ((ix >= start) & (ix <= stop))
    window_series = series[selector]
    return window_series

def area_under_curve(series, sign=None, method="trapz", step="post"):
    """Return the area under the time series curve (Integral)

    :param series: The time series to be integrated
    :type series: :mod:`pandas.Series`

    :param sign: Clip the data for the area in positive
        or negative regions. Can have two values

        - `"+"`
        - `"="`
    :type sign: str

    :param method: The method for area calculation. This can
        be any of the integration methods supported in `numpy`
        or `rect`
    :type param: str

    :param step: The step behaviour for `rect` method
    :type step: str

    *Rectangular Method*

        - Step: Post

            Consider the following time series data

            .. code::

                2            *----*----*----+
                             |              |
                1            |              *----*----+
                             |
                0  *----*----+
                   0    1    2    3    4    5    6    7

            .. code::

                import pandas as pd
                a = [0, 0, 2, 2, 2, 1, 1]
                s = pd.Series(a)

            The area under the curve is:

            .. math::

                \sum_{k=0}^{N-1} (x_{k+1} - {x_k}) \\times f(x_k) \\\\
                (2 \\times 3) + (1 \\times 2) = 8

        - Step: Pre

            .. code::

                2       +----*----*----*
                        |              |
                1       |              +----*----*----+
                        |
                0  *----*
                   0    1    2    3    4    5    6    7

            .. code::

                import pandas as pd
                a = [0, 0, 2, 2, 2, 1, 1]
                s = pd.Series(a)

            The area under the curve is:

            .. math::

                \sum_{k=1}^{N} (x_k - x_{k-1}) \\times f(x_k) \\\\
                (2 \\times 3) + (1 \\times 3) = 9
    """

    if sign == "+":
        series = series.clip_lower(0)
    elif sign == "=":
        series = series.clip_upper(0)

    series = series.dropna()

    if method == "rect":

        if step == "post":
            values = series.values[:-1]
        elif step == "pre":
            values = series.values[1:]
        else:
            raise ValueError("Invalid Value for step: {}".format(step))

        return float((values * np.diff(series.index)).sum())

    if hasattr(np, method):
        np_integ_method = getattr(np, method)
        return np_integ_method(series.values, series.index)
    else:
        raise ValueError("Invalid method: {}".format(method))

def interval_sum(series, value=None, step="post"):
    """A function that returns the sum of the
    intervals where the value of series is equal to
    the expected value. Consider the following time
    series data:

    ====== =======
     Time   Value
    ====== =======
      0      0
      1      0
      2      1
      3      1
      4      1
      5      1
      8      0
      9      1
     10      0
     11      1
     12      1
    ====== =======

    .. note::

        The time/index values, in general, may not be
        uniform. This causes difference in the
        the values of :func:`interval_sum` for **step-pre**
        and **step-post** behaviours

    .. code::

        import pandas

        values = [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1]
        index = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]
        series = pandas.Series(values, index=index)

    The :func:`interval_sum` for the value 1 is calculated differently
    for **step-post** and **step-pre** behaviours as follows:

        - **Step-Post**


          .. code::

            1            *----*----*----*-------------+    *----+    *----*
                         |                            |    |    |    |
            0  *----*----+                            *----+    *----+
               0    1    2    3    4    5    6    7   8    9    10   11   12

          .. math::

            (8-2) + (10-9) + (12-11) = 6 + 1 + 1 = 8

        - **Step-Pre**

          .. code::

            1       +----*----*----*----*              +----*    +----*----*
                    |                   |              |    |    |
            0  *----*                   +--------------*    +----*
               0    1    2    3    4    5    6    7    8    9    10   11   12

          .. math::

                (5-1) + (9-8) + (12-10) = 4 + 1 + 2 = 7

        .. note::

            The asterisks (*) on the plots above represent the values of the time
            series data and these do not vary between the two step styles

    :param series: The time series data
    :type series: :mod:`pandas.Series`

    :param value: The value to checked for in the series. If the
        value is None, the truth value of the elements in the
        series will be used
    :type value: element

    :param step: The step behaviour as described above
        ::

            step="post"
            step="pre
    :type step: str
    """

    index = series.index
    array = series.values

    time_splits = np.append(np.where(np.diff(array) != 0), len(array) - 1)

    prev = 0
    time = 0
    step_post = True

    if step == "pre":
        step_post = False
    elif step != "post":
        raise ValueError("Invalid value for step: {}".format(step))

    for split in time_splits:

        first_val = series.iloc[split]
        check = (first_val == value) if value else first_val
        if check:
            start = prev
            end = split

            if step_post:
                end = split + 1 if split < len(series) - 1 else split
            else:
                start = prev - 1 if prev > 1 else prev

            time += index[end] - index[start]

        prev = split + 1

    return float(time)
