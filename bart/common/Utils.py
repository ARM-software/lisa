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

"""Utility functions for sheye"""

import trappy

def init_run(trace):
    """Initialize the Run Object

    :param trace: Path for the trace file
        or a trace object
    :type trace: str, :mod:`trappy.run.Run`
    """

    if isinstance(trace, basestring):
        return trappy.Run(trace)

    elif isinstance(trace, trappy.Run):
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
