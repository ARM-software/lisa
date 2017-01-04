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

"""Generic functions that can be used in multiple places in trappy
"""

def listify(to_select):
    """Utitlity function to handle both single and
    list inputs
    """

    if not isinstance(to_select, list):
        to_select = [to_select]

    return to_select

def handle_duplicate_index(data,
                           max_delta=0.000001):
    """Handle duplicate values in index

    :param data: The timeseries input
    :type data: :mod:`pandas.Series`

    :param max_delta: Maximum interval adjustment value that
        will be added to duplicate indices
    :type max_delta: float

    Consider the following case where a series needs to be reindexed
    to a new index (which can be required when different series need to
    be combined and compared):
    ::

        import pandas
        values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 1.0, 6.0, 7.0]
        series = pandas.Series(values, index=index)
        new_index = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0]
        series.reindex(new_index)

    The above code fails with:
    ::

        ValueError: cannot reindex from a duplicate axis

    The function :func:`handle_duplicate_axis` changes the duplicate values
    to
    ::

        >>> import pandas
        >>> from trappy.utils import handle_duplicate_index

        >>> values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 1.0, 6.0, 7.0]
        series = pandas.Series(values, index=index)
        series = handle_duplicate_index(series)
        print series.index.values
        >>> [ 0.        1.        1.000001  6.        7.      ]

    """

    index = data.index
    new_index = index.values

    dups = index.get_duplicates()

    for dup in dups:
        # Leave one of the values intact
        dup_index_left = index.searchsorted(dup, side="left")
        dup_index_right = index.searchsorted(dup, side="right") - 1
        num_dups = dup_index_right - dup_index_left + 1

        # Calculate delta that needs to be added to each duplicate
        # index
        try:
            delta = (index[dup_index_right + 1] - dup) / num_dups
        except IndexError:
            # dup_index_right + 1 is outside of the series (i.e. the
            # dup is at the end of the series).
            delta = max_delta

        # Clamp the maximum delta added to max_delta
        if delta > max_delta:
            delta = max_delta

        # Add a delta to the others
        dup_index_left += 1
        while dup_index_left <= dup_index_right:
            new_index[dup_index_left] += delta
            delta += delta
            dup_index_left += 1

    return data.reindex(new_index)
