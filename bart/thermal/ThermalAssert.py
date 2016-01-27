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
"""A thermal specific library to assert certain thermal
behaviours
"""

from bart.common import Utils
from bart.common.Analyzer import Analyzer
import numpy as np


# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
class ThermalAssert(object):

    """A class that accepts a TRAPpy FTrace object and
    provides assertions for thermal behaviours

    :param ftrace: A path to the trace file or a TRAPpy FTrace object
    :type ftrace: str, :mod:`trappy.ftrace.FTrace`
    """

    def __init__(self, ftrace, config=None):

        self._ftrace = Utils.init_ftrace(ftrace)
        self._analyzer = Analyzer(self._ftrace, config)

    def getThermalResidency(self, temp_range, window, percent=False):
        """Return the total time spent in a given temperature range

        :param temp_range: A tuple of (low_temp, high_temp)
            which specifies the range of temperature that
            one intends to calculate the residency for.
        :type temp_range: tuple

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param percent: Returns the residency as a percentage of the total
            duration of the trace
        :type percent: bool

        .. seealso:

            :mod:`bart.thermal.ThermalAssert.ThermalAssert.assertThermalResidency`
        """

        # Get a pivoted thermal temperature data using the grammar
        data = self._analyzer.getStatement("trappy.thermal.Thermal:temp")

        result = {}
        for pivot, data_frame in data.groupby(axis=1, level=0):

            series = data_frame[pivot]
            series = Utils.select_window(series, window)
            mask = (series >= temp_range[0]) & (series <= temp_range[1])
            index = series.index.values
            # pylint fails to recognize numpy members.
            # pylint: disable=no-member
            shift_index = np.roll(index, 1)
            # pylint: enable=no-member
            shift_index[0] = 0

            result[pivot] = sum((index - shift_index)[mask.values])

            if percent:
                result[pivot] = (
                    result[pivot] * 100.0) / self._ftrace.get_duration()

        return result

    def assertThermalResidency(
            self,
            expected_value,
            operator,
            temp_range,
            window,
            percent=False):
        """
        :param expected_value: The expected value of the residency
        :type expected_value: double

        :param operator: A binary operator function that returns
            a boolean. For example:
            ::

                import operator
                op = operator.ge
                assertThermalResidency(temp_range, expected_value, op)

            Will do the following check:
            ::

                getThermalResidency(temp_range) >= expected_value

            A custom function can also be passed:
            ::

                THRESHOLD=5
                def between_threshold(a, expected):
                    return abs(a - expected) <= THRESHOLD

        :param temp_range: A tuple of (low_temp, high_temp)
            which specifies the range of temperature that
            one intends to calculate the residency for.
        :type temp_range: tuple

        :param window: A (start, end) tuple to limit the scope of the
            residency calculation.
        :type window: tuple

        :param percent: Returns the residency as a percentage of the total
            duration of the trace
        :type percent: bool

        .. seealso:

            :mod:`bart.thermal.ThermalAssert.ThermalAssert.assertThermalResidency`
        """

        residency = self.getThermalResidency(temp_range, window, percent)
        return operator(residency, expected_value)
