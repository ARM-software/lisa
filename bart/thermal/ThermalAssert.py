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
"""A thermal specific library to assert certain thermal
behaviours
"""

from bart.common import Utils
from bart.common.Analyzer import Analyzer
import numpy as np


# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
class ThermalAssert(object):

    """A class that accepts a TRAPpy Run object and
    provides assertions for thermal behaviours"""

    def __init__(self, run, config=None):

        self._run = Utils.init_run(run)
        self._analyzer = Analyzer(self._run, config)

    def getThermalResidency(self, temp_range, window, percent=False):
        """Returns the total time spent in a given temperature range
        Args:
            temp_range (tuple): A tuple of (low_temp, high_temp)
                which the specifies the range of temperature that
                one intends to calculate the residency for.
            window (tuple): A (start, end) tuple to limit the scope of the
                residency calculation.
            percent: Returns the residency as a percentage of the total
                duration of the trace
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
                    result[pivot] * 100.0) / self._run.get_duration()

        return result

    def assertThermalResidency(
            self,
            expected_value,
            operator,
            temp_range,
            window,
            percent=False):
        """
        Args:
            expected_value (double): The expected value of the residency
            operator (function): A binary operator function that returns
                a boolean
            temp_range (tuple): A tuple of (low_temp, high_temp)
                which the specifies the range of temperature that
                one intends to calculate the residency for.
            window (tuple): A (start, end) tuple to limit the scope of the
                residency calculation.
            percent: Returns the residency as a percentage of the total
                duration of the trace
        """

        residency = self.getThermalResidency(temp_range, window, percent)
        return operator(residency, expected_value)
