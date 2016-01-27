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

"""Allow the user to assert various conditions
based on the grammar defined in trappy.stats.grammar. The class is
also intended to have aggregator based functionality. This is not
implemented yet.
"""

from trappy.stats.grammar import Parser
import warnings
import numpy as np
import pandas as pd

# pylint: disable=invalid-name


class Analyzer(object):

    """
    :param data: TRAPpy FTrace Object
    :type data: :mod:`trappy.ftrace.FTrace`

    :param config: A dictionary of variables, classes
        and functions that can be used in the statements
    :type config: dict
    """

    def __init__(self, data, config, **kwargs):
        self._parser = Parser(data, config, **kwargs)

    def assertStatement(self, statement, select=None):
        """Solve the statement for a boolean result

        :param statement: A string representing a valid
            :mod:`trappy.stats.grammar` statement
        :type statement: str

        :param select: If the result represents a boolean
            mask and the data was derived from a TRAPpy event
            with a pivot value. The :code:`select` can be
            used to select a particular pivot value
        :type select: :mod:`pandas.DataFrame` column
        """

        result = self.getStatement(statement, select=select)

        if isinstance(result, pd.DataFrame):
            result = result.all().all()
        elif not(isinstance(result, bool) or isinstance(result, np.bool_)): # pylint: disable=no-member
            warnings.warn("solution of {} is not boolean".format(statement))

        return result

    def getStatement(self, statement, reference=False, select=None):
        """Evaluate the statement"""

        result = self._parser.solve(statement)

        # pylint: disable=no-member
        if np.isscalar(result):
            return result
        # pylint: enable=no-member

        if select is not None and len(result):
            result = result[select]
            if reference:
                result = self._parser.ref(result)

        return result
