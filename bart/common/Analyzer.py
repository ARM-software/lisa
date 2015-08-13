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

"""Allow the user to assert various conditions
based on the grammar defined in trappy.stats.grammar. The class is
also intended to have aggregator based functionality. This is not
implemented yet.
"""

from trappy.stats.grammar import Parser
import warnings
import numpy as np

# pylint: disable=invalid-name


class Analyzer(object):

    """
        Args:
            data (trappy.Run): A trappy.Run instance
            config (dict): A dictionary of variables, classes
                and functions that can be used in the statements
    """

    def __init__(self, data, config, topology=None):
        self._parser = Parser(data, config, topology)

    def assertStatement(self, statement):
        """Solve the statement for a boolean result"""

        result = self.getStatement(statement)
        # pylint: disable=no-member
        if not (isinstance(result, bool) or isinstance(result, np.bool_)):
            warnings.warn(
                "solution of {} is not an instance of bool".format(statement))
        return result
        # pylint: enable=no-member

    def getStatement(self, statement):
        """Evaluate the statement"""

        return self._parser.solve(statement)
