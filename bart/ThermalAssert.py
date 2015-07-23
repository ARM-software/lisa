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
# File:        ThermalAssert.py
# ----------------------------------------------------------------
# $
#
"""Allow the user to assert various conditions
based on the grammar defined in cr2.stats.grammar. The class is
also intended to have aggregator based functionality. This is not
implemented yet.
"""

from cr2.stats.grammar import Parser
import warnings
import numpy as np

# pylint: disable=invalid-name

class ThermalAssert(object):

    """
        Args:
            data (cr2.Run): A cr2.Run instance
            config (dict): A dictionary of variables, classes
                and functions that can be used in the statements
    """

    def __init__(self, data, config):
        self._parser = Parser(data, config)

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
