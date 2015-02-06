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
# File:        AbstractDataPlotter.py
# ----------------------------------------------------------------
# $
#
"""This is the template class that all Plotters inherit"""
from abc import abstractmethod
# pylint: disable=R0921
# pylint: disable=R0903


class AbstractDataPlotter(object):

    """This is an Abstract Data Plotting Class defining an interface
       for the various Plotting Classes"""

    @abstractmethod
    def view(self):
        """View the graph"""
        raise NotImplementedError("Method Not Implemented")

    @abstractmethod
    def savefig(self, path):
        """Save the image as a file"""
        raise NotImplementedError("Method Not Implemented")
