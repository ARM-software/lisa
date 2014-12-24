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
