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

"""This is the template class that all Plotters inherit"""
from abc import abstractmethod, ABCMeta
from pandas import DataFrame
from trappy.plotter.Utils import listify
from functools import reduce
# pylint: disable=R0921
# pylint: disable=R0903


class AbstractDataPlotter(object):

    __metaclass__ = ABCMeta

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

    def _check_data(self):
        """Internal function to check the received data"""

        data = listify(self.runs)

        if len(data):
            mask = map(lambda x: isinstance(x, DataFrame), data)
            data_frame = reduce(lambda x, y: x and y, mask)
            if not data_frame and not self.templates:
                raise ValueError(
                    "Cannot understand data. Accepted DataFormats are pandas.DataFrame and trappy.Run (with templates)")
        else:
            raise ValueError("Empty Data received")
