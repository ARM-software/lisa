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

"""Defines a generic indexable ColorMap Class"""
import matplotlib.colors as clrs
import matplotlib.cm as cmx


class ColorMap(object):

    """The Color Map Class to return a gradient method

    :param num_colors: Number or colors for which a gradient
        is needed
    :type num_colors: int
    """

    def __init__(self, num_colors):
        self.color_norm = clrs.Normalize(vmin=0, vmax=num_colors)
        self.scalar_map = cmx.ScalarMappable(norm=self.color_norm, cmap='hsv')
        self.num_colors = num_colors

    def cmap(self, index):
        """
        :param index: Index for the gradient array
        :type index: int

        :return: The color at specified index
        """
        return self.scalar_map.to_rgba(index)

    def cmap_inv(self, index):
        """
        :param index: Index for the gradient array
        :type index: int

        :return: The color at :math:`N_{colors} - i`
        """
        return self.cmap(self.num_colors - index)
