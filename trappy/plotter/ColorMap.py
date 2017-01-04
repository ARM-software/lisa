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

"""Defines a generic indexable ColorMap Class"""
import matplotlib.colors as clrs
import matplotlib.cm as cmx
from matplotlib.colors import ListedColormap, Normalize


def to_dygraph_colors(color_map):
    """Convert a color_map specified as a list of rgb tuples to the
    syntax that dygraphs expect: ["rgb(1, 2, 3)", "rgb(4, 5, 6)",...]

    :param color_map: a list of rgb tuples
    :type color_map: list of tuples
    """

    rgb_list = ["rgb(" + ", ".join(str(i) for i in e) + ")" for e in color_map]

    return '["' + '", "'.join(rgb_list) + '"]'

class ColorMap(object):

    """The Color Map Class to return a gradient method

    :param num_colors: Number or colors for which a gradient
        is needed
    :type num_colors: int
    """

    def __init__(self, num_colors, cmap='hsv'):
        self.color_norm = clrs.Normalize(vmin=0, vmax=num_colors)
        self.scalar_map = cmx.ScalarMappable(norm=self.color_norm, cmap=cmap)
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

    @classmethod
    def rgb_cmap(cls, rgb_list):
        """Constructor for a ColorMap from an rgb_list

        :param rgb_list: A list of rgb tuples for red, green and blue.
            The rgb values should be in the range 0-255.
        :type rgb_list: list of tuples
        """

        rgb_list = [[x / 255.0 for x in rgb[:3]] for rgb in rgb_list]

        rgb_map = ListedColormap(rgb_list, name='default_color_map', N=None)
        num_colors = len(rgb_list)

        return cls(num_colors, cmap=rgb_map)
