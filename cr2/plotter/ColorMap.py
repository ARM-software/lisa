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
# File:        ColorMap.py
# ----------------------------------------------------------------
# $
#
"""Defines a generic indexable ColorMap Class"""
import matplotlib.colors as clrs
import matplotlib.cm as cmx


class ColorMap(object):

    """The Color Map Class to return a gradient method"""

    def __init__(self, num_colors):
        self.color_norm = clrs.Normalize(vmin=0, vmax=num_colors)
        self.scalar_map = cmx.ScalarMappable(norm=self.color_norm, cmap='hsv')
        self.num_colors = num_colors

    def cmap(self, index):
        """Return the color at index"""
        return self.scalar_map.to_rgba(index)

    def cmap_inv(self, index):
        """Return the inverse color"""
        return self.cmap(self.num_colors - index)
