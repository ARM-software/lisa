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

"""These are the default plotting Attributes"""
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

WIDTH = 7
"""Default Width of a MatPlotlib Plot"""
LENGTH = 7
"""Default Length of a MatPlotlib Plot"""
PER_LINE = 2
"""Default Graphs per line"""
CONCAT = False
"""Default value for concat in :mod:`trappy.plotter.LinePlot`
and :mod:`trappy.plotter.ILinePlot`
"""
PIVOT = "__TRAPPY_PIVOT_DEFAULT"
"""Default pivot when None specified"""
PIVOT_VAL = "__TRAPPY_DEFAULT_PIVOT_VAL"
"""Default PivotValue for the default pivot"""
XLIM = None
"""Default value for xlimit"""
YLIM = None
"""Default value for ylim"""
FILL = False
"""Default value for "fill" in :mod:`trappy.plotter.LinePlot`
and :mod:`trappy.plotter.ILinePlot`"""
ALPHA = 0.75
"""Default value for the alpha channel"""
TITLE = None
"""Default figure title (no title)"""
TITLE_SIZE = 24
"""Default size for the figure title"""
LEGEND_NCOL = 3
"""Default number of columns in the legend"""

MPL_STYLE = {
    'axes.axisbelow': True,
    'axes.edgecolor': '#bcbcbc',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'axes.labelcolor': '#555555',
    'axes.labelsize': 'large',
    'axes.linewidth': 1.0,
    'axes.titlesize': 'x-large',
    'figure.edgecolor': 'white',
    'figure.facecolor': 'white',
    'figure.figsize': (6.0, 4.0),
    'figure.subplot.hspace': 0.5,
    'font.size': 10,
    'interactive': True,
    'keymap.all_axes': ['a'],
    'keymap.back': ['left', 'c', 'backspace'],
    'keymap.forward': ['right', 'v'],
    'keymap.fullscreen': ['f'],
    'keymap.grid': ['g'],
    'keymap.home': ['h', 'r', 'home'],
    'keymap.pan': ['p'],
    'keymap.save': ['s'],
    'keymap.xscale': ['L', 'k'],
    'keymap.yscale': ['l'],
    'keymap.zoom': ['o'],
    'legend.fancybox': True,
    'lines.antialiased': True,
    'lines.linewidth': 1.0,
    'patch.antialiased': True,
    'patch.edgecolor': '#EEEEEE',
    'patch.facecolor': '#348ABD',
    'patch.linewidth': 0.5,
    'toolbar': 'toolbar2',
    'xtick.color': '#555555',
    'xtick.direction': 'in',
    'xtick.major.pad': 6.0,
    'xtick.major.size': 0.0,
    'xtick.minor.pad': 6.0,
    'xtick.minor.size': 0.0,
    'ytick.color': '#555555',
    'ytick.direction': 'in',
    'ytick.major.pad': 6.0,
    'ytick.major.size': 0.0,
    'ytick.minor.pad': 6.0,
    'ytick.minor.size': 0.0
}

from distutils.version import LooseVersion
import matplotlib

colors = ['#348ABD', '#7A68A6', '#A60628', '#467821', '#CF4457', '#188487',
          '#E24A33']
if LooseVersion(matplotlib.__version__) < LooseVersion("1.5.1"):
    MPL_STYLE['axes.color_cycle'] = colors
else:
    MPL_STYLE['axes.prop_cycle'] = matplotlib.cycler("color", colors)

ARGS_TO_FORWARD = [
    "marker",
    "markersize",
    "markevery",
    "linestyle",
    "linewidth",
    "drawstyle"]
"""kwargs that will be forwarded to matplotlib API calls
"""
HTML_HEIGHT = 400
"""Default height for HTML based plots"""
DEFAULT_SYNC_ZOOM = False
"""Sync Graph zoom by default in
:mod:`trappy.plotter.ILinePlot` graph groups
"""
EVENT_PLOT_STRIDE = False
"""Default value for stride which enables sampled
EventPlots for :mod:`trappy.plotter.EventPlot`
"""
PLOT_SCATTER = False
"""Default value for creating Scatter Plots"""
POINT_SIZE = 2
"""Default Point Size for plots (in pts)"""
LINE_WIDTH = 1.0
"""Default Line Width for plotter"""
