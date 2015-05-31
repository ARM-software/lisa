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
# File:        AttrConf.py
# ----------------------------------------------------------------
# $
#
"""These are the default plotting Attributes"""
WIDTH = 7
LENGTH = 7
PER_LINE = 2
CONCAT = False
PIVOT = "__CR2_PIVOT_DEFAULT"
PIVOT_VAL = "__CR2_DEFAULT_PIVOT_VAL"
DUPLICATE_VALUE_MAX_DELTA = 0.000001
XLIM = None
YLIM = None
FILL = False
ALPHA = 0.75

MPL_STYLE = {
    'axes.axisbelow': True,
    'axes.color_cycle': ['#348ABD',
                         '#7A68A6',
                         '#A60628',
                         '#467821',
                         '#CF4457',
                         '#188487',
                         '#E24A33'],
    'axes.edgecolor': '#bcbcbc',
    'axes.facecolor': '#eeeeee',
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
ARGS_TO_FORWARD = [
    "marker",
    "markersize",
    "markevery",
    "linestyle",
    "linewidth",
    "drawstyle"]

HTML_WIDTH =  900
HTML_HEIGHT = 400

IPLOT_RESOURCES = {
    "ILinePlot": [
        "http://cdnjs.cloudflare.com/ajax/libs/dygraph/1.1.1/dygraph-combined.js",
        "js/ILinePlot.js"],
    "EventPlot": [
        "http://d3js.org/d3.v3.min.js",
        "http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3.js",
        "js/EventPlot.js"]}

try:
    import IPython
    import os
    ip = IPython.get_ipython()
    if not ip:
        PLOTTER_IPYTHON = False
    else:
        PLOTTER_IPYTHON = True
        PLOTTER_IPYTHON_PROFILE_DIR = ip.config.ProfileDir["location"]
        PLOTTER_STATIC_DATA_DIR = os.path.join(
            PLOTTER_IPYTHON_PROFILE_DIR,
            "static", "plotter_data")
        PLOTTER_SCRIPTS_DIR = "plotter_scripts"
        PLOTTER_SCRIPTS_PATH = os.path.join(
            PLOTTER_IPYTHON_PROFILE_DIR,
            "static",
            PLOTTER_SCRIPTS_DIR)

        if not os.path.isdir(PLOTTER_STATIC_DATA_DIR):
            os.mkdir(PLOTTER_STATIC_DATA_DIR)
        if not os.path.isdir(PLOTTER_SCRIPTS_PATH):
            os.mkdir(PLOTTER_SCRIPTS_PATH)
except:
    PLOTTER_IPYTHON = False
