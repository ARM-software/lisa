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
# File:        __init__.py
# ----------------------------------------------------------------
# $
#

from cr2.compare_runs import summary_plots, compare_runs
from cr2.run import Run
from cr2.plotter.LinePlot import LinePlot
try:
    from cr2.plotter.ILinePlot import ILinePlot
    from cr2.plotter.EventPlot import EventPlot
except ImportError:
    pass
from cr2.dynamic import register_dynamic, register_class

# Load all the modules to make sure all classes are registered with Run
import os
for fname in os.listdir(os.path.dirname(__file__)):
    import_name, extension = os.path.splitext(fname)
    if (extension == ".py") and (fname != "__init__.py"):
        __import__("cr2.{}".format(import_name))

del fname, import_name, extension
