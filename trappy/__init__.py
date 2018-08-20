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
from __future__ import unicode_literals

from builtins import object
import warnings
from trappy.bare_trace import BareTrace
from trappy.compare_runs import summary_plots, compare_runs
from trappy.exception import TrappyParseError
from trappy.ftrace import FTrace
from trappy.systrace import SysTrace
from trappy.version import __version__
try:
    from trappy.plotter.LinePlot import LinePlot
except ImportError as exc:
    class LinePlot(object):
        def __init__(self, *args, **kwargs):
            raise exc
try:
    from trappy.plotter.ILinePlot import ILinePlot
    from trappy.plotter.EventPlot import EventPlot
    from trappy.plotter.BarPlot import BarPlot
except ImportError:
    pass
from trappy.dynamic import register_dynamic_ftrace, register_ftrace_parser, \
    unregister_ftrace_parser
import trappy.nbexport

# We define unregister_dynamic_ftrace() because it undoes what
# register_dynamic_ftrace().  Internally it does exactly the same as
# unregister_ftrace_parser() though but with these two names the API
# makes more sense: register with register_dynamic_ftrace(),
# unregister with unregister_dynamic_ftrace()
unregister_dynamic_ftrace = unregister_ftrace_parser

# Load all the modules to make sure all classes are registered with FTrace
import os
for fname in os.listdir(os.path.dirname(__file__)):
    import_name, extension = os.path.splitext(fname)
    if (extension == ".py") and (fname != "__init__.py") and \
       (fname != "plot_utils.py"):
        __import__("trappy.{}".format(import_name))

del fname, import_name, extension
