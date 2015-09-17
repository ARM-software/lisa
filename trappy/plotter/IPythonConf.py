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

"""IPythonConf provides abstraction for the varying configurations in
different versions of ipython/jupyter packages.
"""
import urllib
import os
import shutil
from distutils.version import StrictVersion as V

IPLOT_RESOURCES = {
    "ILinePlot": [
        "http://cdnjs.cloudflare.com/ajax/libs/dygraph/1.1.1/dygraph-combined.js",
        "js/ILinePlot.js",
        "http://dygraphs.com/extras/synchronizer.js",
        "https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min.js"],
    "EventPlot": [
        "http://d3js.org/d3.v3.min.js",
        "http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3.js",
        "js/EventPlot.js"]}

PROFILE_DIR_IPYTHON_V4 = os.path.expanduser("~/.local/share/jupyter")
"""The location of the IPython webserver in IPython version 4.0+"""
IPYTHON_V4_BASE = "/nbextensions"
"""The webserver base directory for IPython version 4.0+"""
IPYTHON_V3_BASE = "/static"
"""The webserver base directory for IPython version < 4.0"""
PLOTTER_SCRIPTS = "plotter_scripts"
"""The installation directory of plotter JS files in the
IPython webserver"""
PLOTTER_DATA = "plotter_data"
"""The installation directory of plotter data files in the
IPython webserver"""

def install_http_resource(url, to_path):
    """Install a HTTP Resource (eg. javascript) to
    a destination on the disk

    :param url: HTTP URL
    :type url: str

    :param to_path: Destination path on the disk
    :type to_path: str
    """
    try:
        urllib.urlretrieve(url, filename=to_path)
    except IOError:
        raise ImportError("Could not receive Web Resource {}"
                          .format(to_path))


def install_local_resource(from_path, to_path):
    """Move a local resource  to the desired
    a destination.

    :param from_path: Path relative to this file
    :type from_path: str

    :param to_path: Destination path on the disk
    :type to_path: str
    """
    base_dir = os.path.dirname(__file__)
    from_path = os.path.join(base_dir, from_path)
    shutil.copy(from_path, to_path)


def install_resource(from_path, to_path):
    """Install a resource to a location on the disk

    :param from_path: URL or relative path
    :type from_path: str

    :param to_path: Destination path on the disk
    :type to_path: str
    """

    if from_path.startswith("http"):
        if not os.path.isfile(to_path):
            install_http_resource(from_path, to_path)
    else:
        install_local_resource(from_path, to_path)


def iplot_install(module_name):
    """Install the resources for the module to the Ipython
    profile directory

    :param module_name: Name of the module
    :type module_name: str

    :return: A list than can be consumed by requirejs or
        any relative resource dependency resolver
    """

    resources = IPLOT_RESOURCES[module_name]
    for resource in resources:
        resource_name = os.path.basename(resource)
        resource_dest_dir = os.path.join(
            get_scripts_path(),
            module_name)

        # Ensure if the directory exists
        if not os.path.isdir(resource_dest_dir):
            os.mkdir(resource_dest_dir)
        resource_dest_path = os.path.join(resource_dest_dir, resource_name)
        install_resource(resource, resource_dest_path)


def get_ipython():
    """Return an IPython instance. Returns None
    if IPython is not installed"""

    try:
        import IPython
        return IPython.get_ipython()
    except ImportError:
        return None

def check_ipython():
    """A boolean function to check if IPython
    is available"""

    try:
        import IPython
    except ImportError:
        return False

    return True

def get_profile_name():
    """Get the name of the profile of the current IPython
     notebook. This is only relevant to V <= 4.0.0"""

    ipy = get_ipython()
    if not ipy:
        raise ImportError("Cannot Find IPython Profile")

    return ipy.profile

def get_ipython_dir(profile=None):
    """Returns the base directory of the IPython server

    :param profile: The name of the IPython profile
    :type profile: str
    """

    if not check_ipython():
        raise ImportError("Cannot Find IPython Environment")

    import IPython
    # IPython 4.0+ changes the position of files in the profile
    # directory
    if V(IPython.__version__) >= V('4.0.0'):
        return os.path.join(
            PROFILE_DIR_IPYTHON_V4,
            IPYTHON_V4_BASE.strip("/"))
    else:
        if not profile:
            profile = get_profile_name()
        return os.path.join(
            IPython.utils.path.locate_profile(
                profile),
            IPYTHON_V3_BASE.strip("/"))

def add_web_base(path):
    """Add the base of the IPython dependency URLs

    :param path: The path to be augmented with the
        webserver base
    :type path: str
    """

    import IPython
    if V(IPython.__version__) >= V('4.0.0'):
        return os.path.join(IPYTHON_V4_BASE, path)
    else:
        return os.path.join(IPYTHON_V3_BASE, path)

def get_scripts_path(profile=None):
    """Directory where plotter scripts are installed

    :param profile: The name of the IPython profile
    :type profile: str
    """

    dir_name = os.path.join(get_ipython_dir(profile), PLOTTER_SCRIPTS)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

def get_data_path(profile=None):
    """Directory where Plotter Data is stored

    :param profile: The name of the IPython profile
    :type profile: str
    """

    dir_name = os.path.join(get_ipython_dir(profile), PLOTTER_DATA)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name
