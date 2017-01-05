#!/usr/bin/env python
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


"""This is a script to publish a notebook containing Ipython graphs
The static data is published as an anonymous gist. GitHub does not
allow easy deletions of anonymous gists.
"""

import os
import argparse
from IPython.nbformat.sign import TrustNotebookApp
from argparse import RawTextHelpFormatter

# Logging Configuration
import logging
from trappy.plotter import IPythonConf

logging.basicConfig(level=logging.INFO)


def change_resource_paths(txt):
    """Change the resource paths from local to
       Web URLs
    """

    # Replace the path for d3-tip
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/EventPlot/d3.tip.v0.6.3"),
        IPythonConf.D3_TIP_URL)
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/EventPlot/d3.v3.min"),
        IPythonConf.D3_PLOTTER_URL)
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/EventPlot/EventPlot"),
        "https://rawgit.com/sinkap/7f89de3e558856b81f10/raw/46144f8f8c5da670c54f826f0c634762107afc66/EventPlot")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/synchronizer"),
        IPythonConf.DYGRAPH_SYNC_URL)
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/dygraph-combined"),
        IPythonConf.DYGRAPH_COMBINED_URL)
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/ILinePlot"),
        "https://rawgit.com/sinkap/648927dfd6985d4540a9/raw/69d6f1f9031ae3624c15707315ce04be1a9d1ac3/ILinePlot")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/underscore-min"),
        IPythonConf.UNDERSCORE_URL)

    logging.info("Updated Library Paths...")
    return txt


def publish(source, target):
    """Publish the notebook for globally viewable interactive
     plots
    """

    txt = ""

    with open(source, 'r') as file_fh:
        txt = change_resource_paths(file_fh.read())

    with open(target, 'w') as file_fh:
        file_fh.write(txt)

    trust = TrustNotebookApp()
    trust.sign_notebook(target)
    logging.info("Signed and Saved: %s", target)

def main():
    """Command Line Invocation Routine"""

    parser = argparse.ArgumentParser(description="""
    The data for the interactive plots is stored in the  ipython profile.
    In order to make it accessible when the notebook is published or shared,
    a github gist of the data is created and the links in the notebook are
    updated. The library links are also updated to their corresponding publicly
    accessible URLs.
    """,
    prog="publish_interactive_plots.py", formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "-p",
        "--profile",
        help="ipython profile",
        default="default",
        type=str)

    parser.add_argument(
        "-o",
        "--outfile",
        help="name of the output notebook",
        default="",
        type=str)

    parser.add_argument("notebook")
    args = parser.parse_args()

    notebook = args.notebook
    outfile = args.outfile

    if outfile == "":
        outfile = "published_" + os.path.basename(notebook)
        logging.info("Setting outfile as %s", outfile)

    elif not outfile.endswith(".ipynb"):
        outfile += ".ipynb"

    publish(notebook, outfile)

if __name__ == "__main__":
    main()
