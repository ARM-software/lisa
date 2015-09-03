#!/usr/bin/env python
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


"""This is a script to publish a notebook containing Ipython graphs
The static data is published as an anonymous gist. GitHub does not
allow easy deletions of anonymous gists.
"""

import json
import os
import re
import requests
import argparse
import urlparse
from IPython.nbformat.sign import TrustNotebookApp
from requests.auth import HTTPBasicAuth
from argparse import RawTextHelpFormatter
from ConfigParser import ConfigParser

# Logging Configuration
import logging
from trappy.plotter import IPythonConf

logging.basicConfig(level=logging.INFO)

RAWGIT = "rawgit.com"
GITHUB_API_URL = "https://api.github.com/gists"


def change_resource_paths(txt):
    """Change the resource paths from local to
       Web URLs
    """

    # Replace the path for d3-tip
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/EventPlot/d3.tip.v0.6.3"),
        "http://labratrevenge.com/d3-tip/javascripts/d3.tip.v0.6.3")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/EventPlot/d3.v3.min"),
        "http://d3js.org/d3.v3.min")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/EventPlot/EventPlot"),
        "https://rawgit.com/sinkap/7f89de3e558856b81f10/raw/46144f8f8c5da670c54f826f0c634762107afc66/EventPlot")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/synchronizer"),
        "http://dygraphs.com/extras/synchronizer")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/dygraph-combined"),
        "http://cdnjs.cloudflare.com/ajax/libs/dygraph/1.1.1/dygraph-combined")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/ILinePlot"),
        "https://rawgit.com/sinkap/648927dfd6985d4540a9/raw/69d6f1f9031ae3624c15707315ce04be1a9d1ac3/ILinePlot")
    txt = txt.replace(
        IPythonConf.add_web_base("plotter_scripts/ILinePlot/underscore-min"),
        "https://cdnjs.cloudflare.com/ajax/libs/underscore.js/1.8.3/underscore-min")

    logging.info("Updated Library Paths...")
    return txt


def get_url_from_response(response, file_name):
    """Get the URL of gist from GitHub API response"""

    resp_data = response.json()
    url = resp_data["files"][file_name]["raw_url"]
    url = list(urlparse.urlsplit(url))
    url[1] = RAWGIT
    url = urlparse.urlunsplit(url)

    logging.info("gist created at: %s", url)
    return url


def fig_to_json(fig, profile):
    """Get the underlying data file from figure name"""

    data_dir = IPythonConf.get_data_path(profile)

    return os.path.expanduser(
        os.path.join(
            data_dir,
            fig +
            ".json"))


def create_new_gist(fig, profile, login):
    """Create a new gist for the data of the figure"""

    path = fig_to_json(fig, profile)
    file_name = os.path.basename(path)

    with open(path) as file_h:
        content = file_h.read()

    data = {}
    data["description"] = "Gist Data: {}".format(file_name)
    data["public"] = True
    data["files"] = {}
    data["files"][file_name] = {}
    data["files"][file_name]["content"] = content
    response = requests.post(GITHUB_API_URL, data=json.dumps(data), auth=login)
    return get_url_from_response(response, file_name)


def publish(source, target, profile, login):
    """Publish the notebook for globally viewable interactive
     plots
    """

    regex = r"(ILinePlot|EventPlot)\.generate\(\'(fig_.{32})\', '\/(nbextensions|static)\/'\)"
    txt = ""

    with open(source, 'r') as file_fh:

        for line in file_fh:
            match = re.search(regex, line)
            if match:
                plot = match.group(1)
                fig = match.group(2)
                logging.info("Publishing %s : %s", plot, fig)
                line = re.sub(
                    regex,
                    plot + ".generate('" + fig + "', '" +
                    create_new_gist(fig, profile, login) + "')",
                    line)
            txt += line

        txt = change_resource_paths(txt)

    with open(target, 'w') as file_fh:
        file_fh.write(txt)

    trust = TrustNotebookApp()
    trust.sign_notebook(target)
    logging.info("Signed and Saved: %s", target)

def read_login_config(config_file):
    """returns an HTTPBasicAuth object if the
    config exists"""

    if not config_file:
        logging.debug("Anonymous gists will be created")
        return None

    with open(config_file, 'r') as c_fh:
        config = ConfigParser()
        config.readfp(c_fh)
        username = config.get("login", "username")
        token = config.get("login", "token")

        logging.info("Received Login info for: %s", username)

    return HTTPBasicAuth(username, token)

def main():
    """Command Line Invocation Routine"""

    parser = argparse.ArgumentParser(description="""
    The data for the interactive plots is stored in the  ipython profile.
    In order to make it accessible when the notebook is published or shared,
    a github gist of the data is created and the links in the notebook are
    updated. The library links are also updated to their corresponding publicly
    accessible URLs.

    The login credentials can be added to a config file as follows

        1. Go to settings in your github profile and create a 'Personal Access Token'
        2. This token can be used in place of your password for BasicAuth APIs
        3. Create a config file:

        [login]
        username=<your github username>
        token=<personal access token>

    and pass the path to the file as -c <config>.
    The gists can then be viewed in the corresponding github account.

    The absence of this will create an anonymous gist which cannot be deleted/managed.""",
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

    parser.add_argument(
        "-c",
        "--config",
        help="The path to a config file containing github login credentials",
        default=None,
        type=str)

    parser.add_argument("notebook")
    args = parser.parse_args()

    profile = args.profile
    notebook = args.notebook
    outfile = args.outfile
    config = args.config
    login = read_login_config(config)

    if outfile == "":
        outfile = "published_" + os.path.basename(notebook)
        logging.info("Setting outfile as %s", outfile)

    elif not outfile.endswith(".ipynb"):
        outfile += ".ipynb"

    publish(notebook, outfile, profile, login)

if __name__ == "__main__":
    main()
