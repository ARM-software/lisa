#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from setuptools import setup, find_namespace_packages

import importlib
import distutils.cmd
import distutils.log

with open('README.rst', 'r') as fh:
    long_description = fh.read()

with open("lisa/version.py") as f:
    version_globals = dict()
    exec(f.read(), version_globals)
    lisa_version = version_globals['__version__']


packages = find_namespace_packages(include=['lisa*'])
package_data = {
    package: ['*']
    for package in packages
    if package.startswith('lisa.assets.')
}
package_data['lisa.assets'] = ['*']

setup(
    name='LISA',
    version=lisa_version,
    author='Arm Ltd',
    # TODO: figure out which email to put here
    # author_email=
    packages=packages,
    url='https://github.com/ARM-software/lisa',
    project_urls={
        "Bug Tracker": "https://github.com/ARM-software/lisa/issues",
        "Documentation": "https://lisa-linux-integrated-system-analysis.readthedocs.io/",
        "Source Code": "https://github.com/ARM-software/lisa",
    },
    license='LICENSE.txt',
    description='A stick to probe the kernel with',
    long_description=long_description,
    python_requires='>= 3.5',
    install_requires=[
        "psutil >= 4.4.2",
        # Figure.savefig() (without pyplot) does not work in
        # matplotlib < 3.1.0, and that is used for non-interactive plots when
        # building the doc. Unfortunately, extra_requires does not allow
        # overriding that, and recent versions don't support Python 3.5
        # anymore. Since don't want to drop support for Python 3.5 for now, so
        # we mandate a lower version that what is actually required.
        "matplotlib >= 1.4.2",
        "pandas >= 0.23.0",
        "numpy",
        "scipy",
        "ruamel.yaml >= 0.15.81",
        "docutils", # For the HTML output of analysis plots

        # Depdendencies that are shipped as part of the LISA repo as
        # subtree/submodule
        "devlib",
        "trappy",
    ],

    extras_require={
        "notebook": [
            "ipython",
            "jupyterlab",
            "ipywidgets",
            "ipympl", # For %matplotlib widget under jupyter lab
            "sphobjinv", # To open intersphinx inventories
        ],

        "doc": [
            "sphinx >= 1.8",
            "sphinx_rtd_theme",
            "sphinxcontrib-plantuml",
            "nbsphinx",
            # necessary to import some modules
            "ipython",
        ],

        "test": [
            "nose",
        ],
    },

    package_data=package_data,
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        # This is not a standard classifier, as there is nothing defined for
        # Apache 2.0 yet:
        # https://pypi.org/classifiers/
        "License :: OSI Approved :: Apache 2.0",
        # It has not been tested under any other OS
        "Operating System :: POSIX :: Linux",

        "Topic :: System :: Operating System Kernels :: Linux",
        "Topic :: Software Development :: Testing",
        "Intended Audience :: Developers",
    ],
)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
