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

from setuptools import setup

import importlib
import distutils.cmd
import distutils.log

class SystemCheckCommand(distutils.cmd.Command):
    """A custom command to check some prerequisites on the system."""

    description = 'check some requirements on the system, that cannot be satisfied by installing Python packages in a venv'
    user_options = [
        # The format is (long option, short option, description).
    ]

    def initialize_options(self):
        """Set default values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        self.check_system_install()

    def check_system_install(self):
        missing_pkg_msg = '\n\nThe following packages need to be installed using your Linux distribution package manager. On ubuntu, the following packages are needed:\n    apt-get install {}\n\n'
        distro_pkg_list = list()
        def check_pkg(python_pkg, distro_pkg):
            try:
                importlib.import_module(python_pkg)
            except ImportError:
                distro_pkg_list.append(distro_pkg)

        # Some packages are not always present by default on Debian-based
        # systems, even if there are part of the Python standard library
        check_pkg('tkinter', 'python3-tk')
        check_pkg('ensurepip', 'python3-venv')

        if distro_pkg_list:
            self.announce(
                missing_pkg_msg.format(' '.join(distro_pkg_list)),
                level=distutils.log.INFO
            )

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open("lisa/version.py") as f:
    version_globals = dict()
    exec(f.read(), version_globals)
    lisa_version = version_globals['__version__']

setup(
    name='LISA',
    version=lisa_version,
    author='Arm Ltd',
    # TODO: figure out which email to put here
    # author_email=
    packages=['lisa'],
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
        "matplotlib >= 1.4.2",
        "pandas >= 0.23.0",
        "numpy",
        "scipy",
        "ruamel.yaml >= 0.15.81",

        # Depdendencies that are shipped as part of the LISA repo as
        # subtree/submodule
        "devlib",
        "trappy",
        "bart-py",
    ],

    extras_require={
        "notebook": [
            "ipython",
            "jupyterlab",
            "ipywidgets",
        ],

        "doc": [
            "sphinx >= 1.8",
            "sphinx_rtd_theme"
        ],

        "test": [
            "nose",
        ],
    },

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

    # Add extra subcommands to setup.py
    cmdclass={
        'systemcheck': SystemCheckCommand,
    }
)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
