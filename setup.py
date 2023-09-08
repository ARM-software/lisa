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

import os
import sys
import itertools
import platform

from setuptools import setup, find_packages, find_namespace_packages


plat = platform.system()
if plat != 'Linux':
    raise ValueError(f'Only Linux is supported, {plat} is unsupported')


with open('README.rst', 'r') as f:
    long_description = f.read()

with open('LICENSE.txt', 'r') as f:
    license_txt = f.read()

with open("lisa/version.py") as f:
    version_globals = dict()
    exec(f.read(), version_globals)
    lisa_version = version_globals['__version__']

def make_console_script(name):
    mod_name = name.replace('.py', '')
    cli_name = mod_name.replace('_', '-')
    return f'{cli_name}=lisa._cli_tools.{mod_name}:main'

with os.scandir('lisa/_cli_tools/') as scanner:
    console_scripts = [
        make_console_script(entry.name)
        for entry in scanner
        if entry.name.endswith('.py') and entry.is_file()
    ]

packages = ['lisa'] + [
    f'lisa.{pkg}'
    for pkg in sorted(set(itertools.chain(
        find_namespace_packages(where='lisa'),
        find_packages(where='lisa'),
    )))
]
package_data = {
    package: ['*']
    for package in packages
    if package.startswith('lisa._assets.')
}
package_data['lisa._assets'] = ['*']

extras_require={
    "notebook": [
        "jupyterlab >= 3",
    ],

    "dev": [
        "pip-audit",
        "pytest",
        "build",
        "twine",
        "github3.py",
    ],

    "wa": [
        "wlauto",
    ],
}

extras_require["doc"] = [
    # Force ReadTheDocs to use a recent version, rather than the defaults used
    # for old projects.
    "sphinx > 2",
    "sphinx_rtd_theme >= 0.5.2",
    "sphinxcontrib-plantuml",
    "nbsphinx",

    # Add all the other optional dependencies to ensure all modules from lisa
    # can safely be imported
    *itertools.chain.from_iterable(extras_require.values())
]

# "all" extra requires all to install all the optional dependencies
extras_require['all'] = sorted(set(
    itertools.chain.from_iterable(extras_require.values())
))

python_requires = '>= 3.8'

if __name__ == "__main__":

    setup(
        name='lisa-linux',
        license='Apache License 2.0',
        version=lisa_version,
        maintainer='Arm Ltd.',
        packages=packages,
        url='https://github.com/ARM-software/lisa',
        project_urls={
            "Bug Tracker": "https://github.com/ARM-software/lisa/issues",
            "Documentation": "https://lisa-linux-integrated-system-analysis.readthedocs.io/",
            "Source Code": "https://github.com/ARM-software/lisa",
        },
        description='A stick to probe the kernel with',
        long_description=long_description,
        python_requires=python_requires,
        install_requires=[
            "psutil >= 4.4.2",
            # Figure.savefig() (without pyplot) does not work in matplotlib <
            # 3.1.0, and that is used for non-interactive plots when building the
            # doc.
            "matplotlib >= 3.1.0",
            "bokeh",
            # For bokeh static image exports
            "selenium",
            "phantomjs",
            "pillow",

            "holoviews",
            "panel",
            "colorcet",
            # Pandas >= 1.0.0 has support for new nullable dtypes
            # Pandas 1.2.0 has broken barplots:
            # https://github.com/pandas-dev/pandas/issues/38947
            "pandas >=1.0.0, <3.0",
            "numpy",
            "scipy",
            # Earlier versions have broken __slots__ deserialization
            "ruamel.yaml >= 0.16.6",
            # For the HTML output of analysis plots
            "docutils",
            # To open intersphinx inventories
            "sphobjinv",
            # For pandas.to_parquet() dataframe storage
            "pyarrow",

            # For lisa.trace.PerfettoTraceParser
            "perfetto",

            "ipython",
            "ipywidgets",

            # Depdendencies that are shipped as part of the LISA repo as
            # subtree/submodule
            "devlib >= 1.3.4",

            'jinja2',

            "pyelftools", # To get symbol names in kernel module
            "cffi", # unshare syscall

            "typeguard",
            "pycparserext", # For the kernel module build
        ],

        extras_require=extras_require,
        package_data=package_data,
        classifiers=[
            "Programming Language :: Python :: 3 :: Only",
            # This is not a standard classifier, as there is nothing defined for
            # Apache 2.0 yet:
            # https://pypi.org/classifiers/
            # It has not been tested under any other OS
            "Operating System :: POSIX :: Linux",

            "Topic :: System :: Operating System Kernels :: Linux",
            "Topic :: Software Development :: Testing",
            "Intended Audience :: Developers",
        ],
        entry_points={
            'console_scripts': console_scripts,
        },
    )

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
