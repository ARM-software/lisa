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

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='bisector',
    version='1.0',
    author='Arm Ltd',
    # TODO: figure out which email to put here
    # author_email=
    packages=['bisector'],
    # url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Command execution sequencer',
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'bisector=bisector.bisector:main',
        ],
    },
    python_requires='>= 3.5',
    install_requires=[
        # Older versions will have troubles with serializing complex nested
        # objects hierarchy implementing custom __getstate__ and __setstate__
        "ruamel.yaml >= 0.15.72",
        "pandas",
        "scipy",
        "requests",
    ],

    extras_require={
        'dbus': [
            'pydbus',
            'pygobject',
            # You will also need gobject-introspection package from your
            # distribution
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        # This is not a standard classifier, as there is nothing defined for
        # Apache 2.0 yet:
        # https://pypi.org/classifiers/
        "License :: OSI Approved :: Apache 2.0",
        # It has not been tested under any other OS
        "Operating System :: POSIX :: Linux",

        "Topic :: Software Development :: Testing",
        "Intended Audience :: Developers",
    ],
)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
