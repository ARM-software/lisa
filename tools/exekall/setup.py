#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, Arm Limited and contributors.
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

with open('README.rst') as fh:
    long_description = fh.read()

setup(
    name='exekall',
    version='1.0',
    maintainer='Arm Ltd.',
    packages=['exekall'],
    url='https://github.com/ARM-software/lisa',
    project_urls={
        "Bug Tracker": "https://github.com/ARM-software/lisa/issues",
        "Documentation": "https://lisa-linux-integrated-system-analysis.readthedocs.io/",
        "Source Code": "https://github.com/ARM-software/lisa",
    },
    license='LICENSE.txt',
    description='Python expression execution engine',
    long_description=long_description,
    entry_points={
        'console_scripts': ['exekall=exekall.main:main'],
    },
    python_requires='>= 3.5',
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        # This is not a standard classifier, as there is nothing defined for
        # Apache 2.0 yet:
        # https://pypi.org/classifiers/
        "License :: OSI Approved :: Apache 2.0",
        # It has not been tested under any other OS
        "Operating System :: POSIX :: Linux",

        "Intended Audience :: Developers",
    ],
)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
