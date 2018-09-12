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
    name='LISA',
    version='yolo',
    author='Arm Ltd',
    # TODO: figure out which email to put here
    # author_email=
    packages=['lisa'],
    # url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='A stick to probe the kernel with',
    long_description=long_description,
    # obfuscated way of asking for Python >= 2.7 or >= 3.4
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*, !=3.3.*',
    install_requires=[
        "Cython >= 0.25.1",
        "psutil >= 4.4.2",
        "wrapt >= 1.10.8",
        "matplotlib >= 1.4.2, < 3.0",
        "pandas",
        "numpy",
        "future",
        "ruamel.yaml",
        "devlib",
        "trappy",
        "py-bart",
    ],
    extras_require={
        "notebook": [
            "ipython < 6.0.0", # 6.0+ only supports Python 3.3
            "jupyter"
        ],
    }
)
