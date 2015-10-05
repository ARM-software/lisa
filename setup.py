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
from setuptools import setup, find_packages


VERSION = "1.0.1"

LONG_DESCRIPTION = """Behavioural Analysis involves the expressing the general
expectation of the state of the system while targeting a single or set of heuristics.
This is particularly helpful when there are large number of factors that can change
the behaviour of the system and testing all permutations of these input parameters
is impossible. In such a scenario an assertion of the final expectation can be
useful in managing performance and regression.

The Behavioural Analysis and Regression Toolkit is based on TRAPpy. The primary goal is
to assert behaviours using the FTrace output from the kernel
"""

REQUIRES = [
    "TRAPpy==1.0.0",
]

setup(name='bart-py',
      version=VERSION,
      license="Apache v2",
      author="ARM-BART",
      author_email="bart@arm.com",
      description="Behavioural Analysis and Regression Toolkit",
      long_description=LONG_DESCRIPTION,
      url="http://arm-software.github.io/bart",
      packages=find_packages(),
      include_package_data=True,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Environment :: Web Environment",
          "Environment :: Console",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python :: 2.7",
          # As we depend on trace data from the Linux Kernel/FTrace
          "Topic :: System :: Operating System Kernels :: Linux",
          "Topic :: Scientific/Engineering :: Visualization"
      ],
      install_requires=REQUIRES
      )
