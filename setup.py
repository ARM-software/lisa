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


VERSION = "1.0.0"

LONG_DESCRIPTION = """TRAPpy is a framework written in python for
analysing and plotting FTrace data by converting it into standardised
PANDAS DataFrames (tabular times series data representation).The goal is to
allow developers easy and systematic access to FTrace data and leverage
the flexibility of PANDAS for the analysis.

TRAPpy also provides functionality to build complex statistical analysis
based on the underlying FTrace data.
"""
REQUIRES = [
    "matplotlib>=1.3.1",
    "pandas>=0.13.1",
    "ipython>=3.0.0",
    "jupyter>=1.0.0",
]

data_files = {"trappy.plotter": ["js/EventPlot.js",
                                 "js/ILinePlot.js",
                                 "css/EventPlot.css",
                             ]
}

setup(name='TRAPpy',
      version=VERSION,
      license="Apache v2",
      author="ARM-TRAPPY",
      author_email="trappy@arm.com",
      description="Trace Analysis and Plotting",
      long_description=LONG_DESCRIPTION,
      url="http://arm-software.github.io/trappy",
      packages=find_packages(),
      package_data=data_files,
      scripts=["scripts/publish_interactive_plots.py"],
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
