#    Copyright 2013-2015 ARM Limited
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


import warnings
from distutils.core import setup

import daqpower


warnings.filterwarnings('ignore', "Unknown distribution option: 'install_requires'")

params = dict(
    name='daqpower',
    version=daqpower.__version__,
    packages=[
        'daqpower',
    ],
    scripts=[
        'scripts/run-daq-server',
        'scripts/send-daq-command',
    ],
    url='N/A',
    maintainer='workload-automation',
    maintainer_email='workload-automation@arm.com',
    install_requires=[
        'twisted',
        'PyDAQmx',
    ],
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: Other/Proprietary License',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.7',
    ],
)

setup(**params)
