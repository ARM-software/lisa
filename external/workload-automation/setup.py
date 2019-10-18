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

import os
import sys
import warnings
from itertools import chain

try:
    from setuptools import setup
    from setuptools.command.sdist import sdist as orig_sdist
except ImportError:
    from distutils.core import setup
    from distutils.command.sdist import sdist as orig_sdist


wa_dir = os.path.join(os.path.dirname(__file__), 'wa')

sys.path.insert(0, os.path.join(wa_dir, 'framework'))
from version import (get_wa_version, get_wa_version_with_commit,
                     format_version, required_devlib_version)

# happens if falling back to distutils
warnings.filterwarnings('ignore', "Unknown distribution option: 'install_requires'")
warnings.filterwarnings('ignore', "Unknown distribution option: 'extras_require'")

try:
    os.remove('MANIFEST')
except OSError:
    pass

packages = []
data_files = {'': [os.path.join(wa_dir, 'commands', 'postgres_schema.sql')]}
source_dir = os.path.dirname(__file__)
for root, dirs, files in os.walk(wa_dir):
    rel_dir = os.path.relpath(root, source_dir)
    data = []
    if '__init__.py' in files:
        for f in files:
            if os.path.splitext(f)[1] not in ['.py', '.pyc', '.pyo']:
                data.append(f)
        package_name = rel_dir.replace(os.sep, '.')
        package_dir = root
        packages.append(package_name)
        data_files[package_name] = data
    else:
        # use previous package name
        filepaths = [os.path.join(root, f) for f in files]
        data_files[package_name].extend([os.path.relpath(f, package_dir) for f in filepaths])

scripts = [os.path.join('scripts', s) for s in os.listdir('scripts')]

with open("README.rst", "r") as fh:
    long_description = fh.read()

devlib_version = format_version(required_devlib_version)
params = dict(
    name='wlauto',
    description='A framework for automating workload execution and measurement collection on ARM devices.',
    long_description=long_description,
    version=get_wa_version_with_commit(),
    packages=packages,
    package_data=data_files,
    include_package_data=True,
    scripts=scripts,
    url='https://github.com/ARM-software/workload-automation',
    license='Apache v2',
    maintainer='ARM Architecture & Technology Device Lab',
    maintainer_email='workload-automation@arm.com',
    setup_requires=[
        'numpy<=1.16.4; python_version<"3"',
        'numpy; python_version>="3"',
    ],
    install_requires=[
        'python-dateutil',  # converting between UTC and local time.
        'pexpect>=3.3',  # Send/receive to/from device
        'pyserial',  # Serial port interface
        'colorama',  # Printing with colors
        'pyYAML>=5.1b3',  # YAML-formatted agenda parsing
        'requests',  # Fetch assets over HTTP
        'devlib>={}'.format(devlib_version),  # Interacting with devices
        'louie-latest',  # callbacks dispatch
        'wrapt',  # better decorators
        'pandas>=0.23.0,<=0.24.2; python_version<"3.5.3"',  # Data analysis and manipulation
        'pandas>=0.23.0; python_version>="3.5.3"',  # Data analysis and manipulation
        'future',  # Python 2-3 compatiblity
    ],
    dependency_links=['https://github.com/ARM-software/devlib/tarball/master#egg=devlib-{}'.format(devlib_version)],
    extras_require={
        'test': ['nose', 'mock'],
        'notify': ['notify2'],
        'doc': ['sphinx', 'sphinx_rtd_theme'],
        'postgres': ['psycopg2-binary'],
        'daq': ['daqpower'],
    },
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)

all_extras = list(chain(iter(params['extras_require'].values())))
params['extras_require']['all'] = all_extras


class sdist(orig_sdist):

    user_options = orig_sdist.user_options + [
        ('strip-commit', 's',
         "Strip git commit hash from package version ")
    ]

    def initialize_options(self):
        orig_sdist.initialize_options(self)
        self.strip_commit = False

    def run(self):
        if self.strip_commit:
            self.distribution.get_version = get_wa_version
        orig_sdist.run(self)


params['cmdclass'] = {'sdist': sdist}

setup(**params)
