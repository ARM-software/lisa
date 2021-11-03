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

import imp
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


devlib_dir = os.path.join(os.path.dirname(__file__), 'devlib')

sys.path.insert(0, os.path.join(devlib_dir, 'core'))

# happends if falling back to distutils
warnings.filterwarnings('ignore', "Unknown distribution option: 'install_requires'")
warnings.filterwarnings('ignore', "Unknown distribution option: 'extras_require'")

try:
    os.remove('MANIFEST')
except OSError:
    pass


vh_path = os.path.join(devlib_dir, 'utils', 'version.py')
# can load this, as it does not have any devlib imports
version_helper = imp.load_source('version_helper', vh_path)
__version__ = version_helper.get_devlib_version()
commit = version_helper.get_commit()
if commit:
    __version__ = '{}+{}'.format(__version__, commit)


packages = []
data_files = {}
source_dir = os.path.dirname(__file__)
for root, dirs, files in os.walk(devlib_dir):
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

with open("README.rst", "r") as fh:
    long_description = fh.read()

params = dict(
    name='devlib',
    description='A library for interacting with and instrumentation of remote devices.',
    long_description=long_description,
    version=__version__,
    packages=packages,
    package_data=data_files,
    url='https://github.com/ARM-software/devlib',
    license='Apache v2',
    maintainer='ARM Ltd.',
    install_requires=[
        'python-dateutil',  # converting between UTC and local time.
        'pexpect>=3.3',  # Send/recieve to/from device
        'pyserial',  # Serial port interface
        'paramiko', # SSH connection
        'scp', # SSH connection file transfers
        'wrapt',  # Basic for construction of decorator functions
        'future', # Python 2-3 compatibility
        'enum34;python_version<"3.4"', # Enums for Python < 3.4
        'contextlib2;python_version<"3.0"', # Python 3 contextlib backport for Python 2
        'numpy<=1.16.4; python_version<"3"',
        'numpy; python_version>="3"',
        'pandas<=0.24.2; python_version<"3"',
        'pandas; python_version>"3"',
        'lxml', # More robust xml parsing
    ],
    extras_require={
        'daq': ['daqpower>=2'],
        'doc': ['sphinx'],
        'monsoon': ['python-gflags'],
        'acme': ['pandas', 'numpy'],
    },
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)

all_extras = list(chain(iter(params['extras_require'].values())))
params['extras_require']['full'] = all_extras


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
            self.distribution.get_version = lambda : __version__.split('+')[0]
        orig_sdist.run(self)


params['cmdclass'] = {'sdist': sdist}

setup(**params)
