# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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
import logging

from glob import glob
from inspect import isclass
from importlib import import_module

from collections import namedtuple


class Workload(object):
    """
    Base class for Android related workloads
    """
    _availables = None

    _AW = namedtuple('AndroidWorkload',
          ['module_name', 'module', 'class_name', 'ctor'])

    @staticmethod
    def get(te, name='YouTube'):
        """
        Get a reference to the specified Android workload
        """
        if Workload._availables is None:
            Workload.availables(te.target)
        # Build list of case insensitive workload names
        log = logging.getLogger('Workload')
        if name not in Workload._availables:
            log.warning('Workload [%s] not available on target', name)
            return None
        return Workload._availables[name].ctor(te)

    @staticmethod
    def availables(target):
        """
        List the supported android workloads which are available on the target
        """
        if Workload._availables:
            return Workload._availables.keys()

        Workload._availables = {}

        log = logging.getLogger('Workload')
        log.debug('Building list of available workloads...')

        # Add workloads dir to system path
        workloads_dir = os.path.dirname(os.path.abspath(__file__))
        workloads_dir = os.path.join(workloads_dir, 'workloads')
        log.debug('Workdir: %s', workloads_dir)

        sys.path.insert(0, workloads_dir)
        log.debug('Syspath: %s', sys.path)

        for filepath in glob(os.path.join(workloads_dir, '*.py')):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            log.debug('Filename: %s', filename)

            # Ignore __init__ files
            if filename.startswith('__'):
                continue

            # Import the module for inspection
            module = import_module(filename)
            for member in dir(module):
                # Ignore the base class
                if member == 'Workload':
                    continue
                handler = getattr(module, member)
                if handler and isclass(handler) and \
                   issubclass(handler, Workload):
                    class_name = handler.__name__
                    module_name = module.__name__
                    # Check if a package is required and is available on target
                    aw = Workload._AW(module_name, module, class_name, handler)
                    if (Workload._is_available(target, aw)):
                        # Keep track of classes which are 'Android.Workload'
                        Workload._availables[class_name] = aw

        return Workload._availables.keys()

    @staticmethod
    def _is_available(target, aw):
        try:
            package = getattr(aw.ctor, 'package')
        except AttributeError:
            # Assume workloads not requiring a package
            # are always available
            return True

        # Check for the package being available
        count = target.execute('pm list packages | grep {} | wc -l'\
                               .format(package))
        if int(count) >= 1:
            return True

        log = logging.getLogger('Workload')
        log.warning('Package [%s] not installed', package)
        log.warning('Workload [%s] disabled', aw.class_name)
        return False

    def __init__(self, test_env):
        """
        Initialized workloads available on the specified test environment

        test_env: target test environmen
        """
        self.te = test_env
        self.target = test_env.target
        self._log = logging.getLogger('Workload')

        wloads = Workload.availables(self.target)
        self._log.info('Workloads available on target:')
        self._log.info('  %s', wloads)

    def _adb(self, cmd):
        return 'adb -s {} {}'.format(self.target.adb_name, cmd)


    def run(self, exp_dir, **kwargs):
        raise RuntimeError('Not implemeted')


# vim :set tabstop=4 shiftwidth=4 expandtab
