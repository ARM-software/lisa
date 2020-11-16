# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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

"""Base classes and utilities for self-testing LISA's wlgen packages"""

import os
import shutil
import tempfile
from unittest import TestCase

from devlib import LocalLinuxTarget, Platform

dummy_calibration = {}


class TestTarget(LocalLinuxTarget):
    """
    Devlib target for self-testing LISA

    Uses LocalLinuxTarget configured to disallow using root.
    Adds facility to record the commands that were executed for asserting LISA
    behaviour.
    """

    def __init__(self):
        self.execute_calls = []
        working_directory = tempfile.mkdtemp()
        super().__init__(platform=Platform(),
                         working_directory=working_directory,
                         executables_directory=os.path.join(working_directory, 'bin'),
                         load_default_modules=False,
                         connection_settings={'unrooted': True})

    def execute(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        return super().execute(*args, **kwargs)

    @property
    def executed_commands(self):
        return [args[0] if args else kwargs['command']
                for args, kwargs in self.execute_calls]

    def clear_execute_calls(self):
        self.execute_calls = []


class WlgenBase(TestCase):
    """
    Base class for wlgen self-tests

    Creates and sets up a TestTarget.

    Provides directory paths to use for output files. Deletes those paths if
    they already exist, to try and provide a clean test environment. This
    doesn't create those paths, tests should create them if necessary.
    """

    tools = []
    """Tools to install on the 'target' before each test"""

    @property
    def target_run_dir(self):
        """Unique directory to use for creating files on the 'target'"""
        return os.path.join(self.target.working_directory,
                            'lisa_target_{}'.format(self.__class__.__name__))

    @property
    def host_out_dir(self):
        """Unique directory to use for creating files on the host"""
        return os.path.join(
            os.getenv('LISA_HOME'), 'results',
            'lisa_selftest_out_{}'.format(self.__class__.__name__))

    def setup_method(self, method):
        self.target = TestTarget()

        tools_path = os.path.join(os.getenv('LISA_HOME'),
                                  'tools', self.target.abi)
        self.target.setup([os.path.join(tools_path, tool)
                           for tool in self.tools])

        if self.target.directory_exists(self.target_run_dir):
            self.target.remove(self.target_run_dir)

        if os.path.isdir(self.host_out_dir):
            shutil.rmtree(self.host_out_dir)

        self.target.clear_execute_calls()

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
