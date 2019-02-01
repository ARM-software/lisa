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

import logging
import os

from test import LisaTest

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
TESTS_CONF = os.path.join(TESTS_DIRECTORY, "rfc.config")

class RFC(LisaTest):
    """Tests for the Energy-Aware Scheduler"""

    test_conf = TESTS_CONF
    experiments_conf = TESTS_CONF

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(RFC, cls).runExperiments(args, kwargs)

    def test_run(self):
        """A dummy test just to run configured workloads"""
        pass

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
