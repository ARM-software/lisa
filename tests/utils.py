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

from unittest import TestCase
import tempfile
import shutil

from devlib.target import KernelVersion

from lisa.target import Target, TargetConf
from lisa.platforms.platinfo import PlatformInfo


HOST_TARGET_CONF = TargetConf({
    'kind': 'host',
    # Don't load cpufreq, it usually won't work with CI targets
    'devlib': {
        'excluded-modules': ['cpufreq', 'hwmon'],
    },
})

HOST_PLAT_INFO = PlatformInfo({
    # With no cpufreq, we won't be able to do calibration. Provide dummy.
    'rtapp': {
        'calib': {c: 100 for c in list(range(64))},
    },
})

def create_local_target():
    """
    :returns: A localhost :class:`lisa.target.Target` instance
    """
    return Target.from_conf(conf=HOST_TARGET_CONF, plat_info=HOST_PLAT_INFO)

class StorageTestCase(TestCase):
    """
    A base class for tests that also provides a directory
    """
    def setUp(self):
        self.res_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.res_dir)

def nullcontext(enter_result=None):
    """
    Backport of Python 3.7 contextlib.nullcontext
    """

    class CM:
        def __enter__(self):
            return enter_result

        def __exit__(self, *args, **kwargs):
            return

    return CM()
