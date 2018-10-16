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

from lisa.platforms.platinfo import PlatformInfo

from lisa.tests.kernel.test_bundle import TestBundle, ResultBundle
from lisa.tests.lisa.utils import create_local_testenv, StorageTestCase

class DummyTestBundle(TestBundle):
    """
    A dummy bundle that only does some simple target interaction
    """
    def __init__(self, res_dir, shell_output):
        plat_info = PlatformInfo()
        super().__init__(res_dir, plat_info)
        self.shell_output = shell_output

    @classmethod
    def _from_testenv(cls, te, res_dir):
        output = te.target.execute('echo $((21+21))').split()
        return cls(res_dir, output)

    def test_output(self):
        passed = False
        for line in self.shell_output:
            if '42' in line:
                passed = True
                break

        return ResultBundle.from_bool(passed)

class BundleCheck(StorageTestCase):
    """
    A test class that verifies some :class:`lisa.tests.kernel.test_bundle.TestBundle`
    base behaviours.
    """
    def setUp(self):
        super().setUp()
        self.te = create_local_testenv()

    def test_init(self):
        """
        Test that creating a dummy bundle works
        """
        bundle = DummyTestBundle(None, "42")

    def test_target_init(self):
        """
        Test that creating a bundle from a target works
        """
        bundle = DummyTestBundle.from_testenv(self.te)

    def test_bundle_serialization(self):
        """
        Test that bundle serialization works correctly
        """
        bundle = DummyTestBundle(None, "42")
        output = bundle.shell_output

        bundle.to_dir(self.res_dir)
        bundle = DummyTestBundle.from_dir(self.res_dir)

        self.assertEqual(output, bundle.shell_output)
