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

from lisa.tests.base import TestBundle, ResultBundle
from .utils import create_local_target, StorageTestCase

class TestBundle(TestBundle):
    """
    Dummy proxy class so that pytest does not try to build it, since it
    contains "Test" in the name.
    """
    __test__ = False

class DummyTestBundle(TestBundle):
    """
    A dummy bundle that only does some simple target interaction
    """
    __test__ = False

    def __init__(self, res_dir, shell_output):
        plat_info = PlatformInfo()
        super().__init__(res_dir, plat_info)
        self.shell_output = shell_output

    @classmethod
    def _from_target(cls, target, *, res_dir, collector=None) -> 'DummyTestBundle':
        with collector:
            output = target.execute('echo $((21+21))').split()
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
    A test class that verifies some :class:`lisa.tests.base.TestBundle`
    base behaviours.
    """

    def setup_method(self, method):
        super().setup_method(method)
        self.target = create_local_target()

    def test_init(self):
        """
        Test that creating a dummy bundle works
        """
        bundle = DummyTestBundle("/foo", "42")

    def test_target_init(self):
        """
        Test that creating a bundle from a target works
        """
        bundle = DummyTestBundle.from_target(self.target)

    def test_serialization(self):
        """
        Test that bundle serialization works correctly
        """
        bundle = DummyTestBundle("/foo", "42")
        output = bundle.shell_output

        bundle.to_dir(self.res_dir)
        bundle = DummyTestBundle.from_dir(self.res_dir)

        assert output == bundle.shell_output
