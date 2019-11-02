# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
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

from lisa.utils import nullcontext
from lisa.trace import TraceEventChecker, AndTraceEventChecker, OrTraceEventChecker, MissingTraceEventError

""" A test suite for event checking infrastructure."""


class TestEventCheckerBase:
    """
    A test class that verifies checkers work as expected
    """
    EVENTS_SET = {'foo', 'bar', 'baz'}
    expected_success = True

    def test_check_events(self):
        if self.expected_success:
            cm = nullcontext()
        else:
            cm = self.assertRaises(MissingTraceEventError)

        with cm:
            print('Checking: {}'.format(self.checker))
            self.checker.check_events(self.EVENTS_SET)


class TestEventChecker_and1(TestEventCheckerBase, TestCase):
    checker = AndTraceEventChecker.from_events(['foo', 'bar'])


class TestEventChecker_and2(TestEventCheckerBase, TestCase):
    checker = AndTraceEventChecker.from_events(['foo', 'lancelot'])
    expected_success = False


class TestEventChecker_or1(TestEventCheckerBase, TestCase):
    checker = OrTraceEventChecker.from_events(['foo', 'bar'])


class TestEventChecker_or2(TestEventCheckerBase, TestCase):
    checker = OrTraceEventChecker.from_events(['foo', 'lancelot'])


class TestEventChecker_or3(TestEventCheckerBase, TestCase):
    checker = OrTraceEventChecker.from_events(['arthur', 'lancelot'])
    expected_success = False


class TestEventChecker_single1(TestEventCheckerBase, TestCase):
    checker = TraceEventChecker('bar')


class TestEventChecker_single2(TestEventCheckerBase, TestCase):
    checker = TraceEventChecker('non-existing-event')
    expected_success = False


class TestEventChecker_and3(TestEventCheckerBase, TestCase):
    checker = AndTraceEventChecker.from_events([
        TestEventChecker_and1.checker,
        TestEventChecker_or1.checker,
    ])


class TestEventChecker_and4(TestEventCheckerBase, TestCase):
    checker = AndTraceEventChecker.from_events([
        TestEventChecker_and1.checker,
        TestEventChecker_or2.checker,
    ])


class TestEventChecker_and5(TestEventCheckerBase, TestCase):
    checker = AndTraceEventChecker.from_events([
        TestEventChecker_and1.checker,
        TestEventChecker_and2.checker,
    ])
    expected_success = False

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
