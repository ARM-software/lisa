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


# pylint: disable=E0611
# pylint: disable=R0201
import os
import tempfile
from unittest import TestCase

from nose.tools import assert_equal

from wlauto.instrumentation.misc import _diff_interrupt_files


class InterruptDiffTest(TestCase):

    def test_interrupt_diff(self):
        file_dir = os.path.join(os.path.dirname(__file__), 'data', 'interrupts')
        before_file = os.path.join(file_dir, 'before')
        after_file = os.path.join(file_dir, 'after')
        expected_result_file = os.path.join(file_dir, 'result')
        output_file = tempfile.mktemp()

        _diff_interrupt_files(before_file, after_file, output_file)
        with open(output_file) as fh:
            output_diff = fh.read()
        with open(expected_result_file) as fh:
            expected_diff = fh.read()
        assert_equal(output_diff, expected_diff)


