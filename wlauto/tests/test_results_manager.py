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


# pylint: disable=W0231,W0613,E0611,W0603,R0201
from unittest import TestCase

from nose.tools import assert_equal, assert_true, assert_false, assert_raises

from wlauto.core.result import ResultProcessor, ResultManager
from wlauto.exceptions import WAError


class MockResultProcessor1(ResultProcessor):

    name = 'result_processor_with_exception'

    def process_iteration_result(self, result, context):
        raise Exception()

    def process_run_result(self, result, context):
        raise Exception()


class MockResultProcessor2(ResultProcessor):

    name = 'result_processor_with_wa_error'

    def process_iteration_result(self, result, context):
        raise WAError()

    def process_run_result(self, result, context):
        raise WAError()


class MockResultProcessor3(ResultProcessor):

    name = 'result_processor_with_keybaord_interrupt'

    def process_iteration_result(self, result, context):
        raise KeyboardInterrupt()

    def process_run_result(self, result, context):
        raise KeyboardInterrupt()


class MockResultProcessor4(ResultProcessor):

    name = 'result_processor'

    def __init__(self):
        super(MockResultProcessor4, self).__init__()
        self.is_invoked = False

    def process_iteration_result(self, result, context):
        self.is_invoked = True

    def process_run_result(self, result, context):
        self.is_invoked = True


class ResultManagerTest(TestCase):

    def test_keyboard_interrupt(self):
        processor_keyboard_interrupt = _instantiate(MockResultProcessor3)

        # adding the results processor to the result manager
        manager = ResultManager()
        assert_false(manager.processors)

        # adding the results processor to the result manager
        manager.install(processor_keyboard_interrupt)

        assert_equal(len(manager.processors), 1)
        assert_raises(KeyboardInterrupt, manager.add_result, None, None)

    def test_add_result(self):
        processor_generic_exception = _instantiate(MockResultProcessor1)
        processor_wa_error = _instantiate(MockResultProcessor2)
        processor = _instantiate(MockResultProcessor4)

        # adding the results processor to the result manager
        manager = ResultManager()
        assert_false(manager.processors)

        # adding the results processor to the result manager
        manager.install(processor_generic_exception)
        manager.install(processor_wa_error)
        manager.install(processor)

        assert_equal(len(manager.processors), 3)
        manager.add_result(None, None)

        assert_true(processor.is_invoked)

    def test_process_results(self):
        processor_generic_exception = _instantiate(MockResultProcessor1)
        processor_wa_error = _instantiate(MockResultProcessor2)
        processor = _instantiate(MockResultProcessor4)

        # adding the results processor to the result manager
        manager = ResultManager()
        assert_false(manager.processors)

        # adding the results processor to the result manager
        manager.install(processor_generic_exception)
        manager.install(processor_wa_error)
        manager.install(processor)

        assert_equal(len(manager.processors), 3)
        manager.process_run_result(None, None)

        assert_true(processor.is_invoked)


def _instantiate(cls):
    # Needed to get around Extension's __init__ checks
    return cls()
