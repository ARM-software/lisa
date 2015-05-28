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

from nose.tools import assert_equal, raises, assert_true, assert_false

from wlauto import Instrument
from wlauto.core import signal, instrumentation
from wlauto.instrumentation import instrument_is_installed, instrument_is_enabled, clear_instrumentation


class MockInstrument(Instrument):

    name = 'mock'

    def __init__(self):
        Instrument.__init__(self, None)
        self.before = 0
        self.after = 0

    def before_workload_execution(self, context):
        self.before += 1

    def after_workload_execution(self, context):
        self.after += 1


class MockInstrument2(Instrument):

    name = 'mock_2'

    def __init__(self):
        Instrument.__init__(self, None)
        self.before = 0
        self.after = 0
        self.result = 0

    def before_workload_execution(self, context):
        self.before += 1

    def after_workload_execution(self, context):
        self.after += 1

    def after_workload_result_update(self, context):
        self.result += 1


class MockInstrument3(Instrument):

    name = 'mock_3'

    def __init__(self):
        Instrument.__init__(self, None)

    def slow_before_workload_execution(self, context):
        global counter
        counter += 1


class MockInstrument4(Instrument):

    name = 'mock_4'

    def __init__(self):
        Instrument.__init__(self, None)

    def slow_before_first_iteration_boot(self, context):
        global counter
        counter = 4


class MockInstrument5(Instrument):

    name = 'mock_5'

    def __init__(self):
        Instrument.__init__(self, None)

    def fast_before_first_iteration_boot(self, context):
        global counter
        counter += 2


class MockInstrument6(Instrument):

    name = 'mock_6'

    def __init__(self):
        Instrument.__init__(self, None)

    def before_first_iteration_boot(self, context):
        global counter
        counter *= 10


class BadInstrument(Instrument):

    name = 'bad'

    def __init__(self):
        pass

    # Not specifying the context argument.
    def teardown(self):
        pass


counter = 0


class InstrumentationTest(TestCase):

    def tearDown(self):
        clear_instrumentation()

    def test_install(self):
        instrument = _instantiate(MockInstrument)
        instrument2 = _instantiate(MockInstrument2)
        instrumentation.install(instrument)
        instrumentation.install(instrument2)
        signal.send(signal.BEFORE_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_RESULT_UPDATE, self, context=None)
        assert_equal(instrument.before, 1)
        assert_equal(instrument.after, 1)
        assert_equal(instrument2.before, 1)
        assert_equal(instrument2.after, 1)
        assert_equal(instrument2.result, 1)

    def test_enable_disable(self):
        instrument = _instantiate(MockInstrument)
        instrument2 = _instantiate(MockInstrument2)
        instrumentation.install(instrument)
        instrumentation.install(instrument2)

        instrumentation.disable_all()
        signal.send(signal.BEFORE_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_RESULT_UPDATE, self, context=None)
        assert_equal(instrument.before, 0)
        assert_equal(instrument.after, 0)
        assert_equal(instrument2.before, 0)
        assert_equal(instrument2.after, 0)
        assert_equal(instrument2.result, 0)

        instrumentation.enable(instrument)
        signal.send(signal.BEFORE_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_RESULT_UPDATE, self, context=None)
        assert_equal(instrument.before, 1)
        assert_equal(instrument.after, 1)
        assert_equal(instrument2.before, 0)
        assert_equal(instrument2.after, 0)
        assert_equal(instrument2.result, 0)

        instrumentation.enable_all()
        signal.send(signal.BEFORE_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_EXECUTION, self, context=None)
        signal.send(signal.AFTER_WORKLOAD_RESULT_UPDATE, self, context=None)
        assert_equal(instrument.before, 2)
        assert_equal(instrument.after, 2)
        assert_equal(instrument2.before, 1)
        assert_equal(instrument2.after, 1)
        assert_equal(instrument2.result, 1)

    def test_check_enabled(self):
        instrument = _instantiate(MockInstrument)
        instrumentation.install(instrument)
        instrumentation.enable(instrument)
        assert_true(instrument_is_enabled(instrument))
        assert_true(instrument_is_enabled(instrument.name))
        instrumentation.disable(instrument)
        assert_false(instrument_is_enabled(instrument))
        assert_false(instrument_is_enabled(instrument.name))

    def test_local_instrument(self):
        global counter
        counter = 0
        self.install_local_instrument()
        signal.send(signal.BEFORE_WORKLOAD_EXECUTION, self, context=None)
        assert_equal(counter, 1)

    def test_priority_prefix_instrument(self):
        global counter
        counter = 0
        instrument1 = _instantiate(MockInstrument4)
        instrument2 = _instantiate(MockInstrument5)
        instrument3 = _instantiate(MockInstrument6)
        instrumentation.install(instrument1)
        instrumentation.install(instrument2)
        instrumentation.install(instrument3)
        signal.send(signal.BEFORE_FIRST_ITERATION_BOOT, self, context=None)
        assert_equal(counter, 42)

    @raises(ValueError)
    def test_bad_argspec(self):
        instrument = _instantiate(BadInstrument)
        instrumentation.install(instrument)

    def test_check_installed(self):
        instrumentation.install(_instantiate(MockInstrument))
        assert_true(instrument_is_installed('mock'))
        assert_true(instrument_is_installed(MockInstrument))
        assert_false(instrument_is_installed(MockInstrument2))

    def install_local_instrument(self):
        instrument = _instantiate(MockInstrument3)
        instrumentation.install(instrument)

    @raises(ValueError)
    def test_duplicate_install(self):
        instrument = _instantiate(MockInstrument)
        instrument2 = _instantiate(MockInstrument)
        instrumentation.install(instrument)
        instrumentation.install(instrument2)


def _instantiate(cls):
    # Needed to get around Extension's __init__ checks
    return cls()

